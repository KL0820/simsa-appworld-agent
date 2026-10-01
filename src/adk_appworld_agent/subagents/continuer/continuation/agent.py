from __future__ import annotations

import json
import sys
import time
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, ClassVar

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_appworld_agent.contracts.continuation import ContinuationDecision
from adk_appworld_agent.contracts.limits import (
    EXECUTOR_CODE_DIAG_MAX_LENGTH,
    FRAMEWORK_OVERRIDE_PREVIEW_MAX_LENGTH,
    HEADER_DISPLAY_CAP,
    MAX_HISTORY_CYCLES,
    RATIONALE_MAX_LENGTH,
    STDOUT_EXCERPT_MAX_LENGTH,
    STDOUT_FIELD_PREVIEW_MAX_LENGTH,
)
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    RunConfig,
    model_config_from_env,
)
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.continuer.continuation.prompts import (  # noqa: F401
    CONTINUATION_SYSTEM_PROMPT,
)
from adk_appworld_agent.subagents.failure_codes import (
    PLANNER_LLM_RAISED,
    PROVIDER_RATE_LIMITED,
)
from adk_appworld_agent.subagents.utils.data_summarizer import summarize_data_for_llm
from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
    retry_agent_stream,
)


def _fallback_decision(active_idx: int, milestone_count: int) -> ContinuationDecision:
    # When the parse fails, the framework's premature-SUBMIT guard will gate
    # this; here we just emit RETRY which keeps active unchanged regardless of
    # where we are. If milestones are degenerate (count==0) ABORT instead.
    if milestone_count == 0:
        return ContinuationDecision(
            next_action="ABORT",
            rationale="fallback: no milestones to run",
        )
    return ContinuationDecision(
        next_action="RETRY",
        rationale="fallback: retry active milestone",
    )


def _safe_model_config_snapshot(config: object) -> dict[str, object]:
    """Snapshot of inner LlmAgent's generate_content_config for io.input.model_input.

    Mirror of rough_planner._safe_model_config_snapshot so continuation's
    io_record carries the same metadata.
    """
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        try:
            dumped = config.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    snapshot: dict[str, object] = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "candidate_count",
        "seed",
        "max_output_tokens",
    ):
        value = getattr(config, key, None)
        if value is not None:
            snapshot[key] = value
    return snapshot


def _extract_usage(event: Event) -> dict[str, int] | None:
    """Pull token usage off a Gemini event; mirror of rough_planner._extract_usage."""
    usage = getattr(event, "usage_metadata", None)
    if usage is None:
        return None
    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    completion_tokens = getattr(usage, "candidates_token_count", None) or 0
    thoughts_tokens = getattr(usage, "thoughts_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None) or (
        prompt_tokens + completion_tokens + thoughts_tokens
    )
    if (
        prompt_tokens == 0
        and completion_tokens == 0
        and thoughts_tokens == 0
        and total_tokens == 0
    ):
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": total_tokens,
    }


def _parse_decision(
    text: str, *, active_idx: int, milestone_count: int
) -> ContinuationDecision:
    try:
        data = json.loads(text)
        return ContinuationDecision.model_validate(data)
    except Exception as exc:
        sys.stderr.write(
            f"[continuation] failed to parse output: {exc}, falling back\n"
        )
        sys.stderr.flush()
        return _fallback_decision(active_idx, milestone_count)


def _build_inner_llm_agent(model_cfg: ModelConfig) -> LlmAgent:
    config = types.GenerateContentConfig(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=1,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
        thinking_config=thinking_config_from_env(),
    )
    return LlmAgent(
        name="continuation_llm",
        model=model_cfg.name,
        description="History-aware continuation controller; picks the next action.",
        instruction=CONTINUATION_SYSTEM_PROMPT,
        output_schema=ContinuationDecision,
        generate_content_config=config,
    )


def _collapse_call_headers_line(rest: list[dict]) -> str:
    """Summarize the trailing API calls beyond HEADER_DISPLAY_CAP.

    Callers must group by (app, api_name) before invoking this — the
    uniform-API path produces an informative line like:
        ... and 4 more spotify.show_album (album_id in [11, 12, 13, 14])
    The mixed-API path is defensive only (reached if a legacy caller
    bypasses grouping). It enumerates the API set so the reader can
    still see what was hidden, instead of the unhelpful prior wording
    "mixed apps/apis" that vanished entire API names from the prompt
    (425a494-style trap).
    """
    k = len(rest)
    apps = {c.get("app") for c in rest}
    apis = {c.get("api_name") for c in rest}
    if len(apps) != 1 or len(apis) != 1:
        api_breakdown = ", ".join(
            sorted({f"{c.get('app')}.{c.get('api_name')}" for c in rest})
        )
        return f"    ... and {k} more calls ({api_breakdown})"
    app = next(iter(apps))
    api_name = next(iter(apis))
    varying_keys: dict[str, list] = {}
    kw_lists = [c.get("kwargs") or {} for c in rest]
    all_keys: set[str] = set()
    for kw in kw_lists:
        all_keys.update(kw.keys())
    all_keys.discard("access_token")
    for key in all_keys:
        values = [kw.get(key) for kw in kw_lists]
        if len(set(repr(v) for v in values)) > 1:
            varying_keys[key] = values
    if not varying_keys:
        return f"    ... and {k} more {app}.{api_name} (identical kwargs)"
    parts = ", ".join(f"{kk} in {vv!r}" for kk, vv in varying_keys.items())
    return f"    ... and {k} more {app}.{api_name} ({parts})"


def _render_api_trace_lines(lines: list[str], api_trace: object) -> None:
    """Append api_trace evidence under an EXECUTE entry, using the
    generic data summarizer helper for distribution / range / sample output.

    api_trace is a list of normalized api_calls (see appworld_tools.
    summarize_trace_for_prompt). Empty list → render explicit "no API
    invoked" signal (useful for P_LLM_SURRENDER detection).

    Replaces the legacy first-N + 16-key-priority renderer with one that:
      - Lists each call's app.api_name(kwargs) + result_shape
      - Aggregates items across all calls
      - Hands the aggregate to summarize_data_for_llm → schema +
        distributions + ranges + samples (no AppWorld-specific keywords)
    """
    if api_trace is None or not isinstance(api_trace, list):
        return
    if not api_trace:
        lines.append("  api_trace: [] (executor did NOT invoke any apis.* call)")
        return

    lines.append("  api_trace (supporting evidence):")
    real_calls = [
        c for c in api_trace if isinstance(c, dict) and "_more_calls" not in c
    ]
    sentinel_calls = [
        c for c in api_trace if isinstance(c, dict) and "_more_calls" in c
    ]

    # Per-call headers: group by (app, api_name) preserving first-seen
    # order, cap each group at HEADER_DISPLAY_CAP. A global cap would
    # let a high-volume API (e.g. paginated show_liked_songs ×4) eat all
    # the slots and collapse a different API (e.g. show_album ×11) into
    # "... N more (mixed apps/apis)", hiding the second API's name
    # entirely — 425a494-style trap where continuation_planner concluded
    # "show_album was not called" from this prompt and ordered RETRY for
    # 3 cycles. Within each group, original sequence indices are
    # preserved so the reader can still correlate with the time axis.
    groups: "dict[tuple[str, str], list[tuple[int, dict]]]" = {}
    for idx, call in enumerate(real_calls):
        key = (call.get("app", "?"), call.get("api_name", "?"))
        groups.setdefault(key, []).append((idx, call))
    for (app, api_name), entries in groups.items():
        shown = entries[:HEADER_DISPLAY_CAP]
        rest_entries = entries[HEADER_DISPLAY_CAP:]
        for idx, call in shown:
            kwargs = call.get("kwargs") or {}
            kw_repr = ", ".join(
                f"{k}={v!r}" for k, v in kwargs.items() if k != "access_token"
            )
            shape = call.get("result_shape") or {}
            shape_label = _format_shape_label(shape, call.get("_truncated", False))
            lines.append(f"    [{idx}] {app}.{api_name}({kw_repr}) → {shape_label}")
            status = call.get("status")
            if status and status != "ok":
                lines.append(f"        status={status}")
                err = call.get("error")
                if err:
                    lines.append(f"        error={err}")
        if rest_entries:
            rest_calls = [c for _, c in rest_entries]
            lines.append(_collapse_call_headers_line(rest_calls))
    for sentinel in sentinel_calls:
        more = sentinel.get("_more_calls")
        if more:
            lines.append(f"    (... {more} more calls truncated by sandbox)")

    # Aggregate per (app, api_name) group — one schema/distribution/sample
    # block per API. The previous "pick the bigger pool" path silently
    # dropped the smaller pool when one run mixed list-returning and
    # dict-returning APIs (e.g. 4× show_liked_songs returning list[20] →
    # 43 items in the list pool, 11× show_album returning a dict each →
    # 11 items in the dict pool; list pool won, the 11 album dicts +
    # their `genre` field vanished from the prompt). Per-group rendering
    # surfaces every API's schema independently so continuation_planner
    # sees what each API actually returned.
    for (app, api_name), entries in groups.items():
        group_items: list = []
        group_dicts: list[dict] = []
        for _, call in entries:
            ri = call.get("result_items")
            if isinstance(ri, dict) and "items" in ri:
                group_items.extend(ri.get("items") or [])
            elif (
                isinstance(ri, dict)
                and not ri.get("_oversize")
                and not ri.get("_serialization_failed")
            ):
                group_dicts.append(ri)
        if not group_items and not group_dicts:
            continue
        if group_items:
            data_for_helper: Any = group_items
            header = (
                f"    --- aggregated by {app}.{api_name} "
                f"({len(entries)} call(s), {len(group_items)} items) ---"
            )
        elif len(group_dicts) == 1:
            # Single dict return — pass the dict itself so the helper goes
            # through summarize_dict (schema + value). Wrapping as list[1]
            # would trigger summarize_list_of_dicts where N=1 degenerates:
            # every field is trivially "constant" and the constants
            # section dumps each field's full value.
            data_for_helper = group_dicts[0]
            header = f"    --- {app}.{api_name} single-object return ---"
        else:
            data_for_helper = group_dicts
            header = (
                f"    --- aggregated by {app}.{api_name} "
                f"({len(entries)} call(s), {len(group_dicts)} items) ---"
            )
        lines.append(header)
        for line in summarize_data_for_llm(data_for_helper).splitlines():
            lines.append(f"    {line}")


def _format_stdout_field(v: Any) -> str:
    """Format one field of the 4-field stdout for prompt display.

    - None → 'null'
    - scalar (bool / int / float / str): raw value (strings capped)
    - dict / list: route through summarize_data_for_llm so nested
      structures show schema + distributions + samples instead of being
      first-N truncated. Consistent with Entry 7 prior_variables.
    """
    if v is None:
        return "null"
    if isinstance(v, (bool, int, float)):
        return repr(v)
    if isinstance(v, str):
        return (
            v
            if len(v) <= STDOUT_FIELD_PREVIEW_MAX_LENGTH
            else v[:STDOUT_FIELD_PREVIEW_MAX_LENGTH] + "…"
        )
    # dict / list / other JSON-serializable → helper
    rendered = summarize_data_for_llm(v)
    return "\n" + "\n".join("      " + line for line in rendered.splitlines())


def _format_shape_label(shape: dict, truncated: bool) -> str:
    """Render result_shape as 'list[N]' / 'object(N keys)' / 'scalar' label."""
    t = shape.get("type", "?")
    if t == "list":
        cnt = shape.get("count_hint", "?")
        suffix = " (truncated)" if truncated else ""
        return f"list[{cnt}]{suffix}"
    if t == "object":
        keys = shape.get("keys") or []
        return f"object({len(keys)} keys)"
    return str(t)


def _summarize_value_for_oneliner(value: Any) -> str:
    """Brief outcome label for an EXECUTE entry's committed value.

    Used in condensed (older) cycle render. Examples:
      null / count=12 / dict / list[5] / str
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"int={value}"
    if isinstance(value, float):
        return f"float={value}"
    if isinstance(value, str):
        return f"str(len={len(value)})"
    if isinstance(value, dict):
        return f"dict({len(value)} keys)"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _render_cycle_oneliner(cycle_n: int, entries: list[dict]) -> str:
    """Condense one cycle's entries into a single line for older cycles.

    Format: `cycle N  EXEC m<X> <outcome> + PLAN <decision> ✱revised?`
    Skips FIND entries (usually uninformative when successful).
    """
    exec_part = ""
    plan_part = ""

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        phase = entry.get("phase_completed")
        success = entry.get("success")
        m_idx = entry.get("milestone_index")
        agent_output = entry.get("agent_output") or {}

        if phase == "EXECUTE":
            if success:
                stdout_json = agent_output.get("stdout_json")
                if isinstance(stdout_json, dict):
                    outcome = _summarize_value_for_oneliner(stdout_json.get("value"))
                else:
                    outcome = "ok"
            else:
                fc = entry.get("failure_code") or "FAILED"
                outcome = fc
            mid = f"m{m_idx}" if m_idx is not None else ""
            exec_part = f"EXEC {mid} {outcome}".strip()
        elif phase == "PLAN":
            decision = agent_output.get("next_action") or "?"
            revised = agent_output.get("revised_milestones_applied")
            plan_part = f"PLAN {decision}"
            if revised:
                plan_part += " ✱revised milestones"

    parts = [p for p in (exec_part, plan_part) if p]
    return (
        f"  cycle {cycle_n}  " + " + ".join(parts)
        if parts
        else f"  cycle {cycle_n}  (no recorded action)"
    )


def _render_self_assess_lines(lines: list[str], agent_output: dict) -> None:
    """Surface the executor's ADVISORY self-assess hint into a history block.

    The executor's self-check is grounded in the real api_trace. It is NOT a
    verdict — the continuation weighs it against §3a-3f (it can flag a result
    that is actually correct; see the §3b mutation-null exception). Two cases:

    - ``ok is False``: the executor finalized a result but its own check says
      the milestone is NOT actually accomplished. This is the high-signal case
      on a CLAIMED-DONE (success) entry — render it prominently so §3 does not
      ADVANCE/SUBMIT on a false positive.
    - ``ok`` is not False but a ``problem`` is noted: a softer advisory hint.

    A plain OK self-assess with no problem adds nothing and is omitted.
    """
    self_assess = agent_output.get("self_assess")
    if not isinstance(self_assess, dict):
        return
    ok = self_assess.get("ok")
    prob = (self_assess.get("problem") or "").strip()
    if ok is False:
        msg = prob or (self_assess.get("summary") or "").strip() or "(no detail)"
        lines.append(
            f"  ⚠ executor self-assess: NOT OK (HIGH-PRIORITY override of success=true; see §3g): {msg}"
        )
    elif prob:
        lines.append(
            f"  • executor self-assess hint (advisory; weigh vs §3a-3g): {prob}"
        )


def _render_history_entry(entry: dict) -> str:
    """Render one CycleHistoryEntry as a focused block.

    EXECUTE-failure entries surface code + parse_error + stdout_excerpt
    prominently so the LLM can diagnose the prior bug instead of retrying
    blind. Success entries get a one-line summary.
    """
    cycle_n = entry.get("cycle_n")
    phase = entry.get("phase_completed")
    m_idx = entry.get("milestone_index")
    success = entry.get("success")
    failure_code = entry.get("failure_code")
    agent_output = entry.get("agent_output") or {}

    header = f"cycle={cycle_n} {phase}"
    if m_idx is not None:
        header += f" m{m_idx}"
    header += " success" if success else f" FAILED ({failure_code or '?'})"
    if (
        phase == "FIND"
        and success
        and isinstance(agent_output, dict)
        and agent_output.get("reused_from_prior_cycle")
    ):
        header += " (skipped, reused from prior cycle)"

    lines = [header]

    if phase == "EXECUTE" and not success and isinstance(agent_output, dict):
        code = agent_output.get("code")
        if code:
            lines.append("  Prior code emitted by executor:")
            for cl in str(code)[:EXECUTOR_CODE_DIAG_MAX_LENGTH].splitlines():
                lines.append(f"    {cl}")
        parse_error = agent_output.get("parse_error")
        if parse_error:
            lines.append(f"  parse_error: {parse_error}")
        stdout = agent_output.get("stdout_excerpt")
        if stdout:
            lines.append("  stdout_excerpt:")
            for sl in str(stdout)[:STDOUT_EXCERPT_MAX_LENGTH].splitlines():
                lines.append(f"    {sl}")
        llm_raised = agent_output.get("llm_raised")
        if llm_raised:
            lines.append(f"  llm_raised: {llm_raised}")
        rep = agent_output.get("repair_attempt_count")
        if rep:
            lines.append(f"  inner repair attempts: {rep}")
        _render_self_assess_lines(lines, agent_output)
        api_trace = agent_output.get("api_trace")
        _render_api_trace_lines(lines, api_trace)
    elif phase == "EXECUTE" and success and isinstance(agent_output, dict):
        stdout_json = agent_output.get("stdout_json")
        variables = agent_output.get("variables") or []
        if isinstance(stdout_json, dict):
            lines.append("  stdout (4-field):")
            for key in ("value", "summary", "description", "answer"):
                lines.append(f"    {key}: {_format_stdout_field(stdout_json.get(key))}")
            error = stdout_json.get("error")
            if error:
                lines.append(f"  ⚠ stdout error key: {error}")
        elif agent_output.get("summary"):
            # Fallback when stdout_json missing/malformed: at least show summary.
            lines.append(f"  summary: {agent_output.get('summary')}")
        if variables:
            names = [v.get("name") if isinstance(v, dict) else None for v in variables]
            lines.append(f"  variables committed: {[n for n in names if n]}")
        # P1 wiring: a CLAIMED-DONE (success) milestone can still be flagged
        # ok=False by the executor's own grounded self-check. Previously this
        # branch dropped self_assess entirely, so the continuation never saw
        # the problem and ADVANCEd on a false positive. Surface it here.
        _render_self_assess_lines(lines, agent_output)
        api_trace = agent_output.get("api_trace")
        _render_api_trace_lines(lines, api_trace)
    elif phase == "PLAN" and isinstance(agent_output, dict):
        parts = [f"  decision: {agent_output.get('next_action')}"]
        if agent_output.get("revised_milestones_applied"):
            parts.append("(with revised_milestones)")
        rationale = (agent_output.get("rationale") or "")[:RATIONALE_MAX_LENGTH]
        if rationale:
            parts.append(f"({rationale})")
        fo = agent_output.get("framework_override")
        if fo:
            parts.append(
                f"[framework_override: {str(fo)[:FRAMEWORK_OVERRIDE_PREVIEW_MAX_LENGTH]}]"
            )
        lines.append(" ".join(parts))

    return "\n".join(lines)


def _build_history_block(history: list[dict]) -> str:
    """Render history as two-tier sliding window:
      - Last MAX_HISTORY_CYCLES cycles: full detail (existing render)
      - Older cycles: one-liner per cycle

    Token cost goes from O(N cycles × full entry size) to O(N) lines for
    older + O(MAX_HISTORY_CYCLES × full) for recent. Avoids prompt bloat
    on long stuck loops (9dabbc9_2 cycle 14 was 60-75k chars history).
    """
    if not history:
        return "(empty)"

    max_cycle = max(
        (h.get("cycle_n", 0) for h in history if isinstance(h, dict)), default=0
    )
    recent_threshold = max_cycle - MAX_HISTORY_CYCLES + 1

    older_by_cycle: dict[int, list[dict]] = {}
    recent_entries: list[dict] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        cycle_n = h.get("cycle_n", 0)
        if cycle_n < recent_threshold:
            older_by_cycle.setdefault(cycle_n, []).append(h)
        else:
            recent_entries.append(h)

    parts: list[str] = []
    if older_by_cycle:
        for cycle_n in sorted(older_by_cycle.keys()):
            parts.append(_render_cycle_oneliner(cycle_n, older_by_cycle[cycle_n]))
    if recent_entries:
        if parts:
            parts.append("")  # blank separator before full-detail block
        parts.append("\n\n".join(_render_history_entry(h) for h in recent_entries))
    return "\n".join(parts)


def _render_prompt(metadata: dict, instruction: str) -> str:
    cycle_n = metadata.get("cycle_n", 0)
    active_idx = metadata.get("active_milestone_index", 0)
    milestones = metadata.get("milestones") or []
    history = metadata.get("history") or []
    prior_vars = metadata.get("prior_variables_preview") or {}
    last_framework_override = metadata.get("last_framework_override")

    milestone_lines = []
    for i, m in enumerate(milestones):
        intent = m.get("intent", "") if isinstance(m, dict) else ""
        marker = " ← active" if i == active_idx else ""
        milestone_lines.append(f"  [{i}] {intent}{marker}")
    milestones_block = "\n".join(milestone_lines) or "  (no milestones)"

    if history:
        history_block = _build_history_block(history)
    else:
        history_block = "(empty)"

    # prior_variables_preview is already a markdown string from
    # variable_store.summary() (capped at PRIOR_VARIABLES_MAX_LENGTH at source).
    # No json.dumps wrap here — wrapping would escape \n into literal "\n"
    # and break the markdown structure for the LLM.
    prior_vars_block = (
        prior_vars if isinstance(prior_vars, str) and prior_vars else "(none)"
    )

    override_block = ""
    if last_framework_override:
        override_block = (
            f"⚠ FRAMEWORK OVERRIDE FROM LAST CYCLE: {last_framework_override}\n\n"
            "Your previous decision was rejected by the framework. Read the override\n"
            "message above and pick a DIFFERENT action this round.\n\n"
        )

    # No-structural-progress nudge (escalates before the framework's hard
    # give-up). Fires below the give-up window so continuation can course-correct.
    streak = int(metadata.get("no_progress_streak") or 0)
    no_progress_block = ""
    if streak >= 2:
        no_progress_block = (
            f"⚠ NO STRUCTURAL PROGRESS: the executor has produced structurally "
            f"identical work (same APIs called + same committed value) {streak}× in a "
            f"row on the active milestone. Rewording is NOT moving it. Either change "
            f"the STRUCTURAL approach — describe a different operation / different "
            f"state to read or change (never name an API; §1a) — or ADVANCE accepting "
            f"the best value so far. If the streak continues the framework gives up "
            f"on this milestone.\n\n"
        )

    return (
        f"{override_block}"
        f"{no_progress_block}"
        f"Task instruction:\n{instruction}\n\n"
        f"cycle_n: {cycle_n}\n"
        f"active_milestone_index: {active_idx}\n"
        f"milestone_count: {len(milestones)}\n\n"
        f"Current milestones:\n{milestones_block}\n\n"
        f"History (most recent cycles):\n{history_block}\n\n"
        f"Prior variables:\n{prior_vars_block}\n"
    )


class ContinuationPlannerSubagent(BaseSubagent):
    """History-aware controller invoked on cycles >= 1."""

    phase_: ClassVar[Phase] = Phase.PLAN

    inner_agent: LlmAgent

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        metadata = subagent_input.metadata or {}
        milestones = metadata.get("milestones") or []
        active_idx = metadata.get("active_milestone_index", 0)

        prompt = _render_prompt(metadata, subagent_input.task_context.instruction)

        session_service = InMemorySessionService()
        app_name = f"continuation_{subagent_input.task_context.task_id or 'task'}"
        user_id = "continuation"
        session_id = f"cont_{uuid.uuid4().hex[:8]}"
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        runner = Runner(
            agent=self.inner_agent,
            session_service=session_service,
            app_name=app_name,
        )

        last_text: str = ""
        t0 = time.monotonic()
        llm_calls = 0
        llm_call_attempts = 0
        usage_event_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        thoughts_tokens = 0
        total_tokens = 0

        async def _attempt(attempt: int) -> None:
            nonlocal session_id, last_text
            nonlocal llm_calls, llm_call_attempts, usage_event_count
            nonlocal prompt_tokens, completion_tokens, thoughts_tokens, total_tokens
            if attempt > 0:
                session_id = f"cont_{uuid.uuid4().hex[:8]}"
                await session_service.create_session(
                    app_name=app_name, user_id=user_id, session_id=session_id
                )
                last_text = ""
            llm_call_attempts += 1
            async with aclosing(
                runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=types.Content(
                        role="user", parts=[types.Part(text=prompt)]
                    ),
                )
            ) as agen:
                async for event in agen:
                    usage = _extract_usage(event)
                    if usage is not None:
                        llm_calls += 1
                        usage_event_count += 1
                        prompt_tokens += usage["prompt_tokens"]
                        completion_tokens += usage["completion_tokens"]
                        thoughts_tokens += usage["thoughts_tokens"]
                        total_tokens += usage["total_tokens"]
                    text = content_to_text(event.content)
                    if text:
                        last_text = text

        rate_limit_exhausted = False
        try:
            llm_raised = await retry_agent_stream(_attempt, label="continuation")
        except RateLimitExhausted as exc:
            llm_raised = str(exc)
            rate_limit_exhausted = True
        wall_ms = int((time.monotonic() - t0) * 1000)
        metrics = {
            "wall_ms": wall_ms,
            "llm_calls": llm_calls,
            "llm_call_attempts": llm_call_attempts,
            "usage_event_count": usage_event_count,
            "timeout_count": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "total_tokens": total_tokens,
        }

        decision = _parse_decision(
            last_text, active_idx=active_idx, milestone_count=len(milestones)
        )

        payload: dict = {
            "next_action": decision.next_action,
            "revised_milestones": (
                [m.model_dump() for m in decision.revised_milestones]
                if decision.revised_milestones is not None
                else None
            ),
            "rationale": decision.rationale,
        }
        io_block: dict[str, Any] = {}
        if metadata.get("capture_io"):
            from adk_appworld_agent.subagents.utils.io_format import (
                agent_model_input_block,
            )

            io_block = {
                "input": {
                    "rendered_prompt": prompt,
                    "subagent_input": subagent_input.model_dump(mode="json"),
                    "model_input": agent_model_input_block(self.inner_agent, prompt),
                },
                "output": {
                    "raw_llm_text": last_text,
                    "parsed_decision": decision.model_dump(mode="json"),
                },
            }

        if llm_raised is not None:
            payload["llm_raised"] = llm_raised
            env = self.failed(
                attempt=subagent_input.attempt,
                payload=payload,
                failure_code=(
                    PROVIDER_RATE_LIMITED
                    if rate_limit_exhausted
                    else PLANNER_LLM_RAISED
                ),
            )
            env.metrics = metrics
            env.io = io_block
            yield env
            return

        env = self.succeeded(attempt=subagent_input.attempt, payload=payload)
        env.metrics = metrics
        env.io = io_block
        yield env


def build_continuation_planner_subagent(
    name: str = "continuation_planner_subagent",
    *,
    run_config: "RunConfig | None" = None,
) -> ContinuationPlannerSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    inner = _build_inner_llm_agent(model_cfg)
    return ContinuationPlannerSubagent(
        name=name,
        description="History-aware continuation controller for cycles >= 1.",
        inner_agent=inner,
    )


class DeterministicContinuationStub(BaseSubagent):
    """Stub continuation: advance through milestones, then SUBMIT.

    Reads `active_milestone_index` + `milestones` from metadata.
    If the active milestone has a success entry in history:
      - if active is the last milestone → SUBMIT
      - otherwise → ADVANCE (next active = active + 1)
    If not yet succeeded → RETRY (keep active).
    """

    phase_: ClassVar[Phase] = Phase.PLAN

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        metadata = subagent_input.metadata or {}
        active_idx = int(metadata.get("active_milestone_index", 0))
        milestones = metadata.get("milestones") or []
        history = metadata.get("history") or []

        # most recent EXECUTE result for active milestone
        active_succeeded = False
        for h in reversed(history):
            if (
                isinstance(h, dict)
                and h.get("milestone_index") == active_idx
                and h.get("phase_completed") == "EXECUTE"
            ):
                active_succeeded = bool(h.get("success"))
                break

        last_idx = max(0, len(milestones) - 1)

        if not milestones:
            next_action = "ABORT"
            rationale = "stub: no milestones to run"
        elif active_succeeded and active_idx >= last_idx:
            next_action = "SUBMIT"
            rationale = "stub: all milestones satisfied"
        elif active_succeeded:
            next_action = "ADVANCE"
            rationale = "stub: advance to next milestone"
        else:
            next_action = "RETRY"
            rationale = "stub: retry active milestone"

        payload = {
            "next_action": next_action,
            "revised_milestones": None,
            "rationale": rationale,
        }
        yield self.succeeded(attempt=subagent_input.attempt, payload=payload)


def build_continuation_stub_subagent(
    name: str = "continuation_planner_subagent_stub",
) -> DeterministicContinuationStub:
    return DeterministicContinuationStub(
        name=name,
        description="Deterministic continuation stub for testing the cycle loop.",
    )
