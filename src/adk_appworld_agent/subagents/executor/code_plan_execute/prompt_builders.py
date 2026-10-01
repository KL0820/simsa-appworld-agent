"""Prompt & input-rendering builders for the code_plan_execute executor.

Pure string-building helpers (Stage-1/Stage-2 prompts, prior-attempt and
self-assess rendering) extracted from agent.py in the 2026-06-22
behavior-preserving split. Imports are pruned to what this module uses."""

from __future__ import annotations

import json

from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.contracts.limits import (
    PRIOR_ATTEMPT_PARSE_ERROR_MAX_LENGTH,
    PRIOR_ATTEMPT_STDOUT_EXCERPT_MAX_LENGTH,
    PRIOR_ATTEMPT_SUMMARY_MAX_LENGTH,
    PRIOR_ATTEMPT_VALUE_PREVIEW_MAX_LENGTH,
    RATIONALE_MAX_LENGTH,
)
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.subagents.finder.api_specs import get_api_spec
from adk_appworld_agent.subagents.utils.data_summarizer import (
    _infer_schema,
    _render_sample_item,
    summarize_data_for_llm,
)

_PRIOR_ATTEMPTS_CAP = 2

# Self-assess (observe-then-conclude). Calibrated to flag ONLY on concrete
# evidence — defaults to ok=true — because a prior verifier over-flagged
# correct answers (5 gain / 7 loss). The grounding is the real api_trace
# returns + the produced value, not introspection.
SELF_ASSESS_SYS_PROMPT = """\
You describe — grounded in the REAL observed data, not the executor's narration
— what one completed AppWorld milestone actually produced. This is ADVISORY: it
helps the continuation planner; it does NOT gate the milestone, so a precise,
honest hint matters more than catching every edge case.

Return ONLY valid JSON: {"ok": <bool>, "summary": "<one sentence, what actually
happened, grounded in the result>", "problem": "<if something looks off: the
specific evidence + what to fix; else empty>"}. No markdown fences.

Default ok=true. Set ok=false (with a concrete `problem`) ONLY when the observed
data shows a real deviation, e.g.:
- the value is empty/null AND the observed API returns clearly CONTAIN items the
  milestone should have captured, so the empty is a provable MISS (e.g. the data
  has 21 contacts to compare but the result is []). Emptiness ALONE is not a
  deviation: when the observed data genuinely has no matching item, an empty
  result may be the CORRECT answer — do NOT flag it. (A state-changing action
  that correctly commits value=null is likewise not a deviation — its outcome is
  the side effect.)
- the code matched a string/key spelled differently from the data (parsed
  'screen-time-1hr' but the data has 'screen_time_1_hr');
- a delivered answer contradicts the data (a total summed over un-narrowed rows);
- a required mutation has no successful api call, or gates success on a message
  that differs from the API's real return.
Do not flag on a hunch or an unverifiable value.

The value/returns are STRUCTURED SUMMARIES of COMPLETE data: `list[N]` states N
items; `…` / `... K more` are display caps. NEVER infer "truncated/incomplete/
invalid" from the display — judge by the stated counts + concrete content.
"""

SELF_ASSESS_USER_PROMPT = """\
Task: {task_instruction}
Milestone intent: {milestone_intent}

Produced value (committed result):
{value_json}

Observed API returns (REAL data this execute read):
{observed_returns}

Executor code:
{code}

Judge whether the produced value truly satisfies the milestone intent given the
observed returns. Return the JSON verdict.
"""


def _render_observed_from_records(trace_records: list) -> str | None:
    """Flatten read_sandbox_trace_since records ({api_calls:[...]}) into the
    per-(app,api) real-returns view used by self-assess (reuses
    _render_prior_returns)."""
    calls: list = []
    for rec in trace_records:
        if isinstance(rec, dict):
            calls.extend(rec.get("api_calls") or [])
    return _render_prior_returns(calls) if calls else None


def _render_value_for_self_assess(value) -> str:
    """Render a committed value for the self-assess prompt using the STRUCTURED
    summarizer (count + schema + labelled samples), NOT a raw json.dumps[:N] cut.

    A dumb char-cut ends mid-object and makes the reviewer falsely conclude the
    value is "truncated / invalid JSON / incomplete" — but the committed value is
    always a complete, valid object; only the display was cut (confirmed on
    3d9a636: a complete 8496-char contact list cut to 4000 chars read as
    'truncated'). The summarizer shows `list[N items]` so completeness is judged
    on the count, never on whether the text looks cut off. Scalars render in full
    (they are small and the reviewer needs the exact value to judge correctness)."""
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return summarize_data_for_llm(value)
    if isinstance(value, str) and len(value) > 2000:
        return f"<complete string, {len(value)} chars; head only>:\n{value[:2000]}"
    return repr(value)


def _render_prior_returns(api_trace: list) -> str | None:
    """Render the ACTUAL returns a prior EXECUTE attempt observed, per
    (app, api_name): the real field names / string values / sample rows that
    were already captured in api_trace.result_items (bounded raw by the
    sandbox). This is the in-context raw observation the code-writer needs to
    align its filter/key/selection to the real data — it does NOT re-fetch and
    is NOT a separate LLM pass; the rows are already in history."""
    from collections import OrderedDict

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for call in api_trace:
        if not isinstance(call, dict):
            continue
        if call.get("status") and call.get("status") != "ok":
            continue
        key = (str(call.get("app", "")), str(call.get("api_name", "")))
        ri = call.get("result_items")
        g = groups.setdefault(key, {"items": [], "total": 0, "truncated": False})
        if isinstance(ri, dict) and "items" in ri:
            shown = ri.get("items") or []
            g["items"].extend(shown)
            # The sandbox truncates the CAPTURED items for logging but records the
            # REAL count in `_list_total`; the executor received the full set. Use
            # the true count so the reviewer never reads the display cap as a
            # smaller "observed" set (the 3d9a636 false count-discrepancy:
            # _list_total=20, items=[10] -> reviewer must see 20, not 10).
            g["total"] += int(ri.get("_list_total", len(shown)) or len(shown))
            if ri.get("_truncated") or ri.get("_global_truncated"):
                g["truncated"] = True
        elif ri is not None:
            g["items"].append(ri)
            g["total"] += 1

    blocks: list[str] = []
    for (app, api_name), g in groups.items():
        items = g["items"]
        if not items:
            continue
        total = g["total"]
        data = items if len(items) != 1 else items[0]
        rendered = summarize_data_for_llm(data)
        indented = "\n".join("    " + ln for ln in rendered.splitlines())
        head = f"  {app}.{api_name} returned {total} item(s)"
        if g["truncated"] or total != len(items):
            head += (
                f" (display shows {len(items)} — the executor received the full set)"
            )
        blocks.append(f"{head}:\n{indented}")
    return "\n".join(blocks) if blocks else None


def _format_prior_attempts_block(prior_attempts: list) -> str | None:
    """Render the 'Prior attempts' markdown section for code_planner input.

    Caps to the last `_PRIOR_ATTEMPTS_CAP` EXECUTE entries on the active
    milestone (older attempts dropped, no condensed one-liner per spec —
    code_planner only needs actionable evidence). Each attempt shows the
    planner's OWN prior code_plan (plan_steps + construct_step + print_step),
    not the executor's raw Python. Rationale: code_planner learns from its
    own prior plan, not from executor's downstream implementation.

    Continuation rationale is rendered as a SEPARATE top-level section by
    the caller, not embedded here, so the planner can read strategic
    guidance independent of evidence.
    """
    if not prior_attempts:
        return None
    recent = [h for h in prior_attempts if isinstance(h, dict)][-_PRIOR_ATTEMPTS_CAP:]
    if not recent:
        return None
    lines = [f"## Prior attempts (last {len(recent)} on this milestone)", ""]
    for i, h in enumerate(recent, start=1):
        agent_out = h.get("agent_output") or {}
        if not isinstance(agent_out, dict):
            agent_out = {}
        success = h.get("success")
        if success:
            summary = (agent_out.get("summary") or "").strip()[
                :PRIOR_ATTEMPT_SUMMARY_MAX_LENGTH
            ]
            stdout_json = agent_out.get("stdout_json") or {}
            value_preview = ""
            if isinstance(stdout_json, dict) and "value" in stdout_json:
                value_preview = repr(stdout_json.get("value"))[
                    :PRIOR_ATTEMPT_VALUE_PREVIEW_MAX_LENGTH
                ]
            lines.append(f"### attempt {i} — succeeded")
            if summary:
                lines.append(f"summary: {summary}")
            if value_preview:
                lines.append(f"committed value: {value_preview}")
        else:
            fc = h.get("failure_code") or ""
            stdout_excerpt = (agent_out.get("stdout_excerpt") or "").strip()[
                :PRIOR_ATTEMPT_STDOUT_EXCERPT_MAX_LENGTH
            ]
            parse_error = (agent_out.get("parse_error") or "").strip()[
                :PRIOR_ATTEMPT_PARSE_ERROR_MAX_LENGTH
            ]
            lines.append(f"### attempt {i} — failed ({fc})")
            if parse_error:
                lines.append("parse_error:")
                lines.append(f"  {parse_error}")
            if stdout_excerpt:
                lines.append("stdout_excerpt:")
                for cl in stdout_excerpt.splitlines()[:8]:
                    lines.append(f"  {cl}")
            sa = agent_out.get("self_assess")
            if isinstance(sa, dict) and (sa.get("problem") or "").strip():
                lines.append(
                    "self-assess (grounded reason this attempt was judged NOT-DONE — fix exactly this):"
                )
                lines.append(f"  {sa['problem'].strip()[:400]}")
        code_plan = agent_out.get("code_plan")
        if isinstance(code_plan, dict) and code_plan:
            lines.append("code_plan (planner's prior output):")
            plan_steps = code_plan.get("plan_steps") or []
            if plan_steps:
                lines.append("  plan_steps:")
                for j, step in enumerate(plan_steps, start=1):
                    lines.append(f"    {j}. {str(step)[:200]}")
            cs = code_plan.get("construct_step")
            if isinstance(cs, str) and cs.strip():
                lines.append(f"  construct_step: {cs[:300]}")
            ps = code_plan.get("print_step")
            if isinstance(ps, str) and ps.strip():
                lines.append(f"  print_step: {ps[:400]}")
        api_trace = agent_out.get("api_trace")
        # executor_retry_sees_prior_returns: render the prior attempt's REAL
        # returns so the next code aligns to observed data (the 2026 interleaved
        # pattern, in the code-writer's own context) instead of re-writing blind.
        # Toggleable for ablation; data is already in api_trace.result_items.
        returns_block = (
            _render_prior_returns(api_trace)
            if isinstance(api_trace, list)
            and active_config().executor_retry_sees_prior_returns
            else None
        )
        if returns_block:
            lines.append(
                "Observed API returns from your last attempt (real data — align filters/keys to THIS):"
            )
            lines.append(returns_block)
        elif isinstance(api_trace, list) and api_trace:
            lines.append(
                f"api_trace: {len(api_trace)} call(s) (no inspectable returns)"
            )
        else:
            lines.append("api_trace: (none — local-compute milestone)")
        lines.append("")
    return "\n".join(lines).rstrip()


def _render_prior_returns_compact(
    api_trace: list, *, max_samples: int = 2, max_field_chars: int = 80
) -> str | None:
    """Cross-milestone render: per (app, api), just the field SCHEMA plus a
    couple of bounded sample rows.

    Deliberately LIGHTER than `_render_prior_returns` (which also emits
    distributions / ranges / constants / free-form / id analysis). Those extras
    exist to help the SAME milestone's retry align a filter to the observed
    data; ACROSS milestones the downstream code_planner only needs to learn
    which fields an earlier step observed (e.g. `last_name` that the committed
    variable dropped) and roughly what a row looks like (e.g. to re-pick the
    right record). Schema + 1-2 samples covers both the lossy-field and the
    wrong-row case at a fraction of the tokens — so the mechanism stays
    near-zero-footprint on the tasks that don't need it."""
    from collections import OrderedDict

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for call in api_trace:
        if not isinstance(call, dict):
            continue
        if call.get("status") and call.get("status") != "ok":
            continue
        key = (str(call.get("app", "")), str(call.get("api_name", "")))
        ri = call.get("result_items")
        g = groups.setdefault(key, {"items": [], "total": 0})
        if isinstance(ri, dict) and "items" in ri:
            shown = ri.get("items") or []
            g["items"].extend(shown)
            g["total"] += int(ri.get("_list_total", len(shown)) or len(shown))
        elif ri is not None:
            g["items"].append(ri)
            g["total"] += 1

    blocks: list[str] = []
    for (app, api_name), g in groups.items():
        items = g["items"]
        if not items:
            continue
        dict_items = [it for it in items if isinstance(it, dict)]
        lines = [f"  {app}.{api_name} returned {g['total']} item(s):"]
        if dict_items:
            schema = _infer_schema(dict_items)
            lines.append(
                "    fields: {"
                + ", ".join(f"{k}: {v}" for k, v in schema.items())
                + "}"
            )
            for it in dict_items[:max_samples]:
                lines.append(
                    "    e.g. "
                    + _render_sample_item(it, max_field_chars=max_field_chars)
                )
        else:
            sample = items[:max_samples]
            rendered = json.dumps(sample, ensure_ascii=False)
            if len(rendered) > 200:
                rendered = rendered[:200] + "…"
            lines.append(f"    values (sample): {rendered}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) if blocks else None


def _format_prior_milestone_returns_block(prior_milestone_returns: list) -> str | None:
    """Render observed api returns from EARLIER completed milestones.

    Companion to `## Prior milestone variables` (the lossy committed-variable
    preview): this surfaces the RAW field names / values the earlier steps
    actually observed, so the code-writer can recover a field an earlier step
    saw but did not commit into a variable (9016950: last_name). Reuses
    `_render_prior_returns` per milestone. Controller-gated
    (executor_sees_prior_milestone_returns); this only renders when the metadata
    key is present. Each entry: {milestone_index, milestone_intent, api_trace}.
    """
    if not prior_milestone_returns:
        return None
    blocks: list[str] = []
    for entry in prior_milestone_returns:
        if not isinstance(entry, dict):
            continue
        api_trace = entry.get("api_trace")
        if not isinstance(api_trace, list):
            continue
        returns = _render_prior_returns_compact(api_trace)
        if not returns:
            continue
        idx = entry.get("milestone_index")
        intent = (entry.get("milestone_intent") or "").strip()
        header = f"milestone {idx}" if idx is not None else "an earlier milestone"
        if intent:
            header += f" ({intent})"
        blocks.append(f"### {header}\n{returns}")
    if not blocks:
        return None
    return (
        "## Observed API returns from earlier milestones\n"
        "Real data already fetched by earlier steps — reuse these exact field "
        "names/values instead of re-deriving or guessing (a field here that is "
        "missing from the committed variables above is still available; re-read "
        "it from its source rather than fabricating).\n\n" + "\n\n".join(blocks)
    )


# Controller-internal rationale sentinels that carry no actionable content
# for the code_planner. Rendering them adds noise (and consumes attention
# tokens for an "ignore generic fallback" meta-instruction) without
# providing strategic guidance. Skip the section entirely.
_NON_ACTIONABLE_RATIONALES = frozenset(
    {
        "cold-start: rough plan",
        "fallback: no milestones to run",
        "fallback: retry active milestone",
    }
)


def _format_continuation_rationale_block(latest_rationale: str) -> str | None:
    """Render the 'Continuation rationale' section as a top-level markdown
    block. Pulled out of prior_attempts so the planner can read the routing
    layer's strategic guidance independent of execution evidence.

    Returns None when the rationale is missing, empty, or one of the
    controller-internal placeholders that has no actionable content
    (see `_NON_ACTIONABLE_RATIONALES`). Framework-override rationales
    (e.g. "SUBMIT rejected: active=m..., ...") DO carry actionable
    content and are rendered.
    """
    if not latest_rationale or not latest_rationale.strip():
        return None
    text = latest_rationale.strip()
    if text in _NON_ACTIONABLE_RATIONALES:
        return None
    text = text[:RATIONALE_MAX_LENGTH]
    return f"## Continuation rationale\n\n{text}"


def _build_code_plan_prompt(subagent_input: SubagentInput) -> str:
    """Render the code_planner Stage 1 prompt.

    Structure (top-level sections, fixed order):
      1. JSON payload — task / milestone / candidate_apis / user_profile
      2. ## Continuation rationale — routing-layer strategic guidance
         (only rendered when continuation produced a non-empty rationale)
      3. ## Prior milestone variables — data_summarizer markdown preview
      4. ## Prior attempts — last 2 EXECUTE entries on the active milestone
         (only rendered when retry context, prior_attempts non-empty)

    The previous `prior_variables_schema` JSON block was removed — its
    content is fully covered by the `## Prior milestone variables` preview
    (which carries access pattern + accessible_fields + value summary).
    See docs/notes/2026-05-16_input_fixes.md entry 7 + the 2026-05-17
    refactor discussion.
    """
    meta = subagent_input.metadata or {}
    simple_candidates = meta.get("candidate_apis") or []
    full_candidates = _enrich_candidate_apis(simple_candidates)
    prompt_payload = {
        "task_id": subagent_input.task_context.task_id,
        "task_instruction": subagent_input.task_context.instruction,
        "task_datetime": subagent_input.task_context.task_datetime,
        "milestone": {
            "id": meta.get("milestone_id") or "",
            "index": meta.get("milestone_index") or 0,
            "total": meta.get("milestone_total") or 1,
            "intent": meta.get("milestone_intent") or "",
        },
        "candidate_apis": full_candidates,
        "user_profile": _get_user_profile(),
    }

    sections: list[str] = [json.dumps(prompt_payload, ensure_ascii=False, indent=2)]

    rationale_block = _format_continuation_rationale_block(
        meta.get("latest_continuation_rationale") or ""
    )
    if rationale_block:
        sections.append(rationale_block)

    prior_block = (meta.get("prior_variables_preview") or "").strip()
    if prior_block:
        sections.append(f"## Prior milestone variables\n\n{prior_block}")

    prior_ms_returns_block = _format_prior_milestone_returns_block(
        meta.get("prior_milestone_returns") or []
    )
    if prior_ms_returns_block:
        sections.append(prior_ms_returns_block)

    prior_attempts_block = _format_prior_attempts_block(
        meta.get("prior_attempts_for_this_milestone") or []
    )
    if prior_attempts_block:
        sections.append(prior_attempts_block)

    return "\n\n".join(sections)


def _build_code_execute_prompt(
    subagent_input: SubagentInput, plan: CodePlanOutput
) -> str:
    """Build the Stage 2 prompt with the full allowed API specs."""
    meta = subagent_input.metadata or {}
    ms_id = meta.get("milestone_id") or ""
    ms_idx = int(meta.get("milestone_index") or 0) + 1
    ms_total = int(meta.get("milestone_total") or 1)
    ms_intent = meta.get("milestone_intent") or ""

    plan_steps = "\n".join(
        f"{i + 1}. {step}" for i, step in enumerate(plan.numbered_steps())
    )

    simple_candidates = meta.get("candidate_apis") or []
    allowed_api_specs = _enrich_candidate_apis(simple_candidates)

    prior_block = (meta.get("prior_variables_preview") or "").strip()
    profile = _get_user_profile()
    profile_line = ""
    if profile:
        name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        email = profile.get("email", "")
        profile_line = (
            f"Current user: {name} ({email})"
            if name
            else f"Current user email: {email}"
        )

    lines: list[str] = [
        f"Task: {subagent_input.task_context.instruction}",
        f"Datetime: {subagent_input.task_context.task_datetime}",
        "",
        f"Milestone [{ms_id}] ({ms_idx}/{ms_total}):",
        f"  Intent: {ms_intent}",
        "",
        f"Output variable: {plan.output_variable.name}  # {plan.output_variable.description}",
        "",
        "Code plan — implement every step:",
        plan_steps,
        "",
    ]

    if allowed_api_specs:
        lines += [
            "Allowed API specs (authoritative JSON):",
            json.dumps(allowed_api_specs, ensure_ascii=False, indent=2),
            "",
        ]

    if prior_block:
        lines += ["Prior milestone variables:", prior_block, ""]

    prior_ms_returns_block = _format_prior_milestone_returns_block(
        meta.get("prior_milestone_returns") or []
    )
    if prior_ms_returns_block:
        lines += [prior_ms_returns_block, ""]

    rationale_block = _format_continuation_rationale_block(
        meta.get("latest_continuation_rationale") or ""
    )
    if rationale_block:
        lines += [rationale_block, ""]

    prior_attempts_block = _format_prior_attempts_block(
        meta.get("prior_attempts_for_this_milestone") or []
    )
    if prior_attempts_block:
        lines += [prior_attempts_block, ""]

    if profile_line:
        lines += [profile_line, ""]

    lines += [
        "Write a multi-line Python program (each statement on its own line) that implements the plan above.",
        "The last line of your code must be the four-field print contract:",
        'print(json.dumps({"value": <result>, "summary": "<one sentence>", '
        '"description": "<structure of value, may be empty>", '
        '"answer": "<terminal answer string OR the literal \\"null\\" for action / intermediate>"}))',
    ]

    return "\n".join(lines)


def _format_apis_compact(candidates: list[dict]) -> str:
    """Compact single-line API reference: app.name(param1, param2) -> {field1, field2}."""
    lines: list[str] = []
    for api in candidates:
        app = api.get("app_name", "")
        name = api.get("api_name", "")
        if not app or not name:
            continue
        params = [
            p["name"]
            for p in api.get("parameters", [])
            if isinstance(p, dict) and p.get("name") and p["name"] != "access_token"
        ]
        success = api.get("response_schemas", {}).get("success", {})
        is_list_response = isinstance(success, list)
        if is_list_response and success and isinstance(success[0], dict):
            resp_fields = list(success[0].keys())
        elif isinstance(success, dict):
            resp_fields = list(success.keys())
        else:
            resp_fields = []
        param_str = ", ".join(params[:8]) + ("..." if len(params) > 8 else "")
        fields_str = (
            "{"
            + ", ".join(resp_fields[:6])
            + ("..." if len(resp_fields) > 6 else "")
            + "}"
        )
        # Wrap in [] when the API returns a list so the executor knows to iterate
        resp_str = "[" + fields_str + "]" if is_list_response else fields_str
        lines.append(f"  {app}.{name}({param_str}) -> {resp_str}")
    return "\n".join(lines)


# ── API spec enrichment ───────────────────────────────────────────────────────


def _enrich_candidate_apis(simple_candidates: list) -> list[dict]:
    """Upgrade simple {app, name, description, method} candidates to full specs.

    Strips `access_token` from each spec's parameter list. The runtime auto-
    injects it via _ApisProxy, so exposing it to the code agent only invites
    hallucinated `cognizant_access_token = "..."` patterns that 401-cascade.
    """
    enriched: list[dict] = []
    for candidate in simple_candidates:
        if not isinstance(candidate, dict):
            continue
        app = str(candidate.get("app", ""))
        name = str(candidate.get("name", ""))
        full_spec = get_api_spec(app, name)
        spec = full_spec if full_spec is not None else candidate
        params = spec.get("parameters")
        if isinstance(params, list):
            spec = dict(spec)
            spec["parameters"] = [
                p
                for p in params
                if not (isinstance(p, dict) and p.get("name") == "access_token")
            ]
        enriched.append(spec)
    return enriched


def _get_user_profile() -> dict:
    """Return the current user profile; empty dict if unavailable."""
    try:
        from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder

        auth_manager = AppWorldAuthHolder.get_auth_manager()
        profile = auth_manager.get_profile()
        return profile if isinstance(profile, dict) else {}
    except Exception:
        return {}


# ── Stdout parsing ────────────────────────────────────────────────────────────
