"""Pure-logic helpers for the AppWorld controller (AppWorldAgent).

No-progress fingerprinting, executor failure diagnostics, sandbox-trace
capture, plan-payload interpretation, and cycle-limit config. Extracted from
agent.py in the 2026-06-22 behavior-preserving split (control plane unchanged)."""

from __future__ import annotations

import hashlib
import json
import os

from adk_appworld_agent.contracts.continuation import ContinuationDecision
from adk_appworld_agent.contracts.limits import (
    EXECUTOR_CODE_DIAG_MAX_LENGTH,
    LLM_RAISED_MAX_LENGTH,
    PARSE_ERROR_MAX_LENGTH,
    STDOUT_EXCERPT_MAX_LENGTH,
)
from adk_appworld_agent.contracts.plan_ir import Milestone
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.state import RunState
from adk_appworld_agent.subagents.executor.appworld_tools import (
    read_sandbox_trace_since,
    summarize_trace_for_prompt,
)


def _capture_sandbox_trace_for_execute(start_offset: int) -> list[dict]:
    """Capture the API trace for one EXECUTE phase.

    Reads sandbox_api_calls.jsonl entries appended since `start_offset` bytes.
    For multi-attempt EXECUTE (e.g. inner repair loop) the side file gets one
    record per execute_python call. We pick the LAST record — it's the final
    attempt and aligns with the `code` / `raw_stdout` already in the envelope.
    Returns the summarized api_calls (prompt-budget-bounded) or `[]` if no
    record was appended (LLM raised before invoking execute_python).
    """
    records = read_sandbox_trace_since(start_offset)
    if not records:
        return []
    final = records[-1]
    api_calls = final.get("api_calls") if isinstance(final, dict) else None
    if not isinstance(api_calls, list):
        return []
    return summarize_trace_for_prompt(api_calls)


def _collect_prior_milestone_returns(
    state: RunState, active_id: str | None, *, cap: int
) -> list[dict]:
    """Observed api returns from EARLIER, already-completed milestones — the
    intermediate-visibility bridge for the downstream code_planner.

    Walks history newest→oldest, taking the LATEST successful EXECUTE entry per
    DISTINCT milestone that is NOT the active one, keeping only entries whose
    agent_output carries a non-empty api_trace. Returns chronological
    (oldest→newest) list of {milestone_index, milestone_intent, api_trace},
    capped to the `cap` most recent milestones. Prose-free plumbing; rendering
    (and token bounding of the rows themselves) is the prompt layer's job via
    the existing `_render_prior_returns`. Gated by
    executor_sees_prior_milestone_returns — the caller decides whether to call.
    """
    seen: set = set()
    collected: list[dict] = []
    for h in reversed(state.history):
        if h.phase_completed != "EXECUTE" or not h.success:
            continue
        mid = h.milestone_id
        # Skip the active milestone (its own attempts already flow through
        # prior_attempts_for_this_milestone) and any milestone already taken.
        if mid is not None and mid == active_id:
            continue
        dedup_key = mid if mid is not None else ("idx", h.milestone_index)
        if dedup_key in seen:
            continue
        agent_out = h.agent_output if isinstance(h.agent_output, dict) else {}
        api_trace = agent_out.get("api_trace")
        if not (isinstance(api_trace, list) and api_trace):
            continue
        seen.add(dedup_key)
        collected.append(
            {
                "milestone_index": h.milestone_index,
                "milestone_intent": h.milestone_intent,
                "api_trace": api_trace,
            }
        )
        if len(collected) >= cap:
            break
    collected.reverse()  # chronological: oldest completed milestone first
    return collected


def _execute_fingerprint(candidate_apis, api_trace, value) -> str:
    """Structural, prose-free identity of one EXECUTE attempt for no-progress
    detection: {sorted candidate API ids, set of API endpoints actually called,
    EXACT committed value}.

    Deliberately EXCLUDES every natural-language field (milestone wording,
    continuation rationale, code_plan steps, self-assess prose) so a
    reword-every-cycle loop that re-emits structurally-identical work still
    collides, while a genuinely different attempt (different APIs called, or a
    different committed value) does NOT. The value is hashed EXACTLY, not by a
    shape bucket, so distinct results — e.g. 7134.0 vs 0.0 vs the correct sum on
    552869a — never collide, and a converging task is never given up on early.
    """
    apis = sorted(str(a) for a in (candidate_apis or []))
    called = sorted(
        {
            f"{c.get('app', '')}.{c.get('api_name', '')}"
            for c in (api_trace or [])
            if isinstance(c, dict)
        }
    )
    try:
        value_repr = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        value_repr = repr(value)
    blob = json.dumps(
        {"apis": apis, "called": called, "value": value_repr}, ensure_ascii=False
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def _next_milestone_id(state: RunState) -> str:
    """Mint a stable, monotonic milestone id (§4.1). Deterministic per run (no
    uuid / clock — keeps replay byte-stable)."""
    uid = state.next_milestone_uid
    state.next_milestone_uid = uid + 1
    return f"ms{uid}"


def _ensure_milestone_ids(state: RunState) -> None:
    """Assign a fresh stable id to every milestone that lacks one. Called after
    any milestone mutation (cold-start plan or a revise that introduced new
    Milestone objects, which arrive with id=None). Existing ids are preserved by
    the caller positionally BEFORE this runs, so this only fills genuine gaps —
    i.e. mints ids for newly-introduced milestones."""
    for m in state.milestones:
        if not getattr(m, "id", None):
            m.id = _next_milestone_id(state)


def _update_no_progress(
    state: RunState,
    milestone_id: str,
    fingerprint: str,
    window: int,
    attempt_budget: int = 0,
) -> bool:
    """Append `fingerprint` to the active milestone's fingerprint history
    (resetting when the active milestone's STABLE id changed) and return True
    when the milestone has made no progress, by EITHER trigger:

    1. `window` identical fingerprints in a row (executor reproduced
       structurally-identical work N times — the structurally-stuck trigger).
    2. `attempt_budget` total attempts on this milestone, regardless of whether
       the fingerprint changed. The continuation-revise loop slips past
       trigger 1: each revise re-words the milestone → finder re-routes →
       candidate_apis change → fingerprint changes → the identical streak resets
       to 1 → trigger 1 never fires → the milestone would revise to MAX_CYCLES
       (2c544f9: 14 revise cycles, 123 LLM calls, never bounded, burning the
       project token quota → 429 cascade on later tasks). Keying the reset on
       the STABLE milestone id (not the index) makes the count survive re-routes
       AND prerequisite insertions: a re-word keeps the id so the count
       accumulates; an inserted prerequisite has a different id so the stuck
       milestone's count is not falsely reset. `len(active_milestone_fingerprints)`
       is therefore the true re-route-immune per-milestone attempt count.

    Also updates `state.no_progress_streak` (trailing identical count) for the
    ledger/viewer.
    """
    if state.fingerprint_milestone_id != milestone_id:
        state.fingerprint_milestone_id = milestone_id
        state.active_milestone_fingerprints = []
    state.active_milestone_fingerprints.append(fingerprint)
    streak = 0
    for fp in reversed(state.active_milestone_fingerprints):
        if fp == fingerprint:
            streak += 1
        else:
            break
    state.no_progress_streak = streak
    by_streak = window >= 1 and streak >= window
    by_attempts = (
        attempt_budget >= 1
        and len(state.active_milestone_fingerprints) >= attempt_budget
    )
    return by_streak or by_attempts


DEFAULT_MAX_CYCLES = 15


def _env_max_cycles() -> int:
    raw = os.environ.get("APPWORLD_MAX_CYCLES")
    if raw is None:
        return DEFAULT_MAX_CYCLES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CYCLES
    return max(1, value)


def _milestones_from_plan_payload(payload: dict) -> list[Milestone]:
    milestones_raw = payload.get("milestones")
    if isinstance(milestones_raw, list) and milestones_raw:
        out_ms: list[Milestone] = []
        for m in milestones_raw:
            mv = Milestone.model_validate(m) if isinstance(m, dict) else m
            # ids are controller-owned (§4.1); drop any planner-emitted id so the
            # cold-start reconcile mints a fresh stable id per milestone.
            mv.id = None
            out_ms.append(mv)
        return out_ms
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    out: list[Milestone] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        if not isinstance(task, str) or not task.strip():
            continue
        app = item.get("app")
        out.append(
            Milestone(
                intent=task.strip(),
                app=app.strip() if isinstance(app, str) and app.strip() else None,
            )
        )
    return out


def _executor_failure_diagnostics(envelope: SubagentEnvelope | None) -> dict | None:
    """Extract the failed executor's code + stdout + parse_error so the
    continuation LLM can diagnose the prior attempt instead of retrying blind.

    On EXECUTOR_DID_NOT_FINALIZE this typically reveals the model's actual
    Python (truncated), the sandbox stdout (including any traceback), and any
    parse_error / llm_raised string. Without these, history entries on failure
    only carry the failure_code label, which produces doom loops where each
    cycle's continuation re-emits the same buggy code.
    """
    if envelope is None:
        return None
    payload = envelope.payload or {}
    code_execute = payload.get("code_execute") or {}
    code_plan = payload.get("code_plan") or {}
    diag: dict = {}
    code = code_execute.get("code")
    if isinstance(code, str) and code.strip():
        diag["code"] = code[:EXECUTOR_CODE_DIAG_MAX_LENGTH]
    # Surface the planner's prior code_plan structured fields so the
    # next-cycle code_planner can see its own previous output (plan_steps +
    # construct_step + print_step) and diff against it, instead of just
    # seeing the executor's raw Python. Truncated raw code can mislead at
    # the cut boundary (e.g. `add_regex.finda` on 042a9fc_2 attempt 2);
    # the structured plan is compact and lossless.
    if isinstance(code_plan, dict) and code_plan:
        plan_summary: dict = {}
        plan_steps = code_plan.get("plan_steps")
        if isinstance(plan_steps, list) and plan_steps:
            plan_summary["plan_steps"] = [str(s) for s in plan_steps]
        construct_step = code_plan.get("construct_step")
        if isinstance(construct_step, str) and construct_step.strip():
            plan_summary["construct_step"] = construct_step
        print_step = code_plan.get("print_step")
        if isinstance(print_step, str) and print_step.strip():
            plan_summary["print_step"] = print_step
        output_variable = code_plan.get("output_variable")
        if isinstance(output_variable, dict):
            plan_summary["output_variable"] = {
                "name": output_variable.get("name", ""),
                "description": output_variable.get("description", ""),
            }
        if plan_summary:
            diag["code_plan"] = plan_summary
    parse_error = code_execute.get("parse_error")
    if parse_error:
        diag["parse_error"] = str(parse_error)[:PARSE_ERROR_MAX_LENGTH]
    raw_stdout = code_execute.get("raw_stdout")
    if isinstance(raw_stdout, str) and raw_stdout.strip():
        diag["stdout_excerpt"] = raw_stdout[:STDOUT_EXCERPT_MAX_LENGTH]
    llm_raised = code_execute.get("llm_raised")
    if llm_raised:
        diag["llm_raised"] = str(llm_raised)[:LLM_RAISED_MAX_LENGTH]
    tool_call_count = code_execute.get("tool_call_count")
    if isinstance(tool_call_count, int):
        diag["tool_call_count"] = tool_call_count
    repair_attempts = code_execute.get("repair_attempts")
    if isinstance(repair_attempts, list) and repair_attempts:
        diag["repair_attempt_count"] = len(repair_attempts)
    # Self-assess: when the executor's grounded post-execute review flipped the
    # milestone to not-done, it already pinpointed WHY (grounded in api_trace).
    # Surface that actionable reason so continuation revises on the real cause
    # instead of rewording blind (the missing link that caused a doom loop on
    # 552869a: self-assess correctly diagnosed the filter mismatch but the
    # reason never reached the planner).
    self_assess = code_execute.get("self_assess")
    if isinstance(self_assess, dict):
        sa: dict = {}
        problem = self_assess.get("problem")
        if isinstance(problem, str) and problem.strip():
            sa["problem"] = problem[:STDOUT_EXCERPT_MAX_LENGTH]
        summary = self_assess.get("summary")
        if isinstance(summary, str) and summary.strip():
            sa["summary"] = summary[:STDOUT_EXCERPT_MAX_LENGTH]
        if sa:
            diag["self_assess"] = sa
    return diag or None


def _interpret_plan_envelope(
    envelope: SubagentEnvelope, state: RunState, *, cycle_n: int
) -> tuple[ContinuationDecision, list[Milestone] | None]:
    payload = envelope.payload or {}
    if cycle_n == 0:
        # Cold-start: rough_planner just produced the milestone list. Active is
        # already 0 (RunState default); use RETRY so the routing block keeps
        # active=0 and FIND/EXECUTE proceeds against m0. ADVANCE would
        # increment active to 1 and skip m0.
        milestones = _milestones_from_plan_payload(payload)
        decision = ContinuationDecision(
            next_action="RETRY" if milestones else "ABORT",
            revised_milestones=None,
            rationale="cold-start: rough plan",
        )
        return decision, (milestones or None)

    revised_raw = payload.get("revised_milestones")
    revised: list[Milestone] | None
    if isinstance(revised_raw, list):
        # Preserve `[]` so the framework can reject it explicitly — distinct
        # from omitting `revised_milestones` (which means "no revise this round").
        # Strip any LLM-emitted `id`: milestone ids are CONTROLLER-owned (§4.1).
        # The continuation must not set/echo them — the controller restores the
        # stable id positionally (re-word) or mints a fresh one (new milestone).
        revised = []
        for m in revised_raw:
            if isinstance(m, dict):
                mv = Milestone.model_validate(m)
                mv.id = None
                revised.append(mv)
    else:
        revised = None

    raw_action = payload.get("next_action", "RETRY")
    if raw_action not in ("RETRY", "ADVANCE", "SUBMIT", "ABORT"):
        # legacy / malformed value → degrade to RETRY so the wall budget
        # decides task termination, not a parse failure.
        raw_action = "RETRY"
    if raw_action == "ABORT":
        # ABORT was removed from the continuation_planner action space
        # (2026-05-17): every AppWorld task should run to completion or
        # natural timeout. If the LLM hallucinates ABORT despite the
        # prompt no longer offering it, redirect to RETRY so the wall
        # budget gets the final say. The decision's rationale is kept so
        # the analyst can see what the LLM was trying to express.
        raw_action = "RETRY"

    decision = ContinuationDecision(
        next_action=raw_action,
        revised_milestones=revised,
        rationale=str(payload.get("rationale", "") or ""),
    )
    return decision, revised
