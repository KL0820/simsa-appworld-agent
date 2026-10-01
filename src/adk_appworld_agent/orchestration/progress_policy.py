"""Framework guards for the progress-control subagent's decisions.

The model chooses RETRY, ADVANCE, SUBMIT or ABORT. These helpers enforce
the existing plan and budget invariants; they do not make another model call.
"""

from __future__ import annotations

from dataclasses import dataclass

from adk_appworld_agent._controller_helpers import (
    _ensure_milestone_ids,
    _interpret_plan_envelope,
    _next_milestone_id,
)
from adk_appworld_agent.contracts.continuation import ContinuationDecision
from adk_appworld_agent.contracts.limits import (
    PER_MILESTONE_CYCLE_LIMIT,
    RETRY_WITHOUT_REVISE_LIMIT,
)
from adk_appworld_agent.contracts.plan_ir import Milestone
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.state import CycleHistoryEntry, RunState


@dataclass
class _ProgressRouting:
    decision: ContinuationDecision
    milestones_update: list[Milestone] | None
    active: int
    last_idx: int
    old_milestone_ids: list[str | None]
    active_executed_successfully: bool
    framework_override: str | None = None


def apply_progress_decision(
    state: RunState, envelope: SubagentEnvelope
) -> ContinuationDecision:
    """Apply guards in their established order to a controller-owned state copy."""
    routing = _routing_context(state, envelope)
    state.last_framework_override = None
    _guard_submission(state, routing)
    _apply_revisions(state, routing)
    _guard_final_advance(state, routing)
    _guard_retry_streak(state, routing)
    _stabilize_milestones(state, routing)
    _guard_cycle_budget(state, routing)
    _record_decision(state, routing)
    return routing.decision


def _routing_context(state: RunState, envelope: SubagentEnvelope) -> _ProgressRouting:
    decision, milestones = _interpret_plan_envelope(
        envelope, state, cycle_n=state.cycle_n
    )
    active = state.active_milestone_index
    latest_execute = next(
        (
            entry
            for entry in reversed(state.history)
            if entry.phase_completed == "EXECUTE" and entry.milestone_index == active
        ),
        None,
    )
    return _ProgressRouting(
        decision=decision,
        milestones_update=milestones,
        active=active,
        last_idx=max(0, len(state.milestones) - 1),
        old_milestone_ids=[
            getattr(milestone, "id", None) for milestone in state.milestones
        ],
        active_executed_successfully=bool(latest_execute and latest_execute.success),
    )


def _guard_submission(state: RunState, routing: _ProgressRouting) -> None:
    """Require a successful final milestone before submission."""
    if routing.decision.next_action == "SUBMIT":
        if (
            not state.milestones
            or routing.active != routing.last_idx
            or (not routing.active_executed_successfully)
        ):
            routing.framework_override = f"SUBMIT rejected: active=m{routing.active}, last=m{routing.last_idx}, active_executed_successfully={routing.active_executed_successfully}. Forcing RETRY on active milestone — pick ADVANCE next round to walk through remaining milestones before SUBMIT becomes valid."
            routing.decision = ContinuationDecision(
                next_action="RETRY",
                revised_milestones=None,
                rationale=routing.framework_override,
            )
            routing.milestones_update = None


def _apply_revisions(state: RunState, routing: _ProgressRouting) -> None:
    """Apply bounded forward revisions while preserving untouched milestones."""
    if routing.decision.revised_milestones is None:
        return
    if routing.decision.next_action == "RETRY":
        _apply_retry_revision(state, routing)
    elif routing.decision.next_action == "ADVANCE":
        _apply_advance_revision(state, routing)
    routing.last_idx = max(0, len(state.milestones) - 1)


def _apply_retry_revision(state: RunState, routing: _ProgressRouting) -> None:
    """Revise from the active milestone, with bounded prerequisite insertion."""
    if routing.decision.revised_milestones is None:
        return
    if len(routing.decision.revised_milestones) == 0:
        routing.framework_override = "RETRY with empty revised_milestones is invalid — would erase the active milestone. Forcing RETRY with no revise."
        routing.decision = ContinuationDecision(
            next_action="RETRY",
            revised_milestones=None,
            rationale=routing.framework_override,
        )
        routing.milestones_update = None
        return
    revised = list(routing.decision.revised_milestones)
    writable_len = len(state.milestones) - routing.active
    config = active_config()
    active_id = (
        routing.old_milestone_ids[routing.active]
        if routing.active < len(routing.old_milestone_ids)
        else None
    )
    insertions = state.prereq_insertions.get(active_id, 0) if active_id else 0
    if (
        config.continuation_block_plan_growth
        and len(revised) == writable_len + 1
        and (config.prereq_insertion_cap >= 1)
        and (insertions < config.prereq_insertion_cap)
    ):
        state.milestones = state.milestones[: routing.active] + revised
        state.milestones[routing.active].id = _next_milestone_id(state)
        for offset in range(1, len(revised)):
            old_position = routing.active + (offset - 1)
            if (
                old_position < len(routing.old_milestone_ids)
                and routing.old_milestone_ids[old_position]
            ):
                state.milestones[
                    routing.active + offset
                ].id = routing.old_milestone_ids[old_position]
        if active_id:
            state.prereq_insertions[active_id] = insertions + 1
    elif config.continuation_block_plan_growth and len(revised) > writable_len:
        insertion_note = (
            " (the single-prerequisite insertion budget for this milestone is already spent)"
            if active_id and insertions >= config.prereq_insertion_cap
            else ""
        )
        routing.framework_override = f"revised_milestones rejected: sending {len(revised)} item(s) for a writable region of {writable_len} would GROW the plan by more than the one allowed prerequisite{insertion_note}. Recovery may insert AT MOST one upstream prerequisite milestone before the active one ([prerequisite, active, ...]); otherwise refine the ACTIVE milestone in place. If this step needs a loop, rewrite the active milestone as a single 'repeatedly do X until Y' — never unroll it into one-milestone-per-iteration."
        routing.decision = ContinuationDecision(
            next_action="RETRY",
            revised_milestones=None,
            rationale=routing.framework_override,
        )
        routing.milestones_update = None
    elif len(revised) >= writable_len:
        state.milestones = state.milestones[: routing.active] + revised
    else:
        state.milestones = (
            state.milestones[: routing.active]
            + revised
            + state.milestones[routing.active + len(revised) :]
        )


def _apply_advance_revision(state: RunState, routing: _ProgressRouting) -> None:
    """Revise the future plan without rewriting the completed milestone."""
    if routing.decision.revised_milestones is None:
        return
    if len(routing.decision.revised_milestones) == 0:
        routing.framework_override = "ADVANCE with empty revised_milestones is invalid — would advance past the last milestone. Forcing RETRY."
        routing.decision = ContinuationDecision(
            next_action="RETRY",
            revised_milestones=None,
            rationale=routing.framework_override,
        )
        routing.milestones_update = None
        return
    revised = list(routing.decision.revised_milestones)
    writable_len = len(state.milestones) - routing.active - 1
    config = active_config()
    forward_cap = config.continuation_forward_growth_cap
    if (
        config.continuation_block_plan_growth
        and len(revised) > writable_len
        and (forward_cap >= 1)
        and (state.forward_growth_count < forward_cap)
    ):
        state.milestones = (
            state.milestones[: routing.active + 1]
            + revised
            + state.milestones[routing.active + 1 + len(revised) :]
        )
        state.forward_growth_count += 1
    elif config.continuation_block_plan_growth and len(revised) > writable_len:
        routing.framework_override = f"revised_milestones rejected on ADVANCE: sending {len(revised)} item(s) for a writable region of {writable_len} would GROW the plan beyond the bounded forward-growth budget. Recovery must not add more milestones — advance into the existing plan; if a step needs a loop, rewrite THAT milestone as a single 'repeatedly do X until Y'."
        routing.decision = ContinuationDecision(
            next_action="ADVANCE",
            revised_milestones=None,
            rationale=routing.framework_override,
        )
        routing.milestones_update = None
    elif len(revised) >= writable_len:
        state.milestones = state.milestones[: routing.active + 1] + revised
    else:
        state.milestones = (
            state.milestones[: routing.active + 1]
            + revised
            + state.milestones[routing.active + 1 + len(revised) :]
        )


def _guard_final_advance(state: RunState, routing: _ProgressRouting) -> None:
    """Resolve advancement beyond the final milestone."""
    if (
        routing.decision.next_action == "ADVANCE"
        and routing.active >= routing.last_idx
        and (routing.decision.revised_milestones is None)
    ):
        if routing.active_executed_successfully:
            routing.framework_override = f"ADVANCE at last milestone (m{routing.active}) with no revise → no further milestones to advance into. Forcing SUBMIT."
            routing.decision = ContinuationDecision(
                next_action="SUBMIT",
                revised_milestones=None,
                rationale=routing.framework_override,
            )
        else:
            routing.framework_override = f"ADVANCE at last milestone (m{routing.active}) with no revise but active hasn't succeeded. Forcing RETRY."
            routing.decision = ContinuationDecision(
                next_action="RETRY",
                revised_milestones=None,
                rationale=routing.framework_override,
            )


def _guard_retry_streak(state: RunState, routing: _ProgressRouting) -> None:
    """Request a different approach after repeated unchanged retries."""
    if (
        routing.decision.next_action == "RETRY"
        and routing.decision.revised_milestones is None
    ):
        streak = 0
        for h in reversed(state.history):
            if h.phase_completed != "PLAN":
                continue
            if h.cycle_n == 0:
                break
            if not isinstance(h.agent_output, dict):
                break
            if h.agent_output.get("next_action") != "RETRY":
                break
            if h.milestone_index != routing.active:
                break
            if h.agent_output.get("revised_milestones_applied"):
                break
            streak += 1
        if streak >= RETRY_WITHOUT_REVISE_LIMIT:
            routing.framework_override = f"RETRY × {streak + 1} on m{routing.active} without revised_milestones — executor stuck in identical-approach loop. Next round you MUST emit revised_milestones[0] that changes the active milestone's intent / steps in a way the executor can act on, OR pick ADVANCE (if evidence supports the milestone being done despite a wrong-shape committed value). The solvable path is in your inputs; each revise should make the milestone wording point at it more precisely."
            routing.decision = ContinuationDecision(
                next_action="RETRY",
                revised_milestones=None,
                rationale=routing.framework_override,
            )


def _stabilize_milestones(state: RunState, routing: _ProgressRouting) -> None:
    """Install the initial plan and preserve stable milestone identities."""
    if routing.milestones_update is not None and (not state.milestones):
        state.milestones = routing.milestones_update
        routing.last_idx = max(0, len(state.milestones) - 1)
    for i, m in enumerate(state.milestones):
        if (
            not getattr(m, "id", None)
            and i < len(routing.old_milestone_ids)
            and routing.old_milestone_ids[i]
        ):
            m.id = routing.old_milestone_ids[i]
    _ensure_milestone_ids(state)


def _guard_cycle_budget(state: RunState, routing: _ProgressRouting) -> None:
    """Bound repeated work on a milestone using the existing cycle policy."""
    if routing.decision.next_action == "RETRY":
        cycles_on_active = 0
        for h in reversed(state.history):
            if h.phase_completed != "PLAN":
                continue
            if h.cycle_n == 0:
                break
            if h.milestone_index != routing.active:
                break
            cycles_on_active += 1
        if cycles_on_active >= PER_MILESTONE_CYCLE_LIMIT:
            if routing.active < routing.last_idx:
                routing.framework_override = f"per-milestone budget: m{routing.active} consumed {cycles_on_active} cycles (limit {PER_MILESTONE_CYCLE_LIMIT}) without completing — accepting best-so-far committed value and forcing ADVANCE so downstream milestones get budget instead of looping to the wall."
                routing.decision = ContinuationDecision(
                    next_action="ADVANCE",
                    revised_milestones=None,
                    rationale=routing.framework_override,
                )
            elif routing.active_executed_successfully:
                routing.framework_override = f"per-milestone budget: last milestone m{routing.active} consumed {cycles_on_active} cycles; active has a successful EXECUTE — forcing SUBMIT rather than looping to the wall."
                routing.decision = ContinuationDecision(
                    next_action="SUBMIT",
                    revised_milestones=None,
                    rationale=routing.framework_override,
                )


def _record_decision(state: RunState, routing: _ProgressRouting) -> None:
    """Commit the effective action and its diagnostic history."""
    if routing.decision.next_action == "RETRY":
        pass
    elif routing.decision.next_action == "ADVANCE":
        state.active_milestone_index = min(routing.active + 1, routing.last_idx)
    next_milestone_index_for_history = state.active_milestone_index
    state.history.append(
        CycleHistoryEntry(
            cycle_n=state.cycle_n,
            phase_completed="PLAN",
            milestone_index=next_milestone_index_for_history,
            milestone_intent=state.milestones[next_milestone_index_for_history].intent
            if 0 <= next_milestone_index_for_history < len(state.milestones)
            else None,
            success=True,
            agent_output={
                "next_action": routing.decision.next_action,
                "revised_milestones_applied": routing.decision.revised_milestones
                is not None,
                "rationale": routing.decision.rationale,
                "framework_override": routing.framework_override,
            },
        )
    )
    state.last_framework_override = routing.framework_override
