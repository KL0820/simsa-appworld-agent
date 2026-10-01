from __future__ import annotations

from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.state import (
    CurrentStateSnapshot,
    CycleHistoryEntry,
    Phase,
    RunState,
    RunStatus,
    TaskContext,
)
from adk_appworld_agent.orchestration.state_repo import RunStateRepository


def _exec_success(milestone_id: str) -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "executor_result": {
                "answer": "null",
                "milestone_done": True,
                "summary": "ok",
                "variables": [],
            },
            "finalize_called": True,
            "milestone_done": True,
            "submission_candidate": {
                "answer": "null",
                "task_type_hint": "action",
                "answer_type": "null",
                "source": "executor",
            },
            "code_execute": {"stdout_json": {"summary": "ok"}},
            "milestone_id": milestone_id,
        },
    )


def _exec_did_not_finalize() -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={"finalize_called": False},
    )


def _state_with_milestones(n: int) -> RunState:
    return RunState(
        phase=Phase.EXECUTE,
        status=RunStatus.RUNNING,
        task_context=TaskContext(instruction="x"),
        milestones=[{"id": f"m{i}", "intent": f"intent{i}"} for i in range(n)],
        active_milestone_index=0,
    )


def test_runstate_defaults_have_cycle_n_and_history():
    state = RunState()
    assert state.cycle_n == 0
    assert state.history == []


def test_cycle_history_entry_validates_required_fields():
    entry = CycleHistoryEntry(
        cycle_n=1,
        phase_completed="EXECUTE",
        milestone_index=0,
        milestone_intent="do x",
        success=True,
        agent_output={"k": "v"},
    )
    assert entry.phase_completed == "EXECUTE"
    assert entry.success is True
    assert entry.failure_code is None


def test_current_state_snapshot_defaults():
    snap = CurrentStateSnapshot()
    assert snap.cycle_n == 0
    assert snap.active_milestone_index == 0
    assert snap.milestone_count == 0
    assert snap.last_failure_code is None


def test_runstate_roundtrip_preserves_history_and_cycle_n():
    repo = RunStateRepository()
    state = RunState(cycle_n=3)
    state.history.append(
        CycleHistoryEntry(
            cycle_n=1,
            phase_completed="EXECUTE",
            milestone_index=0,
            milestone_intent="do x",
            success=True,
            agent_output={"k": "v"},
        )
    )

    session_state: dict = {}
    session_state.update(repo.save_delta(state))
    loaded = repo.load(session_state, default_instruction="")

    assert loaded.cycle_n == 3
    assert len(loaded.history) == 1
    assert loaded.history[0].milestone_intent == "do x"
    assert loaded.history[0].agent_output == {"k": "v"}
