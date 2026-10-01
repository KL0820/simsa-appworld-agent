from __future__ import annotations

from adk_appworld_agent.agent import build_appworld_controller_agent
from adk_appworld_agent.contracts.subagent_output import SubagentStatus
from adk_appworld_agent.observability.ledger import ExecutionLedgerStore
from adk_appworld_agent.orchestration.state import Phase, RunStatus
from adk_appworld_agent.orchestration.state_repo import RunStateRepository
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from tests.helpers import run_agent


def test_stub_failure_path_smoke():
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        fail_phase=Phase.PLAN,
    )
    session, events = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    latest_plan_output = SubagentOutputStore().read_latest(session.state, Phase.PLAN)
    ledger = ExecutionLedgerStore().read(session.state)

    assert state.phase == Phase.FAILED
    assert state.status == RunStatus.FAILED
    assert latest_plan_output is not None
    assert latest_plan_output.status == SubagentStatus.FAILED
    assert latest_plan_output.failure_code == "PLAN_STUB_FAILURE"
    assert ledger[-1].state_after.phase == Phase.FAILED
    assert ledger[-1].failure_code == "PLAN_STUB_FAILURE"
    assert "PLAN_STUB_FAILURE" in events[-1].content.parts[0].text


class _FailedEvalSubmitter:
    def submit_answer(self, answer: str):
        return {
            "result": "Num Passed Tests : 0\nNum Failed Tests : 1\nNum Total Tests : 1"
        }


def test_submit_eval_failure_does_not_mark_complete():
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        submitter_provider=_FailedEvalSubmitter,
    )
    session, events = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    latest_submit_output = SubagentOutputStore().read_latest(
        session.state, Phase.SUBMIT
    )
    ledger = ExecutionLedgerStore().read(session.state)

    assert state.phase == Phase.FAILED
    assert state.status == RunStatus.FAILED
    assert state.completion_block_reason == "COMPLETION_EVAL_FAILED"
    assert latest_submit_output is not None
    assert latest_submit_output.status == SubagentStatus.FAILED
    assert latest_submit_output.failure_code == "COMPLETION_EVAL_FAILED"
    assert ledger[-1].state_after.phase == Phase.FAILED
    assert ledger[-1].failure_code == "COMPLETION_EVAL_FAILED"
    assert "COMPLETION_EVAL_FAILED" in events[-1].content.parts[0].text


class _SpySubmitter:
    """Submitter that records every submit_answer call for assertions."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def submit_answer(self, answer: str):
        self.calls.append(answer)
        # Mimic AppWorld's per-task evaluation report so the gate has counts
        # to parse — this mirrors the shape returned for a null submission.
        return {
            "result": "Num Passed Tests : 0\nNum Failed Tests : 3\nNum Total Tests : 3"
        }


def test_executor_failure_triggers_forced_null_submission_exactly_once():
    """Executor crashes (no submission_candidate) → orchestrator falls back
    to a null submit so AppWorld evaluator records test counts instead of NA.
    Exactly-once: only one submit_answer call, marker payload set, original
    failure_code preserved on RunState."""
    spy = _SpySubmitter()
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        fail_phase=Phase.EXECUTE,
        submitter_provider=lambda: spy,
    )
    session, events = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    latest_submit_output = SubagentOutputStore().read_latest(
        session.state, Phase.SUBMIT
    )

    # Run remains FAILED with the original executor failure code.
    assert state.phase == Phase.FAILED
    assert state.status == RunStatus.FAILED
    assert state.completion_block_reason == "EXECUTE_STUB_FAILURE"

    # Forced fallback ran exactly once and submitted "null".
    assert spy.calls == ["null"]
    assert latest_submit_output is not None
    assert latest_submit_output.payload.get("forced_fallback") is True
    assert (
        latest_submit_output.payload.get("original_failure_code")
        == "EXECUTE_STUB_FAILURE"
    )

    # The final user-visible message still reflects the executor failure.
    assert "EXECUTE_STUB_FAILURE" in events[-1].content.parts[0].text


def test_forced_fallback_skipped_when_submit_already_ran():
    """When the executor produced a candidate and SUBMIT phase already
    executed (even if eval failed), the post-loop fallback must NOT add a
    second submit_answer call — that's the exactly-once guarantee."""
    spy = _SpySubmitter()
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        submitter_provider=lambda: spy,
    )
    run_agent(agent)

    # The stub executor emits a non-null submission_candidate, so SUBMIT runs
    # the normal path. Spy should record exactly that one call — no forced
    # fallback layered on top.
    assert len(spy.calls) == 1
