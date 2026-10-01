from __future__ import annotations

from adk_appworld_agent.agent import build_appworld_controller_agent
from adk_appworld_agent.contracts.subagent_output import SubagentStatus
from adk_appworld_agent.observability.ledger import ExecutionLedgerStore
from adk_appworld_agent.orchestration.state import Phase, RunStatus
from adk_appworld_agent.orchestration.state_repo import RunStateRepository
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from tests.helpers import run_agent


class _FakeSubmitter:
    def submit_answer(self, answer: str):
        return {
            "result": "Num Passed Tests : 1\nNum Failed Tests : 0\nNum Total Tests : 1"
        }


def test_stub_happy_path_smoke():
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        submitter_provider=_FakeSubmitter,
    )
    session, events = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    outputs = SubagentOutputStore()
    ledger = ExecutionLedgerStore().read(session.state)

    assert state.phase == Phase.COMPLETE
    assert state.status == RunStatus.COMPLETED
    assert (
        outputs.read_latest(session.state, Phase.FIND).status
        == SubagentStatus.SUCCEEDED
    )
    assert (
        outputs.read_latest(session.state, Phase.PLAN).status
        == SubagentStatus.SUCCEEDED
    )
    assert (
        outputs.read_latest(session.state, Phase.EXECUTE).status
        == SubagentStatus.SUCCEEDED
    )
    assert (
        outputs.read_latest(session.state, Phase.SUBMIT).status
        == SubagentStatus.SUCCEEDED
    )
    assert len(ledger) >= 5
    assert ledger[-1].state_after.phase == Phase.COMPLETE
    assert ledger[-1].state_after.status == RunStatus.COMPLETED
    assert events[-1].content.parts[0].text == "Skeleton workflow completed."
