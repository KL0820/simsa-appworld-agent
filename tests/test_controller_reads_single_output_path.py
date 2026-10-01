from __future__ import annotations

from adk_appworld_agent.agent import AppWorldAgent
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.orchestrator import Orchestrator
from adk_appworld_agent.orchestration.state import (
    Phase,
)
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
from adk_appworld_agent.subagents.stubs import build_stub_subagent
from tests.helpers import run_agent


def test_subagent_output_store_reads_only_from_keyed_append_path():
    store = SubagentOutputStore()
    session_state = {
        "find_output": {"status": "bad-decoy"},
        "subagent_outputs": {
            "FIND": [
                SubagentEnvelope(
                    phase=Phase.FIND,
                    subagent_name="finder",
                    attempt=1,
                    status=SubagentStatus.SUCCEEDED,
                    payload={"summary": "real-output"},
                ).model_dump(mode="json")
            ]
        },
    }

    latest = store.read_latest(session_state, Phase.FIND)

    assert latest is not None
    assert latest.payload["summary"] == "real-output"


def test_controller_run_reads_output_via_store():
    class SpySubagentOutputStore(SubagentOutputStore):
        def __init__(self) -> None:
            super().__init__()
            self.read_calls: list[str] = []

        def read(self, session_state: dict, phase: Phase | str):
            phase_key = phase.value if isinstance(phase, Phase) else phase
            self.read_calls.append(phase_key)
            return super().read(session_state, phase)

    store = SpySubagentOutputStore()
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=SubagentRunner(subagent_output_store=store),
    )

    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    planner = build_stub_subagent(name="planner_subagent_stub", phase=Phase.PLAN)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)

    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="controller with spy subagent output store",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        executor_subagent=executor,
        sub_agents=[
            finder.build_agent(),
            planner.build_agent(),
            executor.build_agent(),
        ],
    )

    run_agent(agent)

    # Invariant: every per-phase read goes through the single store path.
    # The cycle-based loop reads PLAN/FIND/EXECUTE per cycle and SUBMIT at the
    # completion gate; sequence depends on cycle count, but every read must be
    # for a real Phase value and PLAN/FIND/EXECUTE/SUBMIT each appear at least
    # once on the happy path.
    assert all(call in {p.value for p in Phase} for call in store.read_calls)
    assert {"PLAN", "FIND", "EXECUTE", "SUBMIT"}.issubset(set(store.read_calls))
