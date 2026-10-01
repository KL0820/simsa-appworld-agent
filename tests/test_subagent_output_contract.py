from __future__ import annotations

from adk_appworld_agent.agent import build_appworld_controller_agent
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from tests.helpers import run_agent


def test_subagent_output_store_is_append_only():
    store = SubagentOutputStore()
    session_state: dict = {}

    first = SubagentEnvelope(
        phase=Phase.FIND,
        subagent_name="finder",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={"summary": "first"},
    )
    second = SubagentEnvelope(
        phase=Phase.FIND,
        subagent_name="finder",
        attempt=2,
        status=SubagentStatus.SUCCEEDED,
        payload={"summary": "second"},
    )

    session_state.update(store.append_delta(session_state, first))
    session_state.update(store.append_delta(session_state, second))

    assert len(session_state["subagent_outputs"]["FIND"]) == 2
    assert session_state["subagent_outputs"]["FIND"][0]["payload"]["summary"] == "first"
    assert (
        session_state["subagent_outputs"]["FIND"][1]["payload"]["summary"] == "second"
    )


def test_invalid_subagent_output_becomes_synthetic_failure():
    agent = build_appworld_controller_agent(
        impls={Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        invalid_phase=Phase.FIND,
    )
    session, _ = run_agent(agent)
    latest = SubagentOutputStore().read_latest(session.state, Phase.FIND)

    assert latest is not None
    assert latest.status == SubagentStatus.FAILED
    assert latest.failure_code == "SUBAGENT_OUTPUT_INVALID"


def test_missing_subagent_output_becomes_synthetic_failure():
    agent = build_appworld_controller_agent(
        impls={Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        missing_phase=Phase.FIND,
    )
    session, _ = run_agent(agent)
    latest = SubagentOutputStore().read_latest(session.state, Phase.FIND)

    assert latest is not None
    assert latest.status == SubagentStatus.FAILED
    assert latest.failure_code == "SUBAGENT_OUTPUT_MISSING"
