"""Intermediate-visibility bridge: earlier milestones' observed api returns
reach the downstream code_planner (vis-gap-apitrace).

Covers the 9016950 mechanism: M0 (search_contacts) observed last_name in its
api_trace but committed only email+phone; a later milestone's code_planner used
to see only the lossy committed variables. With
executor_sees_prior_milestone_returns=True the raw returns are surfaced.
"""

from __future__ import annotations

from adk_appworld_agent._controller_helpers import _collect_prior_milestone_returns
from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import (
    CycleHistoryEntry,
    RunState,
    TaskContext,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompt_builders import (
    _format_prior_milestone_returns_block,
)


def _contacts_trace():
    return [
        {
            "app": "phone",
            "api_name": "search_contacts",
            "status": "ok",
            "result_items": {
                "items": [
                    {
                        "first_name": "Kathryn",
                        "last_name": "Mueller",
                        "email": "k@x.com",
                    },
                    {"first_name": "Nancy", "last_name": "Powell", "email": "n@x.com"},
                ],
                "_list_total": 2,
            },
        }
    ]


def _exec_entry(cycle, idx, mid, *, success=True, api_trace=None):
    return CycleHistoryEntry(
        cycle_n=cycle,
        phase_completed="EXECUTE",
        milestone_index=idx,
        milestone_id=mid,
        milestone_intent=f"do {mid}",
        success=success,
        agent_output={"api_trace": api_trace or []},
    )


def _state_with_history(entries, active_index=1):
    st = RunState(task_context=TaskContext(task_id="t", instruction="i"))
    st.history = entries
    st.active_milestone_index = active_index
    return st


def test_collect_returns_earlier_milestone_trace():
    st = _state_with_history([_exec_entry(1, 0, "m0", api_trace=_contacts_trace())])
    out = _collect_prior_milestone_returns(st, active_id="m1", cap=3)
    assert len(out) == 1
    assert out[0]["milestone_index"] == 0
    assert out[0]["api_trace"] == _contacts_trace()


def test_collect_skips_active_milestone():
    # The active milestone's own attempts flow through prior_attempts already.
    st = _state_with_history([_exec_entry(1, 1, "m1", api_trace=_contacts_trace())])
    assert _collect_prior_milestone_returns(st, active_id="m1", cap=3) == []


def test_collect_dedupes_to_latest_per_milestone():
    older = _contacts_trace()
    newer = _contacts_trace()
    newer[0]["result_items"]["_list_total"] = 99  # distinguishable
    st = _state_with_history(
        [
            _exec_entry(1, 0, "m0", api_trace=older),
            _exec_entry(2, 0, "m0", api_trace=newer),  # retry, latest wins
        ]
    )
    out = _collect_prior_milestone_returns(st, active_id="m1", cap=3)
    assert len(out) == 1
    assert out[0]["api_trace"][0]["result_items"]["_list_total"] == 99


def test_collect_skips_failed_and_empty_trace():
    st = _state_with_history(
        [
            _exec_entry(1, 0, "m0", success=False, api_trace=_contacts_trace()),
            _exec_entry(2, 0, "m0b", api_trace=[]),  # empty trace
        ]
    )
    assert _collect_prior_milestone_returns(st, active_id="m1", cap=3) == []


def test_collect_respects_cap_and_is_chronological():
    entries = [
        _exec_entry(i + 1, i, f"m{i}", api_trace=_contacts_trace()) for i in range(5)
    ]
    out = _collect_prior_milestone_returns(
        _state_with_history(entries, active_index=5), active_id="m5", cap=3
    )
    assert [e["milestone_index"] for e in out] == [
        2,
        3,
        4,
    ]  # 3 most recent, oldest first


def test_render_surfaces_missing_field():
    block = _format_prior_milestone_returns_block(
        [
            {
                "milestone_index": 0,
                "milestone_intent": "read contacts",
                "api_trace": _contacts_trace(),
            }
        ]
    )
    assert block is not None
    assert "last_name" in block and "Mueller" in block
    assert "earlier milestones" in block


def test_render_empty_is_none():
    assert _format_prior_milestone_returns_block([]) is None
    assert (
        _format_prior_milestone_returns_block([{"milestone_index": 0, "api_trace": []}])
        is None
    )


def test_flag_default_off():
    assert RunConfig().executor_sees_prior_milestone_returns is False


def _agent_and_state():
    from adk_appworld_agent.agent import AppWorldAgent
    from adk_appworld_agent.orchestration.orchestrator import Orchestrator
    from adk_appworld_agent.orchestration.state import Milestone, Phase
    from adk_appworld_agent.subagents.stubs import build_stub_subagent

    finder = build_stub_subagent(name="finder_stub", phase=Phase.FIND)
    planner = build_stub_subagent(name="planner_stub", phase=Phase.PLAN)
    executor = build_stub_subagent(name="executor_stub", phase=Phase.EXECUTE)
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="vis-gap wiring test",
        controller=Orchestrator(),
        finder_subagent=finder,
        planner_subagent=planner,
        executor_subagent=executor,
        sub_agents=[
            finder.build_agent(),
            planner.build_agent(),
            executor.build_agent(),
        ],
    )
    st = RunState(task_context=TaskContext(task_id="t", instruction="i"))
    st.milestones = [
        Milestone(id="m0", intent="read contacts", app="phone"),
        Milestone(id="m1", intent="send message", app="phone"),
    ]
    st.active_milestone_index = 1
    st.history = [_exec_entry(1, 0, "m0", api_trace=_contacts_trace())]
    return agent, st


def test_build_exec_input_gated_by_flag():
    agent, st = _agent_and_state()

    set_active_config(RunConfig(executor_sees_prior_milestone_returns=False))
    meta_off = agent._build_exec_input(st, {}).metadata
    assert "prior_milestone_returns" not in meta_off

    set_active_config(RunConfig(executor_sees_prior_milestone_returns=True))
    meta_on = agent._build_exec_input(st, {}).metadata
    assert "prior_milestone_returns" in meta_on
    assert meta_on["prior_milestone_returns"][0]["milestone_index"] == 0
    set_active_config(RunConfig())  # restore default for other tests
