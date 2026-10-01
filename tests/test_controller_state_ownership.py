from __future__ import annotations

import pytest

from adk_appworld_agent.agent import build_appworld_controller_agent
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.orchestration.state_repo import RUN_STATE_KEY
from adk_appworld_agent.subagents.registry import workflow_subagent_impls
from tests.helpers import run_agent


def test_run_state_only_written_by_controller_agent():
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
    )
    session, _ = run_agent(agent)

    authors = [
        event.author
        for event in session.events
        if event.actions.state_delta and RUN_STATE_KEY in event.actions.state_delta
    ]

    assert authors
    assert set(authors) == {agent.name}


def test_every_selectable_impl_combination_builds_complete_controller_workflow():
    for finder in workflow_subagent_impls(Phase.FIND):
        for planner in workflow_subagent_impls(Phase.PLAN):
            executor = workflow_subagent_impls(Phase.EXECUTE)[0]
            impls = {
                Phase.FIND: finder,
                Phase.PLAN: planner,
                Phase.EXECUTE: executor,
            }
            agent = build_appworld_controller_agent(
                impls=impls,
                run_config=RunConfig(impls=impls),
            )

            assert agent.finder_subagent.phase == Phase.FIND
            assert agent.planner_subagent.phase == Phase.PLAN
            assert agent.executor_subagent.phase == Phase.EXECUTE
            assert agent.continuation_subagent.phase == Phase.PLAN


def test_default_controller_builds_current_complete_workflow():
    agent = build_appworld_controller_agent()

    assert agent.finder_subagent.phase == Phase.FIND
    assert agent.planner_subagent.phase == Phase.PLAN
    assert agent.executor_subagent.phase == Phase.EXECUTE


@pytest.mark.parametrize(
    "library,planner,executor",
    [
        ("best", "rough_skill", "code_plan_execute_skill"),
        ("q0", "rough_skill", "code_plan_execute_skill"),
        ("q1", "rough_skill", "code_plan_execute_skill"),
        ("q2", "rough_skill", "code_plan_execute_skill"),
        ("none", "rough_skill", "code_plan_execute_skill"),
        ("best", "rough_skill_native", "code_plan_execute_skill_native"),
        ("q0", "rough_skill_native", "code_plan_execute_skill_native"),
        ("q1", "rough_skill_native", "code_plan_execute_skill_native"),
        ("q2", "rough_skill_native", "code_plan_execute_skill_native"),
    ],
)
def test_every_skill_mode_derived_workflow_builds(
    library: str,
    planner: str,
    executor: str,
):
    impls = {
        Phase.FIND: "community",
        Phase.PLAN: planner,
        Phase.EXECUTE: executor,
    }
    agent = build_appworld_controller_agent(
        impls=impls,
        run_config=RunConfig(impls=impls, skills_library=library),
    )

    assert agent.finder_subagent.phase == Phase.FIND
    assert agent.planner_subagent.phase == Phase.PLAN
    assert agent.executor_subagent.phase == Phase.EXECUTE
