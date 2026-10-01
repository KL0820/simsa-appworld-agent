from __future__ import annotations

from typing import ClassVar

from adk_appworld_agent.agent import AppWorldAgent
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.orchestrator import Orchestrator
from adk_appworld_agent.orchestration.state import (
    Phase,
)
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.subagents.base import BaseSubagent
from tests.helpers import run_agent


def _exec_output(variables: list[dict], *, milestone_id: str) -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "executor_result": {
                "answer": "null",
                "milestone_done": True,
                "summary": "did the thing",
                "variables": variables,
            },
            "finalize_called": True,
            "milestone_done": True,
            "submission_candidate": {
                "answer": "null",
                "task_type_hint": "action",
                "answer_type": "null",
                "source": "executor",
            },
            "milestone_id": milestone_id,
        },
    )


def _failed_exec_output(failure_code: str) -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.FAILED,
        payload={},
        failure_code=failure_code,
    )


class _VarEmittingExecutor(BaseSubagent):
    phase_: ClassVar[Phase] = Phase.EXECUTE
    captured: list = None  # type: ignore[assignment]

    def __init__(self, **data):
        super().__init__(**data)
        self.captured = []

    async def run_subagent(self, subagent_input: SubagentInput, ctx):
        self.captured.append(subagent_input)
        idx = subagent_input.metadata.get("milestone_index", 0)
        ms_id = subagent_input.metadata.get("milestone_id", f"m{idx + 1}")
        variables = (
            [
                {
                    "name": "wife_email",
                    "value_json": '"sarah@ex.com"',
                    "description": "Email address for the user's wife.",
                    "source_milestone_id": ms_id,
                }
            ]
            if idx == 0
            else []
        )
        envelope = SubagentEnvelope(
            phase=Phase.EXECUTE,
            subagent_name=self.name,
            attempt=subagent_input.attempt,
            status=SubagentStatus.SUCCEEDED,
            payload={
                "executor_result": {
                    "answer": "null",
                    "milestone_done": True,
                    "summary": f"done {ms_id}",
                    "variables": variables,
                },
                "finalize_called": True,
                "milestone_done": True,
                "submission_candidate": {
                    "answer": "null",
                    "task_type_hint": "action",
                    "answer_type": "null",
                    "source": "executor",
                },
                "milestone_id": ms_id,
            },
        )
        yield envelope


class _TwoMilestonePlanner(BaseSubagent):
    phase_: ClassVar[Phase] = Phase.PLAN

    async def run_subagent(self, subagent_input: SubagentInput, ctx):
        envelope = SubagentEnvelope(
            phase=Phase.PLAN,
            subagent_name=self.name,
            attempt=subagent_input.attempt,
            status=SubagentStatus.SUCCEEDED,
            payload={
                "milestones": [
                    {"id": "m1", "intent": "get email"},
                    {"id": "m2", "intent": "use email"},
                ]
            },
        )
        yield envelope


class _CapturingFinder(BaseSubagent):
    phase_: ClassVar[Phase] = Phase.FIND
    captured: list = None  # type: ignore[assignment]

    def __init__(self, **data):
        super().__init__(**data)
        self.captured = []

    async def run_subagent(self, subagent_input: SubagentInput, ctx):
        self.captured.append(subagent_input)
        envelope = SubagentEnvelope(
            phase=Phase.FIND,
            subagent_name=self.name,
            attempt=subagent_input.attempt,
            status=SubagentStatus.SUCCEEDED,
            payload={"candidate_apis": []},
        )
        yield envelope


class _NoOpSubmitter:
    def submit_answer(self, answer: str):
        return {
            "result": "Num Passed Tests : 1\nNum Failed Tests : 0\nNum Total Tests : 1"
        }


def test_controller_injects_prior_variables_into_next_milestone():
    finder = _CapturingFinder(name="finder_test", description="finder")
    planner = _TwoMilestonePlanner(name="planner_test", description="planner")
    executor = _VarEmittingExecutor(name="executor_test", description="executor")

    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="var handoff",
        controller=Orchestrator(submitter_provider=_NoOpSubmitter),
        finder_subagent=finder,
        planner_subagent=planner,
        executor_subagent=executor,
        sub_agents=[
            finder.build_agent(),
            planner.build_agent(),
            executor.build_agent(),
        ],
    )

    session, _ = run_agent(agent)

    find_outputs = (
        SubagentOutputStore().read_all(session.state).get(Phase.FIND.value, [])
    )
    assert len(find_outputs) == 2
    assert len(finder.captured) == 2
    assert finder.captured[0].metadata.get("prior_variables") in (None, {})
    assert "wife_email" in finder.captured[1].metadata["prior_variables_preview"]
    assert "sarah@ex.com" in finder.captured[1].metadata["prior_variables_preview"]
    assert len(executor.captured) == 2
    first, second = executor.captured
    assert first.metadata.get("prior_variables") in (None, {})
    assert second.metadata["prior_variables"]["wife_email"]["name"] == "wife_email"
    assert (
        second.metadata["prior_variables"]["wife_email"]["value_json"]
        == '"sarah@ex.com"'
    )
    assert (
        second.metadata["prior_variables"]["wife_email"]["description"]
        == "Email address for the user's wife."
    )
    assert "wife_email" in second.metadata["prior_variables_preview"]
