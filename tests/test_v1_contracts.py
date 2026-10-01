from __future__ import annotations

import pytest

from adk_appworld_agent.contracts.plan import Plan, PlanTask
from adk_appworld_agent.contracts.plan_ir import (
    Milestone,
    MilestoneResult,
    MilestoneStatus,
    PlanIR,
)
from adk_appworld_agent.seams import (
    AppWorldClient,
    AuthManager,
    BudgetPolicy,
    CompletionGate,
    CompletionVerdict,
    FinderProvider,
    FinderRequest,
    FinderResult,
)


def test_milestone_requires_core_fields():
    m = Milestone(id="m1", intent="find friends")
    assert m.id == "m1"
    assert m.intent == "find friends"
    assert m.app is None


def test_milestone_rejects_missing_intent():
    with pytest.raises(ValueError):
        Milestone(id="m1")


def test_plan_ir_holds_ordered_milestones():
    plan = PlanIR(
        milestones=[
            Milestone(id="m1", intent="a"),
            Milestone(id="m2", intent="b"),
        ]
    )
    assert [m.id for m in plan.milestones] == ["m1", "m2"]


def test_plan_holds_ordered_app_labeled_tasks():
    plan = Plan(
        thoughts="Use gmail first, then venmo.",
        tasks=[
            PlanTask(task="Find the relevant email.", app="gmail"),
            PlanTask(task="Send the requested payment.", app="venmo"),
        ],
    )

    assert [task.app for task in plan.tasks] == ["gmail", "venmo"]


def test_plan_task_requires_task_and_app():
    with pytest.raises(ValueError):
        PlanTask(task="Find the relevant email.")


def test_plan_task_rejects_extra_runtime_fields():
    with pytest.raises(ValueError):
        PlanTask(task="Find the relevant email.", app="gmail", id="m1")


def test_milestone_result_roundtrip():
    r = MilestoneResult(milestone_id="m1", status=MilestoneStatus.DONE, summary="ok")
    assert r.model_validate_json(r.model_dump_json()) == r


def test_seam_protocols_accept_structural_implementations():
    class _Auth:
        def get_access_token(self, app_name: str) -> str | None:
            return None

    class _Client:
        def call(self, app: str, function: str, /, **kwargs):
            return {"app": app, "function": function}

    class _Finder:
        def find(self, request: FinderRequest) -> FinderResult:
            return FinderResult(provider_name="stub")

    class _Budget:
        def check(self, name: str, current: int) -> bool:
            return True

    class _Gate:
        def verify(self, task_id: str) -> CompletionVerdict:
            return CompletionVerdict(success=True)

    assert isinstance(_Auth(), AuthManager)
    assert isinstance(_Client(), AppWorldClient)
    assert isinstance(_Finder(), FinderProvider)
    assert isinstance(_Budget(), BudgetPolicy)
    assert isinstance(_Gate(), CompletionGate)
