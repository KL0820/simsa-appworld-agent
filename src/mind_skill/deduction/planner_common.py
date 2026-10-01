"""Shared pieces of the planner deduction-lite harnesses (recon+rubric loop).

spec §7 notes the planners sit at the top of the pipeline: an outcome-grounded
deduction would have to roll out the whole downstream (executor + sandbox),
whose variance swamps the skill signal. So the planner harnesses replay the
GOLD INPUT VERBATIM (the exact prompt text captured in the trajectory) through
the live component LLM — frozen thin prompt + injected skill — and judge the
reconstruction against the gold output with the component recon judge. No
sandbox, no outcome loss.

Per-component logic lives in rough_planner.py / code_planner.py; this module
holds the shared gold/result containers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerGold:
    task_id: str
    instruction: str
    # rough_planner: single prompt; code_planner: one per milestone (final attempt)
    prompts: list[str]
    milestone_intents: list[str]  # code_planner render context ([] for rough)
    milestone_apis: list[list[str]]


@dataclass
class PlannerDeductionResult:
    task_id: str
    component: str
    reconstruction: list[dict]  # parsed component outputs, in order
    render: str

    def reconstruction_render(self) -> str:
        return self.render

    def outcome_feedback(self) -> str:
        return (
            "no environment outcome for planner deduction (recon+rubric loop; "
            "the reconstruction judge's alignment is the execution signal)"
        )

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "component": self.component,
            "reconstruction": self.reconstruction,
        }


__all__ = ["PlannerDeductionResult", "PlannerGold"]
