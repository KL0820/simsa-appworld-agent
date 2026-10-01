"""Reconstruction judge — procedural alignment between τ_C (reference) and τ̂_C.

Prompt is TRAJECTORY_JUDGE_SYSTEM, faithful to paper Fig 10. Frozen (not a
TextGrad variable). Output schema = ReconJudgment.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from mind_skill.runtime import (
    make_journal_callbacks,
    meta_generate_config,
    meta_model_config,
    run_meta_agent_validated,
)


class ReconJudgment(BaseModel):
    """TRAJECTORY_JUDGE_SYSTEM output (Fig 10): procedural alignment, not text match."""

    alignment_score: int = Field(
        ge=0,
        le=10,
        description="0-10 procedural alignment between reference and reconstructed trajectory (10 = same strategy).",
    )
    api_sequence_match: bool = Field(
        description="Same sequence of API-call families (auth -> list -> detail -> action), order and dependencies."
    )
    control_flow_match: bool = Field(
        description="Same control flow: pagination, accumulation loops, early-exit conditions, branching."
    )
    final_state_match: bool = Field(
        description="Deduction agent's final environment observation converges to the same outcome."
    )
    mismatches: list[str] = Field(
        default_factory=list,
        description="Procedural mismatches (ignore variable names, prints, step counts, specific IDs/values).",
    )


# Faithful to Fig 10; the only adaptation is naming our milestone-based executor
# trajectory instead of "ReAct trajectory" (our deduction agent is the milestone
# executor, not a ReAct loop).
TRAJECTORY_JUDGE_SYSTEM = """\
Role: Evaluate whether a deduction agent's execution trajectory follows the same \
procedural strategy as a reference trajectory on the same task.
Evaluate on procedural alignment, not literal text match.
Criteria:
 1. Same sequence of API-call families (e.g., auth -> list -> detail -> action), \
order and dependencies.
 2. Same control flow: pagination, accumulation loops, early-exit conditions, branching.
 3. Deduction agent's final environment observation converges to the same outcome.
Ignore: Different variable names, intermediate print statements, step-count \
differences, specific IDs/values.
Output: JSON with alignment score (0-10), boolean flags for API sequence / control \
flow / final state match, and list of procedural mismatches.
"""


# Component variants: same Fig-10 frame (role / criteria / ignore / output),
# criteria swapped to the component's decision granularity. The executor judge
# compares API-call behavior; the planner judges compare PLANS, so criterion 3
# (environment outcome) becomes coverage/data-flow equivalence instead.
PLAN_JUDGE_SYSTEM = """\
Role: Evaluate whether a planner's task decomposition follows the same \
procedural strategy as a reference decomposition of the same task.
Evaluate on procedural alignment, not literal text match.
Criteria:
 1. Same set of work units: every state read and state change covered by the \
reference is covered, in a dependency-compatible order.
 2. Same data flow: a unit that feeds an identifier / value / evidence to a \
later unit appears before it and is referenced by it.
 3. Same constraint coverage: selection conditions, exact values, output \
format, and the final deliverable appear in the same work units.
Ignore: Different wording, different unit counts when one unit cleanly merges \
or splits another without changing order or data flow, specific IDs/values.
Output: JSON with alignment score (0-10), boolean flags for API sequence \
(here: work-unit sequence) / control flow (here: data flow) / final state \
(here: constraint coverage) match, and list of procedural mismatches.
"""

CODE_PLAN_JUDGE_SYSTEM = """\
Role: Evaluate whether a code-planner's per-milestone step plans follow the \
same procedural strategy as the reference step plans for the same task.
Evaluate on procedural alignment, not literal text match.
Criteria:
 1. Same sequence of API-call families per milestone (which candidate API is \
used for what), order and dependencies.
 2. Same control flow: pagination, iteration, filtering, aggregation, and \
how prior-milestone variables are read.
 3. Same output contract: the constructed result and printed four-field \
contract carry the same data forward.
Ignore: Different step wording, step-count differences, variable names, \
specific IDs/values.
Output: JSON with alignment score (0-10), boolean flags for API sequence / \
control flow / output contract match, and list of procedural mismatches.
"""

RECON_JUDGE_PROMPTS: dict[str, str] = {
    "code_executor": TRAJECTORY_JUDGE_SYSTEM,
    "code_planner": CODE_PLAN_JUDGE_SYSTEM,
    "rough_planner": PLAN_JUDGE_SYSTEM,
}


def build_recon_judge(
    *, component: str = "code_executor", journal_path: Path | None = None
) -> LlmAgent:
    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(
            journal_path, role=f"recon_judge:{component}"
        )
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="recon_judge",
        model=meta_model_config().name,
        description="Scores procedural alignment between reference and reconstructed trajectories.",
        instruction=RECON_JUDGE_PROMPTS[component],
        output_schema=ReconJudgment,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_recon_input(
    *, task_instruction: str, reference_render: str, reconstruction_render: str
) -> str:
    return (
        f"TASK_INSTRUCTION:\n{task_instruction}\n\n"
        f"REFERENCE_TRAJECTORY:\n{reference_render}\n\n"
        f"DEDUCTION_TRAJECTORY:\n{reconstruction_render}"
    )


async def judge_reconstruction(
    agent: LlmAgent,
    *,
    task_instruction: str,
    reference_render: str,
    reconstruction_render: str,
) -> ReconJudgment:
    return await run_meta_agent_validated(
        agent,
        build_recon_input(
            task_instruction=task_instruction,
            reference_render=reference_render,
            reconstruction_render=reconstruction_render,
        ),
        ReconJudgment,
    )


__all__ = [
    "TRAJECTORY_JUDGE_SYSTEM",
    "build_recon_judge",
    "build_recon_input",
    "judge_reconstruction",
]
