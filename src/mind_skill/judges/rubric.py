"""Rubric judge — five-axis skill quality with the GT-leakage counterfactual.

Prompt is RUBRIC_SYSTEM, faithful to paper Fig 9. Frozen (not a TextGrad
variable). Output schema = RubricJudgment. GT-independence is the gating axis
(losses.py applies the gate).
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


class RubricJudgment(BaseModel):
    """RUBRIC_SYSTEM output (Fig 9): five-axis quality scores with GT-leakage counterfactual."""

    gt_independence: int = Field(
        ge=0,
        le=10,
        description="Fraction of content a developer could write from instruction alone (10 = nothing leaked).",
    )
    actionability: int = Field(
        ge=0,
        le=10,
        description="Can a coder use this + API docs to write working code?",
    )
    transferability: int = Field(
        ge=0,
        le=10,
        description="Would this apply to a structurally similar task in a different domain?",
    )
    completeness: int = Field(
        ge=0,
        le=10,
        description="Full procedure chain covered?",
    )
    conciseness: int = Field(
        ge=0,
        le=10,
        description="Information-dense, no redundant boilerplate?",
    )
    leaked_claims: list[str] = Field(
        default_factory=list,
        description="Claims classified (C): unknowable without seeing the reference solution.",
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Issue summary across the five dimensions.",
    )


RUBRIC_SYSTEM = """\
Role: You judge the quality of a procedural skill that will later be used by a \
coder who has NOT seen the reference solution.
You are given:
 (1) The task instruction (the same one the coder will see)
 (2) The skill that was extracted from the (hidden) reference solution
The Central Question: How much of this skill is useful procedural knowledge vs. \
leaked solution details?
For each claim in the skill, classify it:
 (A) Standard convention — a general software pattern the coder can assume without context.
 (B) Inferable from instruction — derivable from the task text alone.
 (C) Leaked from ground truth — unknowable without seeing the reference solution \
(exact API paths, specific algorithm choices, library decisions, hard-coded thresholds).
Score each dimension (0-10):
 1. GT-Independence: fraction of content a developer could write from instruction alone.
 2. Actionability: can a coder use this + API docs to write working code?
 3. Transferability: would this apply to a structurally similar task in a different domain?
 4. Completeness: full procedure chain covered?
 5. Conciseness: information-dense, no redundant boilerplate?
Output: JSON with per-dimension scores, leaked claims, and issue summary.
"""


def build_rubric_judge(*, journal_path: Path | None = None) -> LlmAgent:
    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(journal_path, role="rubric_judge")
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="rubric_judge",
        model=meta_model_config().name,
        description="Scores skill quality on five axes with GT-leakage classification.",
        instruction=RUBRIC_SYSTEM,
        output_schema=RubricJudgment,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_rubric_input(*, task_instruction: str, skill_md: str) -> str:
    return f"TASK_INSTRUCTION:\n{task_instruction}\n\nSKILL:\n{skill_md}"


async def judge_skill(
    agent: LlmAgent, *, task_instruction: str, skill_md: str
) -> RubricJudgment:
    return await run_meta_agent_validated(
        agent,
        build_rubric_input(task_instruction=task_instruction, skill_md=skill_md),
        RubricJudgment,
    )


__all__ = ["RUBRIC_SYSTEM", "build_rubric_judge", "build_rubric_input", "judge_skill"]
