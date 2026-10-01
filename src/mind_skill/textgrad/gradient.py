"""Gradient LLM — textual diagnosis of why the current P_I produced a weak skill.

Prompt faithful to paper Fig 11, including the critical-tradeoff rule that
forbids endorsing the naive "write the correct field name into the skill" fix.
Input deliberately EXCLUDES the source trajectory τ (spec §5: the gradient sees
P_I, the skill, τ̂, and the three-loss feedback — never τ — so leaked details
cannot be copied into the prompt). Plain-text output.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from mind_skill.judges.recon import ReconJudgment
from mind_skill.judges.rubric import RubricJudgment
from mind_skill.runtime import (
    make_journal_callbacks,
    meta_generate_config,
    meta_model_config,
    run_meta_agent,
)

GRADIENT_SYSTEM = """\
Role: You are part of an optimization system that improves an induction agent \
prompt. The induction agent extracts procedural skills from task solutions. A \
deduction agent then uses these skills to reconstruct the solution.
Your job: Analyze cases where quality was low, and give feedback on how to \
improve the induction agent prompt so it produces better skills.
Each case may include:
 1. Rubric scores (0-10): GT-independence, actionability, transferability, \
conciseness, plus specific leaked claims flagged by the judge.
 2. Reconstruction score/issues: deduction agent's trajectory vs. reference — \
shows what the deduction agent got wrong.
 3. Execution result: pass/fail with error messages.
Critical tradeoff: When execution fails because the deduction agent guessed the \
wrong field name, the naive fix is to write the correct name into the skill. \
Do NOT endorse this. It makes execution pass but destroys GT-independence. \
Better fix: improve the procedure wording so it guides the deduction agent to \
inspect API docs for the correct field, rather than hard-coding the field path.
Output: Describe what to change in the induction agent prompt and why. Do NOT \
propose a new prompt.
"""


def build_gradient_agent(*, journal_path: Path | None = None) -> LlmAgent:
    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(journal_path, role="gradient")
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="gradient_llm",
        model=meta_model_config().name,
        description="Diagnoses induction-prompt weaknesses from low-quality rollout cases.",
        instruction=GRADIENT_SYSTEM,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_gradient_input(
    *,
    induction_prompt: str,
    skill_md: str,
    reconstruction_render: str,
    outcome_feedback: str,
    recon_judgment: ReconJudgment,
    rubric_judgment: RubricJudgment,
) -> str:
    """Assemble the gradient case. NOTE: no source trajectory τ in here."""
    rubric_lines = [
        f"GT-independence: {rubric_judgment.gt_independence}/10",
        f"Actionability: {rubric_judgment.actionability}/10",
        f"Transferability: {rubric_judgment.transferability}/10",
        f"Completeness: {rubric_judgment.completeness}/10",
        f"Conciseness: {rubric_judgment.conciseness}/10",
    ]
    if rubric_judgment.leaked_claims:
        rubric_lines.append("Leaked claims flagged by the judge:")
        rubric_lines.extend(f"  - {c}" for c in rubric_judgment.leaked_claims)
    if rubric_judgment.issues:
        rubric_lines.append("Issues:")
        rubric_lines.extend(f"  - {c}" for c in rubric_judgment.issues)

    recon_lines = [
        f"Alignment score: {recon_judgment.alignment_score}/10",
        f"API sequence match: {recon_judgment.api_sequence_match}",
        f"Control flow match: {recon_judgment.control_flow_match}",
        f"Final state match: {recon_judgment.final_state_match}",
    ]
    if recon_judgment.mismatches:
        recon_lines.append("Procedural mismatches:")
        recon_lines.extend(f"  - {m}" for m in recon_judgment.mismatches)

    return "\n\n".join(
        [
            f"CURRENT_INDUCTION_PROMPT:\n{induction_prompt}",
            f"SKILL_PRODUCED:\n{skill_md}",
            f"DEDUCTION_TRAJECTORY:\n{reconstruction_render}",
            "RUBRIC_FEEDBACK:\n" + "\n".join(rubric_lines),
            "RECONSTRUCTION_FEEDBACK:\n" + "\n".join(recon_lines),
            f"EXECUTION_RESULT:\n{outcome_feedback}",
        ]
    )


async def compute_gradient(agent: LlmAgent, case_input: str) -> str:
    return await run_meta_agent(agent, case_input)


__all__ = [
    "GRADIENT_SYSTEM",
    "build_gradient_agent",
    "build_gradient_input",
    "compute_gradient",
]
