"""Optimizer LLM — applies the gradient feedback to produce P_I^(q+1).

Prompt faithful to paper Fig 12. Receives ONLY the current prompt and the
gradient feedback (clean diagnostic-then-apply separation: no rollout cases or
scores reach the optimizer). Rule 4 is adapted to this project's documented
deviation (spec §9.0): the skill format is enforced by `output_schema=InducedSkill`,
so the protected element is the structured-output instruction, not literal
SKILL.md markdown.
"""

from __future__ import annotations

import re
from pathlib import Path

from google.adk.agents import LlmAgent

from mind_skill.runtime import (
    make_journal_callbacks,
    meta_generate_config,
    meta_model_config,
    run_meta_agent,
)

OPTIMIZER_SYSTEM = """\
Role: You are part of an optimization system that improves text prompts. You \
will receive the current prompt and feedback on its weaknesses. Produce an \
improved version.
Rules:
 1. Make targeted changes that address the specific feedback.
 2. Do not break things that are already working.
 3. Keep roughly the same length and structure.
 4. Hard constraint: The prompt MUST keep its output specification intact — the \
final "Output:" instruction that tells the agent to emit the skill as the \
structured fields of the output schema (name / description / overview / \
when_to_apply / procedure / key_patterns / common_pitfalls). Do not remove, \
weaken, or omit it.
Output: The improved prompt wrapped in <IMPROVED_VARIABLE> tags. No explanation \
outside the tags.
"""

_IMPROVED_RE = re.compile(r"<IMPROVED_VARIABLE>(.*?)</IMPROVED_VARIABLE>", re.DOTALL)


def build_optimizer_agent(*, journal_path: Path | None = None) -> LlmAgent:
    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(journal_path, role="optimizer")
        callbacks = {"before_model_callback": before, "after_model_callback": after}
    return LlmAgent(
        name="optimizer_llm",
        model=meta_model_config().name,
        description="Rewrites the induction prompt according to gradient feedback.",
        instruction=OPTIMIZER_SYSTEM,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_optimizer_input(*, induction_prompt: str, gradient_feedback: str) -> str:
    return f"CURRENT_PROMPT:\n{induction_prompt}\n\nFEEDBACK:\n{gradient_feedback}"


def extract_improved_prompt(text: str) -> str | None:
    """Pull the improved prompt out of the <IMPROVED_VARIABLE> tags.

    Returns None when the tags are missing/empty or the protected output
    instruction was dropped — callers keep P_I^(q) for the next iteration
    (a failed optimizer step must not corrupt the prompt variable).
    """
    match = _IMPROVED_RE.search(text or "")
    if not match:
        return None
    improved = match.group(1).strip()
    if not improved:
        return None
    if "output schema" not in improved.lower():
        return None  # rule-4 violation: structured-output instruction lost
    return improved


async def optimize_prompt(
    agent: LlmAgent, *, induction_prompt: str, gradient_feedback: str
) -> str | None:
    text = await run_meta_agent(
        agent,
        build_optimizer_input(
            induction_prompt=induction_prompt, gradient_feedback=gradient_feedback
        ),
    )
    return extract_improved_prompt(text)


__all__ = [
    "OPTIMIZER_SYSTEM",
    "build_optimizer_agent",
    "build_optimizer_input",
    "extract_improved_prompt",
    "optimize_prompt",
]
