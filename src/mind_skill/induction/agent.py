"""Induction agent A_I — extracts an InducedSkill from one (task, component slice).

The ONLY meta agent whose system prompt is a TextGrad variable: `instruction`
is a dynamic provider reading the CURRENT P_I^(q), so the optimizer can update
the prompt between iterations without rebuilding the agent. Output is forced
through `output_schema=InducedSkill` (ADK controlled generation), which is what
keeps the Fig-12 "keep format" constraint automatic — TextGrad edits P_I rules,
never the schema.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.agents import LlmAgent

from mind_skill.induction.prompts import INDUCTION_PROMPTS
from mind_skill.induction.schemas import InducedSkill
from mind_skill.runtime import (
    make_journal_callbacks,
    meta_generate_config,
    meta_model_config,
    run_meta_agent_validated,
)


class PromptHolder:
    """Mutable cell for P_I^(q). The closed loop swaps `.current`; the agent's
    dynamic instruction reads it on every call."""

    def __init__(self, component: str) -> None:
        if component not in INDUCTION_PROMPTS:
            raise KeyError(f"no initial induction prompt for component {component!r}")
        self.component = component
        self.current: str = INDUCTION_PROMPTS[component]

    def reset(self) -> None:
        self.current = INDUCTION_PROMPTS[self.component]


def build_induction_agent(
    holder: PromptHolder, *, journal_path: Path | None = None
) -> LlmAgent:
    def _instruction(_ctx=None) -> str:
        return holder.current

    callbacks: dict = {}
    if journal_path is not None:
        before, after = make_journal_callbacks(
            journal_path, role=f"induction:{holder.component}"
        )
        callbacks = {"before_model_callback": before, "after_model_callback": after}

    return LlmAgent(
        name=f"{holder.component}_induction_agent",
        model=meta_model_config().name,
        description=f"Extracts a reusable {holder.component} skill from one solved-task slice.",
        instruction=_instruction,
        output_schema=InducedSkill,
        generate_content_config=meta_generate_config(),
        **callbacks,
    )


def build_induction_input(task_instruction: str, slice_render: str) -> str:
    """A_I user message = TASK_INSTRUCTION + the component slice render (spec §9)."""
    return f"TASK_INSTRUCTION:\n{task_instruction}\n\n{slice_render}"


async def induce_skill(
    agent: LlmAgent, *, task_instruction: str, slice_render: str
) -> InducedSkill:
    return await run_meta_agent_validated(
        agent,
        build_induction_input(task_instruction, slice_render),
        InducedSkill,
    )


__all__ = [
    "PromptHolder",
    "build_induction_agent",
    "build_induction_input",
    "induce_skill",
]
