"""Frozen minimal executor prompt (MIND-Skill deduction variant) — contract tests.

Covers spec mind_skill/docs/induction_training_spec.md §6:
  - OFF path (variant="full") is byte-identical to the pre-flag behavior.
  - "minimal" renders the frozen thin prompt; skills enter ONLY through the
    Fig-8 `### SKILLS BEGIN/END` slot.
  - Builder wiring: RunConfig.executor_prompt_variant reaches the inner agent.
"""

from __future__ import annotations

import pytest

from adk_appworld_agent.orchestration.run_config import ModelConfig, RunConfig
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT,
    CODE_EXECUTOR_SYSTEM_PROMPT,
    build_code_plan_execute_subagent,
    instruction_text,
    render_executor_instruction,
    render_executor_skills_section,
)

SKILL_A = """---
name: example-skill
description: when to use it
---
## Overview
Generic overview."""

SKILL_B = """---
name: other-skill
description: another retrieval key
---
## Overview
Other overview."""


def test_full_variant_is_byte_identical_to_fat_prompt():
    assert render_executor_instruction(variant="full") == CODE_EXECUTOR_SYSTEM_PROMPT


def test_full_variant_appends_skills():
    # Decoupling: skills are appended on top of the full constraint-preserving
    # prompt (previously rejected). Default skill path stays minimal via config.
    text = render_executor_instruction(variant="full", skill_texts=[SKILL_A])
    assert text.startswith(CODE_EXECUTOR_SYSTEM_PROMPT)
    assert "### SKILLS BEGIN" in text and SKILL_A in text


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        render_executor_instruction(variant="thin")


def test_minimal_no_skills_has_no_slot_markers():
    text = render_executor_instruction(variant="minimal")
    assert text.startswith(CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT)
    assert "### SKILLS BEGIN" not in text
    assert "execute_python" in text and "submit_final" in text


def test_minimal_with_skills_fills_the_slot_in_order():
    text = render_executor_instruction(
        variant="minimal", skill_texts=[SKILL_A, SKILL_B]
    )
    begin = text.index("### SKILLS BEGIN")
    end = text.index("### SKILLS END")
    assert (
        begin
        < text.index("name: example-skill")
        < text.index("name: other-skill")
        < end
    )
    # Skills sit after the frozen necessary-state sections (Fig 8 ordering).
    assert text.index("Sandbox restrictions") < begin


def test_minimal_prompt_is_necessary_state_only():
    text = render_executor_instruction(variant="minimal", skill_texts=[SKILL_A])
    # NO examples, NO restated tool semantics, NO accreted heuristics — the
    # skill library and the tool declarations carry those.
    assert "WRONG —" not in text
    assert "Paginate" not in text
    assert "f-string" not in text
    assert "interrogative" not in text
    assert "Demonstration" not in text
    assert "What you receive" not in text
    assert "NATIVE Python value" not in text  # lives in submit_final's docstring
    # Environment facts (necessary state) stay.
    assert "Sandbox restrictions" in text
    assert "prior_variable_values" in text


def test_skills_section_empty_or_blank_renders_empty():
    assert render_executor_skills_section([]) == ""
    assert render_executor_skills_section(["  ", ""]) == ""


def test_builder_default_uses_fat_prompt():
    subagent = build_code_plan_execute_subagent(
        run_config=RunConfig(model=ModelConfig())
    )
    # Skill-capable instructions are no-templating providers; raw text unchanged.
    assert (
        instruction_text(subagent.inner_executor_agent.instruction)
        == CODE_EXECUTOR_SYSTEM_PROMPT
    )


def test_builder_minimal_variant_uses_thin_prompt():
    cfg = RunConfig(model=ModelConfig(), executor_prompt_variant="minimal")
    subagent = build_code_plan_execute_subagent(run_config=cfg)
    instruction = instruction_text(subagent.inner_executor_agent.instruction)
    assert instruction.startswith(CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT)
    assert instruction != CODE_EXECUTOR_SYSTEM_PROMPT


def test_skill_instruction_with_braces_is_not_templated():
    cfg = RunConfig(
        model=ModelConfig(),
        executor_prompt_variant="minimal",
        executor_skills=["---\nname: x\ndescription: d\n---\nUse {attributes} as-is."],
    )
    subagent = build_code_plan_execute_subagent(run_config=cfg)
    instruction = subagent.inner_executor_agent.instruction
    # callable provider -> ADK skips {identifier} state templating
    assert callable(instruction)
    assert "{attributes}" in instruction(None)
