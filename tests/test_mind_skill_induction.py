"""Offline tests for the mind_skill induction pipeline (no LLM, no RPC).

Covers: losses + lexicographic order,
optimizer <IMPROVED_VARIABLE> extraction, InducedSkill render, PromptHolder
dynamic instruction.
"""

from __future__ import annotations

import pytest

from mind_skill.induction.agent import PromptHolder, build_induction_agent
from mind_skill.induction.prompts import INDUCTION_PROMPTS
from mind_skill.induction.schemas import InducedSkill, KeyPattern, render_skill_md
from mind_skill.judges.losses import (
    GT_INDEPENDENCE_GATE,
    LossTriple,
    outcome_loss,
    recon_loss,
    rubric_loss,
)
from mind_skill.judges.recon import ReconJudgment
from mind_skill.judges.rubric import RubricJudgment
from mind_skill.textgrad.optimizer import extract_improved_prompt

# ── losses ────────────────────────────────────────────────────────────────────


def test_outcome_loss_zero_when_all_pass():
    assert outcome_loss(passed=9, failed=0, total=9) == 0.0
    assert outcome_loss(passed=0, failed=9, total=9) == 1.0
    assert outcome_loss(passed=0, failed=0, total=0) == 1.0


def test_recon_loss_inverts_alignment():
    j = ReconJudgment(
        alignment_score=8,
        api_sequence_match=True,
        control_flow_match=True,
        final_state_match=True,
    )
    assert recon_loss(j) == 2.0


def _rubric(gt: int, others: int = 8) -> RubricJudgment:
    return RubricJudgment(
        gt_independence=gt,
        actionability=others,
        transferability=others,
        completeness=others,
        conciseness=others,
    )


def test_rubric_loss_gates_on_gt_independence():
    below = _rubric(GT_INDEPENDENCE_GATE - 1, others=10)
    assert rubric_loss(below) == 10.0 - (GT_INDEPENDENCE_GATE - 1)
    above = _rubric(10, others=10)
    assert rubric_loss(above) == 0.0


def test_loss_triple_lexicographic():
    # outcome dominates recon dominates rubric
    assert LossTriple(0.0, 9.0, 9.0) < LossTriple(0.1, 0.0, 0.0)
    assert LossTriple(0.0, 1.0, 9.0) < LossTriple(0.0, 2.0, 0.0)
    assert LossTriple(0.0, 1.0, 1.0) < LossTriple(0.0, 1.0, 2.0)


# ── optimizer extraction ──────────────────────────────────────────────────────

GOOD_PROMPT = (
    "Rules...\nOutput: emit the skill as the structured fields of the output schema."
)


def test_extract_improved_prompt_happy_path():
    text = f"<IMPROVED_VARIABLE>\n{GOOD_PROMPT}\n</IMPROVED_VARIABLE>"
    assert extract_improved_prompt(text) == GOOD_PROMPT


def test_extract_improved_prompt_missing_tags_returns_none():
    assert extract_improved_prompt("no tags here") is None
    assert extract_improved_prompt("") is None


def test_extract_improved_prompt_rejects_dropped_output_spec():
    text = "<IMPROVED_VARIABLE>Rules only, format gone.</IMPROVED_VARIABLE>"
    assert extract_improved_prompt(text) is None


# ── schema render + prompt holder ─────────────────────────────────────────────


def test_render_skill_md_shape():
    skill = InducedSkill(
        name="pagination-exhaustion",
        description="Use when a list endpoint returns paged data.",
        overview="Paged sources hide the tail.",
        when_to_apply=["The instruction implies acting on ALL items."],
        procedure=["Loop pages until an empty page returns."],
        key_patterns=[
            KeyPattern(
                name="Pagination Loop", insight="while-loop, break on empty page."
            )
        ],
        common_pitfalls=["Reading only the first page."],
    )
    md = render_skill_md(skill)
    assert md.startswith("---\nname: pagination-exhaustion\n")
    for section in (
        "## Overview",
        "## When to Apply",
        "## Procedure",
        "## Key Patterns",
        "## Common Pitfalls",
    ):
        assert section in md


def test_induced_skill_rejects_non_kebab_name():
    with pytest.raises(ValueError):
        InducedSkill(
            name="Not Kebab",
            description="d",
            overview="o",
            when_to_apply=["w"],
            procedure=["p"],
            key_patterns=[KeyPattern(name="n", insight="i")],
            common_pitfalls=["c"],
        )


def test_prompt_holder_drives_dynamic_instruction():
    holder = PromptHolder("code_executor")
    agent = build_induction_agent(holder)
    assert agent.instruction(None) == INDUCTION_PROMPTS["code_executor"]
    holder.current = "P_I^(1)"
    assert agent.instruction(None) == "P_I^(1)"
    holder.reset()
    assert agent.instruction(None) == INDUCTION_PROMPTS["code_executor"]
