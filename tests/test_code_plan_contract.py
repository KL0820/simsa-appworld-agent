from __future__ import annotations

import pytest
from pydantic import ValidationError

from adk_appworld_agent.contracts.code_plan import CodePlanOutput, VariableSpec

# ── VariableSpec.description forcing function ─────────────────────────────────


def test_variable_spec_description_min_length_enforced():
    """A description shorter than 30 chars must trigger a pydantic
    ValidationError so the planner repair loop re-prompts instead of accepting
    a vague description (the 0a9d82a_2 / 9dabbc9_2 axis_abc_full losses came
    from descriptions that elided the milestone's content/time-window
    constraint)."""
    with pytest.raises(ValidationError):
        VariableSpec(name="x", description="too short.")


def test_variable_spec_description_blank_rejected():
    """An all-whitespace description must also fail; min_length alone is not
    enough — a 30-space string would otherwise pass."""
    with pytest.raises(ValidationError):
        VariableSpec(name="x", description=" " * 40)


def test_variable_spec_description_at_threshold_accepted():
    """30 chars exactly is accepted. Anything longer is accepted. Below 30 is
    rejected. This pins the calibration so the smoke retry-rate observation
    stays interpretable."""
    spec = VariableSpec(
        name="x",
        description="A list of items including ids.",  # 30 chars
    )
    assert len(spec.description) == 30


def test_variable_spec_description_omitted_rejected():
    """`description` is required (no default) — the planner cannot silently
    skip it. Pydantic raises ValidationError on the missing field."""
    with pytest.raises(ValidationError):
        VariableSpec(name="x")


def test_code_plan_output_propagates_description_validation():
    """The constraint flows through CodePlanOutput so the full planner output
    is rejected when output_variable.description is too short — `_parse_code_plan`
    will see a ValidationError and surface it as a parse_error."""
    with pytest.raises(ValidationError):
        CodePlanOutput(
            plan_steps=["Do work."],
            construct_step="Build result.",
            print_step="Print result.",
            output_variable=VariableSpec(name="x", description="short."),
        )
