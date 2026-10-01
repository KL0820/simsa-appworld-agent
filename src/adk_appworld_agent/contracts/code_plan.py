from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class VariableSpec(BaseModel):
    """Schema for the planner's output variable contract.

    `description` carries a min_length=30 forcing function: the
    0a9d82a_2 / 9dabbc9_2 axis_abc_full losses both came from planner
    descriptions that elided the milestone's content / time-window
    constraint (e.g. "note metadata" instead of "notes including their
    content"). A floor on the description length pushes the planner
    toward writing the qualifier verbatim. Calibrate down (to 20 / 25)
    if smoke shows the constraint triggers too many repair retries.
    """

    name: str
    description: str = Field(..., min_length=30)

    @field_validator("name")
    @classmethod
    def _name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("variable name must not be blank")
        return value

    @field_validator("description")
    @classmethod
    def _description_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be empty or whitespace")
        return value


class CodePlanOutput(BaseModel):
    """Three required fields force the planner to produce a complete plan.

    Earlier the schema was a single `plan: list[str]`, which let Gemini Flash
    silently truncate a plan after the data-logic steps and skip the mandatory
    construct-result_dict / print-json.dumps steps — the executor then ran
    code with no `answer` field and the submission became `null` (a30375d_2
    smoke v3, 9dabbc9_2 M4 smoke v3). Splitting the trailing steps into their
    own required string fields turns that failure mode into a pydantic
    ValidationError, which the parser surfaces as a parse_error and the
    repair loop retries — instead of accepting a silently incomplete plan.
    """

    plan_steps: list[str] = Field(min_length=1)
    construct_step: str
    print_step: str
    output_variable: VariableSpec

    @field_validator("plan_steps")
    @classmethod
    def _plan_steps_must_not_be_blank(cls, value: list[str]) -> list[str]:
        cleaned = [step.strip() for step in value]
        if any(not step for step in cleaned):
            raise ValueError("code plan steps must not be blank")
        return cleaned

    @field_validator("construct_step", "print_step")
    @classmethod
    def _trailing_step_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("trailing step must not be blank")
        return cleaned

    def numbered_steps(self) -> list[str]:
        """Return plan_steps followed by construct_step and print_step.

        Callers that render the plan as a numbered list for the executor
        should use this so the executor sees one continuous, complete sequence.
        """
        return [*self.plan_steps, self.construct_step, self.print_step]


__all__ = ["CodePlanOutput", "VariableSpec"]
