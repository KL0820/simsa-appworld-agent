from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from adk_appworld_agent.contracts.limits import (
    RATIONALE_MAX_LENGTH,
)
from adk_appworld_agent.contracts.plan_ir import Milestone

NextAction = Literal["RETRY", "ADVANCE", "SUBMIT", "ABORT"]


class ContinuationDecision(BaseModel):
    next_action: NextAction
    revised_milestones: list[Milestone] | None = None
    # Producer-side target is RATIONALE_TARGET_LENGTH (soft guidance in
    # system prompt §5b). max_length below is the hard schema cap — a
    # buffer for occasional overshoot, NOT the desired length.
    rationale: str = Field(default="", max_length=RATIONALE_MAX_LENGTH)
    # RETRY    + None        → re-attempt active, no change to milestones
    # RETRY    + [m_a, ...]  → re-attempt active, replace milestones[active:] with the list
    # ADVANCE  + None        → active += 1, tail unchanged
    # ADVANCE  + [m_(a+1)..] → active += 1, replace milestones[active+1:] with the list
    # SUBMIT   / ABORT       → terminal; revised_milestones MUST be None


class ContinuationInputView(BaseModel):
    cycle_n: int
    active_milestone_index: int
    milestones: list[Milestone] = Field(default_factory=list)
    history: list[dict] = Field(default_factory=list)
    prior_variables_preview: dict = Field(default_factory=dict)
    last_failure_code: str | None = None
    last_framework_override: str | None = None
