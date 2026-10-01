from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class MilestoneStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    BLOCKED = "BLOCKED"


class Milestone(BaseModel):
    id: str | None = None
    intent: str
    app: str | None = None
    candidate_apis: list[dict] = Field(default_factory=list)


class PlanIR(BaseModel):
    milestones: list[Milestone] = Field(default_factory=list)


class MilestoneResult(BaseModel):
    milestone_id: str
    status: MilestoneStatus
    summary: str = ""
    failure_code: str | None = None
