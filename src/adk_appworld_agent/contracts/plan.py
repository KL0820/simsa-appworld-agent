from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str = Field(
        min_length=1,
        description=(
            "One single-app work unit with one primary action class and one "
            "independently checkable success condition. Preserve the user's "
            "source set and constraints; do not broaden read tasks into a "
            "whole app collection unless requested; do not combine alternative "
            "or opposite mutations; keep source sets consistent with thoughts; "
            "do not mention API implementation details."
        ),
    )
    app: str = Field(
        min_length=1,
        description="Exactly one app name that owns this work unit.",
    )


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thoughts: str = Field(
        default="",
        description=(
            "First identify the task-specific constraints that must be "
            "preserved, then explain the decomposition."
        ),
    )
    tasks: list[PlanTask] = Field(
        default_factory=list,
        description=(
            "Ordered single-app work units. Split reads and separately "
            "verifiable state changes into separate tasks."
        ),
    )


__all__ = ["Plan", "PlanTask"]
