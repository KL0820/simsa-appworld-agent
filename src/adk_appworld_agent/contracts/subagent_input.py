from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from adk_appworld_agent.orchestration.state import Phase, TaskContext


class SubagentInput(BaseModel):
    phase: Phase
    attempt: int
    task_context: TaskContext
    metadata: dict[str, Any] = Field(default_factory=dict)
