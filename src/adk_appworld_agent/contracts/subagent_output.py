from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from adk_appworld_agent.orchestration.state import Phase


class SubagentStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class SubagentEnvelope(BaseModel):
    phase: Phase
    subagent_name: str
    attempt: int
    status: SubagentStatus
    payload: dict[str, Any] = Field(default_factory=dict)
    failure_code: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    # Observability sidecar — rendered_prompt, history, parsed output, etc.
    # Populated by subagents when subagent_input.metadata.capture_io is set.
    # Kept OUT of `payload` so the domain output schema stays clean (downstream
    # consumers like continuation_planner only ever read payload's task-shaped
    # fields). The io sink reads envelope.io (or the equivalent in the
    # serialized dict form) to build subagent_io.jsonl entries.
    io: dict[str, Any] = Field(default_factory=dict)
