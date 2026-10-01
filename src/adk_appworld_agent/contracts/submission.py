from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SubmissionCandidate(BaseModel):
    """Subagent-produced request to submit a final answer.

    Subagents cannot submit directly. They emit a candidate; the `CompletionGate`
    decides whether and how to call the AppWorld evaluator.
    """

    answer: str
    task_type_hint: Literal["query", "action"]
    answer_type: Literal["str", "int", "float", "null"]
    source: str = "executor"


class CompletionStatus(str, Enum):
    SUBMITTED = "SUBMITTED"
    BLOCKED = "BLOCKED"


class CompletionDecision(BaseModel):
    status: CompletionStatus
    submitted_answer: str | None = None
    evaluation_report: Any = None
    block_reason: str | None = None
    candidate: SubmissionCandidate | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
