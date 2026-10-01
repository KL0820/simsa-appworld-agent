from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from adk_appworld_agent.contracts.plan_ir import Milestone


@runtime_checkable
class AuthManager(Protocol):
    def get_access_token(self, app_name: str) -> str | None: ...


@runtime_checkable
class AppWorldClient(Protocol):
    def call(self, app: str, function: str, /, **kwargs: Any) -> Any: ...


class FinderRequest(BaseModel):
    instruction: str
    available_apps: list[str] = Field(default_factory=list)
    active_milestone: Milestone | None = None


class FinderResult(BaseModel):
    candidate_apis: list[dict[str, Any]] = Field(default_factory=list)
    provider_name: str = ""
    debug: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class FinderProvider(Protocol):
    def find(self, request: FinderRequest) -> FinderResult: ...


@runtime_checkable
class BudgetPolicy(Protocol):
    def check(self, name: str, current: int) -> bool: ...


class CompletionVerdict(BaseModel):
    success: bool
    failure_code: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class CompletionGate(Protocol):
    def verify(self, task_id: str) -> CompletionVerdict: ...
