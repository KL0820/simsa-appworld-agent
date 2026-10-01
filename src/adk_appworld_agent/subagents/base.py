from __future__ import annotations

from abc import abstractmethod
from typing import AsyncIterator, ClassVar

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from pydantic import BaseModel, ConfigDict

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.cache.spec import CacheSpec
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.failure_codes import SUBAGENT_INPUT_INVALID


class Subagent(BaseModel):
    """Project-owned subagent contract.

    A Subagent is not an ADK agent. It owns phase contracts, cache/log metadata,
    and local execution logic. `build_agent()` creates the thin ADK adapter used
    by ADK Runner when the orchestrator invokes the subagent.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""

    phase_: ClassVar[Phase]
    input_schema_: ClassVar[type[BaseModel]] = SubagentInput
    output_schema_: ClassVar[type[BaseModel]] = SubagentEnvelope
    log_spec_: ClassVar[object | None] = None
    cache_spec_: ClassVar[CacheSpec | None] = None

    @property
    def phase(self) -> Phase:
        return self.phase_

    @property
    def input_schema(self) -> type[BaseModel]:
        return self.input_schema_

    @property
    def output_schema(self) -> type[BaseModel]:
        return self.output_schema_

    @property
    def log_spec(self) -> object | None:
        return self.log_spec_

    @property
    def cache_spec(self) -> CacheSpec | None:
        return self.cache_spec_

    def build_agent(self) -> BaseAgent:
        return _SubagentAgentAdapter(
            name=self.name,
            description=self.description,
            subagent=self,
        )


class BaseSubagent(Subagent):
    """Base class for subagents implemented as async envelope generators."""

    @abstractmethod
    def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        """Yield one or more envelopes for this subagent run."""

    def succeeded(self, *, attempt: int, payload: dict) -> SubagentEnvelope:
        return SubagentEnvelope(
            phase=self.phase_,
            subagent_name=self.name,
            attempt=attempt,
            status=SubagentStatus.SUCCEEDED,
            payload=payload,
        )

    def failed(
        self,
        *,
        attempt: int,
        payload: dict,
        failure_code: str,
        error: str | None = None,
    ) -> SubagentEnvelope:
        body = dict(payload)
        if error:
            body["error"] = error
        return SubagentEnvelope(
            phase=self.phase_,
            subagent_name=self.name,
            attempt=attempt,
            status=SubagentStatus.FAILED,
            payload=body,
            failure_code=failure_code,
        )


class _SubagentAgentAdapter(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    subagent: Subagent

    async def _run_async_impl(self, ctx):
        try:
            subagent_input = SubagentInput.model_validate_json(
                content_to_text(ctx.user_content)
            )
        except Exception as exc:
            if isinstance(self.subagent, BaseSubagent):
                envelope = self.subagent.failed(
                    attempt=0,
                    payload={},
                    failure_code=SUBAGENT_INPUT_INVALID,
                    error=str(exc),
                )
            else:
                envelope = SubagentEnvelope(
                    phase=self.subagent.phase,
                    subagent_name=self.subagent.name,
                    attempt=0,
                    status=SubagentStatus.FAILED,
                    payload={"error": str(exc)},
                    failure_code=SUBAGENT_INPUT_INVALID,
                )
            yield self._emit(ctx, envelope)
            return

        if not isinstance(self.subagent, BaseSubagent):
            envelope = SubagentEnvelope(
                phase=subagent_input.phase,
                subagent_name=self.subagent.name,
                attempt=subagent_input.attempt,
                status=SubagentStatus.FAILED,
                payload={},
                failure_code="SUBAGENT_NOT_RUNNABLE",
            )
            yield self._emit(ctx, envelope)
            return

        async for envelope in self.subagent.run_subagent(subagent_input, ctx):
            yield self._emit(ctx, envelope)

    def _emit(self, ctx, envelope: SubagentEnvelope) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model",
                parts=[types.Part(text=envelope.model_dump_json())],
            ),
        )


__all__ = ["BaseSubagent", "Subagent"]
