from __future__ import annotations

from typing import Literal

from google.adk.agents import BaseAgent
from google.adk.events import Event
from google.genai import types
from pydantic import ConfigDict, Field

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent


class DeterministicStubSubagent(BaseSubagent):
    phase_name: Phase
    stub_status: SubagentStatus = SubagentStatus.SUCCEEDED
    emit_mode: Literal["valid", "invalid", "missing"] = "valid"
    stub_payload: dict = Field(default_factory=dict)
    failure_code: str | None = None

    @property
    def phase(self) -> Phase:
        return self.phase_name

    def build_agent(self) -> BaseAgent:
        return _DeterministicStubAgent(
            name=self.name,
            description=self.description,
            subagent=self,
        )

    async def run_subagent(self, subagent_input: SubagentInput, ctx):
        payload = (
            dict(self.stub_payload)
            if self.stub_payload
            else {"summary": f"{self.phase_name.value.lower()} stub completed"}
        )
        if self.phase_name == Phase.EXECUTE:
            metadata = subagent_input.metadata or {}
            payload.setdefault("milestone_id", metadata.get("milestone_id") or None)
            payload.setdefault("milestone_index", metadata.get("milestone_index", 0))
            payload.setdefault("milestone_total", metadata.get("milestone_total", 1))
            payload.setdefault("tool_call_count", 0)
        envelope = SubagentEnvelope(
            phase=self.phase_name,
            subagent_name=self.name,
            attempt=subagent_input.attempt,
            status=self.stub_status,
            payload=payload,
            failure_code=self.failure_code,
        )
        yield envelope


class _DeterministicStubAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    subagent: DeterministicStubSubagent

    async def _run_async_impl(self, ctx):
        if self.subagent.emit_mode == "missing":
            if False:
                yield  # pragma: no cover
            return

        if self.subagent.emit_mode == "invalid":
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model", parts=[types.Part(text="not-json")]
                ),
            )
            return

        subagent_input = SubagentInput.model_validate_json(
            content_to_text(ctx.user_content)
        )
        envelope = None
        async for item in self.subagent.run_subagent(subagent_input, ctx):
            envelope = item
            break
        if envelope is None:
            return
        yield Event(
            invocation_id=ctx.invocation_id,
            author=self.name,
            branch=ctx.branch,
            content=types.Content(
                role="model", parts=[types.Part(text=envelope.model_dump_json())]
            ),
        )


def build_stub_subagent(
    *,
    name: str,
    phase: Phase,
    stub_status: SubagentStatus = SubagentStatus.SUCCEEDED,
    emit_mode: Literal["valid", "invalid", "missing"] = "valid",
) -> DeterministicStubSubagent:
    failure_code = None
    payload: dict = {"summary": f"{phase.value.lower()} stub completed"}
    if stub_status == SubagentStatus.FAILED:
        failure_code = f"{phase.value}_STUB_FAILURE"
        payload = {}
    elif phase == Phase.PLAN:
        payload["milestones"] = [
            {
                "id": "m_stub",
                "intent": "stub milestone",
            }
        ]
    elif phase == Phase.EXECUTE:
        payload["executor_result"] = {
            "answer": "stub-answer",
            "milestone_done": True,
            "summary": "execute stub completed",
            "variables": [],
        }
        payload["finalize_called"] = True
        payload["milestone_done"] = True
        payload["submission_candidate"] = {
            "answer": "stub-answer",
            "task_type_hint": "query",
            "answer_type": "str",
            "source": "executor_stub",
        }

    return DeterministicStubSubagent(
        name=name,
        description=f"Deterministic stub subagent for {phase.value}",
        phase_name=phase,
        stub_status=stub_status,
        emit_mode=emit_mode,
        stub_payload=payload,
        failure_code=failure_code,
    )
