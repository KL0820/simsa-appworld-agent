from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from adk_appworld_agent import APP_NAME
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.content_utils import content_from_text
from adk_appworld_agent.orchestration.state import Phase, TaskContext
from adk_appworld_agent.subagents.stubs import build_stub_subagent


def _run_stub_once(subagent_input: SubagentInput) -> SubagentEnvelope:
    async def _run() -> SubagentEnvelope:
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )
        runner = Runner(
            app_name=APP_NAME,
            agent=build_stub_subagent(
                name="executor_subagent_stub", phase=Phase.EXECUTE
            ).build_agent(),
            session_service=session_service,
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="test-user",
            session_id="test-session",
            new_message=content_from_text(
                subagent_input.model_dump_json(), role="user"
            ),
        ):
            if event.content and event.content.parts and event.content.parts[0].text:
                final_text = event.content.parts[0].text
        return SubagentEnvelope.model_validate_json(final_text)

    return asyncio.run(_run())


def test_execute_stub_carries_milestone_metadata():
    envelope = _run_stub_once(
        SubagentInput(
            phase=Phase.EXECUTE,
            attempt=2,
            task_context=TaskContext(
                task_id="task_1",
                instruction="Do the task.",
                task_datetime="2023-05-18T12:00:00",
            ),
            metadata={
                "milestone_id": "m2",
                "milestone_index": 1,
                "milestone_total": 3,
            },
        )
    )

    assert envelope.payload["milestone_id"] == "m2"
    assert envelope.payload["milestone_index"] == 1
    assert envelope.payload["milestone_total"] == 3
    assert envelope.payload["tool_call_count"] == 0
