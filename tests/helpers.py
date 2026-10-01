from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from adk_appworld_agent import APP_NAME
from adk_appworld_agent.orchestration.content_utils import content_from_text


def run_agent(agent, message: str = "Run the skeleton workflow.") -> object:
    async def _run():
        session_service = InMemorySessionService()
        user_id = "test-user"
        session_id = "test-session"
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        runner = Runner(
            app_name=APP_NAME,
            agent=agent,
            session_service=session_service,
        )
        events = []
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content_from_text(message, role="user"),
        ):
            events.append(event)
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        return session, events

    return asyncio.run(_run())
