from __future__ import annotations

import asyncio

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from adk_appworld_agent import APP_NAME
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.content_utils import content_from_text
from adk_appworld_agent.orchestration.state import Phase, TaskContext
from adk_appworld_agent.subagents.finder.community_finder import (
    build_community_finder_subagent,
)
from adk_appworld_agent.subagents.finder.community_finder.prompts import (
    FILTER_SYS_PROMPT,
)
from adk_appworld_agent.subagents.registry import (
    available_subagent_impls,
    build_subagent,
)


def test_finder_filter_prompt_has_action_milestone_rules():
    # Action milestones in full56 v1 frequently lost the verb's API or its
    # required identifier-resolution producer (search_contacts/search_users
    # /show_payment_cards). The filter prompt must explicitly require both.
    assert "Action-milestone rules" in FILTER_SYS_PROMPT
    assert "state-changing verb" in FILTER_SYS_PROMPT
    assert "MUST be" in FILTER_SYS_PROMPT and "selected" in FILTER_SYS_PROMPT
    assert "Never drop the verb's API" in FILTER_SYS_PROMPT
    assert "search_contacts" in FILTER_SYS_PROMPT
    assert "search_users" in FILTER_SYS_PROMPT
    assert "show_payment_cards" in FILTER_SYS_PROMPT


def _run_subagent_once(agent, subagent_input: SubagentInput) -> SubagentEnvelope:
    async def _run():
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=APP_NAME,
            user_id="test-user",
            session_id="test-session",
        )
        runner = Runner(
            app_name=APP_NAME,
            agent=agent.build_agent(),
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


def test_registry_exposes_community_finder():
    assert "community" in available_subagent_impls(Phase.FIND)
    finder = build_subagent(Phase.FIND, "community")
    assert "finder" in finder.name


def test_community_finder_subagent_wraps_routing_result(monkeypatch):
    async def _fake_search(
        task_instruction: str,
        *,
        planned_apps: list[str] | None = None,
        model_cfg=None,
        **kwargs,
    ) -> dict:
        assert "venmo" in task_instruction.lower()
        assert planned_apps == ["venmo", "phone"]
        return {
            "matched_apps": ["venmo", "phone"],
            "fallback_used": False,
            "selected_communities": ["venmo_c0", "phone_c0"],
            "candidate_apis": [
                {
                    "app": "venmo",
                    "name": "add_friend",
                    "description": "add a friend",
                    "method": "POST",
                },
                {
                    "app": "phone",
                    "name": "search_contacts",
                    "description": "search contacts",
                    "method": "GET",
                },
            ],
            "candidate_count": 2,
            "llm_rounds": 4,
            "llm_call_attempts": 5,
            "usage_event_count": 4,
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "thoughts_tokens": 10,
            "total_tokens": 160,
            "retry_count": 1,
            "retry_backoff_ms": 30000,
            "rate_limit_count": 1,
            "provider_error_count": 1,
            "dependency_rounds": 1,
            "routing_trace": [
                {"step": "seed_filter", "selected_api_keys": ["venmo.add_friend"]}
            ],
        }

    monkeypatch.setattr(
        "adk_appworld_agent.subagents.finder.community_finder.agent.search_apis_by_community_routing",
        _fake_search,
    )
    subagent = build_community_finder_subagent()
    envelope = _run_subagent_once(
        subagent,
        SubagentInput(
            phase=Phase.FIND,
            attempt=1,
            task_context=TaskContext(
                task_id="3d9a636_2",
                instruction="Reset Venmo friends to match my phone friends.",
                task_datetime="2023-05-18T12:00:00",
            ),
            metadata={"planned_apps": ["venmo", "phone"]},
        ),
    )

    assert envelope.status == SubagentStatus.SUCCEEDED
    assert envelope.payload["matched_apps"] == ["venmo", "phone"]
    assert envelope.payload["candidate_count"] == 2
    assert envelope.payload["selected_communities"] == ["venmo_c0", "phone_c0"]
    assert envelope.metrics["llm_calls"] == 4
    assert envelope.metrics["llm_call_attempts"] == 5
    assert envelope.metrics["usage_event_count"] == 4
    assert envelope.metrics["prompt_tokens"] == 120
    assert envelope.metrics["completion_tokens"] == 30
    assert envelope.metrics["thoughts_tokens"] == 10
    assert envelope.metrics["total_tokens"] == 160
    assert envelope.metrics["retry_count"] == 1
    assert envelope.metrics["retry_backoff_ms"] == 30000
    assert envelope.metrics["rate_limit_count"] == 1
    assert envelope.metrics["provider_error_count"] == 1
    assert envelope.metrics["dependency_rounds"] == 1


def test_community_finder_subagent_captures_io_when_requested(monkeypatch):
    async def _fake_search(
        task_instruction: str,
        *,
        planned_apps: list[str] | None = None,
        model_cfg=None,
        io_trace: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        assert "venmo" in task_instruction.lower()
        assert planned_apps == ["venmo"]
        if io_trace is not None:
            io_trace.append(
                {
                    "step": "community_select",
                    "model_input_raw": {
                        "system_instruction": "select communities",
                        "messages": [
                            {
                                "role": "user",
                                "parts": [{"text": "[venmo_c0] Venmo friends"}],
                            }
                        ],
                    },
                    "raw_llm_text": '{"selected": ["venmo_c0"]}',
                    "parsed_json": {"selected": ["venmo_c0"]},
                }
            )
        return {
            "matched_apps": ["venmo"],
            "fallback_used": False,
            "selected_communities": ["venmo_c0"],
            "candidate_apis": [
                {
                    "app": "venmo",
                    "name": "add_friend",
                    "description": "add a friend",
                    "method": "POST",
                },
            ],
            "candidate_count": 1,
            "llm_rounds": 1,
            "dependency_rounds": 0,
            "routing_trace": [
                {"step": "planned_app_filter", "planned_apps": ["venmo"]}
            ],
        }

    monkeypatch.setattr(
        "adk_appworld_agent.subagents.finder.community_finder.agent.search_apis_by_community_routing",
        _fake_search,
    )
    subagent = build_community_finder_subagent()
    envelope = _run_subagent_once(
        subagent,
        SubagentInput(
            phase=Phase.FIND,
            attempt=1,
            task_context=TaskContext(
                task_id="3d9a636_2",
                instruction="Reset Venmo friends.",
                task_datetime="2023-05-18T12:00:00",
            ),
            metadata={"capture_io": True, "planned_apps": ["venmo"]},
        ),
    )

    # io now lives on envelope.io (post-refactor); payload stays domain-only.
    io_record = envelope.io
    assert "io_format" not in io_record
    assert io_record["input"]["subagent_input_text"]
    assert io_record["input"]["subagent_input"]["phase"] == "FIND"
    assert io_record["input"]["planned_apps"] == ["venmo"]
    assert io_record["input"]["model_calls"][0]["step"] == "community_select"
    assert io_record["output"]["model_calls"][0]["parsed_json"] == {
        "selected": ["venmo_c0"]
    }
    assert "add_friend" in io_record["output"]["raw_llm_text"]


def test_build_community_text_format_c_capability_split_no_op_names(monkeypatch):
    """Format C (api_graph_refactor_spec.md §Step 7): title + summary + op
    count + (added 2026-06-03) a METHOD-DERIVED read/write capability split.
    The split disambiguates similarly-titled communities the summaries alone
    confuse (file_system_c0 vs c2) but is derived from each op's HTTP method,
    NOT from op names — so the core format-C invariant holds: NO op-name list,
    so the LLM cannot bypass the community abstraction by pattern-matching
    names.

    Regression test: capability split present; op names still absent."""
    from adk_appworld_agent.subagents.finder import routing
    from adk_appworld_agent.subagents.finder.routing import _build_community_text

    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Friend Management",
            "summary": "APIs for managing the venmo friend list.",
            "operations": [
                "venmo__add_friend",
                "venmo__remove_friend",
                "venmo__search_friends",
            ],
        }
    }
    methods = {
        ("venmo", "add_friend"): "POST",
        ("venmo", "remove_friend"): "DELETE",
        ("venmo", "search_friends"): "GET",
    }
    monkeypatch.setattr(
        routing,
        "get_api_spec",
        lambda app, name: {"method": methods.get((app, name), "")},
    )

    text = _build_community_text(communities, app_filter={"venmo"})

    # Count + title + summary + method-derived capability split present
    assert (
        "[venmo_c0] Friend Management (3 APIs: 1 read/retrieve, 2 create/modify/delete)"
        in text
    )
    assert "APIs for managing the venmo friend list." in text

    # CORE INVARIANT: individual op names must NOT leak into the prompt
    assert "add_friend" not in text
    assert "remove_friend" not in text
    assert "search_friends" not in text
