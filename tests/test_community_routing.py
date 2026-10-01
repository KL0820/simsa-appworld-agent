from __future__ import annotations

import asyncio

from adk_appworld_agent.subagents.finder import routing


def _make_api(
    app_name: str,
    api_name: str,
    *,
    required_params: list[str] | None = None,
    response_fields: list[str] | None = None,
    description: str = "",
) -> dict:
    params = [
        {
            "name": name,
            "required": True,
            "description": f"{name} required input.",
        }
        for name in (required_params or [])
    ]
    success = {field: "string" for field in (response_fields or [])}
    return {
        "app_name": app_name,
        "api_name": api_name,
        "method": "GET",
        "description": description or f"{api_name} description",
        "parameters": params,
        "response_schemas": {"success": success},
    }


def test_community_select_graded_scoring_threshold(monkeypatch):
    """community_select uses graded LLM scores + a lenient threshold: a
    high-relevance community is kept even if another scores higher, and
    below-threshold communities are dropped. (552869a/634f342: a binary pick
    dropped a high-relevance community; graded scoring keeps it.)"""
    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Payments",
            "summary": "send/receive money",
            "operations": ["venmo__add_friend"],
            "dependencies": [],
        },
        "venmo_c1": {
            "community_id": "venmo_c1",
            "app": "venmo",
            "index": 1,
            "title": "Account",
            "summary": "subscription mgmt",
            "operations": ["venmo__show_account"],
            "dependencies": [],
        },
    }
    specs = {
        ("venmo", "add_friend"): _make_api(
            "venmo", "add_friend", response_fields=["friendship_id"]
        ),
        ("venmo", "show_account"): _make_api(
            "venmo", "show_account", response_fields=["balance"]
        ),
    }

    async def _fake_llm_json(user_prompt, sys_prompt, *, model_cfg=None):
        if "[venmo_c0]" in user_prompt and "[venmo_c1]" in user_prompt:
            # c0 high (9), c1 below threshold (2) -> only c0 kept
            return {"scores": {"venmo_c0": 9, "venmo_c1": 2}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": ["venmo.add_friend"]}
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {
            ("venmo", "add_friend"): "venmo_c0",
            ("venmo", "show_account"): "venmo_c1",
        },
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    io_trace: list[dict] = []
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Add a venmo friend.",
            planned_apps=["venmo"],
            io_trace=io_trace,
        )
    )
    # c0 (score 9 >= 5) kept; c1 (score 2 < 5) dropped
    assert result["selected_communities"] == ["venmo_c0"]
    trace = next(
        t for t in result["routing_trace"] if t.get("step") == "community_select"
    )
    assert trace["scores"] == {"venmo_c0": 9, "venmo_c1": 2}
    assert trace["min_score"] == 5


def test_search_apis_by_community_routing_selects_seed_and_dependency(monkeypatch):
    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Venmo friends",
            "summary": "Find and add friends on Venmo.",
            "operations": ["venmo__add_friend"],
            "dependencies": [{"community_id": "phone_c0"}],
        },
        "phone_c0": {
            "community_id": "phone_c0",
            "app": "phone",
            "index": 0,
            "title": "Phone contacts",
            "summary": "Look up contacts in the phone book.",
            "operations": ["phone__search_contacts"],
            "dependencies": [],
        },
    }
    specs = {
        ("venmo", "add_friend"): _make_api(
            "venmo",
            "add_friend",
            required_params=["user_email"],
            response_fields=["friendship_id"],
        ),
        ("phone", "search_contacts"): _make_api(
            "phone",
            "search_contacts",
            response_fields=["user_email", "first_name"],
        ),
    }
    dependency_index = {
        "venmo__add_friend": [
            {
                "producer_op": "phone__search_contacts",
                "consumer_op": "venmo__add_friend",
                "id_field": "user_email",
                "source": "llm_group_inference",
                "producer_community_id": "phone_c0",
                "consumer_community_id": "venmo_c0",
                "same_community": False,
            }
        ]
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "[venmo_c0]" in user_prompt and "[phone_c0]" in user_prompt:
            return {"scores": {"venmo_c0": 10, "phone_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": ["venmo.add_friend"]}
        if "Candidate prerequisite APIs for the above consumer:" in user_prompt:
            return {"selected": ["phone.search_contacts"]}
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: dependency_index)
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {
            ("venmo", "add_friend"): "venmo_c0",
            ("phone", "search_contacts"): "phone_c0",
        },
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    io_trace: list[dict] = []
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Reset Venmo friends to match my phone friends.",
            planned_apps=["venmo", "phone"],
            io_trace=io_trace,
        )
    )

    assert result["matched_apps"] == ["venmo", "phone"]
    assert result["routing_trace"][0] == {
        "step": "planned_app_filter",
        "planned_apps": ["venmo", "phone"],
        "matched_apps": ["venmo", "phone"],
        "invalid_apps": [],
        "fallback_used": False,
    }
    assert result["selected_communities"] == ["venmo_c0", "phone_c0"]
    assert result["candidate_count"] == 2
    assert {f"{item['app']}.{item['name']}" for item in result["candidate_apis"]} == {
        "venmo.add_friend",
        "phone.search_contacts",
    }
    assert result["llm_rounds"] == 3
    assert result["dependency_rounds"] == 1
    assert [item["step"] for item in io_trace] == [
        "community_select",
        "seed_filter",
        "dependency_round_1:batched",
    ]
    assert (
        "Available Communities:"
        not in io_trace[0]["model_input_raw"]["messages"][0]["parts"][0]["text"]
    )
    assert (
        "[venmo_c0]"
        in io_trace[0]["model_input_raw"]["messages"][0]["parts"][0]["text"]
    )
    assert io_trace[-1]["parsed_json"] == {"selected": ["phone.search_contacts"]}


def test_search_apis_routing_empty_filter_backstop(monkeypatch):
    """Empty-filter coverage backstop: communities selected but the seed_filter
    rejects EVERY op → must NOT return candidate_apis=[] (which left the executor
    guessing read_text_file/read_file/... → MAX_CYCLES on 0d01c76_2). Falls back
    to the selected community's own ops (graph-grounded; no op-name leak into
    community selection, so format-C / no-op-list is preserved)."""
    communities = {
        "file_system_c0": {
            "community_id": "file_system_c0",
            "app": "file_system",
            "index": 0,
            "title": "File access",
            "summary": "Read and inspect files.",
            "operations": ["file_system__show_file", "file_system__show_directory"],
            "dependencies": [],
        },
    }
    specs = {
        ("file_system", "show_file"): _make_api(
            "file_system", "show_file", response_fields=["content"]
        ),
        ("file_system", "show_directory"): _make_api(
            "file_system", "show_directory", response_fields=["entries"]
        ),
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "App catalog:" in user_prompt:
            return {"apps": []}  # compute milestone — app-select picks no app
        if "[file_system_c0]" in user_prompt:
            return {"scores": {"file_system_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": []}  # the bug scenario: filter rejects everything
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {
            ("file_system", "show_file"): "file_system_c0",
            ("file_system", "show_directory"): "file_system_c0",
        },
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Read the content of each markdown file.",
            planned_apps=["file_system"],
            io_trace=[],
        )
    )

    assert result["selected_communities"] == ["file_system_c0"]
    # Backstop fired: candidate_apis is the selected community's ops, NOT empty.
    assert result["candidate_count"] > 0
    names = {f"{item['app']}.{item['name']}" for item in result["candidate_apis"]}
    assert "file_system.show_file" in names
    assert any(
        step["step"] == "empty_filter_backstop" for step in result["routing_trace"]
    )


def test_empty_filter_no_backstop_when_app_fallback(monkeypatch):
    """When no app was assigned (planned_apps empty → app fallback) AND the
    seed_filter rejects everything, the finder must return candidate_apis=[] —
    NOT the backstop dump. This is the compute-milestone case (operates on a
    prior variable, needs no API); dumping seed_candidates[:40] there injects
    garbage (amazon-first, variant-dependent) and pollutes retrieval ablations.
    Contrast with test_search_apis_routing_empty_filter_backstop, where a real
    app WAS assigned, so the backstop still fires."""
    communities = {
        "file_system_c0": {
            "community_id": "file_system_c0",
            "app": "file_system",
            "index": 0,
            "title": "File access",
            "summary": "Read and inspect files.",
            "operations": ["file_system__show_file", "file_system__show_directory"],
            "dependencies": [],
        },
    }
    specs = {
        ("file_system", "show_file"): _make_api(
            "file_system", "show_file", response_fields=["content"]
        ),
        ("file_system", "show_directory"): _make_api(
            "file_system", "show_directory", response_fields=["entries"]
        ),
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "App catalog:" in user_prompt:
            return {"apps": []}
        if "[file_system_c0]" in user_prompt:
            return {"scores": {"file_system_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": []}  # filter rejects everything (compute milestone)
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {
            ("file_system", "show_file"): "file_system_c0",
            ("file_system", "show_directory"): "file_system_c0",
        },
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    # No planned_apps → app fallback → stage-1 app-select; compute milestone →
    # app-select returns [] → no flood, candidate_apis=[].
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "From the prior list, count items matching a rule.",
            io_trace=[],
        )
    )

    assert result["fallback_used"] is True
    assert result["candidate_apis"] == []
    assert result["candidate_count"] == 0
    sel = next(s for s in result["routing_trace"] if s["step"] == "app_select")
    assert sel["selected_apps"] == []


def test_search_apis_by_community_routing_falls_back_to_all_apps(monkeypatch):
    communities = {
        "file_system_c0": {
            "community_id": "file_system_c0",
            "app": "file_system",
            "index": 0,
            "title": "Directory listing",
            "summary": "Inspect directories and files.",
            "operations": ["file_system__show_directory"],
            "dependencies": [],
        }
    }
    specs = {
        ("file_system", "show_directory"): _make_api(
            "file_system",
            "show_directory",
            response_fields=["file_name"],
        )
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "App catalog:" in user_prompt:
            return {"apps": ["file_system"]}  # stage-1 app-select picks the app
        if "[file_system_c0]" in user_prompt:
            return {"scores": {"file_system_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": ["file_system.show_directory"]}
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {("file_system", "show_directory"): "file_system_c0"},
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    result = asyncio.run(
        routing.search_apis_by_community_routing("Organize files in my work directory.")
    )

    assert result["fallback_used"] is True
    assert result["matched_apps"] == ["file_system"]
    assert result["routing_trace"][0]["step"] == "planned_app_filter"
    assert result["candidate_count"] == 1
    assert result["candidate_apis"] == [
        {
            "app": "file_system",
            "name": "show_directory",
            "description": "show_directory description",
            "method": "GET",
        }
    ]


def test_app_select_scopes_community_select_to_picked_app(monkeypatch):
    """Stage-1 app-select on app-fallback scopes community_select to the picked
    app(s) — it does NOT flood every app's communities (the 986aa4e 8-app path)."""
    communities = {
        "todoist_c0": {
            "community_id": "todoist_c0",
            "app": "todoist",
            "index": 0,
            "title": "Tasks",
            "summary": "Manage tasks.",
            "operations": ["todoist__show_tasks"],
            "dependencies": [],
        },
        "spotify_c0": {
            "community_id": "spotify_c0",
            "app": "spotify",
            "index": 0,
            "title": "Playlists",
            "summary": "Manage playlists.",
            "operations": ["spotify__show_playlists"],
            "dependencies": [],
        },
    }
    specs = {
        ("todoist", "show_tasks"): _make_api(
            "todoist", "show_tasks", response_fields=["tasks"]
        )
    }
    seen: dict = {"community_prompt": None}

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "App catalog:" in user_prompt:
            return {"apps": ["todoist"]}  # app-select picks ONE app
        if "Score EACH community" in user_prompt:
            seen["community_prompt"] = user_prompt
            return {"scores": {"todoist_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": ["todoist.show_tasks"]}
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {("todoist", "show_tasks"): "todoist_c0"},
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    result = asyncio.run(
        routing.search_apis_by_community_routing("Find my Todoist tasks.")
    )

    sel = next(s for s in result["routing_trace"] if s["step"] == "app_select")
    assert sel["selected_apps"] == ["todoist"]
    # community_select was scoped to todoist only — spotify NOT shown (no flood)
    assert seen["community_prompt"] is not None
    assert "todoist_c0" in seen["community_prompt"]
    assert "spotify" not in seen["community_prompt"]
    assert result["candidate_apis"] == [
        {
            "app": "todoist",
            "name": "show_tasks",
            "description": "show_tasks description",
            "method": "GET",
        }
    ]


def test_search_apis_by_community_routing_uses_shared_trimmed_guidelines(monkeypatch):
    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Venmo friends",
            "summary": "Find and add friends on Venmo.",
            "operations": ["venmo__add_friend"],
            "dependencies": [],
        }
    }
    specs = {
        ("venmo", "add_friend"): _make_api(
            "venmo",
            "add_friend",
            required_params=["user_email"],
            response_fields=["friendship_id"],
        ),
        ("venmo", "login"): _make_api(
            "venmo",
            "login",
            response_fields=["access_token"],
        ),
    }
    prompts: list[tuple[str, str]] = []
    behavior_text = "\n".join(
        [
            "# Behavior Guidelines — Universal Routing Rules",
            "",
            "---",
            "",
            "### Rule 1",
            "Keep login when access is required.",
        ]
    )

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        prompts.append((user_prompt, sys_prompt))
        if "[venmo_c0]" in user_prompt:
            return {"scores": {"venmo_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:
            return {"selected": ["venmo.add_friend", "venmo.login"]}
        raise AssertionError(user_prompt)

    def _fake_load_file(path):
        if path == routing._BEHAVIOR_GUIDELINES_PATH:
            return behavior_text
        if path == routing._APP_CONTEXT_PATH:
            return "APP_CONTEXT"
        return ""

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", _fake_load_file)
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {("venmo", "add_friend"): "venmo_c0"},
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    asyncio.run(
        routing.search_apis_by_community_routing(
            "Reset Venmo friends to match my phone friends.",
            planned_apps=["venmo"],
        )
    )

    community_user_prompt, community_sys_prompt = prompts[0]
    filter_user_prompt, filter_sys_prompt = prompts[1]

    assert "Available Communities:" not in community_user_prompt
    assert "[venmo_c0]" in community_user_prompt
    assert "APP_CONTEXT" not in community_sys_prompt
    assert "Keep login when access is required." in community_sys_prompt

    assert "Candidate APIs (select from these):" in filter_user_prompt
    assert "APP_CONTEXT" in filter_sys_prompt
    assert "Keep login when access is required." in filter_sys_prompt


def test_search_apis_by_community_routing_aggregates_usage_metrics(monkeypatch):
    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Venmo friends",
            "summary": "Find and add friends on Venmo.",
            "operations": ["venmo__add_friend", "venmo__login"],
            "dependencies": [],
        }
    }
    specs = {
        ("venmo", "add_friend"): _make_api(
            "venmo",
            "add_friend",
            required_params=["user_email"],
            response_fields=["friendship_id"],
        ),
        ("venmo", "login"): _make_api(
            "venmo",
            "login",
            response_fields=["access_token"],
        ),
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "[venmo_c0]" in user_prompt:
            return {
                "parsed_json": {"scores": {"venmo_c0": 10}},
                "usage": {
                    "usage_event_count": 1,
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "thoughts_tokens": 1,
                    "total_tokens": 15,
                },
                "retry": {
                    "retry_count": 1,
                    "retry_backoff_ms": 30000,
                    "rate_limit_count": 1,
                    "provider_error_count": 1,
                },
            }
        if "Candidate APIs (select from these):" in user_prompt:
            return {
                "parsed_json": {"selected": ["venmo.add_friend", "venmo.login"]},
                "usage": {
                    "usage_event_count": 1,
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "thoughts_tokens": 2,
                    "total_tokens": 27,
                },
                "retry": {
                    "retry_count": 0,
                    "retry_backoff_ms": 0,
                    "rate_limit_count": 0,
                    "provider_error_count": 0,
                },
            }
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: {})
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {("venmo", "add_friend"): "venmo_c0"},
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)

    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Reset Venmo friends to match my phone friends.",
            planned_apps=["venmo"],
        )
    )

    assert result["usage_event_count"] == 2
    assert result["prompt_tokens"] == 30
    assert result["completion_tokens"] == 9
    assert result["thoughts_tokens"] == 3
    assert result["total_tokens"] == 42
    assert result["llm_call_attempts"] == 3
    assert result["retry_count"] == 1
    assert result["retry_backoff_ms"] == 30000
    assert result["rate_limit_count"] == 1
    assert result["provider_error_count"] == 1


# ---------------------------------------------------------------------------
# Retrieval ablation (2x2): community_layer / dependency_expansion switches.
# Shared env = one app (venmo) with TWO communities so that "all communities of
# the app" (no-community pool) is observably wider than an LLM-narrowed subset,
# and a cross-community producer dependency exercises the dep-expansion loop.
# ---------------------------------------------------------------------------
def _setup_two_community_venmo(monkeypatch):
    communities = {
        "venmo_c0": {
            "community_id": "venmo_c0",
            "app": "venmo",
            "index": 0,
            "title": "Venmo friends",
            "summary": "Add friends on Venmo.",
            "operations": ["venmo__add_friend"],
            "dependencies": [],
        },
        "venmo_c1": {
            "community_id": "venmo_c1",
            "app": "venmo",
            "index": 1,
            "title": "Venmo people search",
            "summary": "Search for Venmo users.",
            "operations": ["venmo__search_users"],
            "dependencies": [],
        },
    }
    specs = {
        ("venmo", "add_friend"): _make_api(
            "venmo",
            "add_friend",
            required_params=["user_email"],
            response_fields=["friendship_id"],
        ),
        ("venmo", "search_users"): _make_api(
            "venmo", "search_users", response_fields=["user_email"]
        ),
    }
    dependency_index = {
        "venmo__add_friend": [
            {
                "producer_op": "venmo__search_users",
                "consumer_op": "venmo__add_friend",
                "id_field": "user_email",
                "source": "llm_group_inference",
                "producer_community_id": "venmo_c1",
                "consumer_community_id": "venmo_c0",
                "same_community": False,
            }
        ]
    }

    async def _fake_llm_json(
        user_prompt: str, sys_prompt: str, *, model_cfg=None
    ) -> dict:
        if "[venmo_c0]" in user_prompt:  # community_select (only Variant A / A-no-dep)
            return {"scores": {"venmo_c0": 10}}
        if "Candidate APIs (select from these):" in user_prompt:  # seed_filter
            return {"selected": ["venmo.add_friend"]}
        if (
            "Candidate prerequisite APIs for the above consumer:" in user_prompt
        ):  # dependency filter
            return {"selected": ["venmo.search_users"]}
        raise AssertionError(user_prompt)

    monkeypatch.setattr(routing, "_load_communities", lambda: communities)
    monkeypatch.setattr(routing, "_load_api_dependency_index", lambda: dependency_index)
    monkeypatch.setattr(routing, "_load_file", lambda path: "")
    monkeypatch.setattr(
        routing,
        "_build_api_to_community_map",
        lambda: {
            ("venmo", "add_friend"): "venmo_c0",
            ("venmo", "search_users"): "venmo_c1",
        },
    )
    monkeypatch.setattr(routing, "get_api_spec", lambda app, api: specs.get((app, api)))
    monkeypatch.setattr(routing, "call_llm_json", _fake_llm_json)


def test_routing_variant_b_no_community_keeps_dependency(monkeypatch):
    """Variant B (community_layer=False, dependency_expansion=True): the
    community-select LLM is skipped; the pool is EVERY community of the planned
    app; the dependency loop still recovers the producer."""
    _setup_two_community_venmo(monkeypatch)
    io_trace: list[dict] = []
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Add my contacts as venmo friends.",
            planned_apps=["venmo"],
            community_layer=False,
            dependency_expansion=True,
            io_trace=io_trace,
        )
    )

    # No community_select model call; seed_filter then one dep round.
    assert [c["step"] for c in io_trace] == [
        "seed_filter",
        "dependency_round_1:batched",
    ]
    # Pool = all communities of the app (not an LLM-narrowed subset).
    assert result["selected_communities"] == ["venmo_c0", "venmo_c1"]
    community_step = next(
        s for s in result["routing_trace"] if s["step"] == "community_select"
    )
    assert community_step["skipped"] is True
    assert community_step["selected_communities"] == ["venmo_c0", "venmo_c1"]
    seed_step = next(
        s for s in result["routing_trace"] if s["step"] == "seed_candidates"
    )
    assert seed_step["community_count"] == 2
    # Dependency expansion kept → producer recovered.
    assert result["dependency_rounds"] == 1
    assert {f"{i['app']}.{i['name']}" for i in result["candidate_apis"]} == {
        "venmo.add_friend",
        "venmo.search_users",
    }
    assert result["llm_rounds"] == 2  # seed_filter + dep, no community_select


def test_routing_variant_b2_pure_app_to_api(monkeypatch):
    """Variant B2 (community_layer=False, dependency_expansion=False): pure
    app->api. No community_select, no dependency round; candidate set is exactly
    the single seed_filter pass."""
    _setup_two_community_venmo(monkeypatch)
    io_trace: list[dict] = []
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Add my contacts as venmo friends.",
            planned_apps=["venmo"],
            community_layer=False,
            dependency_expansion=False,
            io_trace=io_trace,
        )
    )

    assert [c["step"] for c in io_trace] == ["seed_filter"]
    assert result["selected_communities"] == ["venmo_c0", "venmo_c1"]
    assert result["dependency_rounds"] == 0
    assert any(
        s["step"] == "dependency_expansion" and s.get("skipped") is True
        for s in result["routing_trace"]
    )
    # Producer NOT recovered (no dependency expansion).
    assert {f"{i['app']}.{i['name']}" for i in result["candidate_apis"]} == {
        "venmo.add_friend"
    }
    assert result["llm_rounds"] == 1


def test_routing_variant_a_no_dep_community_without_dependency(monkeypatch):
    """Variant A-no-dep (community_layer=True, dependency_expansion=False): the
    community-select LLM still narrows the pool; the dependency loop is off."""
    _setup_two_community_venmo(monkeypatch)
    io_trace: list[dict] = []
    result = asyncio.run(
        routing.search_apis_by_community_routing(
            "Add my contacts as venmo friends.",
            planned_apps=["venmo"],
            community_layer=True,
            dependency_expansion=False,
            io_trace=io_trace,
        )
    )

    assert [c["step"] for c in io_trace] == ["community_select", "seed_filter"]
    # Community layer kept → LLM-narrowed to the single chosen community.
    assert result["selected_communities"] == ["venmo_c0"]
    assert result["dependency_rounds"] == 0
    assert {f"{i['app']}.{i['name']}" for i in result["candidate_apis"]} == {
        "venmo.add_friend"
    }
    assert result["llm_rounds"] == 2  # community_select + seed_filter
