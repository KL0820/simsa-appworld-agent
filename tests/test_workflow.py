from __future__ import annotations

from adk_appworld_agent.observability.markdown import (
    render_batch_results_markdown,
    render_task_summary_markdown,
)
from adk_appworld_agent.observability.workflow import (
    build_task_summary,
    build_workflow_records,
)


def test_build_task_summary_includes_execute_api_aggregate():
    summary = build_task_summary(
        task_id="task_1",
        result={
            "status": "FAILED",
            "phase": "FAILED",
            "task_wall_ms": 12345,
            "passed": 1,
            "failed": 1,
            "total": 2,
            "exec_failure": None,
            "submit_block": "COMPLETION_EVAL_FAILED",
            "block_reason": "COMPLETION_EVAL_FAILED",
        },
        final_state={
            "phase": "FAILED",
            "status": "FAILED",
            "task_context": {
                "instruction": "Reset friends on venmo to match my phone contacts.",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [
                {
                    "intent": "Read Venmo friends.",
                    "app": "venmo",
                }
            ],
        },
        subagent_outputs={
            "PLAN": [
                {
                    "payload": {
                        "tasks": [{"task": "Read Venmo friends.", "app": "venmo"}],
                    },
                }
            ],
            "FIND": [
                {
                    "attempt": 1,
                    "payload": {
                        "matched_apps": ["venmo"],
                        "selected_communities": ["venmo_c3"],
                        "candidate_count": 2,
                        "candidate_apis": [
                            {"app": "venmo", "name": "search_friends"},
                            {"app": "venmo", "name": "add_friend"},
                        ],
                        "io": {
                            "input": {
                                "subagent_input": {
                                    "metadata": {
                                        "milestone_index": 0,
                                        "milestone_intent": "Read Venmo friends.",
                                        "planned_apps": ["venmo"],
                                    }
                                }
                            }
                        },
                    },
                    "metrics": {
                        "llm_calls": 3,
                        "dependency_rounds": 1,
                        "wall_ms": 2000,
                    },
                }
            ],
            "EXECUTE": [
                {
                    "status": "SUCCEEDED",
                    "payload": {
                        "milestone_index": 0,
                        "tool_call_count": 4,
                        "submission_candidate": {
                            "answer": "null",
                            "task_type_hint": "action",
                            "answer_type": "null",
                            "source": "executor",
                        },
                        "code_plan": {
                            "plan": [
                                "Read the current Venmo friends.",
                                "Compare the Venmo friends to the phone contacts.",
                                "Print json.dumps(result_dict) as the final stdout line.",
                            ],
                            "output_variable": {
                                "name": "venmo_friend_sync_plan",
                                "description": "Plan for synchronizing Venmo friends.",
                            },
                        },
                        "metrics": {
                            "llm_calls": 5,
                            "wall_ms": 4000,
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "thoughts_tokens": 7,
                        },
                    },
                }
            ],
            "SUBMIT": [
                {
                    "payload": {
                        "status": "BLOCKED",
                        "extra": {"passed": 1, "failed": 1, "total": 2},
                    }
                }
            ],
        },
        sandbox_trace=[
            {
                "api_calls": [
                    {
                        "kind": "api_docs.show_api_descriptions",
                        "app": "venmo",
                        "api_name": "show_api_descriptions",
                        "status": "ok",
                        "result_preview": "[...]",
                    },
                    {
                        "kind": "api_docs.show_api_doc",
                        "app": "venmo",
                        "api_name": "search_friends",
                        "status": "ok",
                        "result_preview": "{...}",
                    },
                    {
                        "kind": "app_api",
                        "app": "venmo",
                        "api_name": "search_friends",
                        "status": "ok",
                        "result_preview": "[{'email': 'a@example.com'}]",
                        "result_shape": {
                            "type": "list",
                            "count_hint": 1,
                            "keys": ["email"],
                        },
                    },
                    {
                        "kind": "app_api",
                        "app": "venmo",
                        "api_name": "search_friends",
                        "status": "ok",
                        "result_preview": "[{'email': 'b@example.com'}]",
                        "result_shape": {
                            "type": "list",
                            "count_hint": 2,
                            "keys": ["email", "first_name"],
                        },
                    },
                    {
                        "kind": "app_api",
                        "app": "venmo",
                        "api_name": "remove_friend",
                        "status": "error",
                        "error": "422",
                    },
                ]
            }
        ],
        workflow_path="logs/x/workflow.jsonl",
        evaluation_report_path="logs/x/evaluation.txt",
    )

    assert (
        summary["instruction"] == "Reset friends on venmo to match my phone contacts."
    )
    assert summary["datetime"] == "2023-05-18T12:00:00"
    assert summary["eval"] == {
        "passed": 1,
        "failed": 1,
        "total": 2,
        "passed_all": False,
    }
    assert summary["overview"]["failure_point"] == "SUBMIT::COMPLETION_EVAL_FAILED"
    assert summary["aggregate_metrics"]["llm_calls"] == 8
    assert summary["aggregate_metrics"]["total_tokens"] == 127
    assert summary["overview"]["milestones_total"] == 1
    milestone = summary["milestones"][0]
    assert milestone["goal"] == "Read Venmo friends."
    assert milestone["find"]["selected_apis"] == [
        "venmo.add_friend",
        "venmo.search_friends",
    ]
    assert milestone["execute"]["status"] == "SUCCEEDED"
    assert milestone["execute"]["called_apis"] == [
        "venmo.remove_friend",
        "venmo.search_friends",
    ]
    assert milestone["execute"]["code_plan"] == {
        "steps": [
            "Read the current Venmo friends.",
            "Compare the Venmo friends to the phone contacts.",
            "Print json.dumps(result_dict) as the final stdout line.",
        ],
        "output_variable": {
            "name": "venmo_friend_sync_plan",
            "description": "Plan for synchronizing Venmo friends.",
        },
    }
    assert milestone["execute"]["failed_api_calls"] == ["venmo.remove_friend"]
    assert milestone["execute"]["unexpected_api_calls"] == ["venmo.remove_friend"]
    assert summary["run"]["workflow_path"] == "logs/x/workflow.jsonl"
    assert summary["run"]["evaluation_report_path"] == "logs/x/evaluation.txt"
    rendered = render_task_summary_markdown(summary)
    assert "LLM calls     8" in rendered
    assert "Tokens        127" in rendered
    assert "TASK INSTRUCTION" in rendered
    assert "Reset friends on venmo to match my phone contacts." in rendered
    assert "MILESTONES" in rendered
    assert "venmo" in rendered and "Read Venmo friends." in rendered
    assert "MILESTONE DETAILS" in rendered
    assert "[Milestone 1] venmo - Read Venmo friends." in rendered
    assert "Find APIs:" in rendered
    assert "venmo.add_friend" in rendered
    assert "venmo.search_friends" in rendered
    assert "Called APIs:" in rendered
    assert "venmo.remove_friend" in rendered
    assert "Failed APIs:" in rendered
    assert "Unexpected APIs:" in rendered
    assert "Code Plan:" in rendered
    assert (
        "Output variable: venmo_friend_sync_plan - Plan for synchronizing Venmo friends."
        in rendered
    )
    assert "1. Read the current Venmo friends." in rendered
    assert "3. Print json.dumps(result_dict) as the final stdout line." in rendered
    assert "Committed variable:" in rendered
    # A1: API call trace section per milestone, with kwargs + shape rendered
    assert "API call trace:" in rendered
    assert "venmo.search_friends" in rendered
    assert "list[" in rendered  # result_shape rendered as list[N]


def test_kwarg_value_does_not_double_quote_repr_strings():
    """`_sanitize_kwargs` runs `repr()` upstream so kwarg string values
    arrive already-quoted (e.g. \"'partner'\"). The renderer must NOT
    wrap them again into "'partner'" — that produces double-quoted
    nonsense in the API call trace."""
    from adk_appworld_agent.observability.markdown import _format_kwarg_value

    assert _format_kwarg_value("'partner'") == "'partner'"
    assert _format_kwarg_value('"already_double"') == '"already_double"'
    # Plain strings (e.g. the access_token <redacted> placeholder) still
    # get wrapped so kwargs without inherent quoting remain readable.
    assert _format_kwarg_value("<redacted>") == '"<redacted>"'
    # Already-structured (list / dict repr) passes through.
    assert _format_kwarg_value("[1, 2]") == "[1, 2]"
    assert _format_kwarg_value("{'a': 1}") == "{'a': 1}"


def test_assertion_error_extraction_captures_multiline_body():
    """Multi-line AssertionError blocks (set ==, dict ==, with 'In right
    but not left' tail) must be captured fully and joined into a single
    line for the sub-test result entry. Otherwise readers miss the
    actual mismatch detail."""
    from adk_appworld_agent.observability.workflow import _extract_assertion_error

    block = (
        "assert model changes match X.\n"
        "```python\n"
        "with test(...): ...\n"
        "```\n"
        "----------\n"
        "AssertionError:\n"
        "set()\n"
        "==\n"
        "{'venmo.User', 'venmo.Transaction'}\n"
        "\n"
        "In right but not left:\n"
        "['venmo.User', 'venmo.Transaction']\n"
    )
    error = _extract_assertion_error(block)
    assert error is not None
    assert "AssertionError" in error
    assert "set()" in error
    assert "venmo.User" in error
    # Joined on ' | ' so it stays one line.
    assert "\n" not in error


def test_submission_sub_test_results_split_pass_fail():
    """A2: live evaluator report parsed into pass/fail entries the renderer
    surfaces under '**Sub-test results**' so analysis sees which sub-tests
    failed without scrolling through the full report."""
    from adk_appworld_agent.observability.workflow import _parse_sub_test_results

    report = (
        ">> Passed Requirement\n"
        "Test 1 description.\n"
        "\n"
        ">> Failed Requirement\n"
        "Test 2 description.\n"
        "\n"
        "```python\n"
        "AssertionError: 5 == 7\n"
        "```\n"
    )
    results = _parse_sub_test_results(report)
    statuses = [r["status"] for r in results]
    assert statuses == ["pass", "fail"]
    assert results[0]["requirement"] == "Test 1 description."
    assert results[1]["requirement"] == "Test 2 description."
    assert results[1]["error"] == "AssertionError: 5 == 7"


def test_committed_variable_renders_when_executor_emits_stdout_value():
    """A4: workflow.py threads stdout_json.value into output_variable_preview;
    markdown surfaces it under **Committed variable** so the milestone's
    output is visible in its own section, not only the next milestone's input."""
    summary = build_task_summary(
        task_id="task_a4",
        result={
            "status": "FAILED",
            "phase": "EXECUTE",
            "instruction": "Test.",
            "datetime": "2026-05-08T00:00:00",
            "wall_s": 1.0,
            "milestones": [{"id": "m1", "intent": "Find note", "app": "simple_note"}],
            "active_milestone_index": 0,
            "named_variables": {},
        },
        subagent_outputs={
            "PLAN": [
                {"payload": {"tasks": [{"task": "Find note", "app": "simple_note"}]}}
            ],
            "FIND": [
                {
                    "payload": {
                        "milestone_index": 0,
                        "metrics": {"wall_ms": 1000, "llm_calls": 1},
                        "candidate_apis": [
                            {"app": "simple_note", "name": "search_notes"},
                        ],
                    }
                }
            ],
            "EXECUTE": [
                {
                    "payload": {
                        "milestone_index": 0,
                        "metrics": {"wall_ms": 500, "llm_calls": 1},
                        "tool_call_count": 1,
                        "code_plan": {
                            "plan_steps": ["search_notes(query='x')[0]"],
                            "construct_step": "result_dict = {'value': note}",
                            "print_step": "print(json.dumps(result_dict))",
                            "output_variable": {
                                "name": "the_note",
                                "description": "First search result.",
                            },
                        },
                        "code_execute": {
                            "stdout_json": {
                                "value": {"note_id": 42, "title": "hello"},
                            },
                        },
                    }
                }
            ],
        },
        sandbox_trace=[],
        workflow_path=None,
        evaluation_report_path=None,
        final_state={"phase": "FAILED", "status": "FAILED"},
    )
    rendered = render_task_summary_markdown(summary)
    assert "Committed variable:" in rendered
    assert "the_note - First search result." in rendered
    assert "note_id" in rendered  # value preview rendered


def test_render_batch_results_markdown_shows_total_pass_and_test_pass_totals():
    rendered = render_batch_results_markdown(
        {
            "run_name": "batch-1",
            "updated_at": "2026-04-28 05:00:00",
            "config": {
                "plan": "rough",
                "find": "community",
                "execute": "stub",
                "policy": "deterministic",
            },
            "tasks": [
                {
                    "task_id": "task_1",
                    "instruction": "First task.",
                    "state": "completed",
                    "result": {
                        "passed": 1,
                        "total": 2,
                        "passed_all": False,
                        "wall_s": 1.5,
                        "llm_calls": 3,
                        "total_tokens": 50,
                    },
                },
                {
                    "task_id": "task_2",
                    "instruction": "Second task.",
                    "state": "completed",
                    "result": {
                        "passed": 3,
                        "total": 3,
                        "passed_all": True,
                        "wall_s": 2.5,
                        "llm_calls": 5,
                        "total_tokens": 75,
                    },
                },
                {"task_id": "task_3", "instruction": "Third task.", "state": "pending"},
            ],
        }
    )

    assert "TGC (tasks passed)" in rendered and "1 / 3 (33.3%)" in rendered
    # The third variant is pending, so scenario completion is unavailable.
    assert "SGC (scenarios all-variant pass)" in rendered
    assert "N/A (requires all three variants" in rendered
    assert "Test pass" in rendered and "4/5 (80.0%)" in rendered
    assert "Task instruction:" in rendered
    assert "First task." in rendered
    assert "[task_3] PENDING" in rendered
    assert "Cache:    none" in rendered


def test_render_batch_results_markdown_shows_cache_info_when_set():
    rendered = render_batch_results_markdown(
        {
            "run_name": "batch-1",
            "updated_at": "2026-05-15 00:00:00",
            "config": {
                "plan": "rough",
                "find": "community",
                "execute": "code_plan_execute",
                "verify": "stub",
                "policy": "deterministic",
                "cache_dir": "data/subagent_cache/cycle_recovery_v1",
                "cache_read": "rough_planner,community_finder",
            },
            "tasks": [],
        }
    )
    assert (
        "Cache:    data/subagent_cache/cycle_recovery_v1 "
        "(read=rough_planner,community_finder)"
    ) in rendered


def test_render_batch_results_markdown_cache_read_none_when_empty():
    rendered = render_batch_results_markdown(
        {
            "run_name": "batch-1",
            "updated_at": "2026-05-15 00:00:00",
            "config": {
                "plan": "rough",
                "cache_dir": "data/subagent_cache/cycle_recovery_v1",
                "cache_read": "",
            },
            "tasks": [],
        }
    )
    assert "Cache:    data/subagent_cache/cycle_recovery_v1 (read=none)" in rendered


def test_build_task_summary_shows_cached_plan_before_find_and_aligns_find_runs():
    summary = build_task_summary(
        task_id="task_1",
        result={
            "status": "RUNNING",
            "phase": "FIND",
            "task_wall_ms": 120000,
            "passed": None,
            "failed": None,
            "total": None,
            "exec_failure": None,
            "submit_block": None,
            "block_reason": None,
        },
        final_state={
            "phase": "FIND",
            "status": "RUNNING",
            "task_context": {
                "instruction": "Reset friends on venmo to match my phone contacts.",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [
                {"intent": "Read Venmo friends.", "app": "venmo"},
                {"intent": "Read phone contacts.", "app": "phone"},
            ],
        },
        subagent_outputs={
            "PLAN": [
                {
                    "payload": {
                        "tasks": [
                            {"task": "Read Venmo friends.", "app": "venmo"},
                            {"task": "Read phone contacts.", "app": "phone"},
                        ],
                        "task_count": 2,
                        "cache_source": "logs/cache/task_1/subagent_io.md",
                        "metrics": {"wall_ms": 0},
                    }
                }
            ],
            "FIND": [
                {
                    "attempt": 1,
                    "payload": {
                        "matched_apps": ["venmo"],
                        "selected_communities": ["venmo_c3"],
                        "candidate_count": 2,
                        "candidate_apis": [
                            {"app": "venmo", "name": "login"},
                            {"app": "venmo", "name": "search_friends"},
                        ],
                        "io": {
                            "input": {
                                "subagent_input": {
                                    "metadata": {
                                        "milestone_index": 0,
                                        "milestone_intent": "Read Venmo friends.",
                                        "planned_apps": ["venmo"],
                                    }
                                }
                            }
                        },
                    },
                    "metrics": {
                        "llm_calls": 2,
                        "llm_call_attempts": 3,
                        "retry_count": 1,
                        "retry_backoff_ms": 30000,
                        "rate_limit_count": 1,
                        "provider_error_count": 1,
                        "dependency_rounds": 0,
                        "wall_ms": 2000,
                    },
                },
                {
                    "attempt": 2,
                    "payload": {
                        "matched_apps": ["phone"],
                        "selected_communities": ["phone_c0"],
                        "candidate_count": 2,
                        "candidate_apis": [
                            {"app": "phone", "name": "login"},
                            {"app": "phone", "name": "search_contacts"},
                        ],
                        "io": {
                            "input": {
                                "subagent_input": {
                                    "metadata": {
                                        "milestone_index": 1,
                                        "milestone_intent": "Read phone contacts.",
                                        "planned_apps": ["phone"],
                                    }
                                }
                            }
                        },
                    },
                    "metrics": {
                        "llm_calls": 2,
                        "dependency_rounds": 0,
                        "wall_ms": 3000,
                    },
                },
            ],
        },
        sandbox_trace=[],
        workflow_path="logs/x/workflow.jsonl",
        evaluation_report_path=None,
    )

    assert summary["overview"]["plan_source"] == "cache"
    assert summary["overview"]["cache_source"] == "logs/cache/task_1/subagent_io.md"
    assert summary["overview"]["milestones_total"] == 2
    assert summary["overview"]["find_completed"] == 2
    assert summary["overview"]["find_retry_count"] == 1
    assert summary["overview"]["find_backoff_s"] == 30.0
    assert summary["overview"]["rate_limit_count"] == 1
    assert summary["overview"]["provider_issue_likely"] is True
    assert summary["milestones"][0]["app"] == "venmo"
    assert summary["milestones"][0]["find"]["selected_apis"] == [
        "venmo.login",
        "venmo.search_friends",
    ]
    assert summary["milestones"][0]["find"]["retry_count"] == 1
    assert summary["milestones"][0]["find"]["backoff_s"] == 30.0
    assert summary["milestones"][0]["find"]["provider_issue_likely"] is True
    assert summary["milestones"][1]["find"]["selected_apis"] == [
        "phone.login",
        "phone.search_contacts",
    ]
    assert summary["milestones"][0]["execute"]["status"] == "MISSING"
    assert summary["milestones"][1]["execute"]["status"] == "MISSING"


def test_build_task_summary_marks_timeout_as_timed_out():
    summary = build_task_summary(
        task_id="task_1",
        result={
            "status": "TIMED_OUT",
            "phase": "FIND",
            "task_wall_ms": 120000,
            "passed": None,
            "failed": None,
            "total": None,
            "exec_failure": None,
            "submit_block": None,
            "block_reason": "TASK_TIMEOUT",
        },
        final_state={
            "phase": "FIND",
            "status": "RUNNING",
            "task_context": {
                "instruction": "Find the thing.",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [
                {"intent": "Read data.", "app": "gmail"},
            ],
            "active_milestone_index": 0,
        },
        subagent_outputs={
            "PLAN": [{"payload": {"tasks": [{"task": "Read data.", "app": "gmail"}]}}],
        },
        sandbox_trace=[],
        workflow_path="logs/x/workflow.jsonl",
        evaluation_report_path=None,
    )

    assert summary["status"] == "TIMED_OUT"
    assert summary["overview"]["block_reason"] == "TASK_TIMEOUT"
    assert summary["overview"]["failure_point"] == "FIND::TASK_TIMEOUT"
    assert summary["milestones"][0]["find"]["status"] == "TIMED_OUT"


def test_build_workflow_records_logs_executor_finalize_summary():
    records = build_workflow_records(
        ledger=[],
        subagent_outputs={
            "EXECUTE": [
                {
                    "subagent_name": "executor",
                    "status": "SUCCEEDED",
                    "payload": {
                        "tool_call_count": 2,
                        "final_response": "working text",
                        "finalize_called": True,
                        "finalize_nudge_used": True,
                        "no_progress_nudge_used": True,
                        "docs_only_nudge_used": True,
                        "auto_finalize_used": True,
                        "milestone_done": True,
                        "milestone_id": "m1",
                        "milestone_index": 0,
                        "milestone_total": 1,
                        "executor_result": {
                            "answer": "null",
                            "milestone_done": True,
                            "summary": "done",
                            "variables": [
                                {
                                    "name": "wife_email",
                                    "preview": "sarah@ex.com",
                                    "value_json": '"sarah@ex.com"',
                                }
                            ],
                        },
                        "submission_candidate": {
                            "answer": "null",
                            "task_type_hint": "action",
                            "answer_type": "null",
                            "source": "executor",
                        },
                    },
                }
            ]
        },
        executor_trace=[],
        sandbox_trace=[],
    )

    result = next(
        record
        for record in records
        if record["phase"] == "EXECUTE" and record["kind"] == "result"
    )

    assert result["data"]["finalize_called"] is True
    assert result["data"]["finalize_nudge_used"] is True
    assert result["data"]["no_progress_nudge_used"] is True
    assert result["data"]["docs_only_nudge_used"] is True
    assert result["data"]["auto_finalize_used"] is True
    assert result["data"]["milestone_done"] is True
    assert result["data"]["variables_count"] == 1
    assert result["data"]["final_response_chars"] == len("working text")
    assert result["data"]["submission_candidate"] == {
        "task_type_hint": "action",
        "answer_type": "null",
        "source": "executor",
    }
    assert "final_response" not in result["data"]
    assert "answer" not in result["summary"]


def test_build_workflow_records_redacts_code_and_result_content():
    records = build_workflow_records(
        ledger=[],
        subagent_outputs={},
        executor_trace=[
            {
                "t_ms": 100,
                "parts": [
                    {
                        "kind": "call",
                        "name": "execute_python",
                        "args": {"code": "print('secret code')"},
                    },
                    {
                        "kind": "resp",
                        "name": "execute_python",
                        "response": {"result": "secret result"},
                    },
                    {
                        "kind": "text",
                        "text": "secret model text",
                    },
                ],
            }
        ],
        sandbox_trace=[
            {
                "api_calls": [
                    {
                        "kind": "app_api",
                        "app": "venmo",
                        "api_name": "search_friends",
                        "status": "ok",
                        "result_preview": "[{'email': 'secret@example.com'}]",
                        "result_shape": {
                            "type": "list",
                            "count_hint": 1,
                            "keys": ["email"],
                        },
                    }
                ]
            }
        ],
    )

    serialized = str(records)
    assert "secret code" not in serialized
    assert "secret result" not in serialized
    assert "secret model text" not in serialized
    assert "secret@example.com" not in serialized
    assert "code_char_count" in serialized
    assert "result_shape" in serialized


def _find_result_event(records: list[dict]) -> dict | None:
    for record in records:
        if record.get("phase") == "FIND" and record.get("kind") == "result":
            return record
    return None


def _replayed_find_subagent_outputs() -> dict:
    return {
        "PLAN": [
            {
                "payload": {"tasks": [{"task": "x", "app": "venmo"}]},
            }
        ],
        "FIND": [
            {
                "attempt": 1,
                "subagent_name": "finder_subagent_community",
                "warnings": ["replayed from run_task:t1"],
                "payload": {
                    "matched_apps": ["venmo"],
                    "selected_communities": ["venmo_c3"],
                    "candidate_count": 2,
                    "candidate_apis": [
                        {"app": "venmo", "name": "search_friends"},
                    ],
                    "cache_source": "run_task:t1",
                    "io": {
                        "input": {
                            "subagent_input": {
                                "metadata": {
                                    "milestone_index": 0,
                                    "milestone_intent": "Read Venmo friends.",
                                    "planned_apps": ["venmo"],
                                }
                            }
                        }
                    },
                },
                "metrics": {"wall_ms": 5, "llm_calls": 0, "replayed": True},
            }
        ],
    }


def test_find_workflow_event_surfaces_cache_source_and_warnings():
    records = build_workflow_records(
        ledger=[],
        subagent_outputs=_replayed_find_subagent_outputs(),
        executor_trace=[],
        sandbox_trace=[],
    )
    event = _find_result_event(records)
    assert event is not None
    data = event["data"]
    assert data["source"] == "cache"
    assert data["cache_source"] == "run_task:t1"
    assert "replayed from run_task:t1" in data["warnings"]
    assert "cache replay" in event["summary"]


def test_task_summary_counts_find_cache_hits_and_flags_milestone():
    subagent_outputs = _replayed_find_subagent_outputs()
    summary = build_task_summary(
        task_id="t1",
        result={
            "status": "SUCCEEDED",
            "phase": "COMPLETE",
            "task_wall_ms": 500,
            "passed": 1,
            "failed": 0,
            "total": 1,
        },
        final_state={
            "phase": "COMPLETE",
            "status": "COMPLETED",
            "task_context": {
                "instruction": "Read Venmo friends.",
                "task_datetime": "2024-01-01T00:00:00",
            },
            "milestones": [{"intent": "Read Venmo friends.", "app": "venmo"}],
            "active_milestone_index": 1,
        },
        subagent_outputs=subagent_outputs,
        sandbox_trace=[],
        command="pytest",
        workflow_path="ignored",
        evaluation_report_path=None,
    )
    assert summary["overview"]["find_cache_hits"] == 1
    assert summary["milestones"][0]["find"]["replayed"] is True
    assert summary["milestones"][0]["find"]["cache_source"] == "run_task:t1"


def test_task_summary_uses_current_run_metrics_for_cache_replay():
    subagent_outputs = _replayed_find_subagent_outputs()
    subagent_outputs["FIND"][0]["metrics"] = {
        "wall_ms": 0,
        "llm_calls": 0,
        "total_tokens": 0,
        "replayed": True,
        "cached_wall_ms": 5000,
        "cached_llm_calls": 4,
        "cached_total_tokens": 900,
    }
    summary = build_task_summary(
        task_id="t1",
        result={
            "status": "SUCCEEDED",
            "phase": "COMPLETE",
            "task_wall_ms": 300,
            "passed": 1,
            "failed": 0,
            "total": 1,
        },
        final_state={
            "phase": "COMPLETE",
            "status": "COMPLETED",
            "task_context": {
                "instruction": "Read Venmo friends.",
                "task_datetime": "2024-01-01T00:00:00",
            },
            "milestones": [{"intent": "Read Venmo friends.", "app": "venmo"}],
            "active_milestone_index": 1,
        },
        subagent_outputs=subagent_outputs,
        sandbox_trace=[],
        command="pytest",
        workflow_path="ignored",
        evaluation_report_path=None,
    )

    assert summary["aggregate_metrics"]["llm_calls"] == 0
    assert summary["aggregate_metrics"]["total_tokens"] == 0
    assert summary["wall_s"] == 0.3


def test_task_summary_surfaces_submitted_answer_and_eval_report():
    long_report = "Test 1: count_correct = PASS\nTest 2: count_correct = FAIL\n" + (
        "filler " * 200
    )
    summary = build_task_summary(
        task_id="task_1",
        result={
            "status": "FAILED",
            "phase": "FAILED",
            "task_wall_ms": 1000,
            "passed": 1,
            "failed": 1,
            "total": 2,
            "submit_block": "COMPLETION_EVAL_FAILED",
            "block_reason": "COMPLETION_EVAL_FAILED",
        },
        final_state={
            "phase": "FAILED",
            "status": "FAILED",
            "task_context": {
                "instruction": "Count things.",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [],
        },
        subagent_outputs={
            "SUBMIT": [
                {
                    "subagent_name": "completion_gate",
                    "status": "FAILED",
                    "failure_code": "COMPLETION_EVAL_FAILED",
                    "payload": {
                        "status": "BLOCKED",
                        "block_reason": "COMPLETION_EVAL_FAILED",
                        "submitted_answer": "8",
                        "evaluation_report": long_report,
                        "extra": {
                            "passed": 1,
                            "failed": 1,
                            "total": 2,
                            "report_parsed": True,
                        },
                    },
                }
            ],
        },
        sandbox_trace=[],
        workflow_path=None,
        evaluation_report_path=None,
    )

    submission = summary["submission"]
    assert submission["submitted_answer"] == "8"
    assert submission["passed"] == 1
    assert submission["failed"] == 1
    assert submission["total"] == 2
    assert submission["report_excerpt"].startswith("Test 1: count_correct = PASS")
    assert not submission["report_excerpt"].endswith("…"), (
        "report_excerpt no longer truncated; full text preserved so failed "
        "sub-test detail remains visible"
    )

    rendered = render_task_summary_markdown(summary)
    assert "SUBMISSION" in rendered
    assert "Submitted answer: 8" in rendered
    assert "Evaluator:" in rendered and "1/2 passed (1 failed)" in rendered
    assert "Test 1: count_correct = PASS" in rendered


def test_task_summary_submission_section_omitted_when_no_submit_phase():
    summary = build_task_summary(
        task_id="task_no_submit",
        result={
            "status": "FAILED",
            "phase": "EXECUTE",
            "task_wall_ms": 500,
            "submit_block": None,
            "block_reason": "EXECUTOR_DID_NOT_FINALIZE",
        },
        final_state={
            "phase": "EXECUTE",
            "status": "FAILED",
            "task_context": {
                "instruction": "x",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [],
        },
        subagent_outputs={},
        sandbox_trace=[],
        workflow_path=None,
        evaluation_report_path=None,
    )
    assert summary["submission"] == {
        "submitted_answer": None,
        "passed": None,
        "failed": None,
        "total": None,
        "report_excerpt": None,
        "sub_test_results": [],
    }
    rendered = render_task_summary_markdown(summary)
    assert "SUBMISSION" in rendered
    submission_block = rendered.split("SUBMISSION", 1)[1].split("FAILURES", 1)[0]
    assert "None" in submission_block


# ── Stage-1 audit: failure_one_liner + chain + markers ───────────────────────


def _build_executor_failure_summary(*, repair_attempts: list[dict]) -> dict:
    """Construct a build_task_summary input that exercises the executor failure
    chain (root EXECUTOR_DID_NOT_FINALIZE → forced fallback → SUBMIT::COMPLETION_EVAL_FAILED).
    """
    return build_task_summary(
        task_id="task_anita",
        result={
            "status": "FAILED",
            "phase": "FAILED",
            "task_wall_ms": 265000,
            "passed": 2,
            "failed": 4,
            "total": 6,
            "exec_failure": "EXECUTOR_DID_NOT_FINALIZE",
            "submit_block": "COMPLETION_EVAL_FAILED",
            "block_reason": "COMPLETION_EVAL_FAILED",
        },
        final_state={
            "phase": "FAILED",
            "status": "FAILED",
            "task_context": {
                "instruction": "Send $427 on venmo to Anita.",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [
                {"intent": "Send $427 to Anita.", "app": "venmo"},
            ],
        },
        subagent_outputs={
            "PLAN": [
                {
                    "payload": {
                        "tasks": [{"task": "Send $427 to Anita.", "app": "venmo"}]
                    }
                }
            ],
            "FIND": [
                {
                    "attempt": 1,
                    "payload": {
                        "matched_apps": ["venmo"],
                        "candidate_apis": [
                            {"app": "venmo", "name": "create_transaction"}
                        ],
                        "io": {
                            "input": {
                                "subagent_input": {"metadata": {"milestone_index": 0}}
                            }
                        },
                    },
                    "metrics": {"wall_ms": 1000, "llm_calls": 1},
                }
            ],
            "EXECUTE": [
                {
                    "attempt": 1,
                    "status": "FAILED",
                    "failure_code": "EXECUTOR_DID_NOT_FINALIZE",
                    "payload": {
                        "milestone_index": 0,
                        "code_execute": {"repair_attempts": repair_attempts},
                        "metrics": {"wall_ms": 100000, "llm_calls": 5},
                    },
                }
            ],
            "SUBMIT": [
                {
                    "payload": {
                        "status": "BLOCKED",
                        "forced_fallback": True,
                        "original_failure_code": "EXECUTOR_DID_NOT_FINALIZE",
                        "extra": {"passed": 2, "failed": 4, "total": 6},
                    }
                }
            ],
        },
        sandbox_trace=[],
        workflow_path=None,
        evaluation_report_path=None,
    )


def test_executor_failure_one_liner_extracted_from_last_repair_attempt():
    summary = _build_executor_failure_summary(
        repair_attempts=[
            {"attempt": 1, "parse_error": "first attempt error"},
            {
                "attempt": 2,
                "parse_error": (
                    "code reported error: Response status code is 422:\n"
                    '{"message":"Your Venmo balance does not have $427.00 to make this transaction."}'
                ),
            },
        ]
    )

    one_liner = summary["overview"]["failure_one_liner"]
    assert one_liner is not None
    # The latest repair attempt wins.
    assert "422" in one_liner
    assert "balance does not have" in one_liner
    assert "first attempt error" not in one_liner
    # Must be a single line (no embedded newlines).
    assert "\n" not in one_liner

    execute_run_one_liner = summary["milestones"][0]["execute"]["failure_one_liner"]
    assert execute_run_one_liner == one_liner


def test_failure_chain_renders_root_cascade_and_one_liner():
    summary = _build_executor_failure_summary(
        repair_attempts=[
            {"attempt": 1, "parse_error": "venmo balance insufficient"},
        ]
    )
    rendered = render_task_summary_markdown(summary)

    failures_block = rendered.split("FAILURES", 1)[1]
    assert "Root: EXECUTOR_DID_NOT_FINALIZE" in failures_block
    assert "venmo balance insufficient" in failures_block
    assert "Cascade: SUBMIT::COMPLETION_EVAL_FAILED" in failures_block
    assert "FORCED null fallback" in failures_block


def test_header_renders_markers_and_failure_one_liner():
    summary = _build_executor_failure_summary(
        repair_attempts=[
            {"attempt": 1, "parse_error": "venmo balance insufficient"},
        ]
    )
    rendered = render_task_summary_markdown(summary)
    header_block = rendered.split("TASK INSTRUCTION", 1)[0]

    assert "Markers:" in header_block
    assert "forced_fallback" in header_block
    assert "milestone_fail×1" in header_block
    assert "Failure: venmo balance insufficient" in header_block


def test_failure_one_liner_falls_back_to_submit_excerpt_when_executor_clean():
    """When the executor finalized but the evaluator rejected the answer, the
    one-liner should still surface so the reader knows where to look."""
    summary = build_task_summary(
        task_id="task_eval_only",
        result={
            "status": "FAILED",
            "phase": "FAILED",
            "task_wall_ms": 50000,
            "passed": 1,
            "failed": 2,
            "total": 3,
            "submit_block": "COMPLETION_EVAL_FAILED",
            "block_reason": "COMPLETION_EVAL_FAILED",
        },
        final_state={
            "phase": "FAILED",
            "status": "FAILED",
            "task_context": {
                "instruction": "x",
                "task_datetime": "2023-05-18T12:00:00",
            },
            "milestones": [{"intent": "y", "app": "venmo"}],
        },
        subagent_outputs={
            "PLAN": [{"payload": {"tasks": [{"task": "y", "app": "venmo"}]}}],
            "EXECUTE": [
                {
                    "attempt": 1,
                    "status": "SUCCEEDED",
                    "payload": {"milestone_index": 0, "metrics": {"wall_ms": 500}},
                }
            ],
            "SUBMIT": [
                {"payload": {"status": "BLOCKED", "extra": {"passed": 1, "total": 3}}}
            ],
        },
        sandbox_trace=[],
        workflow_path=None,
        evaluation_report_path=None,
    )
    one_liner = summary["overview"]["failure_one_liner"]
    assert one_liner is not None
    assert "1/3" in one_liner
