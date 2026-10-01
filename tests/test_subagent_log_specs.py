from __future__ import annotations

import json

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.observability.subagent_logs import (
    SUBAGENT_IO_FILENAME,
    format_metrics,
    io_record_from_subagent_output,
    render_subagent_io_markdown,
)
from adk_appworld_agent.orchestration.state import Phase, TaskContext
from adk_appworld_agent.subagents.finder.logging import (
    COMMUNITY_FINDER_LOG_SPEC,
    community_finder_io_record,
)
from adk_appworld_agent.subagents.planner.rough_planner.logging import (
    ROUGH_PLANNER_LOG_SPEC,
)


def test_subagent_log_specs_define_shared_identity_only():
    assert SUBAGENT_IO_FILENAME == "subagent_io.txt"
    assert ROUGH_PLANNER_LOG_SPEC.agent_name == "rough_planner_subagent"
    assert ROUGH_PLANNER_LOG_SPEC.expected_output == "Plan"
    assert ROUGH_PLANNER_LOG_SPEC.phase.value == "PLAN"
    assert not hasattr(ROUGH_PLANNER_LOG_SPEC, "result_file_stem")

    assert COMMUNITY_FINDER_LOG_SPEC.agent_name == "finder_subagent_community"
    assert COMMUNITY_FINDER_LOG_SPEC.expected_output == "ApiSelection"
    assert COMMUNITY_FINDER_LOG_SPEC.phase.value == "FIND"
    assert not hasattr(COMMUNITY_FINDER_LOG_SPEC, "result_file_stem")


def test_subagent_io_markdown_preserves_system_prompt_without_sanitizing():
    record = {
        "agent": "rough_planner_subagent",
        "status": "SUCCEEDED",
        "input": {
            "phase": "PLAN",
            "attempt": 1,
            "model_input_raw": {
                "system_instruction": "\n".join(
                    [
                        "Available apps and descriptions:",
                        "- supervisor: Benchmark supervisor context such as user profile; not a normal user app.",
                        "- api_docs: API documentation lookup; use only as a fallback planning app when no domain app is identifiable.",
                    ]
                ),
                "messages": [{"parts": [{"text": "Task instruction:\nDo the task."}]}],
            },
        },
        "output": {
            "phase": "PLAN",
            "subagent_name": "rough_planner_subagent",
            "attempt": 1,
            "failure_code": None,
            "raw_llm_text": '{"thoughts": "ok", "tasks": []}',
        },
        "metrics": {
            "wall_s": 1.25,
            "llm_calls": 1,
            "llm_call_attempts": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "thoughts_tokens": 0,
            "total_tokens": 15,
            "timeout_count": 0,
        },
    }

    content = "\n".join(render_subagent_io_markdown(record))

    assert "SUBAGENT IO REPORT" in content
    assert "- supervisor: Benchmark supervisor context" in content
    assert "- api_docs: API documentation lookup" in content
    assert "Task instruction:" in content
    assert '"thoughts": "ok"' in content
    assert '"total_tokens": 15' in content


def test_metric_format_distinguishes_observed_and_attempted_calls():
    formatted = format_metrics(
        {
            "wall_ms": 120000,
            "subagent_calls": 1,
            "envelope_received": 0,
            "llm_calls": 0,
            "llm_call_attempts": 1,
            "timeout_count": 1,
        }
    )

    assert "subagent_calls=1" in formatted
    assert "envelope_received=0" in formatted
    assert "llm_calls(observed/attempted)=0/1" in formatted
    assert "timeouts=1" in formatted


def test_subagent_io_markdown_renders_finder_model_call_trace():
    subagent_input = SubagentInput(
        phase=Phase.FIND,
        attempt=1,
        task_context=TaskContext(task_id="task_1", instruction="Find Venmo APIs."),
        metadata={
            "planned_apps": ["venmo"],
            "milestone_index": 0,
            "milestone_total": 1,
            "milestone_intent": "Find Venmo friend APIs.",
        },
    )
    record = {
        "agent": "finder_subagent_community",
        "status": "SUCCEEDED",
        "input": {
            "phase": "FIND",
            "attempt": 1,
            "subagent_input": subagent_input.model_dump(mode="json"),
            "planned_apps": ["venmo"],
            "milestone_index": 0,
            "milestone_total": 1,
            "milestone_intent": "Find Venmo friend APIs.",
            "model_calls": [
                {
                    "step": "community_select",
                    "model_input_raw": {
                        "system_instruction": "Select communities.",
                        "messages": [{"parts": [{"text": "[venmo_c0] Venmo friends"}]}],
                    },
                }
            ],
        },
        "output": {
            "phase": "FIND",
            "subagent_name": "finder_subagent_community",
            "attempt": 1,
            "failure_code": None,
            "raw_llm_text": '{"candidate_apis": [{"app": "venmo", "name": "add_friend"}]}',
            "model_calls": [
                {
                    "step": "community_select",
                    "raw_llm_text": '{"selected": ["venmo_c0"]}',
                    "parsed_json": {"selected": ["venmo_c0"]},
                }
            ],
        },
        "metrics": {"llm_calls": 1, "llm_call_attempts": 1},
    }

    content = "\n".join(render_subagent_io_markdown(record))

    assert "SUBAGENT INPUT" in content
    assert (
        "MODEL CALLS" not in content
    )  # uppercase MODEL CALLS would be a future section, currently no top-level
    assert "[1] community_select" in content
    assert "  System (" in content  # plain-text system block
    assert "  User:" in content
    assert '"instruction": "Find Venmo APIs."' in content
    assert '"planned_apps": [' in content
    assert "[venmo_c0] Venmo friends" in content
    assert '"selected": [' in content
    assert '"add_friend"' in content


def test_io_record_from_subagent_output_wraps_subagent_payload_io():
    record = io_record_from_subagent_output(
        "PLAN",
        {
            "subagent_name": "rough_planner_subagent",
            "attempt": 1,
            "status": "SUCCEEDED",
            "failure_code": None,
            "metrics": {"llm_calls": 1},
            "payload": {
                "io": {
                    "input": {
                        "model_input": {
                            "system_instruction": "Plan.",
                            "messages": [{"parts": [{"text": "Task instruction."}]}],
                        },
                    },
                    "output": {"raw_llm_text": '{"tasks": []}'},
                }
            },
        },
    )

    assert record is not None
    assert record["input"]["phase"] == "PLAN"
    assert record["output"]["subagent_name"] == "rough_planner_subagent"
    assert "Task instruction." in "\n".join(render_subagent_io_markdown(record))


def test_io_record_from_subagent_output_surfaces_milestone_context():
    subagent_input = SubagentInput(
        phase=Phase.FIND,
        attempt=1,
        task_context=TaskContext(task_id="task_1", instruction="Find Gmail APIs."),
        metadata={
            "planned_apps": ["gmail"],
            "milestone_index": 0,
            "milestone_total": 2,
            "milestone_intent": "Find the email.",
        },
    )

    record = io_record_from_subagent_output(
        "FIND",
        {
            "subagent_name": "finder_subagent_community",
            "attempt": 1,
            "status": "SUCCEEDED",
            "payload": {
                "io": {
                    "input": {
                        "subagent_input": subagent_input.model_dump(mode="json"),
                        "model_input": {
                            "system_instruction": "Find.",
                            "messages": [{"parts": [{"text": "Task instruction."}]}],
                        },
                    },
                    "output": {"raw_llm_text": '{"candidate_apis": []}'},
                }
            },
        },
    )

    assert record is not None
    rendered = "\n".join(render_subagent_io_markdown(record))
    assert '"planned_apps": [' in rendered
    assert '"gmail"' in rendered
    assert '"milestone_index": 0' in rendered
    assert '"milestone_total": 2' in rendered
    assert '"milestone_intent": "Find the email."' in rendered


def test_community_finder_io_record_matches_planner_logging_shape():
    subagent_input = SubagentInput(
        phase=Phase.FIND,
        attempt=1,
        task_context=TaskContext(
            task_id="task_1",
            instruction="Find Venmo APIs.",
            task_datetime="2024-01-01T00:00:00",
        ),
    )
    envelope = SubagentEnvelope(
        phase=Phase.FIND,
        subagent_name="finder_subagent_community",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "io": {
                "input": {
                    "subagent_input": subagent_input.model_dump(mode="json"),
                    "model_input_raw": {
                        "system_instruction": "Select apps.",
                        "messages": [{"parts": [{"text": "Task: Find Venmo APIs."}]}],
                    },
                    "model_calls": [],
                },
                "output": {
                    "raw_llm_text": '{"candidate_apis": []}',
                    "parsed_api_selection": {"candidate_apis": []},
                    "model_calls": [],
                },
            }
        },
    )

    record = community_finder_io_record(
        task_id="task_1",
        status="SUCCEEDED",
        subagent_input=subagent_input,
        subagent_input_text=subagent_input.model_dump_json(),
        envelope=envelope,
        metrics={"wall_ms": 1000, "llm_calls": 1},
    )

    assert record["io_format"] == "subagent_io.v1"
    assert record["agent"] == COMMUNITY_FINDER_LOG_SPEC.agent_name
    assert record["expected_output"] == "ApiSelection"
    assert record["input"]["phase"] == "FIND"
    assert record["output"]["raw_llm_text"] == '{"candidate_apis": []}'


# ── Stage-1 audit: collapse repeated `system` blocks across model_calls ─────


def _record_with_calls(calls: list[dict]) -> dict:
    """Build a minimal subagent IO record with the given model_calls list on
    both input and output. Uses the multi-step renderer path so we can
    exercise the system-prompt collapse logic."""
    return {
        "agent": "executor_subagent_code_plan_execute",
        "status": "SUCCEEDED",
        "input": {
            "phase": "EXECUTE",
            "attempt": 1,
            "model_calls": calls,
        },
        "output": {
            "phase": "EXECUTE",
            "subagent_name": "executor_subagent_code_plan_execute",
            "attempt": 1,
            "failure_code": None,
            "raw_llm_text": "",
            "model_calls": [{} for _ in calls],
        },
        "metrics": {"llm_calls": len(calls), "llm_call_attempts": len(calls)},
    }


def test_user_prompt_body_lines_are_indented_uniformly():
    """In plain-text rendering, the user prompt body — even when it contains
    arbitrary text including markdown-looking headings (e.g. the executor
    repair prompt's '## Repair required' / '### Failure diagnostics') —
    must be uniformly indented from the surrounding structure. Without
    indentation, embedded heading-like lines could be mistaken for log
    structure markers when scanning the file."""
    body_with_pseudo_headings = (
        "## Repair required\n"
        "The previous attempt failed.\n"
        "\n"
        "### Previous code\n"
        "x = 1\n"
        "\n"
        "### Failure diagnostics\n"
        "- error: KeyError\n"
    )
    record = _record_with_calls(
        [
            {
                "step": "code_execute_attempt_2",
                "model_input_raw": {
                    "system_instruction": "exec system",
                    "messages": [{"parts": [{"text": body_with_pseudo_headings}]}],
                },
            },
        ]
    )
    content = "\n".join(render_subagent_io_markdown(record))
    # The user body section starts with "  User:" header at 2-space indent
    user_idx = content.find("  User:")
    assert user_idx != -1, "missing User: section"
    # Body content (## Repair required ... ### Failure diagnostics) must
    # appear AFTER the User: header and each line should be indented (4
    # leading spaces). Specifically, '### Failure diagnostics' should appear
    # at the start of a line ONLY when prefixed by leading whitespace.
    fail_diag_substr = "### Failure diagnostics"
    fail_diag_idx = content.find(fail_diag_substr, user_idx)
    assert fail_diag_idx != -1, "missing failure diagnostics marker in body"
    # The character immediately before this position should be a space or
    # newline+spaces — i.e., the line must be indented.
    line_start = content.rfind("\n", 0, fail_diag_idx) + 1
    leading = content[line_start:fail_diag_idx]
    assert leading.strip() == "", (
        f"### Failure diagnostics not indented — leaked to top level "
        f"(leading={leading!r})"
    )
    assert len(leading) >= 4, (
        f"insufficient indent ({len(leading)} chars) — body could be "
        f"mistaken for log structure"
    )


def test_repeated_system_prompt_collapses_into_hash_banner():
    big_system = "You are the AppWorld code executor.\n" + ("Rule. " * 200)
    record = _record_with_calls(
        [
            {
                "step": "code_execute_attempt_1",
                "model_input_raw": {
                    "system_instruction": big_system,
                    "messages": [{"parts": [{"text": "First user prompt."}]}],
                },
            },
            {
                "step": "code_execute_attempt_2",
                "model_input_raw": {
                    "system_instruction": big_system,
                    "messages": [{"parts": [{"text": "Second user prompt."}]}],
                },
            },
            {
                "step": "code_execute_attempt_3",
                "model_input_raw": {
                    "system_instruction": big_system,
                    "messages": [{"parts": [{"text": "Third user prompt."}]}],
                },
            },
        ]
    )

    content = "\n".join(render_subagent_io_markdown(record))

    # The big system text should appear exactly once across all three calls.
    # In plain-text rendering, body lines are indented 4 spaces; check for a
    # representative substring (sans newlines / indent) instead of the raw
    # block.
    sample = "Rule. Rule. Rule."
    assert content.count(sample) >= 1  # at least once (in first occurrence dump)
    assert "first occurrence" in content
    # Each repeat should announce reuse of the first call by its hash.
    assert content.count("unchanged from") == 2
    assert content.count("code_execute_attempt_1") >= 3  # 1 heading + 2 banners
    # Per-call user prompts must still be visible.
    assert "First user prompt." in content
    assert "Second user prompt." in content
    assert "Third user prompt." in content


def test_changed_system_prompt_in_repair_attempt_renders_in_full():
    base_system = "You are the executor."
    repair_system = base_system + "\n\n## Repair required\nRewrite the program."
    record = _record_with_calls(
        [
            {
                "step": "code_execute_attempt_1",
                "model_input_raw": {
                    "system_instruction": base_system,
                    "messages": [{"parts": [{"text": "First."}]}],
                },
            },
            {
                "step": "code_execute_attempt_2",
                "model_input_raw": {
                    "system_instruction": repair_system,
                    "messages": [{"parts": [{"text": "Second."}]}],
                },
            },
        ]
    )

    content = "\n".join(render_subagent_io_markdown(record))

    # Both system bodies render fully — they have different hashes.
    assert "## Repair required" in content
    assert content.count("first occurrence") == 2
    assert "unchanged from" not in content


def test_repair_attempts_render_as_per_attempt_timeline_with_triggers():
    record = {
        "agent": "executor_subagent_code_plan_execute",
        "status": "FAILED",
        "input": {
            "phase": "EXECUTE",
            "attempt": 1,
            "model_calls": [
                {
                    "step": "code_plan",
                    "model_input_raw": {
                        "system_instruction": "planner system",
                        "messages": [{"parts": [{"text": "Plan task."}]}],
                    },
                },
                {
                    "step": "code_execute",
                    "model_input_raw": {
                        "system_instruction": "executor system",
                        "messages": [{"parts": [{"text": "Run code."}]}],
                    },
                },
            ],
        },
        "output": {
            "phase": "EXECUTE",
            "subagent_name": "executor_subagent_code_plan_execute",
            "attempt": 1,
            "failure_code": "EXECUTOR_DID_NOT_FINALIZE",
            "raw_llm_text": json.dumps(
                {
                    "code_plan": {"steps": []},
                    "code_execute": {
                        "mode": "llm",
                        "code": "print('latest attempt code')",
                        "raw_stdout": "latest stdout",
                        "stdout_json": None,
                        "parse_error": "venmo balance insufficient",
                        "tool_call_count": 1,
                        "tool_calls": [{"name": "execute_python", "args": {}}],
                        "event_diagnostics": [{"id": "e1"}],
                        "llm_raised": None,
                        "repair_attempts": [
                            {"attempt": 1, "parse_error": "no value key in stdout"},
                            {"attempt": 2, "parse_error": "code reported error: 422"},
                        ],
                    },
                }
            ),
            "model_calls": [
                {"step": "code_plan", "raw_llm_text": '{"steps": []}'},
                {
                    "step": "code_execute",
                    "raw_llm_text": "",
                    "code": "print('latest attempt code')",
                    "raw_stdout": "latest stdout",
                    "parse_error": "venmo balance insufficient",
                    "tool_calls": [{"name": "execute_python", "args": {}}],
                    "event_diagnostics": [{"id": "e1"}],
                    "repair_attempts": [
                        {
                            "attempt": 1,
                            "code": "print('first try')",
                            "raw_stdout": "",
                            "parse_error": "no value key in stdout",
                        },
                        {
                            "attempt": 2,
                            "code": "print('second try with try/except')",
                            "raw_stdout": '{"error": "422"}',
                            "parse_error": "code reported error: 422 balance insufficient",
                        },
                    ],
                },
            ],
        },
        "metrics": {"llm_calls": 2, "llm_call_attempts": 2},
    }

    content = "\n".join(render_subagent_io_markdown(record))

    # Per-attempt section appears with attempt count.
    assert "[2] code_execute (2 attempts)" in content
    assert "Attempts (2 code execution attempts on this milestone):" in content
    # Each attempt shown with its code/stdout/parse_error.
    assert "Attempt 1 -- initial" in content
    assert "print('first try')" in content
    assert "no value key in stdout" in content
    # Trigger one-liner for attempt 2 references attempt 1's failure.
    assert "Attempt 2 -- triggered by attempt 1: no value key in stdout" in content
    assert "print('second try with try/except')" in content
    # Top-level dedup: code/tool_calls/event_diagnostics suppressed when
    # repair_attempts present (they would otherwise duplicate the latest
    # attempt's record).
    code_execute_section = content.split("[2] code_execute", 1)[1].split(
        "PARSED OUTPUT", 1
    )[0]
    assert "Tool calls:" not in code_execute_section
    assert "execute_python code:" not in code_execute_section
    assert "Event diagnostics:" not in code_execute_section
    # Compacted parsed output: redundant fields stripped, banner points back.
    parsed_section = content.split("PARSED OUTPUT", 1)[1]
    assert "_attempts_rendered_above" in parsed_section
    assert '"repair_attempts"' not in parsed_section


def test_seen_system_dedups_across_subagent_reports():
    """When the same system text appears across different subagent reports
    (e.g. executor attempt 1 + attempt 2), the second report should collapse
    via the shared seen_system map."""
    big_system = "Executor system. " * 100
    record_attempt_1 = _record_with_calls(
        [
            {
                "step": "code_execute",
                "model_input_raw": {
                    "system_instruction": big_system,
                    "messages": [{"parts": [{"text": "first attempt"}]}],
                },
            },
        ]
    )
    record_attempt_2 = _record_with_calls(
        [
            {
                "step": "code_execute",
                "model_input_raw": {
                    "system_instruction": big_system,
                    "messages": [{"parts": [{"text": "second attempt"}]}],
                },
            },
        ]
    )
    record_attempt_2["input"]["attempt"] = 2
    record_attempt_2["output"]["attempt"] = 2

    seen: dict[str, str] = {}
    out_1 = "\n".join(render_subagent_io_markdown(record_attempt_1, seen_system=seen))
    out_2 = "\n".join(render_subagent_io_markdown(record_attempt_2, seen_system=seen))

    # Full system text appears in attempt 1 only.
    assert out_1.count(big_system.strip()) == 1
    assert out_2.count(big_system.strip()) == 0
    # Attempt 2 collapses with a banner pointing at attempt 1.
    assert "unchanged from" in out_2
    assert "attempt 1" in out_2
