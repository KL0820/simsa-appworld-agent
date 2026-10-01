from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from adk_appworld_agent.observability.paths import (
    SUBAGENT_IO_JSONL_FILENAME,
    artifacts_dir,
)
from adk_appworld_agent.orchestration.state_repo import RUN_STATE_KEY
from adk_appworld_agent.orchestration.subagent_output_store import SUBAGENT_OUTPUTS_KEY


def _load_run_task_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_task.py"
    spec = importlib.util.spec_from_file_location("run_task_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_subagent_io_markdown_orders_plan_before_find(tmp_path):
    run_task = _load_run_task_module()
    subagent_outputs = {
        "FIND": [
            {
                "subagent_name": "finder_subagent_community",
                "attempt": 1,
                "status": "SUCCEEDED",
                "payload": {
                    "io": {
                        "input": {
                            "model_input": {
                                "system_instruction": "Find.",
                                "messages": [{"parts": [{"text": "find user"}]}],
                            }
                        },
                        "output": {"raw_llm_text": '{"candidate_apis": []}'},
                    }
                },
            }
        ],
        "PLAN": [
            {
                "subagent_name": "rough_planner_subagent",
                "attempt": 1,
                "status": "SUCCEEDED",
                "payload": {
                    "io": {
                        "input": {
                            "model_input": {
                                "system_instruction": "Plan.",
                                "messages": [{"parts": [{"text": "plan task"}]}],
                            }
                        },
                        "output": {"raw_llm_text": '{"tasks": []}'},
                    }
                },
            }
        ],
    }

    artifact_dir = artifacts_dir(tmp_path)
    path = run_task._write_subagent_io_markdown(
        tmp_path, artifact_dir, subagent_outputs
    )

    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert content.find('"phase": "PLAN"') < content.find('"phase": "FIND"')
    jsonl_path = artifact_dir / SUBAGENT_IO_JSONL_FILENAME
    jsonl_records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["phase"] for record in jsonl_records] == ["PLAN", "FIND"]
    assert all(
        "envelope" in record and "io_record" in record for record in jsonl_records
    )


def test_task_log_guard_rejects_existing_completed_outputs(tmp_path):
    run_task = _load_run_task_module()
    (tmp_path / "task_summary.txt").write_text("old\n", encoding="utf-8")

    try:
        run_task._ensure_task_log_can_be_written(tmp_path, overwrite=False)
    except FileExistsError as exc:
        assert "--overwrite-task-log" in str(exc)
    else:
        raise AssertionError("expected existing task summary to be rejected")

    run_task._ensure_task_log_can_be_written(tmp_path, overwrite=True)


def test_progress_lines_show_subagent_result_before_post_transition_state():
    run_task = _load_run_task_module()
    event = SimpleNamespace(
        actions=SimpleNamespace(
            state_delta={
                SUBAGENT_OUTPUTS_KEY: {
                    "EXECUTE": [
                        {
                            "subagent_name": "executor_subagent_stub",
                            "attempt": 4,
                            "status": "SUCCEEDED",
                            "metrics": {"llm_calls": 0, "wall_ms": 0},
                            "payload": {
                                "finalize_called": True,
                                "milestone_done": True,
                                "tool_call_count": 0,
                            },
                        }
                    ]
                },
                RUN_STATE_KEY: {
                    "phase": "FIND",
                    "status": "RUNNING",
                    "milestones": [
                        {},
                        {},
                        {},
                        {},
                        {
                            "app": "phone",
                            "intent": "Read friends from the phone contacts app.",
                        },
                    ],
                    "active_milestone_index": 4,
                },
            }
        )
    )

    lines, _ = run_task._event_progress_lines(
        event,
        seen_outputs=set(),
        last_state_line="[state] phase=EXECUTE status=RUNNING active_milestone=4/5",
    )

    assert lines[0].startswith("[EXECUTE")
    assert (
        lines[-1] == "[state] phase=FIND status=RUNNING active_milestone=5/5 app=phone "
        'intent="Read friends from the phone contacts app."'
    )


def test_state_progress_line_includes_active_milestone_instruction():
    run_task = _load_run_task_module()

    line = run_task._state_progress_line(
        {
            "phase": "EXECUTE",
            "status": "RUNNING",
            "milestones": [
                {
                    "app": "venmo",
                    "intent": "Read the current list of friends from Venmo.",
                },
                {
                    "app": "phone",
                    "intent": "Read my friends in my phone.",
                },
            ],
            "active_milestone_index": 0,
        }
    )

    assert line == (
        "[state] phase=EXECUTE status=RUNNING active_milestone=1/2 app=venmo "
        'intent="Read the current list of friends from Venmo."'
    )


def test_live_stage_result_line_includes_milestone_context():
    run_task = _load_run_task_module()
    item = {
        "subagent_name": "finder_subagent_community",
        "attempt": 2,
        "status": "SUCCEEDED",
        "metrics": {"llm_calls": 1, "wall_ms": 1234},
        "payload": {
            "matched_apps": ["gmail"],
            "candidate_count": 4,
            "io": {
                "input": {
                    "subagent_input": {
                        "metadata": {
                            "planned_apps": ["gmail"],
                            "milestone_index": 1,
                            "milestone_total": 3,
                            "milestone_intent": "Find the relevant email before paying.",
                        }
                    }
                }
            },
        },
    }

    lines = run_task._stage_result_lines("FIND", item)
    rendered = "\n".join(lines)

    assert lines[0].startswith("[FIND")
    assert "llm=1" in rendered
    assert "milestone=2/3" in rendered
    assert "app=gmail" in rendered
    assert 'intent="Find the relevant email before paying."' in rendered
    assert "apis=0" in rendered


def test_live_stage_result_lines_include_plan_tasks_and_find_apis():
    run_task = _load_run_task_module()

    plan_lines = run_task._stage_result_lines(
        "PLAN",
        {
            "subagent_name": "rough_planner_subagent",
            "attempt": 1,
            "status": "SUCCEEDED",
            "metrics": {"llm_calls": 0},
            "payload": {
                "cache_source": "cache.jsonl",
                "task_count": 2,
                "tasks": [
                    {"app": "venmo", "task": "Read the current Venmo friends."},
                    {"app": "phone", "task": "Read phone contacts."},
                ],
            },
        },
    )
    assert plan_lines[0].startswith("[PLAN")
    assert "OK cache" in plan_lines[0]
    assert "llm=0" in "\n".join(plan_lines)
    assert "  -> 2 tasks: venmo, phone" in plan_lines

    find_lines = run_task._stage_result_lines(
        "FIND",
        {
            "subagent_name": "finder_subagent_community",
            "attempt": 1,
            "status": "SUCCEEDED",
            "metrics": {"llm_calls": 2, "wall_ms": 1000},
            "payload": {
                "matched_apps": ["venmo"],
                "selected_communities": ["venmo_c3"],
                "candidate_count": 3,
                "candidate_apis": [
                    {"app": "venmo", "name": "login"},
                    {"app": "venmo", "name": "search_friends"},
                    "venmo.remove_friend",
                ],
            },
        },
    )
    assert find_lines[0].startswith("[FIND")
    assert "llm=2" in "\n".join(find_lines)
    assert "  -> matched=venmo apis=3" in find_lines
    assert "     venmo.login, venmo.search_friends, venmo.remove_friend" in find_lines


def test_live_execute_result_lines_include_code_plan_steps():
    run_task = _load_run_task_module()

    lines = run_task._stage_result_lines(
        "EXECUTE",
        {
            "subagent_name": "executor_subagent_code_plan_execute",
            "attempt": 1,
            "status": "SUCCEEDED",
            "metrics": {"llm_calls": 2, "wall_ms": 1000},
            "payload": {
                "finalize_called": True,
                "milestone_done": True,
                "tool_call_count": 1,
                "code_plan": {
                    "plan": [
                        "Read Venmo friends with venmo.search_friends.",
                        "Compare Venmo friends against phone friends.",
                        "Befriend missing phone friends.",
                        "Unfriend extra Venmo friends.",
                    ],
                    "output_variable": {"name": "venmo_friend_sync_result"},
                },
            },
        },
    )
    rendered = "\n".join(lines)

    assert "  -> finalize=True done=True tools=1" in lines
    assert "  -> code_plan steps=4 output=venmo_friend_sync_result" in lines
    assert "     1. Read Venmo friends with venmo.search_friends." in lines
    assert "     2. Compare Venmo friends against phone friends." in lines
    assert "     3. Befriend missing phone friends." in lines
    assert "     ... +1 more" in lines
    assert "Unfriend extra Venmo friends." not in rendered
