from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

from adk_appworld_agent.observability.live_timeline import LiveTimeline
from scripts.demo_dashboard import make_server, read_events
from scripts.run_metrics import extract_run_metrics
from scripts.timeline_stages import build_display_blocks, build_stages


def test_live_timeline_allowlists_fields(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    sandbox = tmp_path / "sandbox.jsonl"
    sandbox.write_text(json.dumps({"api_calls": [
        {"app": "catalog", "api_name": "search_products", "kwargs": {"password": "secret-marker"}, "result_items": ["secret-marker"]}
    ]}) + "\n", encoding="utf-8")
    timeline = LiveTimeline(path, sandbox)
    timeline.record_state({"phase": "PLAN", "active_milestone_index": 0, "status": "RUNNING", "private": "secret-marker"})
    timeline.record_output("PLAN", {"payload": {"tasks": [{"app": "catalog", "task": "Find products"}], "private": "secret-marker"}})
    timeline.record_output("FIND", {"payload": {"matched_apps": ["catalog"], "candidate_apis": [{"app": "catalog", "name": "search_products", "description": "secret-marker"}]}})
    timeline.record_output("EXECUTE", {"payload": {"code_execute": "secret-marker", "executor_result": {"summary": "Found products", "private": "secret-marker"}, "tool_call_count": 1, "milestone_done": True}})
    timeline.record_output("PLAN", {"payload": {"next_action": "continue", "rationale": "One step remains", "revised_milestones": [{"app": "catalog", "task": "Retry safely", "private": "secret-marker"}], "private": "secret-marker"}})
    timeline.record_result({"status": "COMPLETED", "eval": {"passed": 2, "total": 2, "passed_all": True}, "private": "secret-marker"})

    events = read_events(path)
    assert [event["type"] for event in events] == ["phase_started", "plan", "retrieval", "execution", "control", "evaluation"]
    assert events[2]["apis"] == ["catalog.search_products"]
    assert events[3]["called_apis"] == ["catalog.search_products"]
    assert events[4]["revised_milestones"] == [{"app": "catalog", "task": "Retry safely"}]
    assert "secret-marker" not in path.read_text(encoding="utf-8")


def test_live_timeline_reads_revised_milestone_intent(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    timeline = LiveTimeline(path, tmp_path / "sandbox.jsonl")
    timeline.record_output("PLAN", {"payload": {
        "next_action": "RETRY",
        "revised_milestones": [{"app": "spotify", "intent": "Collect all song details"}],
    }})
    assert read_events(path)[0]["revised_milestones"] == [
        {"app": "spotify", "task": "Collect all song details"}
    ]


def test_dashboard_serves_preview_and_events_on_loopback(tmp_path: Path) -> None:
    timeline_path = tmp_path / "events.jsonl"
    timeline_path.write_text(json.dumps({"type": "phase_started", "phase": "PLAN"}) + "\n", encoding="utf-8")
    server = make_server(timeline_path, mode="preview")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/events") as response:
            payload = json.load(response)
        assert payload["mode"] == "preview"
        assert payload["events"][0]["phase"] == "PLAN"
        assert payload["stages"] == [{"type": "plan", "stage_status": "running", "at": None, "milestone_index": None}]
        assert payload["blocks"] == [{"kind": "stage", "stage": payload["stages"][0]}]
        assert payload["metrics"] is None
        with urlopen(base + "/") as response:
            page = response.read()
            assert b"Agent timeline" in page
            assert b"Run summary" in page
            assert b"Subagent aggregate" not in page
        with urlopen(base + "/dashboard.js") as response:
            assert b"ILLUSTRATIVE PREVIEW" in response.read()
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


def test_metrics_allowlist_and_replay_summary_fallback(tmp_path: Path) -> None:
    summary = {
        "wall_s": 69.555,
        "aggregate_metrics": {
            "llm_calls": 8, "llm_call_attempts": 9,
            "prompt_tokens": 27_555, "completion_tokens": 669,
            "thoughts_tokens": 2_969, "total_tokens": 31_193,
            "private": "secret-marker",
        },
        "api_key": "secret-marker",
    }
    expected = {
        "duration_s": 69.555,
        "llm_calls": 8, "llm_call_attempts": 9,
        "prompt_tokens": 27_555, "completion_tokens": 669,
        "thoughts_tokens": 2_969, "total_tokens": 31_193,
    }
    assert extract_run_metrics(summary) == expected
    assert "secret-marker" not in json.dumps(expected)

    timeline_path = tmp_path / "events.jsonl"
    timeline_path.write_text(json.dumps({"type": "evaluation", "passed": 6, "total": 6}) + "\n", encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    server = make_server(timeline_path, mode="replay", summary_path=summary_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/events") as response:
            payload = json.load(response)
        assert payload["metrics"] == expected
        assert "secret-marker" not in json.dumps(payload)
        timeline_path.write_text(
            json.dumps({"type": "metrics", "values": {"duration_s": 12.0, "llm_calls": 2}}) + "\n",
            encoding="utf-8",
        )
        with urlopen(f"http://127.0.0.1:{server.server_port}/api/events") as response:
            updated = json.load(response)
        assert updated["metrics"] == {"duration_s": 12.0, "llm_calls": 2}
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()


def test_stages_pair_loading_with_result_without_duplicate_bootstrap() -> None:
    events = [
        {"type": "phase_started", "phase": "BOOTSTRAP", "at": 1},
        {"type": "task", "task_id": "example", "at": 2},
        {"type": "phase_started", "phase": "BOOTSTRAP", "milestone_index": 0, "at": 3},
        {"type": "phase_started", "phase": "PLAN", "milestone_index": 0, "at": 4},
        {"type": "plan", "milestones": [{"app": "phone", "task": "Send text"}], "at": 5},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0, "at": 6},
        {"type": "retrieval", "apis": ["phone.search_contacts"], "at": 7},
        {"type": "phase_started", "phase": "EXECUTE", "milestone_index": 0, "at": 8},
        {"type": "execution", "called_apis": ["phone.search_contacts"], "at": 9},
        {"type": "phase_started", "phase": "PLAN", "milestone_index": 0, "at": 10},
        {"type": "control", "action": "SUBMIT", "at": 11},
        {"type": "phase_started", "phase": "SUBMIT", "milestone_index": 0, "at": 12},
        {"type": "submission", "status": "SUBMITTED", "at": 13},
        {"type": "phase_started", "phase": "COMPLETE", "milestone_index": 0, "at": 14},
        {"type": "evaluation", "passed": 6, "total": 6, "at": 15},
    ]
    stages = build_stages(events)

    assert [stage["type"] for stage in stages] == [
        "bootstrap", "plan", "retrieval", "execution", "control", "submission", "evaluation"
    ]
    assert all(stage["stage_status"] == "complete" for stage in stages)
    assert stages[0]["at"] == 2
    assert stages[1]["milestones"] == [{"app": "phone", "task": "Send text"}]
    assert stages[2]["milestone_index"] == 0
    assert stages[2]["plan_item"] == {
        "number": 1, "total": 1, "app": "phone", "task": "Send text", "cycle": 1
    }
    assert stages[3]["plan_item"] == stages[2]["plan_item"]
    assert stages[4]["plan_item"] == stages[2]["plan_item"]
    assert stages[5]["status"] == "SUBMITTED"


def test_pending_stage_is_replaced_when_result_arrives() -> None:
    before = build_stages([
        {"type": "phase_started", "phase": "PLAN", "at": 1},
    ])
    after = build_stages([
        {"type": "phase_started", "phase": "PLAN", "at": 1},
        {"type": "plan", "milestones": [{"task": "First"}], "at": 2},
    ])
    assert len(before) == len(after) == 1
    assert before[0]["stage_status"] == "running"
    assert after[0]["stage_status"] == "complete"


def test_two_plan_items_keep_their_own_retrieval_and_execution() -> None:
    events = [
        {"type": "plan", "milestones": [{"task": "First"}, {"task": "Second"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": ["app.first"]},
        {"type": "phase_started", "phase": "EXECUTE", "milestone_index": 0},
        {"type": "execution", "called_apis": ["app.first"]},
        {"type": "phase_started", "phase": "PLAN", "milestone_index": 0},
        {"type": "control", "action": "CONTINUE"},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 1},
        {"type": "retrieval", "apis": ["app.second"]},
        {"type": "phase_started", "phase": "EXECUTE", "milestone_index": 1},
        {"type": "execution", "called_apis": ["app.second"]},
    ]
    stages = build_stages(events)

    assert [stage["type"] for stage in stages] == [
        "plan", "retrieval", "execution", "control", "retrieval", "execution"
    ]
    assert [stage["milestone_index"] for stage in stages if stage["type"] in {"retrieval", "execution"}] == [0, 0, 1, 1]
    assert [stage["plan_item"]["task"] for stage in stages if stage["type"] == "retrieval"] == ["First", "Second"]


def test_repeating_the_same_plan_item_starts_another_cycle() -> None:
    events = [
        {"type": "plan", "milestones": [{"app": "spotify", "task": "Check playlists"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": ["spotify.search"]},
        {"type": "phase_started", "phase": "EXECUTE", "milestone_index": 0},
        {"type": "execution", "called_apis": ["spotify.search"]},
        {"type": "phase_started", "phase": "PLAN", "milestone_index": 0},
        {"type": "control", "action": "CONTINUE"},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": ["spotify.get_playlist"]},
    ]
    stages = build_stages(events)
    assert [stage["plan_item"]["cycle"] for stage in stages if stage["type"] == "retrieval"] == [1, 2]
    blocks = build_display_blocks(stages)
    assert [block["kind"] for block in blocks] == ["stage", "plan_item"]
    assert blocks[1]["item"]["task"] == "Check playlists"
    assert [cycle["number"] for cycle in blocks[1]["cycles"]] == [1, 2]
    assert [[stage["type"] for stage in cycle["stages"]] for cycle in blocks[1]["cycles"]] == [
        ["retrieval", "execution", "control"], ["retrieval"]
    ]


def test_display_blocks_separate_plan_items_and_final_submission() -> None:
    stages = build_stages([
        {"type": "plan", "milestones": [{"task": "First"}, {"task": "Second"}]},
        {"type": "retrieval", "milestone_index": 0},
        {"type": "control", "milestone_index": 0, "action": "CONTINUE"},
        {"type": "retrieval", "milestone_index": 1},
        {"type": "submission", "status": "SUBMITTED"},
    ])
    blocks = build_display_blocks(stages)
    assert [block["kind"] for block in blocks] == ["stage", "plan_item", "plan_item", "stage"]
    assert [block["item"]["task"] for block in blocks[1:3]] == ["First", "Second"]
    assert blocks[-1]["stage"]["type"] == "submission"


def test_retrieval_retry_without_control_stays_in_the_same_cycle() -> None:
    stages = build_stages([
        {"type": "plan", "milestones": [{"app": "spotify", "task": "Check playlists"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": []},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": ["spotify.search"]},
    ])
    assert [stage["plan_item"]["cycle"] for stage in stages if stage["type"] == "retrieval"] == [1, 1]


def test_revised_plan_changes_the_next_cycle_context() -> None:
    events = [
        {"type": "plan", "milestones": [{"app": "spotify", "task": "Initial goal"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": []},
        {"type": "phase_started", "phase": "PLAN", "milestone_index": 0},
        {"type": "control", "action": "RETRY", "revised_milestones": [{"app": "spotify", "task": "Revised goal"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
    ]
    stages = build_stages(events)
    assert stages[-1]["stage_status"] == "running"
    assert stages[-1]["plan_item"]["task"] == "Revised goal"
    assert stages[-1]["plan_item"]["cycle"] == 2


def test_partial_revision_preserves_other_plan_items() -> None:
    stages = build_stages([
        {"type": "plan", "milestones": [
            {"task": "First"}, {"task": "Second"}, {"task": "Third"},
        ]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 1},
        {"type": "retrieval", "apis": []},
        {"type": "control", "action": "RETRY", "revised_milestones": [{"task": "Second, revised"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 1},
        {"type": "retrieval", "apis": []},
        {"type": "control", "action": "ADVANCE"},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 2},
    ])
    items = [stage["plan_item"] for stage in stages if stage.get("plan_item")]
    assert [(item["number"], item["task"], item["cycle"]) for item in items] == [
        (2, "Second", 1), (2, "Second", 1),
        (2, "Second, revised", 2), (2, "Second, revised", 2),
        (3, "Third", 1),
    ]


def test_control_without_phase_state_uses_the_current_plan_item() -> None:
    stages = build_stages([
        {"type": "plan", "milestones": [{"app": "catalog", "task": "Find product"}]},
        {"type": "phase_started", "phase": "FIND", "milestone_index": 0},
        {"type": "retrieval", "apis": ["catalog.search"]},
        {"type": "control", "action": "SUBMIT"},
    ])
    assert stages[-1]["type"] == "control"
    assert stages[-1]["plan_item"]["task"] == "Find product"
