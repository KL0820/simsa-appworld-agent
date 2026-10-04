from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

from adk_appworld_agent.observability.live_timeline import LiveTimeline
from scripts.demo_dashboard import make_server, read_events


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
    timeline.record_output("PLAN", {"payload": {"next_action": "continue", "rationale": "One step remains", "private": "secret-marker"}})
    timeline.record_result({"status": "COMPLETED", "eval": {"passed": 2, "total": 2, "passed_all": True}, "private": "secret-marker"})

    events = read_events(path)
    assert [event["type"] for event in events] == ["phase_started", "plan", "retrieval", "execution", "control", "evaluation"]
    assert events[2]["apis"] == ["catalog.search_products"]
    assert events[3]["called_apis"] == ["catalog.search_products"]
    assert "secret-marker" not in path.read_text(encoding="utf-8")


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
        with urlopen(base + "/") as response:
            assert b"Agent timeline" in response.read()
        with urlopen(base + "/dashboard.js") as response:
            assert b"ILLUSTRATIVE PREVIEW" in response.read()
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()
