from __future__ import annotations

import json
from pathlib import Path

from scripts.build_task_viewer import build_viewer, render


def _summary() -> dict:
    return {
        "task_id": "example_1",
        "instruction": "Find a contact and send a message",
        "status": "COMPLETED",
        "wall_s": 12.5,
        "eval": {"passed": 2, "total": 2},
        "milestones": [
            {
                "goal": "Find the contact",
                "app": "phone",
                "find": {"selected_apis": ["phone.search_contacts"]},
                "execute": {
                    "called_apis": ["phone.search_contacts"],
                    "status": "SUCCEEDED",
                    "api_call_trace": [{"kwargs": {"secret": "DO_NOT_RENDER"}}],
                },
            }
        ],
        "run": {"command": "PRIVATE_VALUE_DO_NOT_RENDER"},
    }


def test_viewer_shows_workflow_without_raw_values() -> None:
    page = render(_summary())
    assert "phone.search_contacts" in page
    assert "2/2" in page
    assert "DO_NOT_RENDER" not in page
    assert "Find APIs" in page


def test_viewer_escapes_task_text(tmp_path: Path) -> None:
    summary = _summary()
    summary["instruction"] = '<script>alert("x")</script>'
    summary_path = tmp_path / "task_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    output = build_viewer(summary_path)
    page = output.read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page
    assert '<script>alert("x")</script>' not in page
