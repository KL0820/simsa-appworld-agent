"""BatchResultsMarkdownSink resume: adopt on-disk summaries from a prior invocation."""

from __future__ import annotations

import json
from pathlib import Path

from adk_appworld_agent.observability.events import BatchStarted, BatchTaskCompleted
from adk_appworld_agent.observability.sinks.batch_results import (
    BatchResultsMarkdownSink,
)


def _write_summary(
    run_dir: Path, task_id: str, *, passed: int, total: int, datetime: str = ""
) -> None:
    artifact_dir = run_dir / task_id / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "task_summary.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "datetime": datetime,
                "instruction": f"instruction for {task_id}",
                "eval": {"passed": passed, "total": total},
                "wall_s": 10.0,
                "aggregate_metrics": {"llm_calls": 5, "total_tokens": 1000},
                "overview": {"block_reason": None},
            }
        ),
        encoding="utf-8",
    )


def test_batch_started_adopts_prior_invocation_summaries(tmp_path):
    # bbb completed FIRST: adoption must order by completion time, not task id
    _write_summary(
        tmp_path, "aaa111_1", passed=3, total=3, datetime="2026-06-12 18:30:00"
    )
    _write_summary(
        tmp_path, "bbb222_1", passed=1, total=4, datetime="2026-06-12 18:10:00"
    )

    sink = BatchResultsMarkdownSink(tmp_path)
    sink.handle(
        BatchStarted(
            run_name="run",
            config={},
            tasks=[{"task_id": "ccc333_1", "instruction": "third task"}],
        )
    )

    ids = [t["task_id"] for t in sink._state["tasks"]]
    assert ids == ["bbb222_1", "aaa111_1", "ccc333_1"]  # completion order, then current
    adopted = sink._state["tasks"][1]
    assert adopted["state"] == "completed"
    assert adopted["result"]["passed_all"] is True
    assert sink._state["tasks"][2]["state"] == "pending"

    rendered = sink.path.read_text(encoding="utf-8")
    assert "aaa111_1" in rendered and "ccc333_1" in rendered


def test_completion_event_still_overwrites_adopted_row(tmp_path):
    _write_summary(tmp_path, "aaa111_1", passed=1, total=3)
    sink = BatchResultsMarkdownSink(tmp_path)
    sink.handle(BatchStarted(run_name="run", config={}, tasks=[]))
    sink.handle(
        BatchTaskCompleted(
            task_id="aaa111_1",
            instruction="instruction for aaa111_1",
            index=1,
            total=1,
            summary={
                "task_id": "aaa111_1",
                "eval": {"passed": 3, "total": 3},
                "wall_s": 20.0,
                "aggregate_metrics": {},
                "overview": {},
            },
            returncode=0,
        )
    )
    assert len(sink._state["tasks"]) == 1
    assert sink._state["tasks"][0]["result"]["passed_all"] is True
