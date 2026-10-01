"""Batch-level sink: maintains a single rolling `batch_results.md`.

Subscribes to `BatchStarted` (initialize task list with all tasks marked
pending) and `BatchTaskCompleted` (overwrite that task's section with its
result). The file is rewritten in full on every event so a partial run
still produces a valid file and Ctrl-C does not corrupt it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from adk_appworld_agent.observability.events import (
    BatchFinished,
    BatchStarted,
    BatchTaskCompleted,
    OrchestrationEvent,
)
from adk_appworld_agent.observability.markdown import render_batch_results_markdown

BATCH_RESULTS_FILENAME = "batch_results.txt"


def _aggregate_metrics(summary: dict | None) -> dict:
    if not isinstance(summary, dict):
        return {}
    aggregate = (
        summary.get("aggregate_metrics")
        if isinstance(summary.get("aggregate_metrics"), dict)
        else {}
    )
    return aggregate


def _result_from_summary(summary: dict | None, returncode: int) -> dict:
    block_reason = "SUBPROCESS_FAILED" if returncode != 0 else None
    if returncode == 124:
        block_reason = "TASK_TIMEOUT"
    if not isinstance(summary, dict):
        return {
            "passed": None,
            "total": None,
            "passed_all": False,
            "wall_s": 0.0,
            "llm_calls": 0,
            "total_tokens": 0,
            "block_reason": block_reason,
            "exec_failure": None,
        }
    eval_block = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    overview = (
        summary.get("overview") if isinstance(summary.get("overview"), dict) else {}
    )
    aggregate = _aggregate_metrics(summary)
    passed = eval_block.get("passed")
    total = eval_block.get("total")
    passed_all = (
        passed is not None and total is not None and total > 0 and passed == total
    )
    summary_block_reason = overview.get("block_reason")
    if block_reason is None:
        block_reason = summary_block_reason
    return {
        "passed": passed,
        "total": total,
        "passed_all": bool(passed_all),
        "wall_s": summary.get("wall_s") or 0.0,
        "llm_calls": aggregate.get("llm_calls") or 0,
        "total_tokens": aggregate.get("total_tokens") or 0,
        "block_reason": block_reason,
        "exec_failure": overview.get("exec_failure"),
        # infra 備註 fields (gemini-hang perturbation visibility)
        "rate_limit_count": aggregate.get("rate_limit_count") or 0,
        "provider_issue_likely": overview.get("provider_issue_likely"),
    }


class BatchResultsMarkdownSink:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / BATCH_RESULTS_FILENAME
        self._state: dict = {"run_name": "", "config": {}, "tasks": []}

    def handle(self, event: OrchestrationEvent) -> None:
        if isinstance(event, BatchStarted):
            self._state = {
                "run_name": event.run_name,
                "config": dict(event.config or {}),
                "tasks": [
                    {
                        "task_id": task.get("task_id", ""),
                        "instruction": task.get("instruction", ""),
                        "state": "pending",
                    }
                    for task in (event.tasks or [])
                ],
            }
            self._adopt_existing_summaries()
            self._flush()
            return
        if isinstance(event, BatchTaskCompleted):
            self._upsert_completed(event)
            self._flush()
            return
        if isinstance(event, BatchFinished):
            self._flush()
            return

    def _adopt_existing_summaries(self) -> None:
        """Resume support: fold task_summary.json files already on disk (from a
        previous invocation into the same run dir) in as completed rows, so a
        resumed batch's results file covers the whole run, not just its own
        task subset. Adopted rows are listed before this invocation's tasks."""
        run_dir = self.path.parent
        known = {task.get("task_id") for task in self._state["tasks"]}
        adopted = []
        for summary_path in run_dir.glob("*/artifacts/task_summary.json"):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            task_id = summary.get("task_id") or summary_path.parent.parent.name
            if task_id in known:
                continue
            adopted.append(
                (
                    summary.get("datetime") or "",
                    {
                        "task_id": task_id,
                        "instruction": summary.get("instruction", ""),
                        "state": "completed",
                        "result": _result_from_summary(summary, 0),
                    },
                )
            )
        if adopted:
            # completion-time order == the original invocation's execution
            # (= split file) order, so the merged table reads as one run
            adopted.sort(key=lambda pair: pair[0])
            self._state["tasks"] = [entry for _, entry in adopted] + self._state[
                "tasks"
            ]

    def _upsert_completed(self, event: BatchTaskCompleted) -> None:
        result = _result_from_summary(event.summary, event.returncode)
        new_entry = {
            "task_id": event.task_id,
            "instruction": event.instruction,
            "state": "completed",
            "result": result,
        }
        for index, task in enumerate(self._state["tasks"]):
            if task.get("task_id") == event.task_id:
                self._state["tasks"][index] = new_entry
                return
        self._state["tasks"].append(new_entry)

    def _flush(self) -> None:
        self._state["updated_at"] = (
            datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            render_batch_results_markdown(self._state), encoding="utf-8"
        )
