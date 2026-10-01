from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.observability.events import OrchestrationEvent, TaskCompleted
from adk_appworld_agent.observability.markdown import render_task_summary_markdown

TASK_SUMMARY_FILENAME = "task_summary.txt"


class TaskSummaryMarkdownSink:
    def __init__(self, log_dir: Path) -> None:
        self.path = log_dir / TASK_SUMMARY_FILENAME
        self._summary: dict | None = None

    def handle(self, event: OrchestrationEvent) -> None:
        if isinstance(event, TaskCompleted):
            self._summary = event.summary

    def flush(self) -> Path | None:
        if self._summary is None:
            return None
        self.path.write_text(
            render_task_summary_markdown(self._summary), encoding="utf-8"
        )
        return self.path
