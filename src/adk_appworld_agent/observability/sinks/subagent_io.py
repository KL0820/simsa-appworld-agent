from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.observability.events import (
    OrchestrationEvent,
    SubagentCompleted,
)
from adk_appworld_agent.observability.markdown import render_subagent_io_markdown
from adk_appworld_agent.observability.subagent_logs import SUBAGENT_IO_FILENAME


class SubagentIoMarkdownSink:
    def __init__(self, log_dir: Path) -> None:
        self.path = log_dir / SUBAGENT_IO_FILENAME
        self._records: list[dict] = []

    def handle(self, event: OrchestrationEvent) -> None:
        if isinstance(event, SubagentCompleted):
            self._records.append(event.io_record)

    def flush(self) -> Path | None:
        if not self._records:
            return None
        lines: list[str] = []
        # Shared across all subagent reports for this task so that identical
        # system prompts on retries / multi-milestone runs collapse to a hash
        # banner instead of being printed in full each time.
        seen_system: dict[str, str] = {}
        for index, record in enumerate(self._records):
            if index:
                lines.extend(["", "---", ""])
            lines.extend(render_subagent_io_markdown(record, seen_system=seen_system))
        self.path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return self.path
