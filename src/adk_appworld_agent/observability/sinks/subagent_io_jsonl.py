from __future__ import annotations

import json
from pathlib import Path

from adk_appworld_agent.observability.events import (
    OrchestrationEvent,
    SubagentCompleted,
)
from adk_appworld_agent.observability.paths import SUBAGENT_IO_JSONL_FILENAME


class SubagentIoJsonlSink:
    """Machine-readable sidecar to subagent_io.md.

    Each line is a record with both the human-facing `io_record` (used by
    markdown renderers) and the raw `envelope` (full SubagentEnvelope dict
    from the subagent output store). `build_cache.py` consumes the
    `envelope` block when deriving cache entries; markdown rendering keeps
    using `io_record`.
    """

    def __init__(self, artifacts_dir: Path) -> None:
        self.path = artifacts_dir / SUBAGENT_IO_JSONL_FILENAME
        self._records: list[dict] = []

    def handle(self, event: OrchestrationEvent) -> None:
        if isinstance(event, SubagentCompleted):
            self._records.append(
                {
                    "phase": event.phase_name,
                    "io_record": event.io_record,
                    "envelope": event.raw_output,
                }
            )

    def flush(self) -> Path | None:
        if not self._records:
            return None
        with self.path.open("w", encoding="utf-8") as sink:
            for record in self._records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        return self.path
