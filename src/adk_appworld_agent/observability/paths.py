from __future__ import annotations

from pathlib import Path

ARTIFACTS_DIRNAME = "artifacts"
SUBAGENT_IO_JSONL_FILENAME = "subagent_io.jsonl"


def artifacts_dir(log_dir: Path) -> Path:
    path = log_dir / ARTIFACTS_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path
