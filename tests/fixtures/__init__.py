"""Deterministic fail-case fixtures extracted from full56 logs.

Each JSON file under `executor_failures/` collects ground-truth fail
cases that a specific fix targets. Tests load fixtures via
`load_executor_failure_fixture(name)` and assert the post-fix behavior
(parse_error class, normalized submission answer, etc.).

Mixing real-log entries with `synthetic: true` adversarial cases is
deliberate — the real cases pin the fix to the observed bug, the
adversarial cases defend against regex / heuristic over-fit.
"""

from __future__ import annotations

import json
from pathlib import Path

_EXECUTOR_FAILURES_ROOT = Path(__file__).parent / "executor_failures"


def load_executor_failure_fixture(name: str) -> list[dict]:
    """Load `tests/fixtures/executor_failures/<name>.json` as a list of dicts."""
    path = _EXECUTOR_FAILURES_ROOT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["load_executor_failure_fixture"]
