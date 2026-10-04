"""Extract only non-sensitive, recorded run counters for the local viewer."""

from __future__ import annotations

import math


COUNT_FIELDS = (
    "llm_calls",
    "llm_call_attempts",
    "prompt_tokens",
    "completion_tokens",
    "thoughts_tokens",
    "total_tokens",
)


def _count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _duration(value: object) -> float | None:
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        return None
    return round(float(value), 3)


def extract_run_metrics(summary: dict) -> dict | None:
    """Use task wall time and recorded subagent aggregates, not billing data."""
    aggregate = summary.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        aggregate = {}
    metrics = {"duration_s": _duration(summary.get("wall_s"))}
    metrics.update({field: _count(aggregate.get(field)) for field in COUNT_FIELDS})
    return metrics if any(value is not None for value in metrics.values()) else None
