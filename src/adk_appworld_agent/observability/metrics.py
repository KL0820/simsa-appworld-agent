from __future__ import annotations

from typing import Any

METRIC_KEYS: tuple[str, ...] = (
    "wall_ms",
    "subagent_calls",
    "envelope_received",
    "llm_calls",
    "llm_call_attempts",
    "usage_event_count",
    "timeout_count",
    "prompt_tokens",
    "completion_tokens",
    "thoughts_tokens",
    "total_tokens",
    "retry_count",
    "retry_backoff_ms",
    "rate_limit_count",
    "provider_error_count",
)


def normalize_metrics(
    metrics: dict[str, Any] | None, *, fallback_wall_ms: int = 0
) -> dict[str, int]:
    raw = metrics or {}
    normalized = {key: int(raw.get(key) or 0) for key in METRIC_KEYS}
    if normalized["wall_ms"] == 0 and fallback_wall_ms:
        normalized["wall_ms"] = int(fallback_wall_ms)
    if normalized["llm_call_attempts"] == 0 and normalized["llm_calls"]:
        normalized["llm_call_attempts"] = normalized["llm_calls"]
    if normalized["total_tokens"] == 0:
        normalized["total_tokens"] = (
            normalized["prompt_tokens"]
            + normalized["completion_tokens"]
            + normalized["thoughts_tokens"]
        )
    return normalized


def zero_metrics(wall_ms: int = 0) -> dict[str, int]:
    return normalize_metrics({}, fallback_wall_ms=wall_ms)


def metrics_with_seconds(metrics: dict[str, Any] | None) -> dict[str, int | float]:
    raw = metrics or {}
    normalized = normalize_metrics(raw)
    wall_s = raw.get("wall_s")
    return {
        "wall_s": wall_s
        if isinstance(wall_s, int | float)
        else round(normalized["wall_ms"] / 1000.0, 3),
        "llm_calls": normalized["llm_calls"],
        "llm_call_attempts": normalized["llm_call_attempts"],
        "usage_event_count": normalized["usage_event_count"],
        "timeout_count": normalized["timeout_count"],
        "prompt_tokens": normalized["prompt_tokens"],
        "completion_tokens": normalized["completion_tokens"],
        "thoughts_tokens": normalized["thoughts_tokens"],
        "total_tokens": normalized["total_tokens"],
        "retry_count": normalized["retry_count"],
    }


def format_metrics(metrics: dict[str, Any] | None) -> str:
    normalized = normalize_metrics(metrics)
    return (
        f"wall_ms={normalized['wall_ms']} "
        f"subagent_calls={normalized['subagent_calls']} "
        f"envelope_received={normalized['envelope_received']} "
        f"llm_calls(observed/attempted)="
        f"{normalized['llm_calls']}/{normalized['llm_call_attempts']} "
        f"timeouts={normalized['timeout_count']} "
        f"tokens(prompt={normalized['prompt_tokens']}, "
        f"completion={normalized['completion_tokens']}, "
        f"thoughts={normalized['thoughts_tokens']}, "
        f"total={normalized['total_tokens']})"
    )


def metrics_from_subagent_item(item: dict[str, Any]) -> dict[str, int]:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    payload_metrics = (
        payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    )
    item_metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    return normalize_metrics(item_metrics or payload_metrics)


def aggregate_subagent_metrics(
    subagent_outputs: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    totals = normalize_metrics({})
    for items in subagent_outputs.values():
        for item in items or []:
            if not isinstance(item, dict):
                continue
            metrics = metrics_from_subagent_item(item)
            for key in METRIC_KEYS:
                totals[key] += metrics.get(key, 0)
    return totals


def compact_number(value: int | float | None) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and not value.is_integer():
        return f"{value:,.1f}"
    return f"{int(value):,}"


def compact_tokens(value: int | float | None) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if numeric >= 1_000_000:
        return f"{numeric / 1_000_000:.1f}M"
    if numeric >= 10_000:
        return f"{numeric / 1000:.1f}k"
    return compact_number(int(numeric))
