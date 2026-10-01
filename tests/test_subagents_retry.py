"""Tests for subagents.utils.retry: transient-aware retry policy for LLM streams."""

from __future__ import annotations

import asyncio

import pytest

from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
    is_provider_error,
    is_rate_limit,
    is_retryable,
    retry_agent_stream,
    retry_llm_call,
)

# ── Predicate matrix ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("429 RESOURCE_EXHAUSTED", True),
        ("429 quota exceeded", True),
        ("503 SERVICE UNAVAILABLE", True),
        ("500 INTERNAL", True),
        ("502 Bad Gateway", True),
        ("504 Gateway Timeout", True),
        ("408 Request Timeout", True),
        ("RESOURCE_EXHAUSTED quota", True),
        ("400 INVALID_ARGUMENT", False),
        ("401 UNAUTHENTICATED", False),
        ("404 not found", False),
        ("malformed JSON response", False),
        ("", False),
    ],
)
def test_is_retryable(msg: str, expected: bool) -> None:
    assert is_retryable(msg) is expected


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("429 RESOURCE_EXHAUSTED", True),
        ("RATE LIMIT exceeded", True),
        ("RATE_LIMIT triggered", True),
        ("503 SERVICE UNAVAILABLE", False),
        ("400 INVALID", False),
    ],
)
def test_is_rate_limit(msg: str, expected: bool) -> None:
    assert is_rate_limit(msg) is expected


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("429 RESOURCE_EXHAUSTED", True),
        ("503 UNAVAILABLE", True),
        ("500 INTERNAL", True),
        ("502 Bad Gateway", True),
        ("DEADLINE_EXCEEDED", True),
        ("400 INVALID_ARGUMENT", False),
        ("404 not found", False),
    ],
)
def test_is_provider_error(msg: str, expected: bool) -> None:
    assert is_provider_error(msg) is expected


# ── retry_agent_stream behavior ───────────────────────────────────────────────


def test_retry_agent_stream_success_on_first_attempt(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts.append(attempt)

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result is None
    assert attempts == [0]
    assert sleeps == []


def test_retry_agent_stream_transient_then_success(monkeypatch) -> None:
    """Retryable error (429) backs off with exp delays, then succeeds."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    raises_then_succeeds = [
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        RuntimeError("503 UNAVAILABLE"),
        None,
    ]
    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        outcome = raises_then_succeeds[attempt]
        if outcome is not None:
            raise outcome

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result is None
    assert attempts_seen == [0, 1, 2]
    # Both classes use the shared provider ladder. The global attempt index
    # advances across the mixed 429 -> 503 sequence.
    assert sleeps == [4, 8]


def test_retry_agent_stream_transient_budget_exhausted(monkeypatch) -> None:
    """Non-rate-limit transients (5xx) still exhaust the finite ladder.

    Rate-limit errors are exempt (see the never-give-up test below); only
    genuine provider faults like 503 use the same finite provider budget."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        raise RuntimeError("503 Service Unavailable")

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result is not None
    assert "503" in result
    assert attempts_seen == [0, 1, 2, 3, 4, 5]  # 6 attempts = 1 initial + 5 retries
    assert sleeps == [4, 8, 16, 32, 64]


def test_retry_agent_stream_rate_limit_recovers_within_ladder(monkeypatch) -> None:
    """429 / RESOURCE_EXHAUSTED that clears within the capped 5-attempt ladder
    succeeds — quota interference within the budget must not surface as a
    component failure."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    failures = 3  # within the 5-attempt rate-limit ladder
    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        if attempt < failures:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result is None
    assert len(attempts_seen) == failures + 1
    assert sleeps == [4, 8, 16]


def test_retry_agent_stream_rate_limit_budget_exhausts(monkeypatch) -> None:
    """A quota outage longer than the capped ~2 min budget fails terminally —
    the task is rerun later instead of waiting forever."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RateLimitExhausted, match="429"):
        asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert len(attempts_seen) == 6  # 1 initial + 5 rate-limit retries
    assert sleeps == [4, 8, 16, 32, 64]


def test_retry_llm_call_rate_limit_recovers_within_ladder(monkeypatch) -> None:
    """retry_llm_call: 429 gets the capped rate-limit ladder regardless of `delays`."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    calls = {"n": 0}

    async def _fn():
        calls["n"] += 1
        if calls["n"] <= 4:  # within the 5-attempt rate-limit ladder
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "ok"

    result = asyncio.run(
        retry_llm_call(_fn, delays=(30, 30, 30), default=None, label="t")
    )

    assert result == "ok"
    assert calls["n"] == 5
    assert sleeps == [4, 8, 16, 32]


def test_retry_llm_call_rate_limit_budget_exhausts(monkeypatch) -> None:
    """retry_llm_call: sustained 429 raises after the budget (unified
    contract: never degrade into `default`; the run must abort with
    PROVIDER_RATE_LIMITED and be rerun later)."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    calls = {"n": 0}

    async def _fn():
        calls["n"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    with pytest.raises(RateLimitExhausted, match="429"):
        asyncio.run(
            retry_llm_call(_fn, delays=(30, 30, 30), default="DEGRADED", label="t")
        )

    # 1 initial + 5 rate-limit retries, then the ladder is exhausted
    assert calls["n"] == 6


def test_retry_agent_stream_timeout_short_circuits(monkeypatch) -> None:
    """asyncio.TimeoutError must NOT retry — same prompt → same timeout."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        raise asyncio.TimeoutError("attempt timed out")

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result is not None
    assert attempts_seen == [0]  # no retries
    assert sleeps == []


def test_retry_agent_stream_non_retryable_short_circuits(monkeypatch) -> None:
    """Non-transient exception (400, parse error, etc.) must NOT retry."""
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        raise ValueError("400 INVALID_ARGUMENT")

    result = asyncio.run(retry_agent_stream(_attempt, label="t"))

    assert result == "400 INVALID_ARGUMENT"
    assert attempts_seen == [0]
    assert sleeps == []


def test_retry_agent_stream_custom_delays(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    attempts_seen: list[int] = []

    async def _attempt(attempt: int) -> None:
        attempts_seen.append(attempt)
        if attempt < 2:
            raise RuntimeError("503 UNAVAILABLE")

    result = asyncio.run(retry_agent_stream(_attempt, delays=(1, 2, 4), label="t"))

    assert result is None
    assert attempts_seen == [0, 1, 2]
    assert sleeps == [1, 2]
