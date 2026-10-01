from __future__ import annotations

import asyncio

import pytest

from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig
from adk_appworld_agent.subagents.finder import llm


@pytest.fixture(autouse=True)
def _reset_active_config():
    yield
    set_active_config(None)


class _FakeResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = None


class _FakeModels:
    def __init__(self, outcomes: list[object]):
        self._outcomes = outcomes
        self.calls = 0

    async def generate_content(self, **_: object):
        out = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(out, Exception):
            raise out
        return out


class _FakeClient:
    def __init__(self, outcomes: list[object]):
        self.aio = type("_Aio", (), {"models": _FakeModels(outcomes)})()


def test_call_llm_json_429_uses_unified_ladder_then_transient_backoff(monkeypatch):
    """429 takes the unified RATE_LIMIT_DELAYS ladder (4s first, with
    grace); a later 503 still uses the finder's own exponential backoff."""
    outcomes = [
        Exception("429 RESOURCE_EXHAUSTED"),
        Exception("503 SERVICE UNAVAILABLE"),
        _FakeResponse('{"selected": ["venmo_c3"]}'),
    ]
    fake_client = _FakeClient(outcomes)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    # finder transient backoff with zero jitter so the 503 delay is exact.
    set_active_config(RunConfig(retry=RetryConfig(finder_jitter_ratio=0.0)))
    monkeypatch.setattr(llm, "_client", lambda: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)

    granted: list[float] = []
    monkeypatch.setattr(llm, "grant_rate_limit_grace", granted.append)

    result = asyncio.run(llm.call_llm_json("u", "s"))

    assert result["parsed_json"] == {"selected": ["venmo_c3"]}
    # 429 → unified ladder first rung (4s, granted as grace);
    # 503 → finder transient backoff at attempt index 1 (2s, no grace).
    assert sleep_calls == [4.0, 2.0]
    assert granted == [4.0]
    assert result["retry"]["retry_count"] == 2
    assert result["retry"]["retry_backoff_ms"] == 6000
    assert result["retry"]["rate_limit_count"] == 1
    assert result["retry"]["provider_error_count"] == 2


def test_call_llm_json_does_not_retry_non_retryable_error(monkeypatch):
    outcomes = [Exception("400 INVALID_ARGUMENT")]
    fake_client = _FakeClient(outcomes)
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float):
        sleep_calls.append(delay)

    monkeypatch.setenv("FINDER_RETRY_MAX_RETRIES", "5")
    monkeypatch.setenv("FINDER_RETRY_JITTER_RATIO", "0")
    monkeypatch.setattr(llm, "_client", lambda: fake_client)
    monkeypatch.setattr(llm.asyncio, "sleep", _fake_sleep)

    result = asyncio.run(llm.call_llm_json("u", "s"))

    assert result["parsed_json"] == {}
    assert sleep_calls == []
    assert result["retry"]["retry_count"] == 0
    assert result["retry"]["retry_backoff_ms"] == 0
    assert result["retry"]["rate_limit_count"] == 0
    assert result["retry"]["provider_error_count"] == 0
