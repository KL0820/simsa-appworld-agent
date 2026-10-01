from __future__ import annotations

import asyncio

from adk_appworld_agent.orchestration.rate_limit_grace import (
    grant_rate_limit_grace,
    set_rate_limit_grace_extender,
)
from adk_appworld_agent.subagents.utils.retry import retry_agent_stream, retry_llm_call


def test_grant_grace_noop_without_extender():
    set_rate_limit_grace_extender(None)
    grant_rate_limit_grace(10.0)  # must not raise


def test_grant_grace_calls_extender_and_ignores_nonpositive():
    seen: list[float] = []
    set_rate_limit_grace_extender(seen.append)
    try:
        grant_rate_limit_grace(5.0)
        grant_rate_limit_grace(0)
        grant_rate_limit_grace(-1)
    finally:
        set_rate_limit_grace_extender(None)
    assert seen == [5.0]


def test_retry_agent_stream_grants_grace_on_transient_backoff(monkeypatch):
    async def _fake_sleep(_d: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    granted: list[float] = []
    set_rate_limit_grace_extender(granted.append)
    calls = {"n": 0}

    async def attempt(_i: int) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return None

    try:
        err = asyncio.run(retry_agent_stream(attempt, delays=(0.01,), label="t"))
    finally:
        set_rate_limit_grace_extender(None)
    assert err is None  # recovered on attempt 2
    # rate-limit errors use the dedicated ladder (4s first), granted as grace
    assert granted == [4]


def test_retry_llm_call_grants_grace_only_for_provider_errors(monkeypatch):
    async def _fake_sleep(_d: float) -> None:
        pass

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    granted: list[float] = []
    set_rate_limit_grace_extender(granted.append)
    try:
        # Non-provider error is retried but is NOT rate-limit grace.
        n = {"i": 0}

        async def fn_nonprovider():
            n["i"] += 1
            if n["i"] == 1:
                raise ValueError("malformed json")
            return "ok"

        assert (
            asyncio.run(retry_llm_call(fn_nonprovider, delays=(0.01,), label="t"))
            == "ok"
        )
        assert granted == []

        # A 429 backoff IS granted as grace.
        m = {"i": 0}

        async def fn_provider():
            m["i"] += 1
            if m["i"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return "ok2"

        assert (
            asyncio.run(retry_llm_call(fn_provider, delays=(0.02,), label="t")) == "ok2"
        )
        # 429 takes the dedicated rate-limit ladder (4s first), as grace
        assert granted == [4]
    finally:
        set_rate_limit_grace_extender(None)
