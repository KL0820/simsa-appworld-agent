"""Unit tests for the per-attempt fresh-loop isolation helper.

The executor stall fix (RunConfig.executor_isolate_llm_loop) runs each Stage-2
LLM attempt on its OWN event loop in a short-lived thread, so a poisoned genai
connection (python-genai #1893 socket stall / #1709 unawaited aclose) dies with
that throwaway loop instead of cascading to the next attempt. These tests pin
the helper's contract: it returns the coro's result, runs it on a DIFFERENT
loop than the caller, and propagates exceptions.
"""

from __future__ import annotations

import asyncio

from adk_appworld_agent.subagents.executor.code_plan_execute.execution import (
    _run_attempt_isolated,
)


def test_isolated_returns_result():
    async def _coro():
        return "ok-value"

    async def _main():
        return await _run_attempt_isolated(lambda: _coro())

    assert asyncio.run(_main()) == "ok-value"


def test_isolated_runs_on_a_different_loop():
    """The attempt must execute on a fresh loop, not the caller's loop — that is
    the whole point (a poisoned connector is loop-bound and must not be shared).
    """

    async def _main():
        caller_loop = asyncio.get_running_loop()

        async def _coro():
            inner_loop = asyncio.get_running_loop()
            return inner_loop is not caller_loop

        return await _run_attempt_isolated(lambda: _coro())

    assert asyncio.run(_main()) is True


def test_isolated_propagates_exceptions():
    class _Boom(RuntimeError):
        pass

    async def _coro():
        raise _Boom("attempt blew up")

    async def _main():
        return await _run_attempt_isolated(lambda: _coro())

    try:
        asyncio.run(_main())
    except _Boom as exc:
        assert "attempt blew up" in str(exc)
    else:  # pragma: no cover - guard
        raise AssertionError("exception was not propagated from the worker loop")


def test_isolated_each_call_gets_a_fresh_loop():
    """Two sequential attempts must run on two DIFFERENT loops — otherwise the
    cascade (every retry reusing the poisoned loop) would not be broken.
    """

    seen: list[asyncio.AbstractEventLoop] = []

    async def _coro():
        seen.append(asyncio.get_running_loop())
        return None

    async def _main():
        await _run_attempt_isolated(lambda: _coro())
        await _run_attempt_isolated(lambda: _coro())

    asyncio.run(_main())
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert all(loop.is_closed() for loop in seen)
