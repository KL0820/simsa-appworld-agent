"""Unified 429 policy tests.

Contract under test (see retry.RATE_LIMIT_DELAYS):
  1. Every call site shares ONE rate-limit ladder — 4s first delay,
     capped at 64s, 5 attempts (~2 min budget).
  2. Ladder exhaustion in the SHARED retry helpers (retry_agent_stream /
     retry_llm_call — executor/planner/continuation) raises RateLimitExhausted.
     EXCEPTION (regression revert 2026-06-22): the FINDER degrades to an empty
     result + a typed `rate_limited` marker instead of raising — an empty recall
     result is survivable (controller re-FINDs next cycle), whereas raising
     turned every transient finder 429 into a terminal PROVIDER_RATE_LIMITED FAIL.
  3. Subagents map a raised exception to PROVIDER_RATE_LIMITED.
  4. The FSM aborts the run on PROVIDER_RATE_LIMITED instead of looping
     into continuation.
"""

from __future__ import annotations

import asyncio

import pytest

from adk_appworld_agent.subagents.failure_codes import PROVIDER_RATE_LIMITED
from adk_appworld_agent.subagents.utils import retry as retry_mod
from adk_appworld_agent.subagents.utils.retry import (
    RATE_LIMIT_DELAYS,
    RateLimitExhausted,
    retry_agent_stream,
    retry_llm_call,
)


@pytest.fixture(autouse=True)
def _reset_active_config():
    """Ladders now come from the active RunConfig; restore defaults after any
    test that installs a shrunk ladder, so config never leaks across tests."""
    yield
    from adk_appworld_agent.orchestration.active_config import set_active_config

    set_active_config(None)


class _Raise429:
    def __init__(self) -> None:
        self.calls = 0

    def exc(self) -> Exception:
        self.calls += 1
        return RuntimeError(
            "429 RESOURCE_EXHAUSTED. Resource exhausted. Please try again later."
        )


@pytest.fixture
def fast_ladder(monkeypatch):
    """Shrink the ladder and zero the sleeps so exhaustion is testable."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig

    set_active_config(
        RunConfig(retry=RetryConfig(provider_backoff_delays=(16.0, 32.0)))
    )
    sleeps: list[float] = []

    async def _no_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(retry_mod.asyncio, "sleep", _no_sleep)
    granted: list[float] = []
    monkeypatch.setattr(retry_mod, "grant_rate_limit_grace", granted.append)
    return sleeps, granted


def test_ladder_starts_at_4_capped_at_64_with_5_attempts():
    assert RATE_LIMIT_DELAYS[0] == 4
    assert max(RATE_LIMIT_DELAYS) == 64
    assert len(RATE_LIMIT_DELAYS) == 5


def test_retry_agent_stream_raises_on_ladder_exhaustion(fast_ladder):
    sleeps, granted = fast_ladder
    source = _Raise429()

    async def _attempt(_attempt_idx: int) -> None:
        raise source.exc()

    with pytest.raises(RateLimitExhausted):
        asyncio.run(retry_agent_stream(_attempt, label="t"))
    # 2 ladder waits + the final attempt that raises
    assert source.calls == 3
    assert sleeps == [16.0, 32.0]
    # every wait was granted as grace (not burned from the wall budget)
    assert granted == [16.0, 32.0]


def test_retry_agent_stream_recovers_within_ladder(fast_ladder):
    sleeps, _ = fast_ladder
    source = _Raise429()

    async def _attempt(_attempt_idx: int) -> None:
        if source.calls < 1:
            raise source.exc()

    assert asyncio.run(retry_agent_stream(_attempt, label="t")) is None
    assert sleeps == [16.0]


def test_retry_llm_call_raises_on_ladder_exhaustion(fast_ladder):
    source = _Raise429()

    async def _fn():
        raise source.exc()

    with pytest.raises(RateLimitExhausted):
        asyncio.run(retry_llm_call(_fn, label="t", default="sentinel"))


def test_retry_agent_stream_non_rate_limit_still_returns_message(fast_ladder):
    async def _attempt(_attempt_idx: int) -> None:
        raise RuntimeError("400 INVALID_ARGUMENT bad request")

    out = asyncio.run(retry_agent_stream(_attempt, label="t"))
    assert out is not None and "400" in out


def test_finder_call_llm_json_degrades_to_empty_on_exhaustion(monkeypatch):
    """FINDER EXCEPTION to the raise-on-exhaustion policy (regression revert
    2026-06-22): the finder is the recall layer where an empty result is
    survivable — the controller re-FINDs next cycle and recovers when the
    provider eases. So the finder DEGRADES a 429-exhausted call to an empty
    result + a typed `rate_limited` marker instead of raising RateLimitExhausted
    (which turned every transient 429 into a terminal PROVIDER_RATE_LIMITED FAIL:
    0 such failures across 324 pre-change tasks vs 0.90/task on the raise branch).
    The shared retry.py helpers (executor/planner/continuation) still raise."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig
    from adk_appworld_agent.subagents.finder import llm as finder_llm

    set_active_config(RunConfig(retry=RetryConfig(provider_backoff_delays=(16.0,))))

    async def _no_sleep(delay):
        return None

    monkeypatch.setattr(finder_llm.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(finder_llm, "grant_rate_limit_grace", lambda d: None)

    class _FakeModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    class _FakeAio:
        models = _FakeModels()

    class _FakeClient:
        aio = _FakeAio()

    monkeypatch.setattr(finder_llm, "_client", lambda: _FakeClient())

    result = asyncio.run(finder_llm.call_llm_json("user prompt", "sys prompt"))
    # degrades to empty, does NOT raise
    assert result["parsed_json"] == {}
    assert result["rate_limited"] is True
    # still observable: the rate-limit was counted, not silently absorbed
    assert int(result["retry"]["rate_limit_count"]) > 0


def _wire_429_finder(monkeypatch):
    """Wire the finder llm to always 429, with zeroed sleeps. Caller sets the
    active config (provider_backoff_delays + flags) before calling call_llm_json."""
    from adk_appworld_agent.subagents.finder import llm as finder_llm

    async def _no_sleep(delay):
        return None

    monkeypatch.setattr(finder_llm.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(finder_llm, "grant_rate_limit_grace", lambda d: None)

    class _FakeModels:
        async def generate_content(self, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    class _FakeAio:
        models = _FakeModels()

    class _FakeClient:
        aio = _FakeAio()

    monkeypatch.setattr(finder_llm, "_client", lambda: _FakeClient())
    return finder_llm


def test_429_probe_disabled_by_default(monkeypatch):
    """The 3-way localization probe must NOT fire unless finder_429_probe_enabled."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig

    # fresh config: finder_429_probe_enabled defaults False
    set_active_config(RunConfig(retry=RetryConfig(provider_backoff_delays=(16.0,))))
    finder_llm = _wire_429_finder(monkeypatch)
    calls = []

    async def _spy(*a, **k):
        calls.append(1)

    monkeypatch.setattr(finder_llm, "_probe_429_three_way", _spy)
    asyncio.run(finder_llm.call_llm_json("u", "s"))
    assert calls == []


def test_429_probe_fires_once_on_first_429_and_never_raises(monkeypatch):
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig

    # two ladder rungs => two 429-classified attempts before degrade; probe ON
    set_active_config(
        RunConfig(
            retry=RetryConfig(provider_backoff_delays=(16.0, 32.0)),
            finder_429_probe_enabled=True,
        )
    )
    finder_llm = _wire_429_finder(monkeypatch)
    calls = []

    async def _spy(*a, **k):
        calls.append(1)
        raise RuntimeError("probe blew up — must be swallowed by the caller path")

    monkeypatch.setattr(finder_llm, "_probe_429_three_way", _spy)
    result = asyncio.run(finder_llm.call_llm_json("u", "s"))
    # fired exactly once (on the first 429, rate_limit_attempt==0), not per rung
    assert calls == [1]
    # a probe that raises must not corrupt the degrade result
    assert result["parsed_json"] == {} and result["rate_limited"] is True


def test_finder_swaps_to_fresh_pool_on_429_retry(monkeypatch):
    """finder_fresh_pool_on_429: the shared lru-cached client 429s, but the retry
    swaps to a FRESH genai.Client (fresh httpx pool) — which here succeeds, so the
    finder returns a real result instead of degrading. Proves we stop re-sending
    into the already-throttled pool."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig
    from adk_appworld_agent.subagents.finder import llm as finder_llm

    set_active_config(
        RunConfig(
            retry=RetryConfig(provider_backoff_delays=(16.0,)),
            finder_fresh_pool_on_429=True,
        )
    )

    async def _no_sleep(delay):
        return None

    monkeypatch.setattr(finder_llm.asyncio, "sleep", _no_sleep)
    monkeypatch.setattr(finder_llm, "grant_rate_limit_grace", lambda d: None)

    class _Models429:
        async def generate_content(self, **kw):
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")

    class _Client429:
        class aio:
            models = _Models429()

    monkeypatch.setattr(finder_llm, "_client", lambda: _Client429())

    class _R:
        text = '{"ok": true}'
        parsed = None
        usage_metadata = None

    class _ModelsOK:
        async def generate_content(self, **kw):
            return _R()

    class _ClientOK:
        class aio:
            models = _ModelsOK()

    built: list = []

    def _fake_fresh_client(**kw):
        built.append(kw)
        return _ClientOK()

    monkeypatch.setattr(finder_llm.genai, "Client", _fake_fresh_client)

    result = asyncio.run(finder_llm.call_llm_json("u", "s"))
    # a fresh pool was built on the 429 retry, and the retry on it succeeded
    assert built, "expected a fresh genai.Client to be built on the 429 retry"
    assert result["parsed_json"] == {"ok": True}
    assert "rate_limited" not in result  # did NOT degrade — recovered on fresh pool


def test_finder_fresh_pool_off_keeps_same_client(monkeypatch):
    """With finder_fresh_pool_on_429=False, no fresh client is built — the ladder
    retries on the shared pool and degrades (old behavior, ablation control)."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetryConfig, RunConfig

    set_active_config(
        RunConfig(
            retry=RetryConfig(provider_backoff_delays=(16.0,)),
            finder_fresh_pool_on_429=False,
        )
    )
    finder_llm = _wire_429_finder(monkeypatch)
    built: list = []
    monkeypatch.setattr(finder_llm.genai, "Client", lambda **kw: built.append(kw))
    result = asyncio.run(finder_llm.call_llm_json("u", "s"))
    assert built == []  # no pool swap
    assert result["rate_limited"] is True  # degraded


def test_fsm_aborts_on_provider_rate_limited_execute():
    """EXECUTE failing with PROVIDER_RATE_LIMITED must end the run
    immediately (status FAILED, block reason set, no continuation retry,
    forced null submission emitted)."""
    from typing import AsyncIterator, ClassVar

    from adk_appworld_agent.agent import AppWorldAgent
    from adk_appworld_agent.contracts.subagent_input import SubagentInput
    from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
    from adk_appworld_agent.orchestration.orchestrator import Orchestrator
    from adk_appworld_agent.orchestration.state import Phase, RunStatus
    from adk_appworld_agent.orchestration.state_repo import RunStateRepository
    from adk_appworld_agent.orchestration.subagent_output_store import (
        SubagentOutputStore,
    )
    from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
    from adk_appworld_agent.subagents.base import BaseSubagent
    from adk_appworld_agent.subagents.continuer.continuation import (
        build_continuation_stub_subagent,
    )
    from adk_appworld_agent.subagents.stubs import build_stub_subagent
    from tests.helpers import run_agent
    from tests.test_cycle_loop import _FakeSubmitter, _MultiMilestonePlannerStub

    class _RateLimitedExecutorStub(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.EXECUTE
        calls: ClassVar[list[int]] = []

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            type(self).calls.append(subagent_input.attempt)
            yield self.failed(
                attempt=subagent_input.attempt,
                payload={"llm_raised": "429 RESOURCE_EXHAUSTED (ladder exhausted)"},
                failure_code=PROVIDER_RATE_LIMITED,
            )

    _RateLimitedExecutorStub.calls = []
    planner = _MultiMilestonePlannerStub(name="mm")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = _RateLimitedExecutorStub(name="rate_limited_executor")
    continuation = build_continuation_stub_subagent()
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="rate-limit abort test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=5,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.FAILED
    assert state.completion_block_reason == PROVIDER_RATE_LIMITED
    # No continuation-driven second attempt: the run aborted on the spot.
    assert len(_RateLimitedExecutorStub.calls) == 1
    # Forced null submission ran so the evaluator records the task.
    submit = store.read(session.state, Phase.SUBMIT)
    assert submit is not None
    assert (submit.payload or {}).get("forced_fallback") is True


def test_provider_rate_limited_is_exported():
    assert PROVIDER_RATE_LIMITED == "PROVIDER_RATE_LIMITED"
