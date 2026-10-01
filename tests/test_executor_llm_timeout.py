"""Inner executor/planner LLM transport timeout — REVERTED (2026-06-23).

History: commit 3c74aed added http_options.timeout to the inner executor/planner
streams to bound a 28-min hang. It turned out to be the root cause of the 2-day
"finder hangs after EXECUTE" bug: one asyncio.run per task means PLAN/FIND/EXECUTE
share ONE event loop and a loop-bound genai aiohttp session. When the transport
timeout fired / a stream was abandoned, genai's only cleanup is a GC-timed,
never-awaited create_task(aclose()), which left a half-dead stream in the shared
loop's connection pool -> the NEXT finder request on that loop hung to its own
deadline (a TIMEOUT, not a 429). See project_finder_429_client_bug.

Fix: (1) offload the sync zerorpc sandbox call off the event loop (dedicated
single worker thread) so EXECUTE never blocks the loop and the per-task
asyncio.timeout (run_task.py) can actually fire; (2) REVERT the inner-agent
http_options.timeout so a stream is never abandoned mid-flight on the shared
session. These tests pin the revert.
"""

from __future__ import annotations

from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.run_config import ModelConfig, RunConfig
from adk_appworld_agent.subagents.executor.code_plan_execute.agent import (
    _build_inner_code_executor_agent,
    _build_inner_code_planner_agent,
    _llm_request_timeout_ms,
)


def test_request_timeout_ms_helper_still_computes():
    # The helper is retained (for an optional future explicit-teardown path) but
    # is NO LONGER wired into the inner agents.
    set_active_config(RunConfig(executor_timeout_s=240.0))
    try:
        assert _llm_request_timeout_ms() == 240_000
    finally:
        set_active_config(None)


def test_executor_agent_has_NO_transport_timeout():
    """REVERTED: a per-request transport deadline on the shared loop-bound genai
    session poisoned the next finder call. The inner executor agent must not set
    http_options.timeout."""
    mc = ModelConfig(name="gemini-2.5-flash")
    ex = _build_inner_code_executor_agent(mc)
    ho = ex.generate_content_config.http_options
    assert ho is None or not getattr(ho, "timeout", None)


def test_planner_agent_has_NO_transport_timeout():
    mc = ModelConfig(name="gemini-2.5-flash")
    pl = _build_inner_code_planner_agent(mc)
    ho = pl.generate_content_config.http_options
    assert ho is None or not getattr(ho, "timeout", None)
