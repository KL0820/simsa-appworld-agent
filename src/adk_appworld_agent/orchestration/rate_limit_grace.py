"""Rate-limit grace: keep transient 429 / provider backoff off the task budget.

A 429 (RESOURCE_EXHAUSTED) is infrastructure noise, not an agent failure, and
must not affect the agent's outcome or state. The retry layer already survives
429 bursts via exponential backoff (`subagents/retry.py`), but those backoff
sleeps run INSIDE the per-task `asyncio.timeout(...)` wall deadline — so a
sustained 429 cascade burns the wall budget and turns into a spurious
TASK_TIMEOUT (an agent "failure" the agent did nothing to earn, which then
contaminates the pass-rate measurement).

This module decouples "rate-limit wait time" from "agent compute budget". The
run harness installs an extender bound to the task's `asyncio.timeout` deadline;
the retry layer calls `grant_rate_limit_grace(slept_seconds)` around each
transient backoff sleep, pushing the deadline forward by exactly the time spent
waiting on the provider. Net effect: a 429 storm extends wall time but cannot
cause a timeout — the task is judged on its own compute, not on rate-limit luck.

The hook is a ContextVar so the worker layer stays decoupled from the harness:
no extender installed (unit tests, ad-hoc scripts) → `grant_rate_limit_grace`
is a safe no-op, so callers can invoke it unconditionally.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable

_extender: contextvars.ContextVar[Callable[[float], None] | None] = (
    contextvars.ContextVar("rate_limit_grace_extender", default=None)
)


def set_rate_limit_grace_extender(fn: Callable[[float], None] | None) -> None:
    """Install (or clear, with None) the deadline extender for the current task."""
    _extender.set(fn)


def grant_rate_limit_grace(seconds: float) -> None:
    """Extend the current task's wall deadline by `seconds` of rate-limit wait.

    No-op when no extender is installed or `seconds <= 0`. Best-effort: a fault
    in the extender must never break the run, so exceptions are swallowed.
    """
    if seconds <= 0:
        return
    fn = _extender.get()
    if fn is None:
        return
    try:
        fn(seconds)
    except Exception:
        pass


__all__ = ["set_rate_limit_grace_extender", "grant_rate_limit_grace"]
