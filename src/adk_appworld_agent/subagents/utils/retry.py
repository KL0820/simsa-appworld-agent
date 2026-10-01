from __future__ import annotations

import asyncio
import logging
import re
import sys
from typing import Any, Awaitable, Callable

from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.rate_limit_grace import grant_rate_limit_grace
from adk_appworld_agent.orchestration.run_config import RetryConfig

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

# The provider retry ladder lives in RetryConfig (the single config source). The
# functions below read the ACTIVE config at call time (active_config().retry.*),
# so an experiment config file changes them with no code edit. These module
# constants expose the schema default under the historical 429 name, kept for
# backward-compatible imports.
#
# 429 and retryable 5xx failures use the same waiting schedule. Their terminal
# contracts remain distinct: exhausted 429 raises RateLimitExhausted, while an
# exhausted non-rate-limit transient returns its provider error to the caller.
RATE_LIMIT_DELAYS: tuple[float, ...] = RetryConfig().provider_backoff_delays


class RateLimitExhausted(RuntimeError):
    """Raised when the unified 429 ladder is fully consumed.

    Signals 'this is the provider's quota problem, not the agent's' —
    callers must surface PROVIDER_RATE_LIMITED instead of a generic
    LLM_RAISED code so the task can be cleanly rerun after the batch.
    """


def _extract_status_code(message: str) -> int | None:
    match = re.search(r"\b(\d{3})\b", message)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_retryable(message: str) -> bool:
    """True for transient provider failures worth retrying (429, 5xx, RESOURCE_EXHAUSTED)."""
    upper = message.upper()
    status_code = _extract_status_code(message)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    return "RESOURCE_EXHAUSTED" in upper


def is_rate_limit(message: str) -> bool:
    """True for rate-limit / quota errors specifically (string fallback).

    Defensive: the bare-substring `"429" in message` was too broad — any
    message that happened to contain "429" (a token count, an id, a byte
    size) would be misclassified as a rate limit and funneled into the
    terminal 429 ladder. We now only treat 429 as a rate limit when it is
    the actual extracted HTTP status code; the specific quota status strings
    are matched directly. Prefer `is_rate_limit_error(exc)` (typed .code)
    over this string path whenever the exception object is available.
    """
    upper = message.upper()
    if "RESOURCE_EXHAUSTED" in upper or "RATE LIMIT" in upper or "RATE_LIMIT" in upper:
        return True
    return _extract_status_code(message) == 429


def is_rate_limit_error(exc: BaseException) -> bool:
    """Structural rate-limit check — prefer the provider exception's typed
    fields (`google.genai.errors.APIError` exposes `.code` int + `.status`
    str) and only fall back to the string heuristic for non-typed errors.
    This avoids both false positives (incidental "429" substrings) and false
    negatives (a real 429 whose message text doesn't spell it out)."""
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code == 429:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED":
        return True
    return is_rate_limit(str(exc))


def is_provider_error(message: str) -> bool:
    """True for any upstream provider-side error (rate limit + 5xx + service issues)."""
    upper = message.upper()
    return (
        is_rate_limit(message)
        or "UNAVAILABLE" in upper
        or "INTERNAL" in upper
        or "DEADLINE_EXCEEDED" in upper
        or "SERVICE UNAVAILABLE" in upper
        or "503" in upper
        or "502" in upper
        or "500" in upper
    )


async def retry_llm_call(
    fn: Callable[[], Awaitable[Any]],
    *,
    delays: tuple[float, ...] = (30, 30, 30),
    default: Any = None,
    logger: logging.Logger | None = None,
    label: str = "",
) -> Any:
    """Retry a simple async LLM call on any exception.

    Rate-limit errors (429 / RESOURCE_EXHAUSTED) use the shared provider
    ladder regardless of `delays`; the wait is granted as rate-limit grace
    so it cannot become a timeout. After the budget, raises
    RateLimitExhausted — the caller maps it to PROVIDER_RATE_LIMITED and
    the task is rerun later instead of degrading the trajectory.
    """
    attempt = 0
    rate_limit_attempt = 0
    provider_delays = active_config().retry.provider_backoff_delays
    while True:
        try:
            return await fn()
        except Exception as exc:
            if is_rate_limit(str(exc)):
                if rate_limit_attempt >= len(provider_delays):
                    msg = f"{label} rate-limit ladder exhausted: {exc}"
                    if logger:
                        logger.warning(msg)
                    else:
                        sys.stderr.write(msg + "\n")
                        sys.stderr.flush()
                    raise RateLimitExhausted(str(exc)) from exc
                delay = provider_delays[rate_limit_attempt]
                msg = (
                    f"{label} attempt {attempt + 1} rate-limited: {exc}, "
                    f"waiting {delay:.0f}s "
                    f"({rate_limit_attempt + 1}/{len(provider_delays)})"
                )
                if logger:
                    logger.warning(msg)
                else:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                grant_rate_limit_grace(delay)
                await asyncio.sleep(delay)
                attempt += 1
                rate_limit_attempt += 1
                continue
            if attempt < len(delays):
                delay = delays[attempt]
                msg = f"{label} attempt {attempt + 1} failed: {exc}, retrying in {delay:.0f}s"
                if logger:
                    logger.warning(msg)
                else:
                    sys.stderr.write(msg + "\n")
                    sys.stderr.flush()
                # Provider/rate-limit waits are infra, not agent compute — keep
                # them off the task wall budget so a 429 cascade can't time out.
                if is_provider_error(str(exc)):
                    grant_rate_limit_grace(delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue

            msg = f"{label} failed after all retries: {exc}"
            if logger:
                logger.warning(msg)
            else:
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()
            return default


async def retry_agent_stream(
    attempt_fn: Callable[[int], Awaitable[None]],
    *,
    delays: tuple[float, ...] | None = None,
    label: str = "",
) -> str | None:
    """Retry a stateful LlmAgent streaming loop with transient-aware policy.

    Only transient provider errors (429, 5xx, RESOURCE_EXHAUSTED) trigger
    exponential backoff retry. TimeoutError and other exceptions return
    immediately — retrying same prompt won't help.

    Rate-limit errors (429 / RESOURCE_EXHAUSTED) use the config provider ladder
    REGARDLESS of `delays` (same contract as retry_llm_call); after that, raises
    RateLimitExhausted (caller maps it to PROVIDER_RATE_LIMITED — rerun the task
    later instead of degrading the trajectory). `delays`, when given, only
    customizes the non-rate-limit transient schedule; other transients return
    their provider error after that ladder.
    """
    # 429 always rides the config provider ladder (4,8,16,32,64) so an artificial
    # or shortened `delays` can never weaken the rate-limit backoff; `delays`
    # customizes only the non-rate-limit transient path.
    rate_limit_delays = active_config().retry.provider_backoff_delays
    transient_delays = delays or rate_limit_delays
    attempt = 0
    while True:
        try:
            await attempt_fn(attempt)
            return None
        except asyncio.TimeoutError as exc:
            sys.stderr.write(
                f"[{label}] attempt {attempt + 1} timed out, not retrying: {exc}\n"
            )
            sys.stderr.flush()
            return str(exc)
        except Exception as exc:
            msg = str(exc)
            if is_rate_limit(msg):
                if attempt >= len(rate_limit_delays):
                    sys.stderr.write(f"[{label}] rate-limit ladder exhausted: {exc}\n")
                    sys.stderr.flush()
                    raise RateLimitExhausted(msg) from exc
                delay = rate_limit_delays[attempt]
                sys.stderr.write(
                    f"[{label}] attempt {attempt + 1} rate-limited ({exc}), "
                    f"waiting {delay:.0f}s "
                    f"({attempt + 1}/{len(rate_limit_delays)})\n"
                )
                sys.stderr.flush()
                grant_rate_limit_grace(delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            if attempt < len(transient_delays) and is_retryable(msg):
                delay = transient_delays[attempt]
                sys.stderr.write(
                    f"[{label}] attempt {attempt + 1} transient ({exc}), "
                    f"retrying in {delay:.0f}s\n"
                )
                sys.stderr.flush()
                # Transient provider backoff is infra wait, not agent compute —
                # extend the task deadline so it can't become a TASK_TIMEOUT.
                grant_rate_limit_grace(delay)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            # Non-retryable exception, or transient budget exhausted.
            sys.stderr.write(
                f"[{label}] attempt {attempt + 1} failed (terminal): {exc}\n"
            )
            sys.stderr.flush()
            return msg


__all__ = [
    "RATE_LIMIT_DELAYS",
    "RateLimitExhausted",
    "is_retryable",
    "is_rate_limit",
    "is_provider_error",
    "retry_llm_call",
    "retry_agent_stream",
]
