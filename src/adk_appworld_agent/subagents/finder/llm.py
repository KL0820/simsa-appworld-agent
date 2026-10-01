from __future__ import annotations

import asyncio
import json
import logging
import random
from functools import lru_cache
from typing import Any

from google import genai
from google.genai import types

from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.rate_limit_grace import grant_rate_limit_grace
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    model_config_from_env,
)
from adk_appworld_agent.subagents.utils.retry import (
    is_provider_error,
    is_rate_limit,
    is_retryable,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> genai.Client:
    timeout_ms = int(active_config().retry.llm_call_timeout_s * 1000)
    return genai.Client(http_options={"timeout": timeout_ms})


def _build_config(
    sys_prompt: str, model_cfg: ModelConfig
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=sys_prompt,
        response_mime_type="application/json",
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=model_cfg.candidate_count,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
        thinking_config=thinking_config_from_env(),
    )


async def _probe_429_three_way(
    user_prompt: str, sys_prompt: str, model_cfg: ModelConfig
) -> None:
    """DIAGNOSTIC (gated by finder_429_probe_enabled): the instant the finder
    hits a 429, fire the BYTE-IDENTICAL request two ways to localize the
    throttle. Every arm is best-effort (exceptions swallowed); this NEVER
    affects the caller's result/metrics. Read the log lines '[429-probe] ...':
      arm A (same lru_cached client) / arm B (fresh in-process client).
      A=B 429 → not client-object state (account/project level); A 429, B ok →
      per-client-object state. (The former arm C = fresh subprocess was removed
      2026-06-25: per-process isolation was DISPROVEN — a fresh subprocess per
      finder call still 429'd consecutively on d194965.)
    """
    contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
    config = _build_config(sys_prompt, model_cfg)
    timeout_ms = int(active_config().retry.llm_call_timeout_s * 1000)

    async def _arm_same_client() -> None:
        await _client().aio.models.generate_content(
            model=model_cfg.name, contents=contents, config=config
        )

    async def _arm_fresh_client() -> None:
        fresh = genai.Client(http_options={"timeout": timeout_ms})
        await fresh.aio.models.generate_content(
            model=model_cfg.name, contents=contents, config=config
        )

    arms = (
        ("A_same_client", _arm_same_client),
        ("B_fresh_client", _arm_fresh_client),
    )
    for arm_name, arm in arms:
        t0 = asyncio.get_event_loop().time()
        try:
            await arm()
            outcome = "OK_200"
            detail = ""
        except Exception as exc:  # noqa: BLE001 — diagnostic, swallow everything
            msg = _error_text(exc)
            outcome = "RATE_LIMITED_429" if is_rate_limit(msg) else "OTHER_ERROR"
            detail = msg[:160]
        latency_ms = int((asyncio.get_event_loop().time() - t0) * 1000)
        logger.warning(
            "[429-probe] arm=%s outcome=%s latency_ms=%d %s",
            arm_name,
            outcome,
            latency_ms,
            detail,
        )


def _usage_from_response(response: object) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {
            "usage_event_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thoughts_tokens": 0,
            "total_tokens": 0,
        }
    return {
        "usage_event_count": 1,
        "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "thoughts_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }


def _empty_retry_metrics() -> dict[str, int | str | None]:
    return {
        "retry_count": 0,
        "retry_backoff_ms": 0,
        "rate_limit_count": 0,
        "provider_error_count": 0,
        "last_error": None,
    }


def _error_text(exc: Exception) -> str:
    return str(exc).strip()


def _record_retry(
    metrics: dict[str, int | str | None], exc: Exception, *, delay_s: float
) -> None:
    message = _error_text(exc)
    metrics["retry_count"] = int(metrics["retry_count"] or 0) + 1
    metrics["retry_backoff_ms"] = int(metrics["retry_backoff_ms"] or 0) + int(
        delay_s * 1000
    )
    if is_rate_limit(message):
        metrics["rate_limit_count"] = int(metrics["rate_limit_count"] or 0) + 1
    if is_provider_error(message):
        metrics["provider_error_count"] = int(metrics["provider_error_count"] or 0) + 1
    metrics["last_error"] = message


def _record_terminal_error(
    metrics: dict[str, int | str | None], exc: Exception
) -> None:
    message = _error_text(exc)
    if is_rate_limit(message):
        metrics["rate_limit_count"] = int(metrics["rate_limit_count"] or 0) + 1
    if is_provider_error(message):
        metrics["provider_error_count"] = int(metrics["provider_error_count"] or 0) + 1
    metrics["last_error"] = message


def _retry_delay_s(retry_index: int) -> float:
    rc = active_config().retry
    base_delay_s = rc.finder_base_delay_s
    max_delay_s = rc.finder_max_delay_s
    jitter_ratio = rc.finder_jitter_ratio

    exp_delay_s = min(max_delay_s, base_delay_s * (2**retry_index))
    jitter_s = random.uniform(0.0, max(0.0, exp_delay_s * jitter_ratio))
    return exp_delay_s + jitter_s


async def call_llm_json(
    user_prompt: str,
    sys_prompt: str,
    *,
    model_cfg: ModelConfig | None = None,
) -> dict[str, Any]:
    logger.info("llm call start (prompt_len=%d)", len(user_prompt))
    resolved_model_cfg = model_cfg or model_config_from_env()
    retry_metrics = _empty_retry_metrics()
    max_retries = max(0, active_config().retry.finder_max_retries)
    provider_delays = active_config().retry.provider_backoff_delays
    timeout_ms = int(active_config().retry.llm_call_timeout_s * 1000)
    # Start on the shared lru-cached client; a 429 retry may swap in a FRESH
    # client (fresh httpx pool) so we don't re-send into the already-throttled
    # pool (finder_fresh_pool_on_429). `None` means "use the shared client".
    pool: dict[str, Any] = {"client": None}

    async def _call_once() -> dict[str, Any]:
        client = pool["client"] or _client()
        response = await client.aio.models.generate_content(
            model=resolved_model_cfg.name,
            contents=[{"role": "user", "parts": [{"text": user_prompt}]}],
            config=_build_config(sys_prompt, resolved_model_cfg),
        )
        logger.info("llm call done")
        usage = _usage_from_response(response)
        text = getattr(response, "text", None) or ""
        if not text and getattr(response, "parsed", None) is not None:
            parsed = response.parsed
            return {
                "parsed_json": parsed if isinstance(parsed, dict) else {},
                "usage": usage,
            }
        parsed = json.loads(text) if text else {}
        return {
            "parsed_json": parsed if isinstance(parsed, dict) else {},
            "usage": usage,
        }

    attempt = 0
    rate_limit_attempt = 0
    while True:
        try:
            result = await _call_once()
            result["retry"] = dict(retry_metrics)
            return result
        except Exception as exc:
            message = _error_text(exc)
            # 429 policy: use the shared provider ladder with grace-extended
            # sleeps, and on exhaustion
            # DEGRADE to an empty result + a typed `rate_limited` marker (see the
            # exhaustion branch below). The marker keeps the event observable
            # (retry_metrics.rate_limit_count) while letting the controller
            # re-FIND next cycle instead of terminally failing the task.
            if is_rate_limit(message):
                if rate_limit_attempt >= len(provider_delays):
                    # REGRESSION REVERT (2026-06-22): the old finder (main:
                    # finding/llm.py) degraded a rate-limit-exhausted call to an
                    # EMPTY result and let the task continue (controller re-FINDs
                    # next cycle, recovers when the provider eases). The interim
                    # `raise RateLimitExhausted` turned every transient 429 into
                    # a TERMINAL PROVIDER_RATE_LIMITED task FAIL — 0 such failures
                    # across 324 pre-change tasks vs 0.90/task on the raise branch.
                    # Degrade-to-empty + a TYPED `rate_limited` marker keeps the
                    # event observable (no silent quota pollution) without killing
                    # an otherwise-recoverable task.
                    _record_terminal_error(retry_metrics, exc)
                    logger.warning(
                        "finder llm rate-limit ladder exhausted (degrade to empty): %s",
                        exc,
                    )
                    return {
                        "parsed_json": {},
                        "usage": _usage_from_response(None),
                        "retry": dict(retry_metrics),
                        "rate_limited": True,
                    }
                delay_s = provider_delays[rate_limit_attempt]
                _record_retry(retry_metrics, exc, delay_s=delay_s)
                logger.warning(
                    "finder llm attempt %d rate-limited: %s, waiting %.0fs (%d/%d)",
                    attempt + 1,
                    exc,
                    delay_s,
                    rate_limit_attempt + 1,
                    len(provider_delays),
                )
                # DIAGNOSTIC: localize the 429 on the FIRST hit only (gated).
                # A diagnostic must NEVER affect the real path, so swallow any
                # failure of the probe itself (the arms already swallow per-arm).
                if active_config().finder_429_probe_enabled and rate_limit_attempt == 0:
                    try:
                        await _probe_429_three_way(
                            user_prompt, sys_prompt, resolved_model_cfg
                        )
                    except Exception:  # noqa: BLE001 — diagnostic, never propagate
                        logger.warning(
                            "[429-probe] probe failed (ignored)", exc_info=True
                        )
                # Swap to a FRESH httpx pool for the retry so we don't keep
                # re-sending into the already-throttled connection (the user's
                # "重送時換一個池"). If the throttle is per-process/server rather
                # than per-pool this is a no-op and the ladder still degrades —
                # the 429-probe (arm B vs arm C) tells which level escapes.
                if active_config().finder_fresh_pool_on_429:
                    try:
                        pool["client"] = genai.Client(
                            http_options={"timeout": timeout_ms}
                        )
                    except Exception:  # noqa: BLE001 — never let pool-swap break the retry
                        logger.warning(
                            "finder fresh-pool swap failed (ignored)", exc_info=True
                        )
                grant_rate_limit_grace(delay_s)
                await asyncio.sleep(delay_s)
                attempt += 1
                rate_limit_attempt += 1
                continue
            retryable = is_retryable(message)
            if retryable and attempt < max_retries:
                delay_s = _retry_delay_s(attempt)
                _record_retry(retry_metrics, exc, delay_s=delay_s)
                logger.warning(
                    "finder llm attempt %d failed: %s, retrying in %.2fs",
                    attempt + 1,
                    exc,
                    delay_s,
                )
                await asyncio.sleep(delay_s)
                attempt += 1
                continue
            _record_terminal_error(retry_metrics, exc)
            if retryable:
                logger.warning("finder llm failed after all retries: %s", exc)
            else:
                logger.warning("finder llm non-retryable failure: %s", exc)
            return {
                "parsed_json": {},
                "usage": _usage_from_response(None),
                "retry": dict(retry_metrics),
            }


__all__ = ["call_llm_json"]
