from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, ClassVar

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.cache.spec import CacheSpec, canonical_key
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    RunConfig,
    model_config_from_env,
)
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.failure_codes import (
    FINDER_FETCH_FAILED,
    PROVIDER_RATE_LIMITED,
)
from adk_appworld_agent.subagents.finder.routing import search_apis_by_community_routing
from adk_appworld_agent.subagents.utils.retry import RateLimitExhausted

logger = logging.getLogger(__name__)


def _community_finder_key(subagent_input: SubagentInput) -> str:
    metadata = subagent_input.metadata or {}
    planned_apps = _planned_apps_from_metadata(metadata)
    return canonical_key(
        {
            "task_id": subagent_input.task_context.task_id,
            "instruction": subagent_input.task_context.instruction,
            "milestone_intent": str(metadata.get("milestone_intent") or ""),
            "planned_apps": sorted(planned_apps),
        }
    )


COMMUNITY_FINDER_CACHE_SPEC = CacheSpec(
    subagent_name="community_finder",
    version="v4",
    key_fn=_community_finder_key,
)


class CommunityFinderSubagent(BaseSubagent):
    phase_: ClassVar[Phase] = Phase.FIND
    cache_spec_: ClassVar[CacheSpec | None] = COMMUNITY_FINDER_CACHE_SPEC

    model_cfg: ModelConfig
    # Retrieval-ablation switches. Defaults reproduce Variant A (the production
    # community finder) exactly. Ablation variants live in ablation_finders.py
    # and flip these via subclasses.
    community_layer: bool = True
    dependency_expansion: bool = True
    # When True (impl "global_seed_filter"), the finder ignores planned_apps and
    # feeds seed_filter the entire 454-op API surface. Implies community_layer
    # off. Default False → unchanged for all existing variants.
    global_pool: bool = False
    # When False (impl "community_noguide"), blank out behavior_guidelines +
    # app_context in the selection prompts. Default True → unchanged.
    use_guidance: bool = True
    # When True (impl "community_forward"), the dep loop also expands FORWARD
    # (producer -> consumer communities). Default False → unchanged.
    forward_dependency_expansion: bool = False

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        t0 = time.monotonic()
        subagent_input_text = subagent_input.model_dump_json()
        capture_io = bool(subagent_input.metadata.get("capture_io"))
        planned_apps = _planned_apps_from_metadata(subagent_input.metadata)
        finder_instruction = _finder_instruction(subagent_input)
        io_trace: list[dict] | None = [] if capture_io else None
        try:
            if capture_io:
                result = await search_apis_by_community_routing(
                    finder_instruction,
                    planned_apps=planned_apps,
                    community_layer=self.community_layer,
                    dependency_expansion=self.dependency_expansion,
                    global_pool=self.global_pool,
                    use_guidance=self.use_guidance,
                    forward_dependency_expansion=self.forward_dependency_expansion,
                    model_cfg=self.model_cfg,
                    io_trace=io_trace,
                )
            else:
                result = await search_apis_by_community_routing(
                    finder_instruction,
                    planned_apps=planned_apps,
                    community_layer=self.community_layer,
                    dependency_expansion=self.dependency_expansion,
                    global_pool=self.global_pool,
                    use_guidance=self.use_guidance,
                    forward_dependency_expansion=self.forward_dependency_expansion,
                    model_cfg=self.model_cfg,
                )
        except Exception as exc:
            finder_failure_code = (
                PROVIDER_RATE_LIMITED
                if isinstance(exc, RateLimitExhausted)
                else FINDER_FETCH_FAILED
            )
            logger.warning("community finder failed (%s): %s", finder_failure_code, exc)
            usage_metrics = _aggregate_usage_metrics(io_trace or [])
            retry_metrics = _aggregate_retry_metrics(io_trace or [])
            metrics = {
                "wall_ms": int((time.monotonic() - t0) * 1000),
                "llm_calls": len(io_trace or []),
                "llm_call_attempts": len(io_trace or [])
                + int(retry_metrics.get("retry_count") or 0),
                "dependency_rounds": 0,
                **usage_metrics,
                **retry_metrics,
            }
            payload: dict = {}
            io_block: dict = {}
            if capture_io:
                io_block = {
                    "input": _finder_input_payload(
                        subagent_input=subagent_input,
                        subagent_input_text=subagent_input_text,
                        finder_instruction=finder_instruction,
                        planned_apps=planned_apps,
                        io_trace=io_trace or [],
                    ),
                    "output": {
                        "phase": self.phase_.value,
                        "subagent_name": self.name,
                        "attempt": subagent_input.attempt,
                        "failure_code": finder_failure_code,
                        "error": str(exc),
                        "model_calls": _model_call_outputs(io_trace or []),
                    },
                }
            envelope = self.failed(
                attempt=subagent_input.attempt,
                payload=payload,
                failure_code=finder_failure_code,
                error=str(exc),
            )
            envelope.metrics = metrics
            envelope.io = io_block
            yield envelope
            return

        payload = {
            "matched_apps": result.get("matched_apps", []),
            "fallback_used": bool(result.get("fallback_used")),
            "selected_communities": result.get("selected_communities", []),
            "candidate_apis": result.get("candidate_apis", []),
            "candidate_count": int(result.get("candidate_count", 0)),
            "routing_trace": result.get("routing_trace", []),
        }
        metrics = {
            "wall_ms": int((time.monotonic() - t0) * 1000),
            "llm_calls": int(result.get("llm_rounds", 0)),
            "llm_call_attempts": int(
                result.get("llm_call_attempts", result.get("llm_rounds", 0))
            ),
            "dependency_rounds": int(result.get("dependency_rounds", 0)),
            "usage_event_count": int(result.get("usage_event_count", 0)),
            "prompt_tokens": int(result.get("prompt_tokens", 0)),
            "completion_tokens": int(result.get("completion_tokens", 0)),
            "thoughts_tokens": int(result.get("thoughts_tokens", 0)),
            "total_tokens": int(result.get("total_tokens", 0)),
            "retry_count": int(result.get("retry_count", 0)),
            "retry_backoff_ms": int(result.get("retry_backoff_ms", 0)),
            "rate_limit_count": int(result.get("rate_limit_count", 0)),
            "provider_error_count": int(result.get("provider_error_count", 0)),
        }
        io_block: dict = {}
        if capture_io:
            io_block = {
                "input": _finder_input_payload(
                    subagent_input=subagent_input,
                    subagent_input_text=subagent_input_text,
                    finder_instruction=finder_instruction,
                    planned_apps=planned_apps,
                    io_trace=io_trace or [],
                ),
                "output": _finder_output_payload(
                    subagent_name=self.name,
                    attempt=subagent_input.attempt,
                    payload=payload,
                    io_trace=io_trace or [],
                ),
            }
        envelope = self.succeeded(attempt=subagent_input.attempt, payload=payload)
        envelope.metrics = metrics
        envelope.io = io_block
        yield envelope


def _finder_input_payload(
    *,
    subagent_input: SubagentInput,
    subagent_input_text: str,
    finder_instruction: str,
    planned_apps: list[str],
    io_trace: list[dict],
) -> dict:
    model_calls = _model_call_inputs(io_trace)
    payload = {
        "subagent_input_text": subagent_input_text,
        "subagent_input": subagent_input.model_dump(mode="json"),
        "task_instruction": subagent_input.task_context.instruction,
        "finder_instruction": finder_instruction,
        "task_datetime": subagent_input.task_context.task_datetime,
        "planned_apps": planned_apps,
        "model_calls": model_calls,
    }
    if model_calls:
        payload["model_input_raw"] = model_calls[0]["model_input_raw"]
    return payload


def _planned_apps_from_metadata(metadata: dict) -> list[str]:
    raw = metadata.get("planned_apps")
    if not isinstance(raw, list):
        return []
    apps: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        app = item.strip()
        if app and app not in seen:
            seen.add(app)
            apps.append(app)
    return apps


def _finder_instruction(subagent_input: SubagentInput) -> str:
    metadata = subagent_input.metadata or {}
    intent = metadata.get("milestone_intent")
    if not isinstance(intent, str) or not intent.strip():
        return subagent_input.task_context.instruction or ""

    lines = [
        f"Overall task: {subagent_input.task_context.instruction}",
        f"Current milestone: {intent.strip()}",
    ]
    return "\n".join(lines)


def _finder_output_payload(
    *,
    subagent_name: str,
    attempt: int,
    payload: dict,
    io_trace: list[dict],
) -> dict:
    result_payload = {
        "matched_apps": payload.get("matched_apps", []),
        "fallback_used": payload.get("fallback_used"),
        "selected_communities": payload.get("selected_communities", []),
        "candidate_apis": payload.get("candidate_apis", []),
        "candidate_count": payload.get("candidate_count", 0),
        "routing_trace": payload.get("routing_trace", []),
    }
    return {
        "phase": Phase.FIND.value,
        "subagent_name": subagent_name,
        "attempt": attempt,
        "failure_code": None,
        "raw_llm_text": json.dumps(result_payload, ensure_ascii=False),
        "parsed_api_selection": result_payload,
        "model_calls": _model_call_outputs(io_trace),
    }


def _model_call_inputs(io_trace: list[dict]) -> list[dict]:
    return [
        {
            "step": str(call.get("step", "")),
            "model_input_raw": call.get("model_input_raw") or {},
        }
        for call in io_trace
    ]


def _model_call_outputs(io_trace: list[dict]) -> list[dict]:
    return [
        {
            "step": str(call.get("step", "")),
            "raw_llm_text": str(call.get("raw_llm_text", "")),
            "parsed_json": call.get("parsed_json")
            if isinstance(call.get("parsed_json"), dict)
            else {},
            "usage": call.get("usage") if isinstance(call.get("usage"), dict) else {},
            "retry": call.get("retry") if isinstance(call.get("retry"), dict) else {},
        }
        for call in io_trace
    ]


def _aggregate_usage_metrics(io_trace: list[dict]) -> dict[str, int]:
    totals = {
        "usage_event_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
    }
    for call in io_trace:
        usage = call.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return totals


def _aggregate_retry_metrics(io_trace: list[dict]) -> dict[str, int]:
    totals = {
        "retry_count": 0,
        "retry_backoff_ms": 0,
        "rate_limit_count": 0,
        "provider_error_count": 0,
    }
    for call in io_trace:
        retry = call.get("retry")
        if not isinstance(retry, dict):
            continue
        for key in totals:
            totals[key] += int(retry.get(key) or 0)
    return totals


def build_community_finder_subagent(
    name: str = "finder_subagent_community",
    *,
    run_config: "RunConfig | None" = None,
) -> CommunityFinderSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    return CommunityFinderSubagent(
        name=name,
        description="Community-routing AppWorld API finder.",
        model_cfg=model_cfg,
    )


__all__ = ["CommunityFinderSubagent", "build_community_finder_subagent"]
