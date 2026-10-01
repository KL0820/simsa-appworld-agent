"""Leaf primitives for the code_plan_execute executor.

Pure, dependency-free helpers (timeouts, model/usage/name helpers, answer-type
regexes) extracted from the monolithic ``agent.py`` as part of the
2026-06-22 behavior-preserving split. No imports from sibling executor modules
— this is the bottom of the dependency DAG.
"""

from __future__ import annotations

import re

from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.run_config import ModelConfig

_MAX_CODE_EXECUTE_ATTEMPTS = 4
_DEFAULT_LLM_STALL_TIMEOUT_SECONDS = 60.0
EXECUTOR_LLM_STALL_TIMEOUT_ENV = "EXECUTOR_LLM_STALL_TIMEOUT_S"

# ── regex for answer type inference ───────────────────────────────────────────
_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)$"
)


def _llm_stall_timeout_seconds() -> float:
    """Per-event stall timeout for the Stage 2 executor LLM stream.

    The timer resets every time the stream yields an event, so legitimate
    long tool-call work (e.g. multi-API sandbox scripts) is not bounded.
    Only fires when the LLM stops emitting events entirely — strands of
    Gemini latency degradation or quota cascade. Sandbox infinite loops
    are already bounded by the zerorpc client's per-call timeout.
    """
    # Unified knob: the executor stall timer == the gemini-call max-wait that
    # also governs the finder deadline (same root cause: gemini streaming hang).
    value = active_config().retry.llm_call_timeout_s
    if value <= 0:
        return _DEFAULT_LLM_STALL_TIMEOUT_SECONDS
    return value


def _llm_request_timeout_ms() -> int:
    """HARD transport-level timeout (ms) for the inner LLM calls (planner +
    executor streams).

    The per-event stall timer (`asyncio.wait_for` on `agen.__anext__()`) cannot
    interrupt a SYNCHRONOUSLY-blocking stream read inside the genai client: the
    blocked read holds the event loop, so the asyncio timeout callback never
    runs. Observed: a 28-min hang on 8749218 that neither the 60s stall nor the
    asyncio timeout caught (CPU ~0, blocked on a socket read). Setting
    `http_options.timeout` makes the TRANSPORT abort a hung read -> the stream
    raises -> caught -> EXECUTOR_LLM_RAISED/STALLED -> infra-abort -> the batch
    continues instead of hanging. (The finder already does this via
    `genai.Client(http_options={"timeout": ...})`; this brings the executor's
    ADK LlmAgent path in line.) Bound = executor_timeout_s: well above any
    legitimate generation, far below the runaway hang."""
    return int(active_config().executor_timeout_s * 1000)


def _model_input_raw(
    *, model: str, system_prompt: str, prompt: str, tool_config: dict | None = None
) -> dict:
    from adk_appworld_agent.subagents.utils.io_format import model_input_block

    return model_input_block(
        model=model, instruction=system_prompt, prompt=prompt, tool_config=tool_config
    )


def _model_name(model_cfg: ModelConfig | None, use_llm: bool) -> str:
    if not use_llm:
        return "deterministic_skeleton"
    if model_cfg is None:
        return "unknown"
    return model_cfg.name


def _output_variable_name(metadata: dict) -> str:
    milestone_id = str(metadata.get("milestone_id") or "").strip()
    if milestone_id:
        return _snake_case(f"{milestone_id}_result")
    milestone_index = int(metadata.get("milestone_index") or 0)
    return f"milestone_{milestone_index + 1}_result"


def _snake_case(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")
    return name or "milestone_result"


def _empty_usage() -> dict[str, int]:
    return {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
    }
