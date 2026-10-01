"""Canonical per-LLM-call io block, shared by every subagent.

Module-swap contract: every impl of every layer logs its model calls in the
SAME shape, so output files and the analysis viewer read identical fields no
matter which impl produced them:

    {
      "model": <model name str>,
      "system_instruction": <ACTUAL instruction text — providers unwrapped>,
      "messages": [{"role": "user", "parts": [{"text": <prompt>}]}],
      "generate_content_config": {...} | absent,
      "tool_config": {...} | absent,
    }

Always build this through `model_input_block(...)`; never hand-roll the dict
(hand-rolled blocks twice hardcoded a stale system_instruction).
"""

from __future__ import annotations

from typing import Any

from adk_appworld_agent.subagents.utils.instructions import instruction_text


def config_snapshot(config: Any) -> dict:
    """Plain-dict snapshot of a GenerateContentConfig (best effort)."""
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        try:
            dumped = config.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    snapshot: dict = {}
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "candidate_count",
        "seed",
        "max_output_tokens",
    ):
        value = getattr(config, key, None)
        if value is not None:
            snapshot[key] = value
    return snapshot


def model_input_block(
    *,
    model: str | Any,
    instruction: Any,
    prompt: str,
    generate_content_config: Any = None,
    tool_config: dict | None = None,
) -> dict:
    """The canonical model_input dict (see module docstring)."""
    model_name = model if isinstance(model, str) else str(model)
    block: dict = {
        "model": model_name,
        "system_instruction": instruction_text(instruction),
        "messages": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    snapshot = config_snapshot(generate_content_config)
    if snapshot:
        block["generate_content_config"] = snapshot
    if tool_config is not None:
        block["tool_config"] = tool_config
    return block


def agent_model_input_block(
    agent: Any, prompt: str, *, tool_config: dict | None = None
) -> dict:
    """model_input_block derived directly from an LlmAgent (preferred form)."""
    return model_input_block(
        model=getattr(agent, "model", "unknown"),
        instruction=getattr(agent, "instruction", None),
        prompt=prompt,
        generate_content_config=getattr(agent, "generate_content_config", None),
        tool_config=tool_config,
    )


__all__ = ["agent_model_input_block", "config_snapshot", "model_input_block"]
