"""Gemini thinking-budget config, shared by all main-flow LLM sites.

gemini-2.5-flash defaults to DYNAMIC (unbounded) thinking, which can hang the
response stream — observed as executor [stall-cancel], finder 429/504, and
rollouts silently eating the wall (see the project_gemini_streaming_stall_thinking
investigation). Bounding the thinking phase to a finite token budget eliminates
the hang (verified 2026-06-29: thinking_budget=4096 turned a 5x-stall FAIL into a
clean PASS on a rough rollout).

This is a single env-gated knob so the budget applies pipeline-wide (finder,
rough/code planner, executor, continuation, skill retrieval, answer extraction)
with one switch. Default is 4096 so normal batch runs get a bounded thinking
phase without requiring shell setup. Set GEMINI_THINKING_BUDGET=off (or blank)
to send no thinking_config for an explicit ablation.
"""

from __future__ import annotations

import os

from google.genai import types

_DEFAULT_THINKING_BUDGET = 4096


def thinking_config_from_env():
    """Return bounded Gemini thinking unless explicitly disabled."""
    raw = os.environ.get("GEMINI_THINKING_BUDGET", str(_DEFAULT_THINKING_BUDGET))
    if not raw or not raw.strip() or raw.strip().lower() in {"off", "none", "false"}:
        return None
    try:
        return types.ThinkingConfig(thinking_budget=int(raw))
    except (ValueError, TypeError):
        return None


__all__ = ["thinking_config_from_env"]
