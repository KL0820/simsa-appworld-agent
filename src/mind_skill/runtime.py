"""Shared runtime utilities for the induction meta agents.

Every meta agent (induction / recon judge / rubric judge / gradient / optimizer)
is a single-turn, tool-less `LlmAgent` on gemini-2.5-flash. This module owns the
pieces they share:

  - `meta_model_config()`        — the project ModelConfig (temp 0 / seed 123).
  - `meta_generate_config()`     — GenerateContentConfig derived from it.
  - `make_journal_callbacks()`   — before/after model callbacks appending every
                                   meta-LLM request/response to a JSONL journal
                                   (reproducibility appendix material).
  - `run_meta_agent()`           — one prompt -> final text, via a nested Runner
                                   (LlmAgents must run under their own Runner).
  - `run_meta_agent_validated()` — same, plus the paper-B.3 retry contract:
                                   empty response -> same-message retry; response
                                   that fails schema validation -> appended back
                                   as an assistant turn followed by a fix
                                   instruction. 3 attempts total.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, ValidationError

from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.utils.retry import is_retryable

META_MODEL_NAME = "gemini-2.5-flash"
MAX_LLM_ATTEMPTS = 3  # paper B.3: every LLM call wrapped in a 3-attempt retry
# 429 / 5xx between same-message retries. Long tail on purpose: the per-minute
# quota is shared with concurrently running experiments, so a call must be able
# to ride out several minutes of contention before giving up.
_TRANSIENT_BACKOFF_S = (20.0, 45.0, 90.0, 180.0, 300.0, 300.0)

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def meta_model_config() -> ModelConfig:
    """All induction-pipeline LLM roles share the deployment alignment config."""
    return ModelConfig(name=META_MODEL_NAME, temperature=0.0, seed=123)


def meta_generate_config(
    model_cfg: ModelConfig | None = None,
) -> types.GenerateContentConfig:
    cfg = model_cfg or meta_model_config()
    return types.GenerateContentConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        candidate_count=cfg.candidate_count,
        seed=cfg.seed,
        max_output_tokens=cfg.max_output_tokens or None,
    )


# ── Observability callbacks ───────────────────────────────────────────────────


def make_journal_callbacks(journal_path: Path, *, role: str):
    """before/after model callbacks that append request/response JSONL records.

    Pure observers: they never modify the request or short-circuit the call
    (no hidden routing — AGENTS.md callback rule).
    """
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _append(record: dict) -> None:
        record["ts"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
        record["role"] = role
        with journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def before_model(callback_context, llm_request):
        try:
            contents = [
                {
                    "role": getattr(c, "role", None),
                    "text": "\n".join(
                        p.text
                        for p in (getattr(c, "parts", None) or [])
                        if getattr(p, "text", None)
                    ),
                }
                for c in (getattr(llm_request, "contents", None) or [])
            ]
            _append({"event": "request", "contents": contents})
        except Exception:
            pass
        return None

    def after_model(callback_context, llm_response):
        try:
            content = getattr(llm_response, "content", None)
            text = ""
            if content is not None:
                text = "\n".join(
                    p.text for p in (content.parts or []) if getattr(p, "text", None)
                )
            usage = getattr(llm_response, "usage_metadata", None)
            _append(
                {
                    "event": "response",
                    "text": text,
                    "usage": {
                        "prompt_tokens": getattr(usage, "prompt_token_count", None),
                        "completion_tokens": getattr(
                            usage, "candidates_token_count", None
                        ),
                        "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
                    }
                    if usage is not None
                    else None,
                }
            )
        except Exception:
            pass
        return None

    return before_model, after_model


# ── Single-turn execution ─────────────────────────────────────────────────────


async def _run_turn(
    runner: Runner, *, user_id: str, session_id: str, message: str
) -> str:
    final_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=message)]),
    ):
        if event.content and event.content.parts:
            text = "\n".join(
                p.text for p in event.content.parts if getattr(p, "text", None)
            )
            if text:
                final_text = text
    return final_text


async def _run_turn_with_transient_retry(
    runner: Runner, *, user_id: str, session_id: str, message: str, label: str
) -> str:
    """_run_turn + paper-B.3 transient handling: 429 / 5xx / RESOURCE_EXHAUSTED
    raised by the provider get a same-message retry with backoff instead of
    killing the whole task loop."""
    for i, delay in enumerate((*_TRANSIENT_BACKOFF_S, None)):
        try:
            return await _run_turn(
                runner, user_id=user_id, session_id=session_id, message=message
            )
        except Exception as exc:
            if delay is None or not is_retryable(str(exc)):
                raise
            sys.stderr.write(
                f"[mind_skill:{label}] transient ({type(exc).__name__}), "
                f"retry {i + 1}/{len(_TRANSIENT_BACKOFF_S)} in {delay:.0f}s\n"
            )
            sys.stderr.flush()
            await asyncio.sleep(delay)
    return ""  # unreachable


async def run_meta_agent(agent: LlmAgent, prompt: str) -> str:
    """Run one tool-less LlmAgent for one prompt under its own nested Runner.

    Empty responses (rate limits, transient outages) get a same-message retry
    in a FRESH session, up to MAX_LLM_ATTEMPTS.
    """
    last = ""
    for _ in range(MAX_LLM_ATTEMPTS):
        session_service = InMemorySessionService()
        app_name = f"mind_skill_{agent.name}"
        user_id = "mind_skill"
        session_id = f"{agent.name}_{uuid.uuid4().hex[:8]}"
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        runner = Runner(agent=agent, session_service=session_service, app_name=app_name)
        last = await _run_turn_with_transient_retry(
            runner,
            user_id=user_id,
            session_id=session_id,
            message=prompt,
            label=agent.name,
        )
        if last.strip():
            return last
    return last


def _strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[: -len("```")]
    return cleaned.strip()


async def run_meta_agent_validated(
    agent: LlmAgent,
    prompt: str,
    schema: type[SchemaT],
    *,
    postprocess: Callable[[dict], dict] | None = None,
) -> SchemaT:
    """Run a schema-output meta agent and return the validated model.

    Paper B.3 recovery contract:
      - empty response          -> same-message retry (fresh session)
      - schema-invalid response -> the bad turn stays in the SAME session as an
        assistant turn; a follow-up user message carries the fix instruction.

    Raises the last ValidationError / ValueError after MAX_LLM_ATTEMPTS.
    """
    last_error: Exception | None = None
    for _ in range(MAX_LLM_ATTEMPTS):
        session_service = InMemorySessionService()
        app_name = f"mind_skill_{agent.name}"
        user_id = "mind_skill"
        session_id = f"{agent.name}_{uuid.uuid4().hex[:8]}"
        await session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
        runner = Runner(agent=agent, session_service=session_service, app_name=app_name)

        message = prompt
        for _fix_round in range(MAX_LLM_ATTEMPTS):
            text = await _run_turn_with_transient_retry(
                runner,
                user_id=user_id,
                session_id=session_id,
                message=message,
                label=agent.name,
            )
            if not text.strip():
                break  # empty -> same-message retry in a fresh session
            try:
                raw: Any = json.loads(_strip_json_fences(text))
                if postprocess is not None and isinstance(raw, dict):
                    raw = postprocess(raw)
                return schema.model_validate(raw)
            except (ValueError, ValidationError) as exc:  # json error or schema error
                last_error = exc
                # The invalid turn is already in this session's history as the
                # assistant turn; send the fix instruction as the next user turn.
                message = (
                    "Your previous response failed validation:\n"
                    f"{exc}\n"
                    "Re-emit ONLY the corrected JSON object for the required schema."
                )
    raise (
        last_error
        if last_error is not None
        else RuntimeError(
            f"{agent.name}: no non-empty response after {MAX_LLM_ATTEMPTS} attempts"
        )
    )


__all__ = [
    "MAX_LLM_ATTEMPTS",
    "META_MODEL_NAME",
    "make_journal_callbacks",
    "meta_generate_config",
    "meta_model_config",
    "run_meta_agent",
    "run_meta_agent_validated",
]
