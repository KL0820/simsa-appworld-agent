"""AppWorld final-answer extractor.

Ported from cuga-agent's FinalAnswerAgent (AppWorld mode). The extractor
runs after the last EXECUTE milestone and before the AppWorld submit
call. It takes the task_instruction plus the last executor's structured
output (stdout_json with value/answer/summary) and returns a bare
final answer string the evaluator can compare directly.

Implementation: ADK `LlmAgent` with `output_schema=FinalAnswerOutput`.
Gemini's structured-output mode enforces schema-conformant JSON, which
matters here because cuga's prompt has 14 ```json``` markdown-fenced
few-shot examples — without schema enforcement the model mirrors that
fenced format and downstream `json.loads` fails.

Source attribution:
- Prompt: cuga-agent/.../prompts/system_appworld.jinja2 (verbatim port)
- Schema: cuga-agent/.../prompts/load_prompt.py FinalAnswerAppworldOutput
- Integration shape: function-call hook in our orchestrator (we don't
  have langgraph; cuga's `FinalAnswerAgent` graph node maps to our
  orchestrator `_run_completion_gate` pre-submit step).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import aclosing
from pathlib import Path
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import ValidationError

from adk_appworld_agent.contracts.final_answer import FinalAnswerOutput
from adk_appworld_agent.contracts.limits import FINAL_ANSWER_TEXT_MAX_LENGTH
from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
    retry_agent_stream,
)

logger = logging.getLogger(__name__)


_PROMPT_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "final_answer_appworld_system.txt").read_text(
    encoding="utf-8"
)


def _format_value_for_prompt(value: Any) -> str:
    """Render the executor's `value` field for the LLM, with a sane cap.

    Long lists / dicts get truncated so the prompt stays under control on
    tasks that produce thousands of records (the LLM only needs enough
    structure to extract the bare answer).
    """
    if value is None:
        return "null"
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    if len(text) > FINAL_ANSWER_TEXT_MAX_LENGTH:
        return text[:FINAL_ANSWER_TEXT_MAX_LENGTH] + " …(truncated)"
    return text


def build_executor_output_text(stdout_json: dict[str, Any] | None) -> str:
    """Render last executor stdout_json into the `system_answer` slot.

    cuga's prompt template only takes `last_planner_answer` (a single
    string). Our executor produces the 4-field unified contract
    `{value, summary, description, answer}` — we flatten the
    extractor-relevant subset (summary, value, answer) into a structured
    block the extractor LLM can read.

    Including the `answer` line restores the action-vs-query
    discriminator the extractor needs: when the planner / executor
    emitted `"answer": "null"` (action terminal), the extractor sees
    that literal and propagates it. The 325d6ec_2 regression came from
    dropping this line — the extractor then type-inferred a query
    answer from value=song_dict on an action task.
    """
    if not isinstance(stdout_json, dict):
        return "null"
    parts: list[str] = []
    summary = stdout_json.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append(f"summary: {summary.strip()}")
    if "value" in stdout_json:
        parts.append(f"value: {_format_value_for_prompt(stdout_json.get('value'))}")
    if "answer" in stdout_json:
        answer = stdout_json.get("answer")
        if answer is None:
            answer_str = "null"
        else:
            answer_str = str(answer).strip() or "null"
        parts.append(f"answer: {answer_str}")
    if not parts:
        return "null"
    return "\n".join(parts)


def _build_user_prompt(task_instruction: str, executor_output_text: str) -> str:
    return f"`user_intent`: {task_instruction}\n`system_answer`: {executor_output_text}"


def _parse_final_answer_response(raw_text: str) -> FinalAnswerOutput | None:
    """Validate the LlmAgent's final text against FinalAnswerOutput.

    With `output_schema=FinalAnswerOutput`, Gemini's structured output
    emits JSON conforming to the schema. We still validate to surface
    any contract drift cleanly.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    try:
        return FinalAnswerOutput.model_validate_json(raw_text)
    except ValidationError as exc:
        logger.warning(
            "final_answer_extractor: schema validation failed: %s (raw=%r)",
            exc,
            raw_text[:300],
        )
        return None
    except Exception as exc:
        logger.warning("final_answer_extractor: parse failed: %s", exc)
        return None


def _build_extractor_llm_agent(model_cfg: ModelConfig) -> LlmAgent:
    config = types.GenerateContentConfig(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=1,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
        thinking_config=thinking_config_from_env(),
    )
    return LlmAgent(
        name="appworld_final_answer_extractor",
        model=model_cfg.name,
        description=(
            "Extracts the bare final answer from a task instruction + last "
            "executor output, formatted for the AppWorld evaluator."
        ),
        instruction=_SYSTEM_PROMPT,
        output_schema=FinalAnswerOutput,
        generate_content_config=config,
    )


async def _run_extractor_session(
    *,
    inner_agent: LlmAgent,
    user_prompt: str,
) -> str:
    """Run a single LlmAgent turn and return the last text emitted.

    Mirrors the rough_planner pattern: InMemorySessionService + Runner +
    retry_agent_stream wrapper. Each call gets a fresh session_id so
    no state leaks across tasks.
    """
    session_service = InMemorySessionService()
    app_name = "appworld_final_answer_extractor"
    user_id = "extractor"
    session_id = f"extract_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(
        agent=inner_agent, session_service=session_service, app_name=app_name
    )

    last_text = ""

    async def _attempt(attempt: int) -> None:
        nonlocal last_text, session_id
        if attempt > 0:
            session_id = f"extract_{uuid.uuid4().hex[:8]}"
            await session_service.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            last_text = ""
        async with aclosing(
            runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=user_prompt)]
                ),
            )
        ) as agen:
            async for event in agen:
                text = content_to_text(event.content)
                if text:
                    last_text = text

    await retry_agent_stream(_attempt, label="appworld_final_answer_extractor")
    return last_text


async def extract_appworld_final_answer(
    *,
    task_instruction: str,
    last_executor_output: dict[str, Any] | None,
    model_cfg: ModelConfig,
) -> FinalAnswerOutput | None:
    """Run the cuga-style extractor on the last executor output.

    Returns FinalAnswerOutput on success. Returns None if the LLM call
    fails or the response cannot be validated against the schema — the
    caller is expected to fall back to the executor's own answer string.
    """
    executor_output_text = build_executor_output_text(last_executor_output)
    user_prompt = _build_user_prompt(task_instruction, executor_output_text)

    try:
        inner_agent = _build_extractor_llm_agent(model_cfg)
    except Exception as exc:
        logger.warning("final_answer_extractor: failed to build LlmAgent: %s", exc)
        return None

    try:
        raw_text = await _run_extractor_session(
            inner_agent=inner_agent, user_prompt=user_prompt
        )
    except asyncio.CancelledError:
        raise
    except RateLimitExhausted:
        # Unified 429 contract: propagate so the completion gate fails the
        # run with PROVIDER_RATE_LIMITED — a quota-degraded answer is
        # silent pollution; rerun the task instead.
        raise
    except Exception as exc:
        logger.warning("final_answer_extractor: llm session raised: %s", exc)
        return None

    return _parse_final_answer_response(raw_text)


__all__ = [
    "build_executor_output_text",
    "extract_appworld_final_answer",
    "_build_extractor_llm_agent",
    "_build_user_prompt",
    "_parse_final_answer_response",
    "_run_extractor_session",
]
