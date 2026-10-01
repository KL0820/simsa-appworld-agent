from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import aclosing
from typing import Any, AsyncIterator, ClassVar

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_appworld_agent.contracts.limits import FALLBACK_MILESTONE_TASK_MAX_LENGTH
from adk_appworld_agent.contracts.plan import Plan, PlanTask
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.cache.policy import replay_metrics
from adk_appworld_agent.orchestration.cache.spec import CacheSpec, canonical_key
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    RunConfig,
    model_config_from_env,
)
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    static_instruction_provider,
)
from adk_appworld_agent.subagents.failure_codes import PROVIDER_RATE_LIMITED
from adk_appworld_agent.subagents.planner.rough_planner.prompts import (  # noqa: F401
    ALLOWED_APP_NAMES,
    APP_CATALOG,
    APP_CATALOG_SOURCE,
    APP_DESCRIPTIONS,
    APP_SELECTION_CONTEXT,
    ROUGH_PLANNER_INSTRUCTION,
    ROUGH_PLANNER_MINIMAL_INSTRUCTION,
    render_app_catalog,
    render_rough_planner_instruction,
    render_rough_planner_prompt,
)
from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
    retry_agent_stream,
)

_ALLOWED_APP_SET = set(ALLOWED_APP_NAMES)


def _debug_enabled() -> bool:
    return os.getenv("ROUGH_PLANNER_DEBUG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _fallback_plan(instruction: str) -> Plan:
    task = instruction.strip() or "Complete the user task."
    return Plan(
        thoughts="Fallback single-task rough plan.",
        tasks=[
            PlanTask(task=task[:FALLBACK_MILESTONE_TASK_MAX_LENGTH], app="file_system")
        ],
    )


def _normalize_app_name(app: str) -> str | None:
    candidate = app.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "simplenote": "simple_note",
        "simple_notes": "simple_note",
        "filesystem": "file_system",
        "file": "file_system",
        "files": "file_system",
    }
    candidate = aliases.get(candidate, candidate)
    if candidate in _ALLOWED_APP_SET:
        return candidate
    return None


def _normalize_plan(plan: Plan, instruction: str) -> Plan:
    tasks: list[PlanTask] = []
    for item in plan.tasks:
        app = _normalize_app_name(item.app)
        if app is None:
            sys.stderr.write(f"[rough_planner] unknown app in plan: {item.app!r}\n")
            sys.stderr.flush()
            return _fallback_plan(instruction)
        tasks.append(PlanTask(task=item.task.strip(), app=app))

    if not tasks:
        return _fallback_plan(instruction)

    return Plan(thoughts=plan.thoughts.strip(), tasks=tasks)


def _parse_plan(text: str, instruction: str) -> Plan:
    try:
        data = json.loads(text)
        return _normalize_plan(Plan.model_validate(data), instruction)
    except Exception as exc:
        sys.stderr.write(
            f"[rough_planner] failed to parse output: {exc}, falling back\n"
        )
        sys.stderr.flush()
        return _fallback_plan(instruction)


def _extract_usage(event: Event) -> dict[str, int] | None:
    usage = getattr(event, "usage_metadata", None)
    if usage is None:
        return None

    prompt_tokens = getattr(usage, "prompt_token_count", None) or 0
    completion_tokens = getattr(usage, "candidates_token_count", None) or 0
    thoughts_tokens = getattr(usage, "thoughts_token_count", None) or 0
    total_tokens = getattr(usage, "total_token_count", None) or (
        prompt_tokens + completion_tokens + thoughts_tokens
    )
    if (
        prompt_tokens == 0
        and completion_tokens == 0
        and thoughts_tokens == 0
        and total_tokens == 0
    ):
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thoughts_tokens": thoughts_tokens,
        "total_tokens": total_tokens,
    }


def _build_inner_llm_agent(
    model_cfg: ModelConfig, *, instruction=None, skills_dir=None
) -> LlmAgent:
    extra: dict = {}
    if skills_dir is not None:
        from adk_appworld_agent.subagents.executor.code_plan_execute import (
            build_skill_toolset,
        )

        toolset = build_skill_toolset(skills_dir)
        if toolset is not None:
            extra["tools"] = [toolset]
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
        name="rough_planner_llm",
        model=model_cfg.name,
        description="Single-turn rough planner that decomposes AppWorld tasks into single-app work units.",
        instruction=(
            instruction
            if callable(instruction)
            else static_instruction_provider(instruction)
            if instruction is not None
            else ROUGH_PLANNER_INSTRUCTION
        ),
        output_schema=Plan,
        generate_content_config=config,
        **extra,
    )


def _safe_model_config_snapshot(config: object) -> dict[str, object]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        try:
            dumped = config.model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    snapshot: dict[str, object] = {}
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


def _rough_planner_key(subagent_input: SubagentInput) -> str:
    task_context = subagent_input.task_context
    return canonical_key(
        {
            "task_id": task_context.task_id,
            "instruction": task_context.instruction,
            "task_datetime": task_context.task_datetime,
        }
    )


def _rough_planner_adapt_replay(
    envelope: SubagentEnvelope, subagent_input: SubagentInput, source: str
) -> SubagentEnvelope:
    next_env = envelope.model_copy(deep=True)
    next_env.attempt = subagent_input.attempt
    warning = f"replayed from {source}"
    if warning not in next_env.warnings:
        next_env.warnings = [*next_env.warnings, warning]
    payload = dict(next_env.payload or {})
    payload["cache_source"] = source
    rendered_prompt = render_rough_planner_prompt(
        task_id=subagent_input.task_context.task_id,
        instruction=subagent_input.task_context.instruction,
        task_datetime=subagent_input.task_context.task_datetime,
    )
    if subagent_input.metadata.get("capture_io"):
        raw_plan = {
            "thoughts": payload.get("thoughts", ""),
            "tasks": payload.get("tasks", []),
        }
        next_env.io = {
            "input": {
                "rendered_prompt": rendered_prompt,
                "subagent_input": subagent_input.model_dump(mode="json"),
                "model_input": {
                    "model": "rough_plan_cache",
                    "system_instruction": ROUGH_PLANNER_INSTRUCTION,
                    "messages": [
                        {"role": "user", "parts": [{"text": rendered_prompt}]}
                    ],
                    "cache_source": source,
                },
            },
            "output": {
                "raw_llm_text": json.dumps(raw_plan, ensure_ascii=False),
                "parsed_plan": raw_plan,
            },
        }
    metrics = replay_metrics(next_env.metrics)
    next_env.payload = payload
    next_env.metrics = metrics
    return next_env


ROUGH_PLANNER_CACHE_SPEC = CacheSpec(
    subagent_name="rough_planner",
    version="v2",
    key_fn=_rough_planner_key,
    adapt_replay=_rough_planner_adapt_replay,
)


class RoughPlannerSubagent(BaseSubagent):
    """Produces single-app work units for an AppWorld task."""

    phase_: ClassVar[Phase] = Phase.PLAN
    cache_spec_: ClassVar[CacheSpec | None] = ROUGH_PLANNER_CACHE_SPEC

    inner_agent: LlmAgent

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        prompt = render_rough_planner_prompt(
            task_id=subagent_input.task_context.task_id,
            instruction=subagent_input.task_context.instruction,
            task_datetime=subagent_input.task_context.task_datetime,
        )

        session_service = InMemorySessionService()
        app_name = f"rough_planner_{subagent_input.task_context.task_id or 'task'}"
        user_id = "rough_planner"
        session_id = f"rough_plan_{uuid.uuid4().hex[:8]}"
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
        )
        runner = Runner(
            agent=self.inner_agent,
            session_service=session_service,
            app_name=app_name,
        )

        last_text = ""
        t0 = time.monotonic()
        llm_calls = 0
        skill_load_counter = {"n": 0}
        llm_call_attempts = 0
        usage_event_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        thoughts_tokens = 0
        total_tokens = 0

        async def _attempt(attempt: int) -> None:
            nonlocal last_text, session_id
            nonlocal llm_calls, llm_call_attempts, usage_event_count
            nonlocal prompt_tokens, completion_tokens, thoughts_tokens, total_tokens

            if attempt > 0:
                session_id = f"rough_plan_{uuid.uuid4().hex[:8]}"
                await session_service.create_session(
                    app_name=app_name,
                    user_id=user_id,
                    session_id=session_id,
                )
                last_text = ""

            llm_call_attempts += 1
            async with aclosing(
                runner.run_async(
                    user_id=user_id,
                    session_id=session_id,
                    new_message=types.Content(
                        role="user",
                        parts=[types.Part(text=prompt)],
                    ),
                )
            ) as agen:
                async for event in agen:
                    nonlocal_parts = getattr(event.content, "parts", None) or []
                    for part in nonlocal_parts:
                        fc = getattr(part, "function_call", None)
                        if fc is not None and fc.name == "load_skill":
                            skill_load_counter["n"] += 1
                    usage = _extract_usage(event)
                    if usage is not None:
                        llm_calls += 1
                        usage_event_count += 1
                        prompt_tokens += usage["prompt_tokens"]
                        completion_tokens += usage["completion_tokens"]
                        thoughts_tokens += usage["thoughts_tokens"]
                        total_tokens += usage["total_tokens"]
                    text = content_to_text(event.content)
                    if text:
                        last_text = text
                        if _debug_enabled():
                            sys.stderr.write(
                                f"[rough_planner] raw output (len={len(text)}): {text[:300]}\n"
                            )
                            sys.stderr.flush()

        rate_limit_exhausted = False
        try:
            llm_raised = await retry_agent_stream(_attempt, label="rough_planner")
        except RateLimitExhausted as exc:
            llm_raised = str(exc)
            rate_limit_exhausted = True
        plan = _parse_plan(last_text, subagent_input.task_context.instruction)
        wall_ms = int((time.monotonic() - t0) * 1000)
        metrics = {
            "wall_ms": wall_ms,
            "llm_calls": llm_calls,
            "skill_loads": skill_load_counter["n"],
            "llm_call_attempts": llm_call_attempts,
            "usage_event_count": usage_event_count,
            "timeout_count": 0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "total_tokens": total_tokens,
        }

        payload: dict[str, Any] = {
            "thoughts": plan.thoughts,
            "tasks": [task.model_dump() for task in plan.tasks],
            "task_count": len(plan.tasks),
        }
        if llm_raised is not None:
            payload["llm_raised"] = llm_raised

        io_block: dict[str, Any] = {}
        if subagent_input.metadata.get("capture_io"):
            model_name = getattr(self.inner_agent, "model", "")
            if not isinstance(model_name, str):
                model_name = str(model_name)
            from adk_appworld_agent.subagents.utils.io_format import (
                agent_model_input_block,
            )

            io_block = {
                "input": {
                    "rendered_prompt": prompt,
                    "subagent_input": subagent_input.model_dump(mode="json"),
                    "model_input": agent_model_input_block(self.inner_agent, prompt),
                },
                "output": {
                    "raw_llm_text": last_text,
                    "parsed_plan": plan.model_dump(mode="json"),
                },
            }

        if rate_limit_exhausted:
            # Unified 429 contract: do NOT degrade into an empty plan
            # (PLAN_EMPTY) — surface the provider problem so the run aborts
            # with PROVIDER_RATE_LIMITED and the task is rerun later.
            envelope = self.failed(
                attempt=subagent_input.attempt,
                payload=payload,
                failure_code=PROVIDER_RATE_LIMITED,
            )
        else:
            envelope = self.succeeded(attempt=subagent_input.attempt, payload=payload)
        envelope.metrics = metrics
        envelope.io = io_block
        yield envelope


class RoughPlannerSkillSubagent(RoughPlannerSubagent):
    """rough planner with per-task skill resolution (registry impl: rough_skill).

    Resolves this task's skills (train: per-task library entry; held-out:
    description retrieval, spec §12) INSIDE the subagent, swaps the frozen
    minimal instruction's SKILLS slot, and reports the choice in the standard
    payload — identical logging surface to every other impl.
    """

    skills_source: object | None = None
    skill_instruction: object | None = None  # MutableInstruction
    skills_prompt_variant: str = "minimal"

    async def run_subagent(self, subagent_input, ctx):
        resolved = None
        if self.skills_source is not None and self.skill_instruction is not None:
            from adk_appworld_agent.subagents.planner.rough_planner.prompts import (
                render_rough_planner_instruction,
            )

            try:
                resolved = await self.skills_source.resolve(
                    subagent_input.task_context.task_id,
                    subagent_input.task_context.instruction,
                )
            except RateLimitExhausted as exc:
                yield self.failed(
                    attempt=subagent_input.attempt,
                    payload={"llm_raised": str(exc)},
                    failure_code=PROVIDER_RATE_LIMITED,
                )
                return
            self.skill_instruction.current = render_rough_planner_instruction(
                variant=self.skills_prompt_variant, skill_texts=resolved.texts
            )
        async for envelope in super().run_subagent(subagent_input, ctx):
            if resolved is not None and isinstance(envelope.payload, dict):
                envelope.payload["skills"] = {
                    "names": resolved.names,
                    "source": resolved.source,
                    "library": str(self.skills_source.library),
                }
            yield envelope


def build_rough_planner_skill_subagent(
    name: str = "rough_planner_subagent_skill",
    *,
    run_config: "RunConfig | None" = None,
    native: bool = False,
) -> RoughPlannerSubagent:
    """Registry impls rough_skill / rough_skill_native."""
    from adk_appworld_agent.subagents.utils.skills_source import (
        MutableInstruction,
        SkillsSource,
    )

    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    root = run_config.skills_root if run_config is not None else None
    library = run_config.skills_library if run_config is not None else "best"
    k = run_config.skills_k if run_config is not None else 3
    selector = run_config.skills_selector if run_config is not None else "llm"
    embedding_model = (
        run_config.skills_embedding_model
        if run_config is not None
        else "all-mpnet-base-v2"
    )
    thin_text = render_rough_planner_instruction(variant="minimal")

    if native:
        inner = _build_inner_llm_agent(
            model_cfg,
            instruction=thin_text,
            skills_dir=(root / "rough_planner" / library) if root is not None else None,
        )
        return RoughPlannerSubagent(
            name=name,
            description="Rough planner, frozen thin prompt + native SkillToolset (model-pull).",
            inner_agent=inner,
        )

    holder = MutableInstruction(thin_text)
    inner = _build_inner_llm_agent(model_cfg, instruction=holder)
    source = (
        SkillsSource(
            component="rough_planner",
            root=root,
            library=library,
            model_cfg=model_cfg,
            k=k,
            selector=selector,
            embedding_model=embedding_model,
        )
        if root is not None and library != "none"
        else None
    )
    return RoughPlannerSkillSubagent(
        name=name,
        description="Rough planner, frozen thin prompt + per-task skill injection.",
        inner_agent=inner,
        skills_source=source,
        skill_instruction=holder,
        skills_prompt_variant=(
            run_config.skills_prompt_variant if run_config is not None else "minimal"
        ),
    )


def build_rough_planner_subagent(
    name: str = "rough_planner_subagent",
    *,
    run_config: "RunConfig | None" = None,
) -> RoughPlannerSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    instruction = None
    if run_config is not None and run_config.rough_planner_prompt_variant != "full":
        from adk_appworld_agent.subagents.planner.rough_planner.prompts import (
            render_rough_planner_instruction,
        )

        instruction = render_rough_planner_instruction(
            variant=run_config.rough_planner_prompt_variant,
            skill_texts=list(run_config.rough_planner_skills),
        )
    inner = _build_inner_llm_agent(
        model_cfg,
        instruction=instruction,
        skills_dir=(
            run_config.rough_planner_skills_dir if run_config is not None else None
        ),
    )
    return RoughPlannerSubagent(
        name=name,
        description="ADK RoughPlanner subagent for single-app work-unit decomposition.",
        inner_agent=inner,
    )


__all__ = [
    "ROUGH_PLANNER_CACHE_SPEC",
    "RoughPlannerSkillSubagent",
    "RoughPlannerSubagent",
    "build_rough_planner_skill_subagent",
    "build_rough_planner_subagent",
]
