from __future__ import annotations

import sys
import time
from typing import AsyncIterator, ClassVar

from google.adk.agents import LlmAgent

from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    RunConfig,
    model_config_from_env,
)
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.executor.appworld_tools import (
    read_sandbox_trace_since,
    sandbox_trace_byte_offset,
)
from adk_appworld_agent.subagents.executor.code_plan_execute._primitives import (
    _empty_usage,
    _model_input_raw,
    _model_name,
)
from adk_appworld_agent.subagents.executor.code_plan_execute._primitives import (
    _llm_request_timeout_ms as _llm_request_timeout_ms,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.builders import (
    _build_failure_payload,
    _build_inner_code_executor_agent,
    _build_inner_code_planner_agent,
    _build_io_payload,
    _execute_python_tool_config_json,
    build_skill_toolset,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.execution import (
    _build_deterministic_execute_run,
    _build_deterministic_plan,
    _CodeExecuteRun,
    _CodePlanRun,
    _parse_code_plan,
    _run_llm_code_executor,
    _run_llm_code_planner,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompt_builders import (
    SELF_ASSESS_SYS_PROMPT,
    SELF_ASSESS_USER_PROMPT,
    _build_code_execute_prompt,
    _build_code_plan_prompt,
    _render_observed_from_records,
    _render_value_for_self_assess,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompt_builders import (
    _format_prior_attempts_block as _format_prior_attempts_block,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompt_builders import (
    _render_prior_returns as _render_prior_returns,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompts import (
    CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT,
    CODE_EXECUTOR_SYSTEM_PROMPT,
    CODE_PLANNER_MINIMAL_SYSTEM_PROMPT,
    CODE_PLANNER_SYSTEM_PROMPT,
    instruction_text,
    render_code_planner_instruction,
    render_executor_instruction,
    render_executor_skills_section,
    static_instruction_provider,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.results import (
    _build_executor_result,
    _parse_execute_stdout,
)
from adk_appworld_agent.subagents.failure_codes import (
    EXECUTOR_DID_NOT_FINALIZE,
    EXECUTOR_LLM_RAISED,
    EXECUTOR_LLM_STALLED,
    EXECUTOR_MILESTONE_NOT_DONE,
    PROVIDER_RATE_LIMITED,
)
from adk_appworld_agent.subagents.finder.llm import call_llm_json
from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
)


class CodePlanExecuteSubagent(BaseSubagent):
    """M11 executor: Stage 1 LLM code planner + Stage 2 LLM code executor."""

    phase_: ClassVar[Phase] = Phase.EXECUTE

    inner_agent: LlmAgent | None = None
    inner_executor_agent: LlmAgent | None = None
    model_cfg: ModelConfig | None = None
    use_llm_code_plan: bool = True
    use_llm_code_execute: bool = True
    self_assess: bool = False

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        t0 = time.monotonic()
        meta = subagent_input.metadata or {}
        milestone_id = str(meta.get("milestone_id") or "")
        milestone_index = int(meta.get("milestone_index") or 0)
        milestone_total = int(meta.get("milestone_total") or 1)

        # ── Stage 1: Code Planning ─────────────────────────────────────────
        plan_run = await self._run_code_planner(subagent_input)
        if plan_run.plan is None:
            payload, io_block, metrics_block = _build_failure_payload(
                subagent_input=subagent_input,
                plan_run=plan_run,
                milestone_id=milestone_id,
                milestone_index=milestone_index,
                milestone_total=milestone_total,
                t0=t0,
            )
            env = self.failed(
                attempt=subagent_input.attempt,
                payload=payload,
                failure_code=(
                    PROVIDER_RATE_LIMITED
                    if plan_run.rate_limit_exhausted
                    else EXECUTOR_LLM_RAISED
                ),
            )
            env.io = io_block
            env.metrics = metrics_block
            yield env
            return

        plan = plan_run.plan

        # ── Stage 2: Code Execution ────────────────────────────────────────
        # Capture the byte offset before the execute so self-assess can read
        # exactly THIS execute's api_trace (the real returns) afterwards.
        trace_offset = sandbox_trace_byte_offset() if self.self_assess else 0
        exec_run = await self._run_code_executor(subagent_input, plan)

        executor_result, candidate, finalize_called = _build_executor_result(
            exec_run=exec_run,
            plan=plan,
            milestone_id=milestone_id,
        )

        # ── Stage 2.5: grounded self-assess (ADVISORY observe-then-explain) ──
        # The in-code summary was written BEFORE the value existed (blind). When
        # enabled, run ONE grounded review of (intent + value + code + this
        # execute's real api_trace) and attach a GROUNDED summary + optional
        # `problem` hint for the continuation planner.
        #
        # ADVISORY ONLY: it does NOT flip milestone_done. The done/not-done
        # judgement stays with continuation §3, which already encodes every
        # exception (mutation value=null §3b, display truncation §3e, multi-path
        # §3d, ...). A self-assess gate that flipped milestone_done kept
        # FALSE-flagging correct results — intermediate-read subsets, paginated
        # over-counts, mutation value=null — and each regression demanded a new
        # carve-out rule (patch accumulation). Making it advisory removes that
        # whole failure class while keeping the value: a grounded summary +
        # problem hint that continuation weighs alongside its own §3 review.
        self_assess_block: dict | None = None
        if (
            self.self_assess
            and finalize_called
            and executor_result.milestone_done
            and self.use_llm_code_execute
        ):
            verdict = await self._run_self_assess(
                subagent_input, plan, exec_run, trace_offset
            )
            if verdict is not None:
                self_assess_block = verdict
                grounded = (verdict.get("summary") or "").strip()
                if grounded:
                    executor_result.summary = grounded

        code_execute = {
            "mode": "llm" if self.use_llm_code_execute else "deterministic_skeleton",
            "code": exec_run.code,
            "raw_stdout": exec_run.raw_stdout,
            "stdout_json": exec_run.stdout_json,
            "parse_error": exec_run.parse_error,
            "tool_call_count": exec_run.tool_call_count,
            "tool_calls": exec_run.tool_calls,
            "event_diagnostics": exec_run.event_diagnostics,
            "llm_raised": exec_run.llm_raised,
            "repair_attempts": exec_run.repair_attempts,
        }
        if self_assess_block is not None:
            code_execute["self_assess"] = self_assess_block

        wall_ms = int((time.monotonic() - t0) * 1000)
        total_usage = {
            k: plan_run.usage.get(k, 0) + exec_run.usage.get(k, 0)
            for k in (
                "llm_calls",
                "prompt_tokens",
                "completion_tokens",
                "thoughts_tokens",
            )
        }

        payload = {
            "executor_result": executor_result.model_dump(mode="json"),
            "milestone_done": executor_result.milestone_done,
            "submission_candidate": candidate.model_dump(mode="json"),
            "finalize_called": finalize_called,
            "tool_call_count": exec_run.tool_call_count,
            "milestone_id": milestone_id or None,
            "milestone_index": milestone_index,
            "milestone_total": milestone_total,
            "code_plan": plan.model_dump(mode="json"),
            "code_execute": code_execute,
        }
        io_block = _build_io_payload(
            subagent_input=subagent_input,
            plan_run=plan_run,
            exec_run=exec_run,
            code_execute=code_execute,
        )
        metrics_block = {
            "wall_ms": wall_ms,
            **total_usage,
        }

        def _emit(failure_code: str | None) -> SubagentEnvelope:
            if failure_code is None:
                env = self.succeeded(attempt=subagent_input.attempt, payload=payload)
            else:
                env = self.failed(
                    attempt=subagent_input.attempt,
                    payload=payload,
                    failure_code=failure_code,
                )
            env.io = io_block
            env.metrics = metrics_block
            return env

        if exec_run.rate_limit_exhausted:
            yield _emit(PROVIDER_RATE_LIMITED)
            return
        if exec_run.llm_raised is not None:
            failure_code = (
                EXECUTOR_LLM_STALLED if exec_run.stalled else EXECUTOR_LLM_RAISED
            )
            yield _emit(failure_code)
            return
        if not finalize_called:
            yield _emit(EXECUTOR_DID_NOT_FINALIZE)
            return
        if not executor_result.milestone_done:
            yield _emit(EXECUTOR_MILESTONE_NOT_DONE)
            return

        yield _emit(None)

    # ── Stage 1 ──────────────────────────────────────────────────────────────

    async def _run_code_planner(self, subagent_input: SubagentInput) -> _CodePlanRun:
        prompt = _build_code_plan_prompt(subagent_input)
        planner_instruction = (
            instruction_text(self.inner_agent.instruction)
            if self.inner_agent is not None
            else CODE_PLANNER_SYSTEM_PROMPT
        )
        model_input_raw = _model_input_raw(
            model=_model_name(self.model_cfg, self.use_llm_code_plan),
            system_prompt=planner_instruction,
            prompt=prompt,
        )
        if not self.use_llm_code_plan:
            plan = _build_deterministic_plan(subagent_input.metadata or {})
            return _CodePlanRun(
                plan=plan,
                prompt=prompt,
                model_input_raw=model_input_raw,
                raw_text=plan.model_dump_json(),
                parsed_json=plan.model_dump(mode="json"),
                parse_error=None,
                llm_raised=None,
                usage=_empty_usage(),
            )
        if self.inner_agent is None:
            return _CodePlanRun(
                plan=None,
                prompt=prompt,
                model_input_raw=model_input_raw,
                raw_text="",
                parsed_json=None,
                parse_error="code planner inner_agent is not configured",
                llm_raised="code planner inner_agent is not configured",
                usage=_empty_usage(),
            )
        return await _run_llm_code_planner(
            inner_agent=self.inner_agent,
            subagent_input=subagent_input,
            prompt=prompt,
            model_input_raw=model_input_raw,
        )

    # ── Stage 2 ──────────────────────────────────────────────────────────────

    async def _run_code_executor(
        self, subagent_input: SubagentInput, plan: CodePlanOutput
    ) -> _CodeExecuteRun:
        prompt = _build_code_execute_prompt(subagent_input, plan)
        executor_instruction = (
            instruction_text(self.inner_executor_agent.instruction)
            if self.inner_executor_agent is not None
            else CODE_EXECUTOR_SYSTEM_PROMPT
        )
        model_input_raw = _model_input_raw(
            model=_model_name(self.model_cfg, self.use_llm_code_execute),
            system_prompt=executor_instruction,
            prompt=prompt,
            tool_config=_execute_python_tool_config_json()
            if self.use_llm_code_execute
            else None,
        )
        if not self.use_llm_code_execute:
            return _build_deterministic_execute_run(
                subagent_input.metadata or {}, plan, model_input_raw
            )
        if self.inner_executor_agent is None:
            err = "code executor inner_executor_agent is not configured"
            return _CodeExecuteRun(
                stdout_json=None,
                finalize_args=None,
                submit_final_args=None,
                raw_stdout="",
                code=None,
                model_input_raw=model_input_raw,
                raw_text="",
                parse_error=err,
                llm_raised=err,
                tool_call_count=0,
                tool_calls=[],
                event_diagnostics=[],
                usage=_empty_usage(),
            )
        return await _run_llm_code_executor(
            inner_agent=self.inner_executor_agent,
            subagent_input=subagent_input,
            prompt=prompt,
            model_input_raw=model_input_raw,
        )

    async def _run_self_assess(
        self,
        subagent_input: SubagentInput,
        plan: CodePlanOutput,
        exec_run: "_CodeExecuteRun",
        trace_offset: int,
    ) -> dict | None:
        """Observe-then-conclude: ONE grounded review of (milestone intent +
        produced value + code + this execute's real api_trace) → a done verdict
        {ok, summary, problem}. Best-effort: any failure returns None (the blind
        verdict stands). Calibrated to flag only on concrete evidence (the
        prompt defaults to ok=true) to avoid the prior verifier's over-flag."""
        stdout_json = exec_run.stdout_json or {}
        value = stdout_json.get("value")
        value_json = _render_value_for_self_assess(value)
        trace_records = read_sandbox_trace_since(trace_offset)
        returns_view = (
            _render_observed_from_records(trace_records) or "(no API returns observed)"
        )
        ms_intent = (subagent_input.metadata or {}).get("milestone_intent") or ""
        user_prompt = SELF_ASSESS_USER_PROMPT.format(
            task_instruction=subagent_input.task_context.instruction,
            milestone_intent=ms_intent,
            value_json=value_json,
            observed_returns=returns_view,
            code=(exec_run.code or "")[:4000],
        )
        try:
            result = await call_llm_json(
                user_prompt, SELF_ASSESS_SYS_PROMPT, model_cfg=self.model_cfg
            )
        except Exception:
            return None
        parsed = result.get("parsed_json") if isinstance(result, dict) else None
        if not isinstance(parsed, dict) or "ok" not in parsed:
            sys.stderr.write(
                "[self-assess] no verdict (parse failed) — blind summary stands\n"
            )
            sys.stderr.flush()
            return None
        verdict = {
            "ok": bool(parsed.get("ok")),
            "summary": str(parsed.get("summary") or ""),
            "problem": str(parsed.get("problem") or ""),
        }
        # Live visibility (→ run.log). ADVISORY: this verdict does NOT gate the
        # milestone (continuation §3 owns that); it is a grounded summary + an
        # optional `problem` hint. So there is no flip-suppression backstop here —
        # an imperfect hint is low-stakes (continuation weighs it against §3),
        # not a regression. "note" marks an advisory concern, not a NOT-DONE.
        flag = "OK" if verdict["ok"] else "note"
        detail = verdict["summary"] if verdict["ok"] else verdict["problem"]
        sys.stderr.write(f"[self-assess] {flag}: {detail[:160]}\n")
        sys.stderr.flush()
        return verdict


# ── Prompt builders ───────────────────────────────────────────────────────────


class CodePlanExecuteSkillSubagent(CodePlanExecuteSubagent):
    """Two-stage executor with per-task skill resolution (impl: code_plan_execute_skill).

    Resolves skills for BOTH inner agents (code_planner + code_executor) once
    per task (cached), swaps the frozen minimal prompts' SKILLS slots, and
    reports the choices in the standard payload.
    """

    planner_skills_source: object | None = None
    executor_skills_source: object | None = None
    planner_skill_instruction: object | None = None
    executor_skill_instruction: object | None = None
    skills_prompt_variant: str = "minimal"

    async def run_subagent(self, subagent_input, ctx):
        task = subagent_input.task_context
        _meta = subagent_input.metadata or {}
        skills_meta: dict = {}
        if (
            self.planner_skills_source is not None
            and self.planner_skill_instruction is not None
        ):
            try:
                resolved = await self.planner_skills_source.resolve(
                    task.task_id,
                    task.instruction,
                    milestone_intent=_meta.get("milestone_intent"),
                    milestone_index=_meta.get("milestone_index"),
                )
            except RateLimitExhausted as exc:
                yield self.failed(
                    attempt=subagent_input.attempt,
                    payload={"llm_raised": str(exc)},
                    failure_code=PROVIDER_RATE_LIMITED,
                )
                return
            self.planner_skill_instruction.current = render_code_planner_instruction(
                variant=self.skills_prompt_variant, skill_texts=resolved.texts
            )
            skills_meta["code_planner"] = {
                "names": resolved.names,
                "source": resolved.source,
            }
        if (
            self.executor_skills_source is not None
            and self.executor_skill_instruction is not None
        ):
            try:
                resolved = await self.executor_skills_source.resolve(
                    task.task_id,
                    task.instruction,
                    milestone_intent=_meta.get("milestone_intent"),
                    milestone_index=_meta.get("milestone_index"),
                )
            except RateLimitExhausted as exc:
                yield self.failed(
                    attempt=subagent_input.attempt,
                    payload={"llm_raised": str(exc)},
                    failure_code=PROVIDER_RATE_LIMITED,
                )
                return
            self.executor_skill_instruction.current = render_executor_instruction(
                variant=self.skills_prompt_variant, skill_texts=resolved.texts
            )
            skills_meta["code_executor"] = {
                "names": resolved.names,
                "source": resolved.source,
            }
        async for envelope in super().run_subagent(subagent_input, ctx):
            if skills_meta and isinstance(envelope.payload, dict):
                envelope.payload["skills"] = skills_meta
            yield envelope


def build_code_plan_execute_skill_subagent(
    name: str = "executor_subagent_code_plan_execute_skill",
    *,
    run_config: "RunConfig | None" = None,
    native: bool = False,
) -> CodePlanExecuteSubagent:
    """Registry impls code_plan_execute_skill / code_plan_execute_skill_native."""
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
    per_milestone = run_config.skills_per_milestone if run_config is not None else False
    planner_thin = render_code_planner_instruction(variant="minimal")
    executor_thin = render_executor_instruction(variant="minimal")

    if native:
        return CodePlanExecuteSubagent(
            name=name,
            description="Two-stage executor, frozen thin prompts + native SkillToolset (model-pull).",
            inner_agent=_build_inner_code_planner_agent(
                model_cfg,
                instruction=planner_thin,
                skills_dir=(root / "code_planner" / library)
                if root is not None
                else None,
            ),
            inner_executor_agent=_build_inner_code_executor_agent(
                model_cfg,
                instruction=executor_thin,
                skills_dir=(root / "code_executor" / library)
                if root is not None
                else None,
            ),
            model_cfg=model_cfg,
            use_llm_code_plan=True,
            use_llm_code_execute=True,
            self_assess=(
                run_config.executor_self_assess if run_config is not None else False
            ),
        )

    planner_holder = MutableInstruction(planner_thin)
    executor_holder = MutableInstruction(executor_thin)
    common = dict(
        model_cfg=model_cfg,
        k=k,
        selector=selector,
        embedding_model=embedding_model,
        per_milestone=per_milestone,
    )
    return CodePlanExecuteSkillSubagent(
        name=name,
        description="Two-stage executor, frozen thin prompts + per-task skill injection.",
        inner_agent=_build_inner_code_planner_agent(
            model_cfg, instruction=planner_holder
        ),
        inner_executor_agent=_build_inner_code_executor_agent(
            model_cfg, instruction=executor_holder
        ),
        model_cfg=model_cfg,
        use_llm_code_plan=True,
        use_llm_code_execute=True,
        self_assess=(
            run_config.executor_self_assess if run_config is not None else False
        ),
        planner_skills_source=(
            SkillsSource(component="code_planner", root=root, library=library, **common)
            if root is not None and library != "none"
            else None
        ),
        executor_skills_source=(
            SkillsSource(
                component="code_executor", root=root, library=library, **common
            )
            if root is not None and library != "none"
            else None
        ),
        planner_skill_instruction=planner_holder,
        executor_skill_instruction=executor_holder,
        skills_prompt_variant=(
            run_config.skills_prompt_variant if run_config is not None else "minimal"
        ),
    )


# ── Builder ───────────────────────────────────────────────────────────────────


def build_code_plan_execute_subagent(
    name: str = "executor_subagent_code_plan_execute",
    *,
    run_config: "RunConfig | None" = None,
    use_llm_code_plan: bool = True,
    use_llm_code_execute: bool = True,
) -> CodePlanExecuteSubagent:
    model_cfg = run_config.model if run_config is not None else model_config_from_env()
    executor_variant = (
        run_config.executor_prompt_variant if run_config is not None else "full"
    )
    executor_skills = list(run_config.executor_skills) if run_config is not None else []
    planner_variant = (
        run_config.code_planner_prompt_variant if run_config is not None else "full"
    )
    planner_skills = (
        list(run_config.code_planner_skills) if run_config is not None else []
    )
    inner_agent = (
        _build_inner_code_planner_agent(
            model_cfg,
            instruction=render_code_planner_instruction(
                variant=planner_variant,
                skill_texts=planner_skills if planner_variant != "full" else None,
            ),
            skills_dir=(
                run_config.code_planner_skills_dir if run_config is not None else None
            ),
        )
        if use_llm_code_plan
        else None
    )
    inner_executor_agent = (
        _build_inner_code_executor_agent(
            model_cfg,
            instruction=render_executor_instruction(
                variant=executor_variant,
                skill_texts=executor_skills if executor_variant != "full" else None,
            ),
            skills_dir=(
                run_config.executor_skills_dir if run_config is not None else None
            ),
        )
        if use_llm_code_execute
        else None
    )
    return CodePlanExecuteSubagent(
        name=name,
        description="M11 code-plan-execute AppWorld executor.",
        inner_agent=inner_agent,
        inner_executor_agent=inner_executor_agent,
        model_cfg=model_cfg,
        use_llm_code_plan=use_llm_code_plan,
        use_llm_code_execute=use_llm_code_execute,
        self_assess=(
            run_config.executor_self_assess if run_config is not None else False
        ),
    )


# ── Shared utilities ──────────────────────────────────────────────────────────


__all__ = [
    "CODE_PLANNER_SYSTEM_PROMPT",
    "CODE_PLANNER_MINIMAL_SYSTEM_PROMPT",
    "CODE_EXECUTOR_SYSTEM_PROMPT",
    "CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT",
    "build_skill_toolset",
    "instruction_text",
    "render_code_planner_instruction",
    "render_executor_instruction",
    "render_executor_skills_section",
    "static_instruction_provider",
    "CodePlanExecuteSkillSubagent",
    "CodePlanExecuteSubagent",
    "build_code_plan_execute_skill_subagent",
    "build_code_plan_execute_subagent",
    "_build_code_plan_prompt",
    "_build_code_execute_prompt",
    "_parse_code_plan",
    "_parse_execute_stdout",
]
