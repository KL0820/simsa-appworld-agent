"""Agent/tool/payload builders for the code_plan_execute executor.

Constructs the inner LlmAgents (planner + executor), execute_python tool
config, skill toolset, and the IO/failure SubagentEnvelope payloads.
Extracted from agent.py in the 2026-06-22 behavior-preserving split."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from adk_appworld_agent.subagents.utils.native_skills import build_skill_toolset

if TYPE_CHECKING:
    from .execution import _CodeExecuteRun, _CodePlanRun

from google.adk.agents import LlmAgent
from google.genai import types

from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.gemini_thinking import thinking_config_from_env
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.executor.appworld_tools import (
    EXECUTE_PYTHON_TOOL,
    SUBMIT_FINAL_TOOL,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompts import (
    CODE_EXECUTOR_SYSTEM_PROMPT,
    CODE_PLANNER_SYSTEM_PROMPT,
    static_instruction_provider,
)


def _build_io_payload(
    *,
    subagent_input: SubagentInput,
    plan_run: _CodePlanRun,
    exec_run: _CodeExecuteRun,
    code_execute: dict,
) -> dict:
    plan = plan_run.plan
    if plan is None:
        raise ValueError("successful code execution requires a parsed code plan")
    return {
        "expected_output": "executor_result, milestone_done, submission_candidate, finalize_called, tool_call_count",
        "input": {
            "subagent_input": subagent_input.model_dump(mode="json"),
            "model_calls": [
                {
                    "step": "code_plan",
                    "model_input_raw": plan_run.model_input_raw,
                },
                {
                    "step": "code_execute",
                    "model_input_raw": exec_run.model_input_raw,
                },
            ],
        },
        "output": {
            "model_calls": [
                {
                    "step": "code_plan",
                    "raw_llm_text": plan_run.raw_text,
                    "parsed_json": plan_run.parsed_json,
                },
                {
                    "step": "code_execute",
                    "raw_llm_text": exec_run.raw_text,
                    "raw_stdout": exec_run.raw_stdout,
                    "parsed_json": exec_run.stdout_json,
                    "code": exec_run.code,
                    "parse_error": exec_run.parse_error,
                    "tool_calls": exec_run.tool_calls,
                    "event_diagnostics": exec_run.event_diagnostics,
                    "repair_attempts": exec_run.repair_attempts,
                },
            ],
            "raw_llm_text": json.dumps(
                {"code_plan": plan_run.parsed_json, "code_execute": code_execute},
                ensure_ascii=False,
            ),
            "parsed_json": {
                "code_plan": plan_run.parsed_json,
                "code_execute": code_execute,
            },
        },
    }


def _build_failure_payload(
    *,
    subagent_input: SubagentInput,
    plan_run: _CodePlanRun,
    milestone_id: str,
    milestone_index: int,
    milestone_total: int,
    t0: float,
) -> tuple[dict, dict, dict]:
    """Return (payload, io, metrics) for the stage-1 failure path.

    Caller is responsible for setting envelope.io / envelope.metrics so the
    domain payload stays free of observability sidecars.
    """
    code_plan_output: dict = {
        "raw_llm_text": plan_run.raw_text,
        "parsed_json": plan_run.parsed_json,
    }
    if plan_run.parse_error:
        code_plan_output["parse_error"] = plan_run.parse_error
    if plan_run.llm_raised:
        code_plan_output["llm_raised"] = plan_run.llm_raised
    io = {
        "expected_output": "executor_result, milestone_done, submission_candidate, finalize_called, tool_call_count",
        "input": {
            "subagent_input": subagent_input.model_dump(mode="json"),
            "model_calls": [
                {"step": "code_plan", "model_input_raw": plan_run.model_input_raw}
            ],
        },
        "output": {
            "model_calls": [{"step": "code_plan", **code_plan_output}],
            "raw_llm_text": plan_run.raw_text,
            "parsed_json": plan_run.parsed_json,
        },
    }
    payload = {
        "final_response": "",
        "tool_call_count": 0,
        "submission_candidate": {
            "answer": "null",
            "task_type_hint": "action",
            "answer_type": "null",
            "source": "code_plan_execute_stage1_failed",
        },
        "executor_result": {
            "answer": "",
            "milestone_done": False,
            "summary": "",
            "variables": [],
        },
        "finalize_called": False,
        "milestone_done": False,
        "milestone_id": milestone_id or None,
        "milestone_index": milestone_index,
        "milestone_total": milestone_total,
        "code_plan": None,
        "code_execute": None,
    }
    metrics = {
        "wall_ms": int((time.monotonic() - t0) * 1000),
        **plan_run.usage,
    }
    return payload, io, metrics


# ── Agent builders ────────────────────────────────────────────────────────────


def _build_inner_code_planner_agent(
    model_cfg: ModelConfig, *, instruction=None, skills_dir=None
) -> LlmAgent:
    # Native skills: output_schema + tools together is supported natively on
    # the Vertex AI variant with Gemini >= 2 (verified on google-adk 1.31.0).
    extra: dict = {}
    if skills_dir is not None:
        toolset = build_skill_toolset(skills_dir)
        if toolset is not None:
            extra["tools"] = [toolset]
    config = types.GenerateContentConfig(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=model_cfg.candidate_count,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
        thinking_config=thinking_config_from_env(),
    )
    return LlmAgent(
        name="code_planner_llm",
        model=model_cfg.name,
        description="LLM that writes a CodePlanOutput for one AppWorld milestone.",
        instruction=(
            instruction
            if callable(instruction)
            else static_instruction_provider(instruction)
            if instruction is not None
            else CODE_PLANNER_SYSTEM_PROMPT
        ),
        output_schema=CodePlanOutput,
        generate_content_config=config,
        **extra,
    )


def _build_inner_code_executor_agent(
    model_cfg: ModelConfig, *, instruction=None, skills_dir=None
) -> LlmAgent:
    # ANY mode forces the model to always emit a function_call.
    # The event loop breaks after a non-empty execute_python result (or after 4 calls),
    # so there is no infinite loop.
    extra_tools: list = []
    allow_load_skill = False
    if skills_dir is not None:
        toolset = build_skill_toolset(skills_dir)
        if toolset is not None:
            extra_tools = [toolset]
            allow_load_skill = True  # tools and whitelist change together
    config = types.GenerateContentConfig(
        temperature=model_cfg.temperature,
        top_p=model_cfg.top_p,
        top_k=model_cfg.top_k,
        candidate_count=model_cfg.candidate_count,
        seed=model_cfg.seed,
        max_output_tokens=model_cfg.max_output_tokens or None,
        tool_config=_execute_python_tool_config(allow_load_skill=allow_load_skill),
        thinking_config=thinking_config_from_env(),
    )
    return LlmAgent(
        name="code_executor_llm",
        model=model_cfg.name,
        description="LLM that writes Python code to execute one AppWorld milestone plan.",
        instruction=(
            instruction
            if callable(instruction)
            else static_instruction_provider(instruction)
            if instruction is not None
            else CODE_EXECUTOR_SYSTEM_PROMPT
        ),
        tools=[EXECUTE_PYTHON_TOOL, SUBMIT_FINAL_TOOL, *extra_tools],
        generate_content_config=config,
    )


def _execute_python_tool_config(*, allow_load_skill: bool = False) -> types.ToolConfig:
    allowed = ["execute_python", "submit_final"]
    if allow_load_skill:
        # Native-skill condition only: load_skill joins the ANY whitelist
        # (list_skills is redundant; run_skill_script stays blocked).
        allowed = [*allowed, "load_skill"]
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY,
            allowed_function_names=allowed,
        )
    )


def _execute_python_tool_config_json() -> dict:
    return _execute_python_tool_config().model_dump(mode="json", by_alias=True)
