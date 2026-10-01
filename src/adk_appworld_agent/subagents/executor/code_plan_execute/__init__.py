"""Two-stage code planner and executor package.

Runtime builders and prompt-selection helpers form the public API. Private
compatibility names are imported from their defining modules so existing
research tests keep working without replacing this package in ``sys.modules``.
"""

from adk_appworld_agent.subagents.utils.native_skills import build_skill_toolset

from ._primitives import (
    EXECUTOR_LLM_STALL_TIMEOUT_ENV as EXECUTOR_LLM_STALL_TIMEOUT_ENV,
)
from ._primitives import _llm_stall_timeout_seconds as _llm_stall_timeout_seconds
from .agent import (
    CodePlanExecuteSkillSubagent,
    CodePlanExecuteSubagent,
    build_code_plan_execute_skill_subagent,
    build_code_plan_execute_subagent,
)
from .builders import (
    _build_inner_code_executor_agent as _build_inner_code_executor_agent,
)
from .execution import (
    _build_code_execute_repair_prompt as _build_code_execute_repair_prompt,
)
from .execution import (
    _clean_code_arg as _clean_code_arg,
)
from .execution import (
    _CodeExecuteRun as _CodeExecuteRun,
)
from .execution import (
    _extract_tool_calls as _extract_tool_calls,
)
from .execution import (
    _no_execute_python_call_error as _no_execute_python_call_error,
)
from .execution import (
    _parse_code_plan as _parse_code_plan,
)
from .execution import (
    _run_code_executor_attempt as _run_code_executor_attempt,
)
from .execution import (
    _should_repair_code_execute_run as _should_repair_code_execute_run,
)
from .prompt_builders import (
    _build_code_execute_prompt as _build_code_execute_prompt,
)
from .prompt_builders import (
    _build_code_plan_prompt as _build_code_plan_prompt,
)
from .prompt_builders import (
    _enrich_candidate_apis as _enrich_candidate_apis,
)
from .prompt_builders import (
    _format_prior_attempts_block as _format_prior_attempts_block,
)
from .prompt_builders import (
    _render_prior_returns as _render_prior_returns,
)
from .prompts import (
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
from .results import _build_executor_result as _build_executor_result
from .results import _parse_execute_stdout as _parse_execute_stdout

__all__ = [
    "CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT",
    "CODE_EXECUTOR_SYSTEM_PROMPT",
    "CODE_PLANNER_MINIMAL_SYSTEM_PROMPT",
    "CODE_PLANNER_SYSTEM_PROMPT",
    "CodePlanExecuteSkillSubagent",
    "CodePlanExecuteSubagent",
    "build_code_plan_execute_skill_subagent",
    "build_code_plan_execute_subagent",
    "build_skill_toolset",
    "instruction_text",
    "render_code_planner_instruction",
    "render_executor_instruction",
    "render_executor_skills_section",
    "static_instruction_provider",
]
