"""Skill registry impls (rough_skill / code_plan_execute_skill ± native)."""

from __future__ import annotations

from pathlib import Path

import pytest

from adk_appworld_agent.orchestration.run_config import ModelConfig, RunConfig
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    CodePlanExecuteSkillSubagent,
    instruction_text,
)
from adk_appworld_agent.subagents.planner.rough_planner import (
    RoughPlannerSkillSubagent,
)
from adk_appworld_agent.subagents.registry import (
    available_subagent_impls,
    build_subagent,
)
from adk_appworld_agent.subagents.utils.skills_source import (
    MutableInstruction,
    SkillsSource,
)

SKILLS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "mind_skill"
    / "skills"
    / "release"
    / "thesis_final"
)

needs_libraries = pytest.mark.skipif(
    not (SKILLS_ROOT / "code_executor" / "best").is_dir(),
    reason="skill libraries not built",
)


def _cfg(**kw) -> RunConfig:
    return RunConfig(model=ModelConfig(), skills_root=SKILLS_ROOT, **kw)


def test_registry_exposes_skill_impls():
    assert {"rough_skill", "rough_skill_native"} <= set(
        available_subagent_impls(Phase.PLAN)
    )
    assert {"code_plan_execute_skill", "code_plan_execute_skill_native"} <= set(
        available_subagent_impls(Phase.EXECUTE)
    )


@needs_libraries
def test_push_impls_carry_sources_and_thin_prompts():
    plan = build_subagent(Phase.PLAN, "rough_skill", run_config=_cfg())
    execute = build_subagent(
        Phase.EXECUTE, "code_plan_execute_skill", run_config=_cfg()
    )
    assert isinstance(plan, RoughPlannerSkillSubagent)
    assert isinstance(execute, CodePlanExecuteSkillSubagent)
    assert plan.skills_source.component == "rough_planner"
    assert execute.planner_skills_source.component == "code_planner"
    assert execute.executor_skills_source.component == "code_executor"
    # thin prompts in the holders, no skills yet
    assert "### SKILLS BEGIN" not in instruction_text(plan.inner_agent.instruction)
    assert instruction_text(execute.inner_executor_agent.instruction).startswith(
        "You are the AppWorld code executor"
    )


@needs_libraries
def test_native_impls_mount_toolsets():
    plan = build_subagent(Phase.PLAN, "rough_skill_native", run_config=_cfg())
    execute = build_subagent(
        Phase.EXECUTE, "code_plan_execute_skill_native", run_config=_cfg()
    )
    assert any(type(t).__name__ == "SkillToolset" for t in plan.inner_agent.tools)
    assert any(
        type(t).__name__ == "SkillToolset" for t in execute.inner_executor_agent.tools
    )
    allowed = execute.inner_executor_agent.generate_content_config.tool_config.function_calling_config.allowed_function_names
    assert "load_skill" in allowed


def test_thin_library_none_disables_sources():
    plan = build_subagent(
        Phase.PLAN, "rough_skill", run_config=_cfg(skills_library="none")
    )
    assert plan.skills_source is None


@needs_libraries
def test_per_task_resolution_swaps_holder():
    import asyncio

    source = SkillsSource(
        component="code_executor",
        root=SKILLS_ROOT,
        library="best",
        model_cfg=ModelConfig(),
    )
    resolved = asyncio.run(
        source.resolve("07b42fd_2", "irrelevant — per-task dir exists")
    )
    assert resolved.source == "per_task" and resolved.texts
    holder = MutableInstruction("base")
    holder.current = "base\n" + resolved.texts[0]
    assert "name:" in holder.text
