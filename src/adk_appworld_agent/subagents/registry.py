from __future__ import annotations

from collections.abc import Callable

from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import Subagent
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    build_code_plan_execute_skill_subagent,
    build_code_plan_execute_subagent,
)
from adk_appworld_agent.subagents.finder.ablation_finders import (
    build_app_scope_finder_subagent,
    build_app_scope_nodep_finder_subagent,
    build_community_forward_finder_subagent,
    build_community_nodep_finder_subagent,
    build_community_noguide_finder_subagent,
    build_global_seed_filter_finder_subagent,
)
from adk_appworld_agent.subagents.finder.community_finder import (
    build_community_finder_subagent,
)
from adk_appworld_agent.subagents.planner import build_rough_planner_subagent
from adk_appworld_agent.subagents.planner.rough_planner import (
    build_rough_planner_skill_subagent,
)
from adk_appworld_agent.subagents.stubs import build_stub_subagent

SubagentFactory = Callable[[RunConfig | None], Subagent]


def _stub_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)


def _stub_planner(run_config: RunConfig | None = None) -> Subagent:
    return build_stub_subagent(name="planner_subagent_stub", phase=Phase.PLAN)


def _stub_executor(run_config: RunConfig | None = None) -> Subagent:
    return build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)


def _community_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_community_finder_subagent(run_config=run_config)


# Retrieval-ablation finder variants (see finding/ablation_finders.py).
def _app_scope_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_app_scope_finder_subagent(run_config=run_config)


def _app_scope_nodep_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_app_scope_nodep_finder_subagent(run_config=run_config)


def _community_nodep_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_community_nodep_finder_subagent(run_config=run_config)


def _global_seed_filter_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_global_seed_filter_finder_subagent(run_config=run_config)


def _community_noguide_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_community_noguide_finder_subagent(run_config=run_config)


def _community_forward_finder(run_config: RunConfig | None = None) -> Subagent:
    return build_community_forward_finder_subagent(run_config=run_config)


def _rough_planner(run_config: RunConfig | None = None) -> Subagent:
    return build_rough_planner_subagent(run_config=run_config)


def _code_plan_execute_executor(run_config: RunConfig | None = None) -> Subagent:
    return build_code_plan_execute_subagent(run_config=run_config)


# MIND-Skill impls: frozen thin prompts + skill libraries (RunConfig.skills_*).
def _rough_skill_planner(run_config: RunConfig | None = None) -> Subagent:
    return build_rough_planner_skill_subagent(run_config=run_config)


def _rough_skill_native_planner(run_config: RunConfig | None = None) -> Subagent:
    return build_rough_planner_skill_subagent(run_config=run_config, native=True)


def _code_plan_execute_skill_executor(run_config: RunConfig | None = None) -> Subagent:
    return build_code_plan_execute_skill_subagent(run_config=run_config)


def _code_plan_execute_skill_native_executor(
    run_config: RunConfig | None = None,
) -> Subagent:
    return build_code_plan_execute_skill_subagent(run_config=run_config, native=True)


SUBAGENT_REGISTRY: dict[Phase, dict[str, SubagentFactory]] = {
    Phase.FIND: {
        "stub": _stub_finder,
        "community": _community_finder,
        # Retrieval ablation (2x2): see finding/ablation_finders.py
        "app_scope": _app_scope_finder,  # B:  no community, keep dep
        "app_scope_nodep": _app_scope_nodep_finder,  # B2: pure app->api
        "community_nodep": _community_nodep_finder,  # A-no-dep: community, no dep
        "global_seed_filter": _global_seed_filter_finder,  # no narrowing: seed_filter over all 454
        "community_noguide": _community_noguide_finder,  # community but guidance (app_context+behavior) off
        "community_forward": _community_forward_finder,  # production community + forward dependency expansion
    },
    Phase.PLAN: {
        "stub": _stub_planner,
        "rough": _rough_planner,
        # MIND-Skill: thin prompt + per-task skill (push) / native load_skill (pull)
        "rough_skill": _rough_skill_planner,
        "rough_skill_native": _rough_skill_native_planner,
    },
    Phase.EXECUTE: {
        "stub": _stub_executor,
        "code_plan_execute": _code_plan_execute_executor,
        # MIND-Skill: thin prompts + per-task skills (push) / native load_skill (pull)
        "code_plan_execute_skill": _code_plan_execute_skill_executor,
        "code_plan_execute_skill_native": _code_plan_execute_skill_native_executor,
    },
}

WORKFLOW_IMPL_OPTIONS: dict[Phase, tuple[str, ...]] = {
    # Test-only stubs and skill-derived implementations are intentionally
    # excluded. Skill mode maps the base planner/executor to the matching
    # push/native implementation.
    Phase.FIND: (
        "community",
        "community_forward",
        "community_nodep",
        "community_noguide",
        "app_scope",
        "app_scope_nodep",
        "global_seed_filter",
    ),
    Phase.PLAN: ("rough",),
    Phase.EXECUTE: ("code_plan_execute",),
}


def available_subagent_impls(phase: Phase) -> list[str]:
    return sorted(SUBAGENT_REGISTRY[phase].keys())


def workflow_subagent_impls(phase: Phase) -> tuple[str, ...]:
    """Return user-selectable base implementations for complete workflows."""
    return WORKFLOW_IMPL_OPTIONS[phase]


def build_subagent(
    phase: Phase,
    impl: str,
    *,
    run_config: RunConfig | None = None,
) -> Subagent:
    factories = SUBAGENT_REGISTRY[phase]
    if impl not in factories:
        raise KeyError(
            f"No subagent impl '{impl}' for phase {phase.value}. "
            f"Available: {available_subagent_impls(phase)}"
        )
    return factories[impl](run_config)


__all__ = [
    "SubagentFactory",
    "SUBAGENT_REGISTRY",
    "WORKFLOW_IMPL_OPTIONS",
    "available_subagent_impls",
    "build_subagent",
    "workflow_subagent_impls",
]
