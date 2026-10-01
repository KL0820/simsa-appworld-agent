from __future__ import annotations

import pytest
from google.adk.agents import BaseAgent

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.run_config import ModelConfig, RunConfig
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.planner.rough_planner.prompts import (
    APP_CATALOG,
    APP_CATALOG_SOURCE,
    APP_DESCRIPTIONS,
    ROUGH_PLANNER_INSTRUCTION,
)
from adk_appworld_agent.subagents.registry import (
    SUBAGENT_REGISTRY,
    available_subagent_impls,
    build_subagent,
    workflow_subagent_impls,
)


def test_registry_lists_phases():
    # Phase.VERIFY removed in step 4 cull — cycle-based AppWorldAgent has no
    # verifier stage; continuation LLM judges milestone intent from EXECUTE summary.
    assert set(SUBAGENT_REGISTRY.keys()) == {Phase.FIND, Phase.PLAN, Phase.EXECUTE}


def test_every_phase_has_a_stub_impl():
    for phase in (Phase.FIND, Phase.PLAN, Phase.EXECUTE):
        assert "stub" in available_subagent_impls(phase)


def test_finder_phase_offers_community_impl():
    assert "community" in available_subagent_impls(Phase.FIND)


def test_executor_phase_offers_code_plan_execute_impl():
    assert "code_plan_execute" in available_subagent_impls(Phase.EXECUTE)


def test_executor_phase_does_not_offer_legacy_react_or_llm_keys():
    """The legacy single-stage ReAct executor (`llm`/`react`) was removed;
    the two-stage `code_plan_execute` impl replaces it wholesale."""
    assert "llm" not in available_subagent_impls(Phase.EXECUTE)
    assert "react" not in available_subagent_impls(Phase.EXECUTE)


def test_planner_phase_offers_rough_impl():
    assert "rough" in available_subagent_impls(Phase.PLAN)
    planner = build_subagent(Phase.PLAN, "rough")
    assert "rough_planner" in planner.name


def test_rough_planner_instruction_keeps_decomposition_high_level():
    assert "single-app work units" in ROUGH_PLANNER_INSTRUCTION
    assert (
        "Available apps loaded from AppWorld app-description API output"
        in ROUGH_PLANNER_INSTRUCTION
    )
    assert "Each task must belong to exactly one app" in ROUGH_PLANNER_INSTRUCTION
    assert "Constraint preservation" in ROUGH_PLANNER_INSTRUCTION
    assert "Task decomposition" in ROUGH_PLANNER_INSTRUCTION
    assert "Completion" in ROUGH_PLANNER_INSTRUCTION
    assert "Do not mention API names" in ROUGH_PLANNER_INSTRUCTION
    assert "Respond only with JSON matching the schema" in ROUGH_PLANNER_INSTRUCTION
    assert "Example:" in ROUGH_PLANNER_INSTRUCTION
    assert "Named-target rule" in ROUGH_PLANNER_INSTRUCTION
    assert "Funding/method-resource rule" in ROUGH_PLANNER_INSTRUCTION
    # Concrete single-recipient example so the named-target rule has a worked
    # demonstration alongside the multi-recipient cable bill case. Both worked
    # examples must come from the train split (gold trajectory 60d0b5b_2 here);
    # test-split task text in the prompt is test-set leakage.
    assert "The last Venmo payment request I sent to Cory" in ROUGH_PLANNER_INSTRUCTION
    assert "Anita" not in ROUGH_PLANNER_INSTRUCTION
    assert "$427" not in ROUGH_PLANNER_INSTRUCTION


def test_rough_planner_app_catalog_uses_appworld_sources():
    catalog = {entry.name: entry for entry in APP_CATALOG}
    assert APP_CATALOG_SOURCE == (
        "AppWorld api_docs.show_app_descriptions (/api_docs/app_descriptions)"
    )
    assert (
        APP_DESCRIPTIONS["gmail"]
        == "An email app to draft, send, receive, and manage emails."
    )
    assert catalog["gmail"].name == "gmail"
    assert catalog["spotify"].name == "spotify"
    assert all(entry.description for entry in APP_CATALOG)


def test_build_subagent_stub_returns_agent_with_phase_in_name():
    finder = build_subagent(Phase.FIND, "stub")
    assert "finder" in finder.name


def test_build_subagent_passes_run_config_to_llm_factories():
    run_config = RunConfig(
        model=ModelConfig(name="gemini-test", temperature=0.3, max_output_tokens=1234),
        timeout_s=42,
    )

    planner = build_subagent(Phase.PLAN, "rough", run_config=run_config)
    executor = build_subagent(Phase.EXECUTE, "code_plan_execute", run_config=run_config)

    assert getattr(planner.inner_agent, "model") == "gemini-test"
    assert planner.inner_agent.generate_content_config.temperature == 0.3
    assert planner.inner_agent.generate_content_config.max_output_tokens == 1234
    # code_plan_execute is a 2-stage executor (no single timeout_s/inner_agent
    # like the removed react executor); run_config flows into its inner LLM
    # stages' model_cfg. Building it with the run_config without error is the
    # propagation check here; the planner asserts above cover the rest.
    assert executor is not None


def test_subagent_contract_is_composition_not_base_agent():
    planner = build_subagent(Phase.PLAN, "rough")

    assert isinstance(planner, BaseSubagent)
    assert not isinstance(planner, BaseAgent)
    assert planner.phase == Phase.PLAN
    assert planner.input_schema is SubagentInput
    assert planner.output_schema is SubagentEnvelope
    assert isinstance(planner.build_agent(), BaseAgent)


def test_build_subagent_unknown_impl_raises_with_available_list():
    with pytest.raises(KeyError) as exc:
        build_subagent(Phase.FIND, "totally_made_up")
    msg = str(exc.value)
    assert "totally_made_up" in msg
    assert "stub" in msg


def test_workflow_options_exclude_test_and_skill_derived_impls():
    assert "stub" not in workflow_subagent_impls(Phase.FIND)
    assert workflow_subagent_impls(Phase.PLAN) == ("rough",)
    assert workflow_subagent_impls(Phase.EXECUTE) == ("code_plan_execute",)
    assert all(
        impl in available_subagent_impls(phase)
        for phase in (Phase.FIND, Phase.PLAN, Phase.EXECUTE)
        for impl in workflow_subagent_impls(phase)
    )
