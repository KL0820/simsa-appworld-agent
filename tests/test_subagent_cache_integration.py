from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from google.adk.events import Event
from google.genai import types

from adk_appworld_agent.agent import AppWorldAgent
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.cache import (
    CacheLayer,
    CachePolicy,
    JsonlSubagentCacheStore,
)
from adk_appworld_agent.orchestration.cache.spec import CacheSpec, canonical_key
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.orchestrator import Orchestrator
from adk_appworld_agent.orchestration.state import Phase
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.stubs import build_stub_subagent
from tests.helpers import run_agent

_PLANNER_CALL_COUNT: list[int] = [0]


def _planner_key(subagent_input: SubagentInput) -> str:
    ctx = subagent_input.task_context
    return canonical_key(
        {
            "task_id": ctx.task_id,
            "instruction": ctx.instruction,
            "task_datetime": ctx.task_datetime,
        }
    )


COUNTING_PLANNER_CACHE_SPEC = CacheSpec(
    subagent_name="counting_planner",
    version="v1",
    key_fn=_planner_key,
)


class CountingPlannerStub(BaseSubagent):
    phase_: ClassVar[Phase] = Phase.PLAN
    cache_spec_: ClassVar[CacheSpec | None] = COUNTING_PLANNER_CACHE_SPEC

    async def run_subagent(self, subagent_input, ctx):
        _PLANNER_CALL_COUNT[0] += 1
        envelope = self.succeeded(
            attempt=subagent_input.attempt,
            payload={
                "thoughts": "counted",
                "tasks": [{"task": "do x", "app": "file_system"}],
                "task_count": 1,
                "metrics": {},
            },
        )
        yield envelope

    async def _run_async_impl(self, ctx):
        async for envelope in self.run_subagent(
            SubagentInput.model_validate_json(content_to_text(ctx.user_content)), ctx
        ):
            yield Event(
                invocation_id=ctx.invocation_id,
                author=self.name,
                branch=ctx.branch,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=envelope.model_dump_json())],
                ),
            )


def _build_agent(cache_layer: CacheLayer, invocation_source: str) -> AppWorldAgent:
    store = SubagentOutputStore()
    invoker = SubagentRunner(
        subagent_output_store=store,
        cache_layer=cache_layer,
        invocation_source=invocation_source,
    )
    controller = Orchestrator(subagent_output_store=store, subagent_runner=invoker)

    planner = CountingPlannerStub(
        name="counting_planner_stub",
        description="planner that counts its run_subagent invocations",
    )
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)
    return AppWorldAgent(
        name="appworld_controller_agent",
        description="integration test controller",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        executor_subagent=executor,
        sub_agents=[
            planner.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )


@pytest.fixture(autouse=True)
def _reset_call_count():
    _PLANNER_CALL_COUNT[0] = 0
    yield


def test_first_run_writes_cache_and_second_run_short_circuits(tmp_path: Path):
    store = JsonlSubagentCacheStore(tmp_path)
    layer = CacheLayer(
        store=store,
        policy=CachePolicy(read={"counting_planner"}, write={"counting_planner"}),
    )

    first = _build_agent(layer, invocation_source="first")
    run_agent(first)
    assert _PLANNER_CALL_COUNT[0] == 1
    assert (tmp_path / "counting_planner.jsonl").exists()

    second = _build_agent(layer, invocation_source="second")
    session, _ = run_agent(second)
    assert _PLANNER_CALL_COUNT[0] == 1, (
        "second run must hit cache and not re-invoke planner"
    )

    plan_outputs = session.state["subagent_outputs"]["PLAN"]
    # The cold-start (cycle 0) planner output is at index 0; later cycles'
    # continuation planner emits its own Phase.PLAN envelopes onto the tail.
    envelope = SubagentEnvelope.model_validate(plan_outputs[0])
    assert envelope.status == SubagentStatus.SUCCEEDED
    assert envelope.payload["cache_source"] == "first"
    assert any("replayed from first" in w for w in envelope.warnings)


def test_disabling_reads_reruns_subagent(tmp_path: Path):
    write_layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=set(), write={"counting_planner"}),
    )
    run_agent(_build_agent(write_layer, invocation_source="seed"))
    assert _PLANNER_CALL_COUNT[0] == 1

    no_read_layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=set(), write=set()),
    )
    run_agent(_build_agent(no_read_layer, invocation_source="fresh"))
    assert _PLANNER_CALL_COUNT[0] == 2


def test_version_override_causes_cache_miss(tmp_path: Path):
    seed_layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=None, write=None),
    )
    run_agent(_build_agent(seed_layer, invocation_source="v1"))
    assert _PLANNER_CALL_COUNT[0] == 1

    bumped = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=None, write=set(), version_override="v2"),
    )
    run_agent(_build_agent(bumped, invocation_source="v2"))
    assert _PLANNER_CALL_COUNT[0] == 2
