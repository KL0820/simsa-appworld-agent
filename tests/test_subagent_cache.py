from __future__ import annotations

from pathlib import Path

import pytest

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
from adk_appworld_agent.orchestration.state import Phase, TaskContext
from adk_appworld_agent.subagents.finder.community_finder import (
    COMMUNITY_FINDER_CACHE_SPEC,
    CommunityFinderSubagent,
)
from adk_appworld_agent.subagents.planner.rough_planner import (
    ROUGH_PLANNER_CACHE_SPEC,
    RoughPlannerSubagent,
)


def _plan_input(
    task_id="t1",
    instruction="Pay my phone bill.",
    datetime_="2024-01-01T00:00:00",
    attempt=1,
):
    return SubagentInput(
        phase=Phase.PLAN,
        attempt=attempt,
        task_context=TaskContext(
            task_id=task_id, instruction=instruction, task_datetime=datetime_
        ),
        metadata={"capture_io": True},
    )


def _find_input(intent="Find the amount", apps=("phone",)):
    return SubagentInput(
        phase=Phase.FIND,
        attempt=1,
        task_context=TaskContext(
            task_id="t1", instruction="Pay bill.", task_datetime="2024-01-01"
        ),
        metadata={
            "milestone_intent": intent,
            "planned_apps": list(apps),
        },
    )


def _plan_envelope(attempt=1):
    return SubagentEnvelope(
        phase=Phase.PLAN,
        subagent_name="rough_planner_subagent",
        attempt=attempt,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "thoughts": "use phone",
            "tasks": [{"task": "open bill", "app": "phone"}],
            "task_count": 1,
        },
        metrics={"llm_calls": 2, "total_tokens": 50, "wall_ms": 1000},
    )


def test_subagent_classes_expose_cache_spec():
    assert RoughPlannerSubagent.cache_spec_ is ROUGH_PLANNER_CACHE_SPEC
    assert CommunityFinderSubagent.cache_spec_ is COMMUNITY_FINDER_CACHE_SPEC


def test_jsonl_store_roundtrip(tmp_path: Path):
    store = JsonlSubagentCacheStore(tmp_path)
    env = _plan_envelope()
    store.put("rough_planner", "k1", "v1", env, meta={"source": "unit"})

    hit = store.get("rough_planner", "k1", "v1")
    assert hit is not None
    roundtrip, meta = hit
    assert roundtrip.model_dump() == env.model_dump()
    assert meta["source"] == "unit"

    assert (tmp_path / "rough_planner.jsonl").exists()

    reopened = JsonlSubagentCacheStore(tmp_path).get("rough_planner", "k1", "v1")
    assert reopened is not None


def test_rough_planner_key_is_stable_and_sensitive(tmp_path):
    key_a = ROUGH_PLANNER_CACHE_SPEC.key_fn(_plan_input(attempt=1))
    key_b = ROUGH_PLANNER_CACHE_SPEC.key_fn(_plan_input(attempt=5))
    assert key_a == key_b, "attempt must not affect the cache key"

    key_other = ROUGH_PLANNER_CACHE_SPEC.key_fn(_plan_input(datetime_="2025-06-01"))
    assert key_other != key_a


def test_cache_layer_hit_adapts_replay(tmp_path: Path):
    layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read={"rough_planner"}, write={"rough_planner"}),
    )
    wi = _plan_input()
    layer.save(ROUGH_PLANNER_CACHE_SPEC, wi, _plan_envelope(), source="run-A")

    retry = _plan_input(attempt=3)
    hit = layer.lookup(ROUGH_PLANNER_CACHE_SPEC, retry)
    assert hit is not None
    replayed = layer.adapt_replay(ROUGH_PLANNER_CACHE_SPEC, hit, retry)

    assert replayed.attempt == 3
    assert "replayed from run-A" in replayed.warnings
    assert replayed.payload["cache_source"] == "run-A"
    # io now lives on envelope.io, not payload["io"], so the domain payload
    # stays free of observability sidecars (see SubagentEnvelope contract).
    assert replayed.io
    assert replayed.io["output"]["parsed_plan"]["tasks"][0]["app"] == "phone"
    assert replayed.metrics["replayed"] is True
    assert replayed.metrics["llm_calls"] == 0
    assert replayed.metrics["total_tokens"] == 0
    assert "metrics" not in replayed.payload


@pytest.mark.parametrize(
    "policy, expect_hit",
    [
        (CachePolicy(read=set(), write=set()), False),
        (CachePolicy(read={"community_finder"}, write=set()), False),
        (CachePolicy(read={"rough_planner"}, write=set()), True),
        (CachePolicy(read=None, write=set()), True),
    ],
)
def test_policy_controls_reads(tmp_path: Path, policy: CachePolicy, expect_hit: bool):
    writer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=set(), write={"rough_planner"}),
    )
    wi = _plan_input()
    writer.save(ROUGH_PLANNER_CACHE_SPEC, wi, _plan_envelope(), source="seed")

    reader = CacheLayer(store=JsonlSubagentCacheStore(tmp_path), policy=policy)
    hit = reader.lookup(ROUGH_PLANNER_CACHE_SPEC, wi)
    assert (hit is not None) == expect_hit


def test_policy_blocks_writes(tmp_path: Path):
    layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=None, write=set()),
    )
    layer.save(ROUGH_PLANNER_CACHE_SPEC, _plan_input(), _plan_envelope(), source="x")
    assert not (tmp_path / "rough_planner.jsonl").exists()


def test_community_finder_key_uses_milestone_and_apps():
    base = COMMUNITY_FINDER_CACHE_SPEC.key_fn(_find_input())
    same = COMMUNITY_FINDER_CACHE_SPEC.key_fn(_find_input())
    assert base == same

    different_intent = COMMUNITY_FINDER_CACHE_SPEC.key_fn(
        _find_input(intent="different")
    )
    assert base != different_intent

    different_apps = COMMUNITY_FINDER_CACHE_SPEC.key_fn(_find_input(apps=("gmail",)))
    assert base != different_apps

    order_insensitive = COMMUNITY_FINDER_CACHE_SPEC.key_fn(
        _find_input(apps=("phone", "gmail"))
    )
    reversed_order = COMMUNITY_FINDER_CACHE_SPEC.key_fn(
        _find_input(apps=("gmail", "phone"))
    )
    assert order_insensitive == reversed_order


def test_version_override_invalidates_existing_entries(tmp_path: Path):
    store = JsonlSubagentCacheStore(tmp_path)
    writer = CacheLayer(store=store, policy=CachePolicy(read=None, write=None))
    wi = _plan_input()
    writer.save(ROUGH_PLANNER_CACHE_SPEC, wi, _plan_envelope(), source="v1")

    reader_override = CacheLayer(
        store=store,
        policy=CachePolicy(read=None, write=set(), version_override="bump"),
    )
    assert reader_override.lookup(ROUGH_PLANNER_CACHE_SPEC, wi) is None

    reader_default = CacheLayer(store=store, policy=CachePolicy(read=None, write=set()))
    assert reader_default.lookup(ROUGH_PLANNER_CACHE_SPEC, wi) is not None


def test_replayable_filter_rejects_failed_envelopes(tmp_path: Path):
    layer = CacheLayer(
        store=JsonlSubagentCacheStore(tmp_path),
        policy=CachePolicy(read=None, write={"rough_planner"}),
    )
    failed = SubagentEnvelope(
        phase=Phase.PLAN,
        subagent_name="rough_planner_subagent",
        attempt=1,
        status=SubagentStatus.FAILED,
        payload={},
        failure_code="SUBAGENT_OUTPUT_MISSING",
    )
    layer.save(ROUGH_PLANNER_CACHE_SPEC, _plan_input(), failed, source="bad")
    assert not (tmp_path / "rough_planner.jsonl").exists()
