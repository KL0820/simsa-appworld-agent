from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from adk_appworld_agent.orchestration.run_config import (
    RunConfig,
    dump_resolved_config,
    load_run_config,
)
from adk_appworld_agent.orchestration.state import Phase


def test_retry_defaults_use_one_shared_provider_ladder():
    c = RunConfig()
    assert c.retry.provider_backoff_delays == (4, 8, 16, 32, 64)
    assert c.retry.executor_stall_backoff == (16, 32, 64, 128)
    assert c.retry.finder_max_retries == 5
    assert c.retry.finder_base_delay_s == 1.0
    assert c.retry.finder_max_delay_s == 60.0
    assert c.retry.finder_jitter_ratio == 0.25
    # unified gemini-call max-wait (finder deadline + executor stall timer),
    # default 150 (above the ~125s thinking-hang).
    assert c.retry.llm_call_timeout_s == 150.0
    assert c.executor_timeout_s == 240.0
    assert c.retrieval.dependency_index_file == "api_dependency_index.json"
    assert c.retrieval.batched_dependency is True
    assert c.retrieval.forward_dependency_expansion is False
    assert c.debug.rough_planner_debug is False


def test_removed_duplicate_transient_ladder_is_rejected():
    with pytest.raises(ValidationError):
        RunConfig(
            retry={
                "provider_backoff_delays": (4, 8, 16),
                "transient_delays": (16, 32, 64),
            }
        )


def test_config_file_round_trips(tmp_path):
    """One experiment config file -> RunConfig -> file is lossless, incl.
    enum-keyed impls, tuple ladders, and Path fields."""
    original = RunConfig(
        impls={
            Phase.FIND: "community",
            Phase.PLAN: "rough",
            Phase.EXECUTE: "code_plan_execute",
        },
        retrieval={
            "dependency_index_file": "api_dependency_index_llm_only.json",
            "batched_dependency": False,
        },
        retry={"provider_backoff_delays": (1, 2, 3)},
    )
    path = tmp_path / "experiment.json"
    path.write_text(original.to_json(), encoding="utf-8")

    loaded = load_run_config(path)

    assert loaded == original
    assert loaded.impls[Phase.FIND] == "community"
    assert (
        loaded.retrieval.dependency_index_file == "api_dependency_index_llm_only.json"
    )
    assert loaded.retrieval.batched_dependency is False
    assert loaded.retry.provider_backoff_delays == (1, 2, 3)


def test_dump_resolved_config_writes_full_settings(tmp_path):
    """Every run records its exact settings on disk (no un-logged env)."""
    c = RunConfig(retrieval={"embedding": True})
    out = dump_resolved_config(c, tmp_path)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["retrieval"]["embedding"] is True
    assert data["retry"]["provider_backoff_delays"] == [4, 8, 16, 32, 64]


def test_dependency_index_file_is_config_selectable():
    """The dependency graph is chosen by config (hybrid vs pure-LLM), not a
    per-worktree env — and switching reloads (cache keyed by resolved path)."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RetrievalConfig
    from adk_appworld_agent.subagents.finder.routing import (
        _api_dependency_index_path,
        _load_api_dependency_index,
    )

    try:
        set_active_config(RunConfig())  # default -> hybrid graph
        assert _api_dependency_index_path().name == "api_dependency_index.json"
        hybrid = _load_api_dependency_index()

        set_active_config(
            RunConfig(
                retrieval=RetrievalConfig(
                    dependency_index_file="api_dependency_index_llm_only.json"
                )
            )
        )
        assert _api_dependency_index_path().name == "api_dependency_index_llm_only.json"
        llm_only = _load_api_dependency_index()

        # Different graphs -> different consumer sets (hybrid 291 vs LLM 316).
        assert hybrid.keys() != llm_only.keys()
        assert len(hybrid) > 0 and len(llm_only) > 0
    finally:
        set_active_config(None)


def test_unknown_key_is_rejected(tmp_path):
    """A typo in the experiment config file must fail loudly, not be ignored."""
    path = tmp_path / "bad.json"
    path.write_text('{"retrieval": {"batchd_dependency": true}}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_run_config(path)
