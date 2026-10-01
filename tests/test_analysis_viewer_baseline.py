from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_VIEWER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_analysis_viewer.py"
)


def _load_viewer_module():
    spec = importlib.util.spec_from_file_location(
        "build_analysis_viewer_baseline_under_test", _VIEWER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_analysis_viewer_baseline_under_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


viewer = _load_viewer_module()


def test_default_baseline_uses_config_when_not_self_compare(tmp_path):
    history_path = tmp_path / "baseline_history.txt"

    baselines, source = viewer._resolve_default_baselines(
        ["reference_a", "reference_b"],
        current_exp_name="current_best",
        history_path=history_path,
    )

    assert baselines == ["reference_a", "reference_b"]
    assert source == "config.json"


def test_default_baseline_falls_back_to_previous_history_on_self_compare(tmp_path):
    history_path = tmp_path / "baseline_history.txt"
    history_path.write_text(
        "2026-05-18 promoted: prev=reference_a,reference_b (23P) "
        "→ current_best (29P, +6)\n",
        encoding="utf-8",
    )

    baselines, source = viewer._resolve_default_baselines(
        ["current_best"],
        current_exp_name="current_best",
        history_path=history_path,
    )

    assert baselines == ["reference_a", "reference_b"]
    assert source == "baseline_history.txt fallback"


def test_promote_baseline_appends_history_without_mutating_config(tmp_path):
    """Post-2026-05-19 contract: auto-promotion writes the audit-trail line
    only. `config.json["baseline"]` stays put so the next rebuild doesn't
    silently flip to self-compare. The operator promotes manually when
    they've decided the new high-water is the canonical baseline."""
    config_path = tmp_path / "config.json"
    history_path = tmp_path / "baseline_history.txt"
    config_path.write_text(
        '{"baseline": ["reference_a"], "timeout_blacklist": []}\n',
        encoding="utf-8",
    )

    viewer._promote_baseline(
        config_path,
        history_path,
        previous_baselines=["reference_a"],
        new_baseline="current_best",
        base_pass=23,
        cur_pass=29,
    )

    # config.json is untouched — the registry is read-only from auto-promote.
    assert viewer._read_baseline_config(config_path) == ["reference_a"]

    # Future runs therefore still compare against the unchanged config value
    # until the operator deliberately swaps it.
    future_baselines, future_source = viewer._resolve_default_baselines(
        ["reference_a"],
        current_exp_name="future_candidate",
        history_path=history_path,
    )
    assert future_baselines == ["reference_a"]
    assert future_source == "config.json"

    # The fallback path remains live for the rare case where an operator
    # later sets config to the same run being rebuilt.
    rebuild_baselines, rebuild_source = viewer._resolve_default_baselines(
        ["current_best"],
        current_exp_name="current_best",
        history_path=history_path,
    )
    assert rebuild_baselines == ["reference_a"]
    assert rebuild_source == "baseline_history.txt fallback"

    written = history_path.read_text(encoding="utf-8")
    assert "prev=reference_a (23P)" in written
    assert "→ current_best (29P, +6)" in written
