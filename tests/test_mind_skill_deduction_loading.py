"""Offline integrity tests for the deduction harness gold loaders (no LLM/RPC)."""

from __future__ import annotations

import pytest

from mind_skill.deduction.code_executor import load_gold_task
from mind_skill.deduction.gold import gold_corpus_task_ids, pick_gold_trajectory
from mind_skill.loop import load_executor_slice
from mind_skill.paths import GOLD_TRAJECTORIES_DIR
from mind_skill.trajectory.render import render_executor_trajectory

RUNS = GOLD_TRAJECTORIES_DIR

pytestmark = pytest.mark.skipif(
    not RUNS.exists(), reason="gold runs corpus not present"
)


def test_corpus_has_thirty_cleared_tasks():
    assert len(gold_corpus_task_ids(RUNS)) == 30


@pytest.mark.parametrize("task_id", gold_corpus_task_ids(RUNS) if RUNS.exists() else [])
def test_load_gold_task_yields_full_teacher_forcing_context(task_id: str):
    gold = load_gold_task(RUNS, task_id)
    assert gold.instruction
    assert gold.milestones
    for gm in gold.milestones:
        assert gm.intent, f"{task_id} m{gm.index}: empty intent"
        assert gm.plan.plan_steps, f"{task_id} m{gm.index}: empty plan"
        # candidate_apis CAN be empty (c901732_2's gold run had none and still
        # passed) — teacher-forcing replays whatever the gold run received.
        for spec in gm.candidate_apis:
            assert isinstance(spec, dict) and ("api_name" in spec or "name" in spec), (
                f"{task_id} m{gm.index}: candidate spec shape unexpected: {list(spec)[:6]}"
            )


def test_pick_gold_prefers_blind_then_openbook_then_blind_fwd():
    path = pick_gold_trajectory(RUNS, "e3d6c94_2")
    assert path.parent.name == "blind_fwd"


def test_render_executor_trajectory_smoke():
    text = render_executor_trajectory(
        [
            {
                "milestone": "read the things",
                "apis": ["app.list_things"],
                "code_plan": {"plan_steps": ["call the api"]},
                "code": "x = 1",
                "result_summary": "ok",
                "api_calls": ["app.list_things:ok"],
                "turns": 2,
                "repair_count": 0,
            }
        ]
    )
    assert "--- milestone 0: read the things" in text
    assert "```python" in text and "api calls:" in text


def test_executor_slice_is_derived_without_sidecar(tmp_path):
    import shutil

    task_id = gold_corpus_task_ids(RUNS)[0]
    trajectory = pick_gold_trajectory(RUNS, task_id)
    task_dir = tmp_path / task_id
    task_dir.mkdir()
    shutil.copyfile(trajectory, task_dir / "trajectory.json")
    sidecar = task_dir / "induction_slices.json"
    assert not sidecar.exists()

    derived = load_executor_slice(tmp_path, task_id)

    assert derived["task_id"] == task_id
    assert derived["instruction"]
    assert not sidecar.exists()
