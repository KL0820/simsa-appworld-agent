"""Finalize step: keep-list copy logs -> training_runs (drop scratch) + run_manifest,
and publish per-q libraries -> induced/. No LLM/RPC — pure file logic."""

from __future__ import annotations

import json

from mind_skill.cli.finalize import (
    KEEP_PER_Q,
    finalize_run,
    finalize_run_level,
    finalize_task,
    publish_induced,
)


def _w(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _fake_run(out_dir, tasks=("t1", "t2"), qs=("q0", "q1")):
    for t in tasks:
        for q in qs:
            qd = out_dir / t / q
            for f in KEEP_PER_Q:
                _w(qd / f, f"{f} of {t}/{q}")
            # run scratch that MUST NOT be kept
            _w(qd / "sandbox_api_calls_deadbeef.jsonl", "{}")
            _w(qd / "meta.json", "{}")
        _w(
            out_dir / t / "result.json",
            json.dumps({"task_id": t, "best_q": 0, "iterations": []}),
        )
    # run-level
    _w(out_dir / "batch_summary.json", "{}")
    _w(out_dir / "journal.jsonl", "{}")
    _w(out_dir / "batch_results.txt", "overall")
    _w(out_dir / "batch_results_q0.txt", "q0")


def _fake_skills_root(root, comp, tasks=("t1", "t2"), libs=("q0", "q1", "best")):
    for lib in libs:
        for t in tasks:
            _w(
                root / comp / lib / t / f"skill-{t}" / "SKILL.md",
                f"---\nname: skill-{t}\n---\nbody",
            )
    _w(root / comp / "assembly_summary.json", "{}")  # must NOT be published to induced


def test_finalize_run_keeps_only_manifest_list(tmp_path):
    out_dir = tmp_path / "logs"
    tr_dir = tmp_path / "training_runs"
    _fake_run(out_dir)

    res = finalize_run(
        out_dir,
        component="code_executor",
        run_tag="RT1",
        training_runs_dir=tr_dir,
        manifest_extra={"corpus": "explicit", "q_iterations": 2},
    )
    run = tr_dir / "code_executor" / "RT1"
    assert res["n_tasks"] == 2 and (run).is_dir()

    for t in ("t1", "t2"):
        assert (run / t / "result.json").exists()
        for q in ("q0", "q1"):
            for f in KEEP_PER_Q:
                assert (run / t / q / f).exists(), f"{f} should be kept"
            # scratch dropped
            assert not list((run / t / q).glob("sandbox_api_calls*"))
            assert not (run / t / q / "meta.json").exists()
    # run-level kept
    for f in (
        "batch_summary.json",
        "journal.jsonl",
        "batch_results.txt",
        "batch_results_q0.txt",
    ):
        assert (run / f).exists()
    # run_manifest
    man = json.loads((run / "run_manifest.json").read_text())
    assert man["component"] == "code_executor" and man["n_tasks"] == 2
    assert sorted(man["tasks"]) == ["t1", "t2"] and man["corpus"] == "explicit"


def test_finalize_task_is_incremental(tmp_path):
    """Each task copies the moment it finishes: after one finalize_task only that task
    is in training_runs; after the second, both are — no batch barrier."""
    out_dir = tmp_path / "logs"
    tr_dir = tmp_path / "training_runs"
    _fake_run(out_dir, tasks=("t1", "t2"))
    run = tr_dir / "code_executor" / "RT1"

    finalize_task(
        out_dir / "t1",
        component="code_executor",
        run_tag="RT1",
        training_runs_dir=tr_dir,
    )
    assert (run / "t1" / "result.json").exists()
    assert not (run / "t2").exists()  # t2 not copied yet
    for f in KEEP_PER_Q:
        assert (run / "t1" / "q0" / f).exists()
    assert not list((run / "t1" / "q0").glob("sandbox_api_calls*"))  # scratch dropped

    finalize_task(
        out_dir / "t2",
        component="code_executor",
        run_tag="RT1",
        training_runs_dir=tr_dir,
    )
    assert (run / "t2" / "result.json").exists()

    # run-level finalize adds manifest + run files without re-copying tasks
    finalize_run_level(
        out_dir,
        component="code_executor",
        run_tag="RT1",
        tasks=["t1", "t2"],
        training_runs_dir=tr_dir,
    )
    man = json.loads((run / "run_manifest.json").read_text())
    assert man["n_tasks"] == 2 and (run / "batch_summary.json").exists()


def test_publish_induced_copies_libraries(tmp_path):
    sr = tmp_path / "skill_libraries"
    ind = tmp_path / "induced"
    _fake_skills_root(sr, "code_executor")

    res = publish_induced(sr, component="code_executor", run_tag="RT1", induced_dir=ind)
    base = ind / "code_executor" / "RT1"
    assert res["libraries"] == {"best": 2, "q0": 2, "q1": 2}
    for lib in ("q0", "q1", "best"):
        assert len(list((base / lib).rglob("SKILL.md"))) == 2
    # assembly_summary.json is not a library dir -> not published
    assert not (base / "assembly_summary.json").exists()
