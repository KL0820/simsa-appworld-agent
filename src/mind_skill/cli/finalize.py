"""Finalize a training run into the canonical MANIFEST layout.

Training writes everything (incl. run scratch) to logs/mind_skill/<comp>/<run_tag>/.
The keep-list is copied into data/mind_skill/training_runs/<comp>/<run_tag>/ so that
sandbox_api_calls_*.jsonl and the resume-only meta.json never reach the preserved
record. Copying is DYNAMIC/incremental: `finalize_task` copies one task the moment it
finishes (so data/ fills in live and survives a mid-run crash); `finalize_run_level`
adds the run-level files + run_manifest.json once, at the end. `finalize_run` does
both for the whole dir (standalone / tests). `publish_induced` copies the assembled
per-q libraries to skills/induced/<comp>/<run_tag>/{q0,q1,q2,best}/ (run at the
component's end, after the whole-library name-dedup).

Roots are injectable (default = the paths.py canonical dirs) so the copy logic is unit
testable without touching real data/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from mind_skill.paths import INDUCED_DIR, TRAINING_RUNS_DIR

# MANIFEST retention (per q and per run). Anything not here stays in the logs/ scratch.
KEEP_PER_Q = (
    "induction_prompt.txt",
    "skill.md",
    "source_component_trajectory.json",
    "reconstructed_trajectory.json",
    "deduction.json",
    "judgments.json",
    "gradient.txt",
)
KEEP_RUN = ("batch_summary.json", "journal.jsonl")  # + batch_results*.txt (globbed)


def _run_dir(component: str, run_tag: str, training_runs_dir: Path | None) -> Path:
    dst = Path(training_runs_dir or TRAINING_RUNS_DIR) / component / run_tag
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def finalize_task(
    task_dir: Path,
    *,
    component: str,
    run_tag: str,
    training_runs_dir: Path | None = None,
) -> str:
    """Copy ONE finished task's keep-list into training_runs/<comp>/<run_tag>/<task>/.
    Incremental + idempotent (overwrites) -> call the moment a task completes."""
    src = Path(task_dir)
    tdst = _run_dir(component, run_tag, training_runs_dir) / src.name
    tdst.mkdir(parents=True, exist_ok=True)
    if (src / "result.json").exists():
        shutil.copyfile(src / "result.json", tdst / "result.json")
    for qd in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("q")):
        qdst = tdst / qd.name
        qdst.mkdir(parents=True, exist_ok=True)
        for f in KEEP_PER_Q:
            if (qd / f).exists():
                shutil.copyfile(qd / f, qdst / f)
    return src.name


def finalize_run_level(
    out_dir: Path,
    *,
    component: str,
    run_tag: str,
    tasks: list[str],
    training_runs_dir: Path | None = None,
    manifest_extra: dict | None = None,
) -> dict:
    """Copy the run-level keep-list (batch_summary/journal/batch_results*) and write
    run_manifest.json. Per-task dirs are assumed already copied via finalize_task."""
    src = Path(out_dir)
    dst = _run_dir(component, run_tag, training_runs_dir)
    for f in KEEP_RUN:
        if (src / f).exists():
            shutil.copyfile(src / f, dst / f)
    for br in sorted(src.glob("batch_results*.txt")):
        shutil.copyfile(br, dst / br.name)

    manifest = {
        "component": component,
        "run_tag": run_tag,
        "n_tasks": len(tasks),
        "tasks": list(tasks),
        "kept_per_q": list(KEEP_PER_Q),
        "kept_run": list(KEEP_RUN)
        + ["batch_results*.txt", "result.json", "run_manifest.json"],
    }
    if manifest_extra:
        manifest.update(manifest_extra)
    (dst / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"training_run": str(dst), "n_tasks": len(tasks)}


def finalize_run(
    out_dir: Path,
    *,
    component: str,
    run_tag: str,
    training_runs_dir: Path | None = None,
    manifest_extra: dict | None = None,
) -> dict:
    """Copy every task's keep-list + the run-level files (standalone / tests). The live
    pipeline instead calls finalize_task per task + finalize_run_level once."""
    src = Path(out_dir)
    tasks = [
        finalize_task(
            rp.parent,
            component=component,
            run_tag=run_tag,
            training_runs_dir=training_runs_dir,
        )
        for rp in sorted(src.glob("*/result.json"))
    ]
    return finalize_run_level(
        out_dir,
        component=component,
        run_tag=run_tag,
        tasks=tasks,
        training_runs_dir=training_runs_dir,
        manifest_extra=manifest_extra,
    )


def publish_induced(
    skills_root: Path,
    *,
    component: str,
    run_tag: str,
    induced_dir: Path | None = None,
) -> dict:
    """Copy the assembled per-q + best libraries (skills_root/<comp>/{q*,best}) into
    the canonical skills/induced/<comp>/<run_tag>/. Non-destructive to skills_root."""
    src = Path(skills_root) / component
    dst = Path(induced_dir or INDUCED_DIR) / component / run_tag
    if dst.exists():
        shutil.rmtree(dst)
    copied: dict[str, int] = {}
    if src.is_dir():
        for lib in sorted(
            p
            for p in src.iterdir()
            if p.is_dir() and (p.name.startswith("q") or p.name == "best")
        ):
            shutil.copytree(lib, dst / lib.name)
            copied[lib.name] = sum(1 for _ in (dst / lib.name).rglob("SKILL.md"))
    return {"induced": str(dst), "libraries": copied}


__all__ = [
    "KEEP_PER_Q",
    "KEEP_RUN",
    "finalize_task",
    "finalize_run_level",
    "finalize_run",
    "publish_induced",
]
