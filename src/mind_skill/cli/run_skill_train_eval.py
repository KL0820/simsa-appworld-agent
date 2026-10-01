"""REAL-pipeline run of the 30 train tasks with per-task best skills injected.

For each gold-corpus task: full FSM (plan -> find -> execute -> continue ->
submit), NO gold-trajectory replay; the three trained layers use their FROZEN
minimal deduction prompts with the task's own best SKILL.md deterministically
injected. This is the training-set fit check (does the trained stack solve the
tasks its skills came from, end to end) — held-out eval comes separately.

  PYTHONPATH=src .venv/bin/python src/mind_skill/run_skill_train_eval.py \
      [--tasks all] [--run_tag skill_train30] [--no_skills] [--layers exec,code,rough]

--no_skills: thin prompts WITHOUT skills (the thin-baseline condition).
--layers:    which layers run thin(+skill); others stay deployment-full.

Each task is a scripts/run_task.py subprocess (standard per-task logging under
logs/<experiment>/<ts>/<task_id>/), so /view-build tooling applies as usual.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
PROJECT_ROOT = HERE.parents[3]  # <root> (worktree/repo root; has scripts/ + src/)

LAYER_ALIASES = {
    "exec": "executor",
    "executor": "executor",
    "code": "code_planner",
    "code_planner": "code_planner",
    "rough": "rough_planner",
    "rough_planner": "rough_planner",
}
LAYER_TO_COMPONENT = {
    "executor": "code_executor",
    "code_planner": "code_planner",
    "rough_planner": "rough_planner",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' = the --corpus gold corpus; comma ids; or use --dataset",
    )
    parser.add_argument(
        "--corpus",
        default="_1",
        choices=["_1", "_2"],
        help="which gold corpus '--tasks all' resolves to "
        "(default _1 = current canonical; gold dir holds both)",
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "dev", "test_normal", "test_challenge"],
        default=None,
        help="Tracked project task set; overrides --tasks.",
    )
    parser.add_argument("--variant", choices=["full", "1", "2", "3"], default="full")
    parser.add_argument(
        "--skills",
        default="best",
        help="run_task --skills condition (best/q0/.../thin/native_best/...)",
    )
    parser.add_argument(
        "--skills_root",
        type=Path,
        default=None,
        help="REQUIRED when --skills reads a library (not off/thin): library root "
        "<component>/<library>/<task_id>/<name>/SKILL.md, e.g. "
        "data/mind_skill/skills/skill_libraries. No silent default (the old "
        "src/mind_skill/skills path no longer exists).",
    )
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--experiment_name", default="skill_train30")
    parser.add_argument(
        "--rpc_url",
        default=os.environ.get("MIND_SKILL_RPC_URL", "tcp://127.0.0.1:4243"),
    )
    parser.add_argument(
        "--timeout", type=float, default=2400.0
    )  # match run_batch TASK_TIMEOUT_S
    parser.add_argument(
        "--layers",
        default="exec,code,rough",
        help="comma list of layers to run thin(+skill); others stay deployment-full",
    )
    parser.add_argument(
        "--no_skills",
        action="store_true",
        help="thin prompts WITHOUT skills (thin-baseline condition)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from adk_appworld_agent.task_sets import task_set_path
    from mind_skill.deduction.gold import gold_corpus_task_ids
    from mind_skill.paths import GOLD_TRAJECTORIES_DIR

    layers = sorted(
        {LAYER_ALIASES[x.strip()] for x in args.layers.split(",") if x.strip()}
    )
    if args.dataset:
        task_set_file = task_set_path(args.dataset)
        task_ids = [
            line.split()[0]
            for line in task_set_file.read_text().splitlines()
            if line.strip()
            and not line.startswith("#")
            and (args.variant == "full" or line.split()[0].endswith(f"_{args.variant}"))
        ]
    elif args.tasks.strip().lower() == "all":
        task_ids = gold_corpus_task_ids(GOLD_TRAJECTORIES_DIR, suffix=args.corpus)
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_name = args.run_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = PROJECT_ROOT / "logs"
    run_dir = log_root / args.experiment_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "batch_results.txt"

    rows: list[dict] = []

    def _flush() -> None:
        # Merge with on-disk state: a partial invocation (single-task rerun)
        # must not clobber rows of tasks finished by earlier invocations.
        # de-dup first: in-flight rows win over disk rows added by earlier flushes
        by_id: dict = {}
        for r in rows:
            if r["task_id"] not in by_id or r.get("status") is not None:
                by_id[r["task_id"]] = r
        rows[:] = list(by_id.values())
        current_ids = set(by_id)
        for summary_path in sorted(run_dir.glob("*/artifacts/task_summary.json")):
            tid = summary_path.parent.parent.name
            if tid in current_ids:
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            ev = summary.get("eval") or {}
            rows.append(
                {
                    "task_id": tid,
                    "status": "PASS"
                    if (ev.get("failed") == 0 and ev.get("total"))
                    else "FAIL",
                    "passed": ev.get("passed"),
                    "total": ev.get("total"),
                    "wall_s": float(summary.get("wall_s") or 0.0),
                    "skills": [],
                }
            )
            current_ids.add(tid)
        rows.sort(key=lambda r: r["task_id"])
        done = [r for r in rows if r.get("status") is not None]
        passed = [r for r in done if r["status"] == "PASS"]
        test_pass = sum(r.get("passed") or 0 for r in done)
        test_total = sum(r.get("total") or 0 for r in done)
        lines = [
            "=" * 64,
            f"Skill train-eval — {args.experiment_name}/{run_name}",
            "=" * 64,
            "",
            f"  Updated:  {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Layers thin{'+skill' if not args.no_skills else ' (NO skills)'}: {', '.join(layers)}",
            "  Pipeline: full FSM (plan/find/execute/continue/submit), no replay",
            "",
            "AGGREGATE",
            "---------",
            "",
            f"  Tasks done    {len(done)} / {len(task_ids)}",
        ]
        if done:
            lines.append(
                f"  Tasks passed  {len(passed)} / {len(done)} "
                f"({100.0 * len(passed) / len(done):.1f}%)"
            )
            if test_total:
                lines.append(
                    f"  Test pass     {test_pass}/{test_total} "
                    f"({100.0 * test_pass / test_total:.1f}%)"
                )
        lines += ["", "PER TASK", "--------", ""]
        for r in rows:
            if r.get("status") is None:
                lines.append(f"[{r['task_id']}] pending")
            else:
                counts = (
                    f" {r['passed']}/{r['total']}"
                    if isinstance(r.get("passed"), int) and r.get("total")
                    else ""
                )
                skills = ",".join(r.get("skills") or []) or "none"
                lines.append(
                    f"[{r['task_id']}] {r['status']}{counts} | {r['wall_s']:.1f}s | skills: {skills}"
                )
        results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"tasks={len(task_ids)}  layers={layers}  skills={'OFF' if args.no_skills else 'per-task best'}"
    )
    print(f"Batch results (live-updated): {results_path}")

    # Fail-fast: a skill-reading run must have a real library, or it would
    # silently run skill-less while still reporting "skills: best".
    skills_mode = "thin" if args.no_skills else args.skills
    if skills_mode not in ("off", "thin"):
        if args.skills_root is None:
            print(
                f"ERROR: --skills_root is required when --skills reads a library "
                f"(got --skills {skills_mode}); e.g. data/mind_skill/skills/skill_libraries"
            )
            return 2
        for layer in layers:
            lib = args.skills_root / LAYER_TO_COMPONENT[layer] / skills_mode
            if not lib.is_dir() or not any(lib.glob("*/*/SKILL.md")):
                print(
                    f"ERROR: --skills {skills_mode} requested but no skills under "
                    f"{lib} (check --skills_root / library / that this layer was trained)"
                )
                return 2

    for i, task_id in enumerate(task_ids, 1):
        summary_path = run_dir / task_id / "artifacts" / "task_summary.json"
        if args.resume and summary_path.exists():
            print(f"[{i}/{len(task_ids)}] {task_id} — already done, skip")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            ev = summary.get("eval") or {}
            rows.append(
                {
                    "task_id": task_id,
                    "status": "PASS"
                    if (ev.get("failed") == 0 and ev.get("total"))
                    else "FAIL",
                    "passed": ev.get("passed"),
                    "total": ev.get("total"),
                    "wall_s": float(summary.get("wall_s") or 0.0),
                    "skills": [],
                }
            )
            _flush()
            continue

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_task.py"),
            "--task_id",
            task_id,
            "--experiment_name",
            args.experiment_name,
            "--run_name",
            run_name,
            "--log_root",
            str(log_root),
            "--rpc_url",
            args.rpc_url,
            "--plan",
            "rough",
            "--find",
            "community",
            "--execute",
            "code_plan_execute",
            "--timeout",
            str(args.timeout),
            "--overwrite-task-log",
        ]
        used_skills: list[str] = []
        cmd += ["--skills", skills_mode]
        if skills_mode not in ("off", "thin"):
            cmd += ["--skills_root", str(args.skills_root)]
            used_skills = list(layers)
        if args.layers != "exec,code,rough":
            # layer-subset ablation: explicit per-layer overrides on top
            for layer in ("executor", "code_planner", "rough_planner"):
                if layer not in layers:
                    cmd += [f"--{layer}_prompt_variant", "full"]

        row = {"task_id": task_id, "status": None, "skills": used_skills}
        rows.append(row)
        _flush()
        print(
            f"[{i}/{len(task_ids)}] {task_id} — full pipeline (skills: {','.join(used_skills) or 'none'})",
            flush=True,
        )
        t0 = time.monotonic()
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        row["wall_s"] = time.monotonic() - t0

        passed = failed = total = None
        if summary_path.exists():
            ev = json.loads(summary_path.read_text(encoding="utf-8")).get("eval") or {}
            passed, failed, total = ev.get("passed"), ev.get("failed"), ev.get("total")
        row.update(
            {
                "status": "PASS" if (failed == 0 and total) else "FAIL",
                "passed": passed,
                "total": total,
            }
        )
        if proc.returncode != 0 and total is None:
            row["status"] = "ERROR"
            (run_dir / task_id).mkdir(parents=True, exist_ok=True)
            (run_dir / task_id / "wrapper_stderr.log").write_text(
                proc.stderr[-20000:], encoding="utf-8"
            )
        _flush()
        print(
            f"    {row['status']} {passed}/{total}  ({row['wall_s']:.0f}s)", flush=True
        )

    done = [r for r in rows if r.get("status")]
    n_pass = sum(1 for r in done if r["status"] == "PASS")
    print(f"\nDONE: {n_pass}/{len(done)} tasks passed -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
