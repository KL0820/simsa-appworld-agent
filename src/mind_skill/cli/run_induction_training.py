"""Run the MIND-Skill closed-loop induction training over gold trajectories.

Prereqs:
  - DEDICATED AppWorld RPC server on 4243 (NOT the ablation server on 4242):
      cd ../appworld && RPC_PORT=4243 .venv/bin/python rpc/server.py
  - GOOGLE_API_KEY in the repo .env (loaded below).

Usage (from the repo/worktree root, with PYTHONPATH=<root>/src):
  # --skills_root is REQUIRED (no silent default); point it at the library root.
  # pilot: 2 tasks x Q=3
  .venv/bin/python src/mind_skill/cli/run_induction_training.py \
      --tasks 07b42fd_1,302c169_1 --skills_root data/mind_skill/skills/skill_libraries
  # full _1 corpus (30); bottom-up order: code_executor -> code_planner -> rough_planner
  .venv/bin/python src/mind_skill/cli/run_induction_training.py \
      --tasks all --corpus _1 --component code_executor \
      --skills_root data/mind_skill/skills/skill_libraries

Artifacts: <repo>/logs/mind_skill/<component>/<run_tag>/...   (every q kept)
Libraries: <skills_root>/<component>/{q0,q1,q2,best}/...   (skills_dir handoff data)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
# this module lives at <root>/src/mind_skill/cli/run_induction_training.py
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
PROJECT_ROOT = HERE.parents[3]  # <root> (repo or worktree root)
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Main repo root: worktrees live at <main>/.claude/worktrees/<name>.
_MAIN_REPO = (
    PROJECT_ROOT.parents[2] if PROJECT_ROOT.parent.name == "worktrees" else PROJECT_ROOT
)
DEFAULT_APPWORLD_DATA = Path(
    os.environ.get("APPWORLD_DATA_ROOT", _MAIN_REPO.parent / "appworld" / "data")
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        default="all",
        help="'all' (the --corpus 30-task corpus) or comma-separated task ids",
    )
    parser.add_argument(
        "--corpus",
        default="_1",
        choices=["_1", "_2", "_3"],
        help="which gold corpus '--tasks all' resolves to (default _1 = current "
        "canonical bottom-up corpus). The gold dir holds _1/_2/_3; this "
        "picks the suffix. Ignored when --tasks is an explicit id list.",
    )
    parser.add_argument(
        "--component",
        default="code_executor",
        choices=["code_executor", "rough_planner", "code_planner"],
        help="trained component: code_executor = full closed loop with sandbox "
        "outcome; planners = recon+rubric deduction-lite (no RPC needed)",
    )
    parser.add_argument(
        "--q",
        type=int,
        default=3,
        help="closed-loop iterations (default 3 -> compare q0/q1/q2)",
    )
    parser.add_argument(
        "--rpc_url",
        default=os.environ.get("MIND_SKILL_RPC_URL", "tcp://127.0.0.1:4243"),
    )
    parser.add_argument(
        "--run_tag", default=None, help="output dir tag (default: UTC timestamp)"
    )
    parser.add_argument(
        "--appworld_data",
        type=Path,
        default=DEFAULT_APPWORLD_DATA,
        help="AppWorld data root (ground truth + evaluator for deduction)",
    )
    parser.add_argument(
        "--skills_root",
        type=Path,
        default=None,
        help="REQUIRED. Library root: assemble_libraries WRITES the "
        "{q0,q1,q2,best} libraries here, and planner/outcome runs READ the "
        "bottom-up upstream libraries from here too. e.g. "
        "data/mind_skill/skills/skill_libraries. No silent default — omitting it errors "
        "(the old src/mind_skill/skills default no longer exists). Point at a "
        "SEPARATE dir for partial retrains (smoke) so production is not clobbered.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip tasks whose result.json already exists in the run dir",
    )
    parser.add_argument(
        "--outcome",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="planners (code_planner / rough_planner): real outcome loss via the "
        "sandbox (needs RPC). DEFAULT ON (paper baseline); --no-outcome = recon-only "
        "(separate RQ). Downstream code_planner/executor run thin + their per-task "
        "skill from --skills_root (deployment-matched, Path B), so bottom-up order "
        "matters: code_executor -> code_planner -> rough_planner.",
    )
    return parser.parse_args()


def _check_rpc(rpc_url: str) -> None:
    import zerorpc

    client = zerorpc.Client(timeout=5)
    client.connect(rpc_url)
    try:
        # Any cheap call proves liveness; init/close of a known train task.
        client._zerorpc_ping()
    finally:
        client.close()


async def _amain(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv

    from adk_appworld_agent.repo_paths import repo_env_path

    load_dotenv(dotenv_path=repo_env_path())

    from mind_skill.curation.dedup import uniquify_component
    from mind_skill.deduction.gold import gold_corpus_task_ids
    from mind_skill.loop import (
        assemble_libraries,
        run_planner_task_loop,
        run_task_loop,
    )
    from mind_skill.paths import GOLD_TRAJECTORIES_DIR

    runs_dir = GOLD_TRAJECTORIES_DIR
    if args.tasks.strip().lower() == "all":
        task_ids = gold_corpus_task_ids(runs_dir, suffix=args.corpus)
        corpus_label = f"all/{args.corpus}"
    else:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
        corpus_label = "explicit"

    run_tag = args.run_tag or _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    # Run artifacts follow the repo logs/ convention, segregated by trained
    # component so code_executor / code_planner / rough_planner runs are
    # distinguishable at a glance (mind_skill/<component>/<run_tag>/...).
    out_dir = PROJECT_ROOT / "logs" / "mind_skill" / args.component / run_tag

    # --skills_root is the assemble WRITE target AND (for planner/outcome runs)
    # the bottom-up upstream READ root. There is NO silent default: the old
    # fallback (src/mind_skill/skills) does not exist post data/ move, so omitting
    # the flag used to silently write to a phantom tree / read no skills. Require
    # it, and validate the bottom-up upstream library is present before spending.
    if args.skills_root is None:
        print(
            "ERROR: --skills_root is required (no silent default). Point it at the "
            "library root, e.g. data/mind_skill/skills/skill_libraries"
        )
        return 2
    skills_root = args.skills_root
    if args.component in ("code_planner", "rough_planner") and args.outcome:
        need = (
            ["code_executor"]
            if args.component == "code_planner"
            else ["code_executor", "code_planner"]
        )
        for comp in need:
            lib = skills_root / comp / "best"
            if not lib.is_dir() or not any(lib.glob("*/*/SKILL.md")):
                print(
                    f"ERROR: {args.component} --outcome needs the upstream library "
                    f"{lib} (empty/missing). Bottom-up order: build code_executor "
                    f"-> code_planner -> rough_planner first, into the same "
                    f"--skills_root."
                )
                return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.appworld_data.exists():
        print(f"ERROR: appworld data root not found: {args.appworld_data}")
        return 2
    if args.component == "code_executor" or args.outcome:
        try:
            _check_rpc(args.rpc_url)
        except Exception as exc:
            print(f"ERROR: AppWorld RPC server not reachable at {args.rpc_url}: {exc}")
            print(
                "Start it with: cd ../appworld && RPC_PORT=4243 .venv/bin/python rpc/server.py"
            )
            return 2

    print(
        f"run_tag={run_tag}  component={args.component}  tasks={len(task_ids)}  "
        f"Q={args.q}  rpc={args.rpc_url if (args.component == 'code_executor' or args.outcome) else 'n/a'}"
    )
    span = f"{task_ids[0]} .. {task_ids[-1]}" if task_ids else "(none)"
    print(f"corpus={corpus_label}  n={len(task_ids)}  [{span}]")
    print(f"artifacts -> {out_dir}")
    print(f"Batch results (live-updated): {out_dir / 'batch_results.txt'}")

    from mind_skill.cli.report_induction_run import write_batch_results

    def _refresh_batch_results() -> list[dict]:
        # run_batch.py convention: rewrite the whole file after every task so
        # the batch is inspectable while still running.
        parsed = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out_dir.glob("*/result.json"))
        ]
        if parsed:
            write_batch_results(out_dir, parsed, [f"q{n}" for n in range(args.q)])
        return parsed

    from mind_skill.cli.finalize import (
        finalize_run_level,
        finalize_task,
        publish_induced,
    )

    done, failed, finalized_ids = 0, [], []
    for i, task_id in enumerate(task_ids, 1):
        result_path = out_dir / task_id / "result.json"
        if args.resume and result_path.exists():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if len(prior.get("iterations") or []) >= args.q:
                print(f"[{i}/{len(task_ids)}] {task_id} — already done, skip")
                done += 1
                continue
            # fewer iterations than requested Q -> re-enter; per-q resume
            # reloads the finished iterations and only runs the missing ones.
        t0 = time.monotonic()
        print(
            f"[{i}/{len(task_ids)}] {task_id} — Q={args.q} closed loop ...", flush=True
        )
        try:
            if args.component == "code_executor":
                result = await run_task_loop(
                    task_id,
                    runs_dir=runs_dir,
                    out_dir=out_dir,
                    appworld_data_root=args.appworld_data,
                    q_iterations=args.q,
                    rpc_url=args.rpc_url,
                )
            else:
                result = await run_planner_task_loop(
                    task_id,
                    component=args.component,
                    runs_dir=runs_dir,
                    out_dir=out_dir,
                    appworld_data_root=args.appworld_data,
                    q_iterations=args.q,
                    outcome=args.outcome,
                    rpc_url=args.rpc_url,
                    skills_root=skills_root,
                )
        except Exception as exc:  # keep batch going; record the failure
            failed.append({"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        done += 1
        losses = " | ".join(
            f"q{it.q}: out={it.losses.outcome:.2f} rec={it.losses.recon:.0f} "
            f"rub={it.losses.rubric:.1f} clr="
            + (
                "-"
                if it.deduction_cleared is None
                else ("Y" if it.deduction_cleared else "N")
            )
            for it in result.iterations
        )
        print(
            f"    best_q={result.best_q}  {losses}  ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
        _refresh_batch_results()
        # dynamic finalize: copy THIS task's keep-list into training_runs the moment
        # it finishes, so data/ fills in live and survives a mid-run crash.
        finalize_task(out_dir / task_id, component=args.component, run_tag=run_tag)
        finalized_ids.append(task_id)

    summary = assemble_libraries(
        out_dir, skills_root, q_iterations=args.q, component=args.component
    )
    # post-assembly hygiene: make same-name skills uniquely addressable by a
    # DETERMINISTIC rename (name-2, name-3; no LLM, no merge). Semantic merging of
    # similar skills is deliberately the curation stage's job, not induction's.
    uniq_summary = uniquify_component(skills_root / args.component)
    n_actions = sum(len(acts) for acts in uniq_summary["libraries"].values())
    print(
        f"name-uniquify: renamed {n_actions} same-name collision(s) (no LLM, no merge)"
    )
    summary["name_uniquify"] = uniq_summary
    _refresh_batch_results()
    (out_dir / "batch_summary.json").write_text(
        json.dumps(
            {
                "run_tag": run_tag,
                "component": args.component,
                "tasks": len(task_ids),
                "done": done,
                "failed": failed,
                "libraries": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\ndone={done}/{len(task_ids)}  failed={len(failed)}")
    print(f"libraries: {summary['libraries']}  quarantined: {summary['quarantined']}")
    print(f"skill libraries -> {skills_root / args.component}")

    # Per-task dirs were already copied live via finalize_task; here just add the
    # run-level files + run_manifest.json, then publish induced (after assemble +
    # the whole-library name-dedup, which needs all tasks present).
    fin = finalize_run_level(
        out_dir,
        component=args.component,
        run_tag=run_tag,
        tasks=finalized_ids,
        manifest_extra={
            "corpus": corpus_label,
            "q_iterations": args.q,
            "done": done,
            "failed_task_ids": [f["task_id"] for f in failed],
            "outcome": bool(args.outcome),
            "skills_root": str(skills_root),
        },
    )
    pub = publish_induced(skills_root, component=args.component, run_tag=run_tag)
    print(f"finalized -> {fin['training_run']}  ({fin['n_tasks']} tasks kept)")
    print(f"induced -> {pub['induced']}  {pub['libraries']}")
    return 0 if not failed else 1


def main() -> int:
    return asyncio.run(_amain(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
