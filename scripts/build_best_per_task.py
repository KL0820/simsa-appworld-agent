#!/usr/bin/env python3
"""Build a per-task BEST-RESULT registry across all historical runs.

Priority (per task): PASS first, then fewest LLM calls, then fewest total tokens,
then fastest wall. This is the canonical PASS-path reference for diff-vs-fail
analysis (see feedback_diff_fail_vs_best_pass). It also COPIES each task's best
PASS trajectory artifacts into a stable folder so they survive worktree cleanup
(feedback_preserve_eval_logs).

Output: logs/_registry/best_per_task/
  index.json              full machine record (best + all candidates per task)
  best_per_task.md        human table
  tasks/<tid>/            copied artifacts of the best PASS path + SOURCE.txt

Re-run after new batches:  uv run python scripts/build_best_per_task.py
"""

import glob
import json
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs/_registry/best_per_task")


def parse_config(cmd):
    def g(flag):
        m = re.search(rf"--{flag}\s+(\S+)", cmd or "")
        return m.group(1) if m else None

    return {
        "find": g("find"),
        "plan": g("plan"),
        "execute": g("execute"),
        "skills": g("skills"),
        "skills_prompt": g("skills_prompt"),
        "model": g("model_name"),
        "experiment": g("experiment_name"),
        "run_name": g("run_name"),
    }


def collect():
    records = {}
    for ts in glob.glob(
        os.path.join(ROOT, "**/artifacts/task_summary.json"), recursive=True
    ):
        # skip the registry's own copies
        if "/_registry/best_per_task/" in ts:
            continue
        try:
            d = json.load(open(ts))
        except Exception:
            continue
        tid = d.get("task_id")
        if not tid:
            continue
        ev = d.get("eval") or {}
        am = d.get("aggregate_metrics") or {}
        cmd = (d.get("run") or {}).get("command", "")
        rec = {
            "task_id": tid,
            "passed_all": bool(ev.get("passed_all")),
            "eval": f"{ev.get('passed', '?')}/{ev.get('total', '?')}",
            "passed": ev.get("passed"),
            "total": ev.get("total"),
            "llm_calls": am.get("llm_calls"),
            "total_tokens": am.get("total_tokens"),
            "wall_ms": am.get("wall_ms"),
            "status": d.get("status"),
            "block": (
                (d.get("submission") or {}).get("block_reason")
                or (d.get("overview") or {}).get("block_reason")
                if isinstance(d.get("overview"), dict)
                else None
            ),
            "artifacts_dir": os.path.dirname(ts),
            "run_dir": os.path.relpath(os.path.dirname(os.path.dirname(ts)), ROOT),
            "config": parse_config(cmd),
        }
        records.setdefault(tid, []).append(rec)
    return records


BIG = float("inf")


def key_pass(r):
    return (
        r["llm_calls"] if r["llm_calls"] is not None else BIG,
        r["total_tokens"] if r["total_tokens"] is not None else BIG,
        r["wall_ms"] if r["wall_ms"] is not None else BIG,
    )


def key_effort(r):
    ratio = (
        (r["passed"] / r["total"]) if (r["passed"] is not None and r["total"]) else -1
    )
    return (-ratio,) + key_pass(r)


def pick_best(records):
    best = {}
    for tid, recs in records.items():
        passes = [r for r in recs if r["passed_all"]]
        if passes:
            b = dict(sorted(passes, key=key_pass)[0])
            b["pass"] = True
        else:
            b = dict(sorted(recs, key=key_effort)[0])
            b["pass"] = False
        b["n_runs_passed"] = len(passes)
        b["n_runs_total"] = len(recs)
        best[tid] = b
    return best


def main():
    records = collect()
    best = pick_best(records)
    os.makedirs(os.path.join(OUT, "tasks"), exist_ok=True)
    # copy artifacts for PASS bests
    copied = 0
    for tid, b in best.items():
        if not b["pass"]:
            continue
        src = b["artifacts_dir"]
        dst = os.path.join(OUT, "tasks", tid)
        os.makedirs(dst, exist_ok=True)
        for f in ("subagent_io.jsonl", "sandbox_api_calls.jsonl", "task_summary.json"):
            s = os.path.join(src, f)
            if os.path.exists(s):
                shutil.copy2(s, os.path.join(dst, f))
        with open(os.path.join(dst, "SOURCE.txt"), "w") as fh:
            fh.write(
                f"task_id: {tid}\nsource_run: {b['run_dir']}\nconfig: {b['config']}\n"
                f"eval: {b['eval']}\nllm_calls: {b['llm_calls']}\ntotal_tokens: {b['total_tokens']}\n"
                f"wall_ms: {b['wall_ms']}\nstatus: {b['status']}\n"
            )
        copied += 1
    json.dump(
        {"best": best, "candidates": records},
        open(os.path.join(OUT, "index.json"), "w"),
        indent=1,
        default=str,
    )
    # markdown
    passed = [t for t, b in best.items() if b["pass"]]
    never = [t for t, b in best.items() if not b["pass"]]
    lines = [
        "# Best-per-task registry",
        "",
        "Per task, the best historical result. Priority: **PASS → fewest LLM calls → fewest tokens → fastest wall**.",
        "Best PASS-path artifacts are copied under `tasks/<tid>/` (stable; survives worktree cleanup).",
        "Regenerate: `uv run python scripts/build_best_per_task.py`",
        "",
        f"- distinct tasks seen: **{len(best)}**",
        f"- tasks with ≥1 PASS (have a gold path): **{len(passed)}**",
        f"- tasks never passed by any run: **{len(never)}**",
        "",
        "| task | best | eval | llm | tokens | source run | config |",
        "|---|---|---|---|---|---|---|",
    ]
    for tid in sorted(best, key=lambda t: (not best[t]["pass"], t)):
        b = best[tid]
        c = b["config"]
        cfg = f"{c.get('plan')}/{c.get('find')}/skills={c.get('skills')},{c.get('skills_prompt')}"
        lines.append(
            f"| {tid} | {'PASS' if b['pass'] else 'never'} | {b['eval']} | "
            f"{b['llm_calls']} | {b['total_tokens']} | {b['run_dir']} | {cfg} |"
        )
    if never:
        lines += ["", "## Never passed by any run", "", " ".join(sorted(never))]
    open(os.path.join(OUT, "best_per_task.md"), "w").write("\n".join(lines))
    print(
        f"distinct tasks={len(best)}  passed={len(passed)}  never={len(never)}  artifacts_copied={copied}"
    )
    print(f"written: {OUT}/index.json , best_per_task.md , tasks/<tid>/")


if __name__ == "__main__":
    main()
