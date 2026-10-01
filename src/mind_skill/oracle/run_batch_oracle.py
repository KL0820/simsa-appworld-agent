"""Oracle batch + watchdog — drive a set of AppWorld tasks through the clean
`claude -p` oracle (run_oracle_auto.py), one at a time, resumable, and write a live
`runs/batch_results.txt` (same spirit as the agent's batch_results.txt) so progress is
checkable at any time.

Watchdog: claude_call retries rate-limits internally; tasks that still fail are retried
on the next pass. Loops passes until all done or max passes.

Default task set = the `_1` variant of every train scenario (~30, one per scenario;
current canonical corpus). Override the variant or pass an explicit set:
  python src/mind_skill/run_batch_oracle.py                    # _1 set (default)
  ORACLE_CORPUS=_2 python src/mind_skill/run_batch_oracle.py   # original _2 set
  TASKS="a_1 b_1" python src/mind_skill/run_batch_oracle.py    # explicit set
Needs the dedicated AppWorld RPC server up (default 4243).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# this module lives at <root>/src/mind_skill/oracle/run_batch_oracle.py
HERE = Path(__file__).resolve()
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
AGENT_ROOT = HERE.parents[3]  # <root> (worktree/repo root; has scripts/ + .venv)
sys.path.insert(0, str(AGENT_ROOT / "src"))
from adk_appworld_agent.task_sets import task_set_path  # noqa: E402

RUNS = MIND_SKILL / "runs"
PY = AGENT_ROOT / ".venv" / "bin" / "python"
TRAIN = task_set_path("train")
MODEL = os.environ.get("ORACLE_CLAUDE_MODEL", "opus")
RPC = os.environ.get("ORACLE_RPC_URL", "tcp://127.0.0.1:4243")
MAX_PASSES = int(os.environ.get("ORACLE_MAX_PASSES", "6"))
PER_TASK_TIMEOUT = int(os.environ.get("ORACLE_PER_TASK_TIMEOUT", "5400"))
RESULTS = RUNS / "batch_results.txt"
# Which scenario variant the default task set picks. The gold dir holds multiple
# corpora (_1 / _2); default _1 = current canonical (matches gold.py suffix="_1").
# Override with ORACLE_CORPUS=_2 for the original set, or TASKS="..." for explicit.
CORPUS = os.environ.get("ORACLE_CORPUS", "_1")
_TS = timezone(timedelta(hours=8))


def task_set() -> list[str]:
    env = os.environ.get("TASKS")
    if env:
        return env.split()
    ids = [l.strip() for l in TRAIN.read_text().split() if l.strip()]
    return [t for t in ids if t.endswith(CORPUS)]


def done(t: str) -> bool:
    return (RUNS / t / "trajectory.json").exists()


def _summary(t: str):
    g = sorted((RUNS / t).glob("logs/**/task_summary.json"))
    if not g:
        return None
    try:
        return json.loads(Path(g[-1]).read_text(encoding="utf-8"))
    except Exception:
        return None


def write_results(tasks: list[str]) -> None:
    done_n = pass_n = tp = tt = 0
    wall = 0.0
    body = []
    for t in tasks:
        d = _summary(t)
        if not d:
            continue
        ev = d.get("eval", {}) or {}
        passed, total, failed = (
            ev.get("passed", 0),
            ev.get("total", 0),
            ev.get("failed", 0),
        )
        cleared = bool(ev.get("passed_all")) or (failed == 0 and total > 0)
        w = float(d.get("wall_s", 0.0) or 0.0)
        ncalls = len(list((RUNS / t).glob("io/req_*.txt")))
        instr = (d.get("instruction") or "").strip().replace("\n", " ")
        block = (d.get("overview") or {}).get("block_reason")
        done_n += 1
        pass_n += 1 if cleared else 0
        tp += passed
        tt += total
        wall += w
        body.append(
            f"[{t}] {'PASS' if cleared else 'FAIL'} {passed}/{total}  | {w:.1f}s | {ncalls} claude calls"
        )
        body.append(f"  {instr[:240]}")
        if not cleared and block:
            body.append(f"  block: {block}")
        body.append("")
    total_n = len(tasks)
    pct = lambda a, b: f"{100 * a / b:.1f}%" if b else "-"
    head = [
        "=" * 64,
        "Oracle Batch Results (mind_skill — claude -p clean oracle)",
        "=" * 64,
        "",
        f"  Updated:  {datetime.now(_TS).strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model:    {MODEL} (claude -p; --setting-sources '' / --allowedTools '' = clean+blind)",
        "  Plan/Find/Execute: rough / community / code_plan_execute",
        "",
        "AGGREGATE",
        "-" * 9,
        f"  Tasks done    {done_n} / {total_n}",
        f"  Tasks passed  {pass_n} / {total_n} ({pct(pass_n, total_n)})",
        f"  Test pass     {tp} / {tt} ({pct(tp, tt)})",
        f"  Wall (s)      {wall:.0f}   (avg {wall / done_n:.0f})"
        if done_n
        else "  Wall (s)      0",
        "",
    ]
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(head + body), encoding="utf-8")


def run_one(t: str) -> None:
    run_dir = RUNS / t
    env = {
        **os.environ,
        "ORACLE_TASK_ID": t,
        "ORACLE_RUN_DIR": str(run_dir),
        "ORACLE_CLAUDE_MODEL": MODEL,
        "ORACLE_RPC_URL": RPC,
    }
    print(f"[batch] >>> {t} (model={MODEL})", flush=True)
    try:
        subprocess.run(
            [str(PY), str(HERE.parent / "run_oracle_auto.py")],
            cwd=str(AGENT_ROOT),
            env=env,
            timeout=PER_TASK_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        print(f"[batch] {t} TIMEOUT after {PER_TASK_TIMEOUT}s", flush=True)
    # build trajectory if the run produced logs (PASS or FAIL — capture either)
    if next(iter((run_dir).glob("logs/**/task_summary.json")), None):
        subprocess.run(
            [
                str(PY),
                str(MIND_SKILL / "trajectory" / "build_trajectory.py"),
                str(run_dir),
            ],
            cwd=str(AGENT_ROOT),
        )
    print(f"[batch] <<< {t} done={done(t)}", flush=True)


def main() -> int:
    tasks = task_set()
    print(
        f"[batch] {len(tasks)} tasks; already done: {sum(done(t) for t in tasks)}",
        flush=True,
    )
    write_results(tasks)
    for p in range(MAX_PASSES):
        remaining = [t for t in tasks if not done(t)]
        if not remaining:
            print("[batch] all done", flush=True)
            break
        print(
            f"[batch] pass {p + 1}/{MAX_PASSES}: {len(remaining)} remaining", flush=True
        )
        for t in remaining:
            run_one(t)
            write_results(tasks)
            time.sleep(3)
        time.sleep(30)  # let subscription rate-limit cool between passes
    write_results(tasks)
    n_done = sum(done(t) for t in tasks)
    print(
        f"[batch] FINISHED: {n_done}/{len(tasks)} have trajectories. See {RESULTS}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
