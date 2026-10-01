"""Open-book recovery for the few tasks blind-solve fails (DISCLOSED).

Gives claude -p the task's gold solution as a CORRECTNESS reference
(ORACLE_SOLUTION_PATH); outputs are still derived from each step's input (the
backend prompt forbids pasting). Writes to runs/<task>/openbook/ — the blind FAIL
at runs/<task>/ is PRESERVED, never overwritten (kept as a safeguard + the
blind-FAIL ↔ open-book-PASS contrast). Marks run_meta.open_book=true + reference.

  python src/mind_skill/run_openbook.py            # the 3 known fails
  TASKS="a_2 b_2" python src/mind_skill/run_openbook.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

# this module lives at <root>/src/mind_skill/oracle/run_openbook.py
HERE = Path(__file__).resolve()
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
AGENT_ROOT = HERE.parents[3]  # <root> (worktree/repo root)
# Worktrees live at <main>/.claude/worktrees/<name>; derive the MAIN repo so the
# appworld data sibling resolves from a worktree (AGENT_ROOT.parent there is
# .claude/worktrees, not the repo). Mirrors run_induction_training._MAIN_REPO.
_MAIN_REPO = (
    AGENT_ROOT.parents[2] if AGENT_ROOT.parent.name == "worktrees" else AGENT_ROOT
)
RUNS = MIND_SKILL / "runs"
PY = AGENT_ROOT / ".venv" / "bin" / "python"
TASKS = os.environ.get("TASKS", "22cc237_2 3c13f5a_2 e3d6c94_2").split()
MODEL = os.environ.get("ORACLE_CLAUDE_MODEL", "opus")
RPC = os.environ.get("ORACLE_RPC_URL", "tcp://127.0.0.1:4243")


def _data_root() -> Path:
    # APPWORLD_DATA_ROOT (repo convention) makes this worktree-portable; the
    # fallback is derived from the MAIN repo, NOT AGENT_ROOT.parent (which is
    # .claude/worktrees from a worktree and silently skips every task).
    base = os.environ.get("APPWORLD_DATA_ROOT")
    return Path(base) if base else _MAIN_REPO.parent / "appworld" / "data"


def sol_path(t: str) -> Path:
    return _data_root() / "tasks" / t / "ground_truth" / "solution.py"


def main() -> int:
    root = _data_root()
    if not root.is_dir():
        print(
            f"[openbook] ERROR: appworld data root not found: {root}\n"
            f"  set APPWORLD_DATA_ROOT, or check the repo layout — without it every "
            f"task silently skips with 'NO solution.py'.",
            flush=True,
        )
        return 2
    results = []
    for t in TASKS:
        ob = RUNS / t / "openbook"
        sp = sol_path(t)
        if not sp.exists():
            print(f"[openbook] {t}: NO solution.py — skip", flush=True)
            continue
        print(
            f"[openbook] >>> {t}  (blind FAIL preserved at runs/{t}/; ref={sp.name})",
            flush=True,
        )
        env = {
            **os.environ,
            "ORACLE_TASK_ID": t,
            "ORACLE_RUN_DIR": str(ob),
            "ORACLE_SOLUTION_PATH": str(sp),
            "ORACLE_CLAUDE_MODEL": MODEL,
            "ORACLE_RPC_URL": RPC,
        }
        try:
            subprocess.run(
                [str(PY), str(HERE.parent / "run_oracle_auto.py")],
                cwd=str(AGENT_ROOT),
                env=env,
                timeout=5400,
            )
        except subprocess.TimeoutExpired:
            print(f"[openbook] {t} TIMEOUT", flush=True)
        if next(iter(ob.glob("logs/**/task_summary.json")), None):
            subprocess.run(
                [
                    str(PY),
                    str(MIND_SKILL / "trajectory" / "build_trajectory.py"),
                    str(ob),
                ],
                cwd=str(AGENT_ROOT),
            )
        tj = ob / "trajectory.json"
        if tj.exists():
            d = json.loads(tj.read_text(encoding="utf-8"))
            d.setdefault("run_meta", {})["open_book"] = True
            d["run_meta"]["reference"] = str(sp)
            tj.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
            r = d.get("result") or {}
            results.append((t, r.get("passed"), r.get("total"), r.get("cleared")))
            print(
                f"[openbook] <<< {t}  open-book = {r.get('passed')}/{r.get('total')} cleared={r.get('cleared')}",
                flush=True,
            )
        else:
            results.append((t, None, None, None))
            print(f"[openbook] <<< {t}  no trajectory", flush=True)
    print(
        "\n[openbook] SUMMARY (blind FAIL still at runs/<t>/; open-book at runs/<t>/openbook/):",
        flush=True,
    )
    for t, p, tot, c in results:
        print(f"  {t}: open-book {p}/{tot} cleared={c}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
