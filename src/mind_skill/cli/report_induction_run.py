"""Summarize an induction training run: per-task q0/q1/q2 losses + library counts.

  PYTHONPATH=src .venv/bin/python src/mind_skill/report_induction_run.py <run_tag> [--write]

Reads <repo>/logs/mind_skill/<run_tag>/*/result.json (falls back to the legacy
src/mind_skill/induction_runs/ location). Pure offline.
--write also drops batch_results.txt + batch_results_q{n}.txt into the run dir
(same convention as logs/<experiment>/<ts>/batch_results.txt for agent batches:
one line per task, header with config), so each training iteration q reads like
its own batch.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

MIND_SKILL = Path(__file__).resolve().parents[1]  # <root>/src/mind_skill
LOGS_ROOT = MIND_SKILL.parents[1] / "logs" / "mind_skill"


def _fmt_iteration(it: dict) -> str:
    losses = it["losses"]
    cleared = it.get("deduction_cleared")
    flags = "n/a " if cleared is None else ("PASS" if cleared else "FAIL")
    return (
        f"outcome={losses['outcome']:.2f} recon={losses['recon']:>4.1f} "
        f"rubric={losses['rubric']:>4.1f} {flags}"
    )


def _load_deduction(run_dir: Path, task_id: str, qk: str) -> dict:
    path = run_dir / task_id / qk / "deduction.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {}


def _aggregate_block(rows: list[dict]) -> list[str]:
    """rows: per-task dicts with cleared(bool|None), passed/failed/total, wall_s,
    leak(bool). Mirrors the AGGREGATE block of logs/<exp>/<ts>/batch_results.txt."""
    n = len(rows)
    with_outcome = [r for r in rows if r["cleared"] is not None]
    passed_tasks = sum(1 for r in with_outcome if r["cleared"])
    test_pass = sum(r.get("passed") or 0 for r in with_outcome)
    test_total = sum(r.get("total") or 0 for r in with_outcome)
    wall = [r["wall_s"] for r in rows if r.get("wall_s")]
    lines = ["AGGREGATE", "---------", ""]
    lines.append(f"  Tasks reported   {n}")
    if with_outcome:
        pct = 100.0 * passed_tasks / len(with_outcome)
        lines.append(
            f"  Deduction passed {passed_tasks} / {len(with_outcome)} ({pct:.1f}%)"
        )
        if test_total:
            lines.append(
                f"  Test pass        {test_pass}/{test_total} "
                f"({100.0 * test_pass / test_total:.1f}%)"
            )
    else:
        lines.append("  Deduction passed n/a (planner loop: recon+rubric only)")
    if wall:
        lines.append(
            f"  Deduction wall   total {sum(wall):.1f}s   avg {sum(wall) / len(wall):.1f}s"
        )
    lines.append("")
    return lines


def write_batch_results(run_dir: Path, results: list[dict], q_keys: list[str]) -> None:
    """batch_results.txt (overall) + batch_results_q{n}.txt (one per iteration),
    matching the logs/ batch convention: header -> AGGREGATE -> PER TASK."""
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _header(title: str) -> list[str]:
        return [
            "=" * 64,
            f"MIND-Skill induction batch — {title}",
            "=" * 64,
            "",
            f"  Updated:  {stamp}",
            "  Model:    gemini-2.5-flash (self-generation, temp 0 / seed 123)",
            "  Loop:     induce -> deduction -> recon/rubric -> TextGrad",
            "",
        ]

    results = sorted(results, key=lambda x: x["task_id"])

    # Collect per-(task, q) rows enriched with deduction.json counts.
    rows_by_q: dict[str, list[dict]] = {qk: [] for qk in q_keys}
    best_rows: list[dict] = []
    for r in results:
        for it in r.get("iterations", []):
            qk = f"q{it['q']}"
            if qk not in rows_by_q:
                continue
            ded = _load_deduction(run_dir, r["task_id"], qk)
            rows_by_q[qk].append(
                {
                    "task_id": r["task_id"],
                    "it": it,
                    "cleared": it.get("deduction_cleared"),
                    "passed": ded.get("passed"),
                    "failed": ded.get("failed"),
                    "total": ded.get("total"),
                    "wall_s": ded.get("wall_s"),
                }
            )
        if r.get("best_q") is not None:
            qk = f"q{r['best_q']}"
            match = next(
                (
                    row
                    for row in rows_by_q.get(qk, [])
                    if row["task_id"] == r["task_id"]
                ),
                None,
            )
            if match is not None:
                best_rows.append(match)

    def _task_line(row: dict) -> str:
        it = row["it"]
        cleared = row["cleared"]
        verdict = "n/a " if cleared is None else ("PASS" if cleared else "FAIL")
        counts = (
            f" {row['passed']}/{row['total']}"
            if isinstance(row.get("passed"), int) and row.get("total")
            else ""
        )
        wall = f" | {row['wall_s']:.1f}s" if row.get("wall_s") else ""
        losses = it["losses"]
        return (
            f"[{row['task_id']}] {verdict}{counts}{wall} | "
            f"out={losses['outcome']:.2f} rec={losses['recon']:.1f} "
            f"rub={losses['rubric']:.1f}"
        )

    # Per-q files: each training iteration reads as its own batch.
    for qk in q_keys:
        rows = rows_by_q[qk]
        lines = _header(f"{run_dir.name} — iteration {qk}")
        lines += _aggregate_block(rows)
        lines += ["PER TASK", "--------", ""]
        lines += [_task_line(row) for row in rows]
        (run_dir / f"batch_results_{qk}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # Overall file: best-of-Q selection (the library that ships) + per-q detail.
    lines = _header(f"{run_dir.name} — best-of-Q selection")
    lines += _aggregate_block(best_rows)
    lines += ["PER TASK", "--------", ""]
    for r in results:
        best = f"best=q{r['best_q']}" if r.get("best_q") is not None else "no-best"
        lines.append(f"[{r['task_id']}]  {best}")
        for it in r.get("iterations", []):
            lines.append(f"    q{it['q']}: {_fmt_iteration(it)}")
    (run_dir / "batch_results.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    run_tag = args[0] if args else None
    legacy_root = MIND_SKILL / "induction_runs"
    if run_tag is None:
        # Current layout: LOGS_ROOT/<component>/<run_tag>. Legacy/pre-segregation:
        # flat <run_tag> under LOGS_ROOT or induction_runs. List runs that have
        # results in either shape.
        tags = sorted(
            {
                f"{p.parent.name}/{p.name}"
                for p in LOGS_ROOT.glob("*/*")
                if p.is_dir() and any(p.glob("*/result.json"))
            }
            | {
                p.name
                for root in (LOGS_ROOT, legacy_root)
                if root.exists()
                for p in root.iterdir()
                if p.is_dir() and any(p.glob("*/result.json"))
            }
        )
        print(
            "usage: report_induction_run.py <run_tag> [--write]; available:",
            ", ".join(tags),
        )
        return 2
    # Resolve <run_tag> across components first (LOGS_ROOT/<component>/<run_tag>),
    # then fall back to the pre-segregation flat layout and the legacy root. A bare
    # tag, or an explicit "<component>/<run_tag>", both resolve.
    matches = sorted(LOGS_ROOT.glob(f"*/{run_tag}"))
    if matches:
        if len(matches) > 1:
            print(
                f"warning: {run_tag} exists under {len(matches)} components; using {matches[0].parent.name}"
            )
        run_dir = matches[0]
    elif (LOGS_ROOT / run_tag).exists():
        run_dir = LOGS_ROOT / run_tag
    elif (legacy_root / run_tag).exists():
        run_dir = legacy_root / run_tag
    else:
        run_dir = LOGS_ROOT / run_tag
    result_paths = sorted(run_dir.glob("*/result.json"))
    if not result_paths:
        print(f"no result.json under {run_dir}")
        return 1
    parsed_results = [json.loads(p.read_text(encoding="utf-8")) for p in result_paths]

    q_keys: list[str] = []
    rows = []
    cleared_by_q: Counter = Counter()
    loss_sums: dict[str, dict[str, float]] = {}
    best_counter: Counter = Counter()
    for r in parsed_results:
        row = {"task": r["task_id"], "best_q": r.get("best_q")}
        best_counter[f"q{r['best_q']}" if r.get("best_q") is not None else "none"] += 1
        for it in r.get("iterations", []):
            qk = f"q{it['q']}"
            if qk not in q_keys:
                q_keys.append(qk)
            losses = it["losses"]
            row[qk] = (
                f"({losses['outcome']:.2f},{losses['recon']:.0f},{losses['rubric']:.1f})"
                + ("✓" if it.get("deduction_cleared") else "✗")
            )
            cleared_by_q[qk] += 1 if it.get("deduction_cleared") else 0
            sums = loss_sums.setdefault(
                qk, {"outcome": 0.0, "recon": 0.0, "rubric": 0.0, "n": 0}
            )
            for k in ("outcome", "recon", "rubric"):
                sums[k] += losses[k]
            sums["n"] += 1
        rows.append(row)

    width = max(len(r["task"]) for r in rows) + 2
    header = "task".ljust(width) + "best  " + "  ".join(k.ljust(16) for k in q_keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = "  ".join(str(row.get(k, "—")).ljust(16) for k in q_keys)
        print(f"{row['task']:<{width}}q={str(row['best_q']):<4}{cells}")

    print()
    n_tasks = len(rows)
    print(f"tasks: {n_tasks}   best_q distribution: {dict(best_counter)}")
    for qk in q_keys:
        sums = loss_sums[qk]
        n = max(1, sums["n"])
        print(
            f"{qk}: deduction cleared {cleared_by_q[qk]}/{sums['n']}   "
            f"mean losses outcome={sums['outcome'] / n:.3f} "
            f"recon={sums['recon'] / n:.2f} rubric={sums['rubric'] / n:.2f}"
        )

    batch = run_dir / "batch_summary.json"
    if batch.exists():
        b = json.loads(batch.read_text(encoding="utf-8"))
        print(
            f"\nbatch: done={b.get('done')}/{b.get('tasks')} failed={len(b.get('failed') or [])}"
        )
        print(f"libraries: {(b.get('libraries') or {}).get('libraries')}")
    if write:
        write_batch_results(run_dir, parsed_results, q_keys)
        print(
            f"\nwrote batch_results.txt + {', '.join(f'batch_results_{k}.txt' for k in q_keys)} -> {run_dir}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
