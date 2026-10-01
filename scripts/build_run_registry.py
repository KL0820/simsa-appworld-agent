#!/usr/bin/env python3
"""Scan logs/ for full56-class batches and (re)generate the run registry.

Auto-extracts from each `logs/<experiment>/<timestamp>/batch_results.txt`:

    - date (from the timestamp dir name)
    - run dir (`<experiment>/<timestamp>`)
    - PASS / total
    - avg wall per task (seconds)
    - avg LLM calls per task
    - viewer link if `logs/<experiment>/analysis/viewer/index.html` exists

Editorial fields (only place humans need to write):

    - headline change (one-line summary of what's different from previous row)
    - notes (free text)
    - group (optional — combine multiple sharded run dirs into one logical row)

Editorial fields live in `logs/_registry/editorial.json`, keyed by the
run dir `<experiment>/<timestamp>`. Past entries are preserved across
regenerations.

Usage:
    uv run python scripts/build_run_registry.py            # scan + regenerate
    uv run python scripts/build_run_registry.py --min 30   # change task-count floor
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOGS_DIR = REPO_ROOT / "logs"
DEFAULT_REGISTRY_DIR = DEFAULT_LOGS_DIR / "_registry"
DEFAULT_EDITORIAL = DEFAULT_REGISTRY_DIR / "editorial.json"
DEFAULT_OUT = DEFAULT_REGISTRY_DIR / "runs.md"
TS_RE = re.compile(r"^20\d{6}_\d+")


def load_editorial(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"WARN: editorial JSON invalid: {e}", file=sys.stderr)
        return {}


def save_editorial(path: Path, data: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def parse_batch_results(path: Path) -> dict[str, Any] | None:
    """Return {n_done, n_pass, walls, llms} or None if unparseable."""
    if not path.exists():
        return None
    n_done = 0
    n_pass = 0
    walls: list[float] = []
    llms: list[int] = []
    # batch_results format example:
    #   [624f342_2] FAIL NA   | 1200.2s | 55 llm | 154,441 tok
    #   [9dabbc9_2] PASS 9/9  | 321.4s  | 20 llm | 76,210 tok
    line_re = re.compile(
        r"^\[(?P<tid>[a-z0-9_]+)\]\s+(?P<verdict>PASS|FAIL)\s+\S+\s+\|\s+"
        r"(?P<wall>[0-9.]+)s\s+\|\s+(?P<llm>\d+)\s+llm"
    )
    for line in path.read_text().splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        n_done += 1
        if m.group("verdict") == "PASS":
            n_pass += 1
        try:
            walls.append(float(m.group("wall")))
            llms.append(int(m.group("llm")))
        except (ValueError, TypeError):
            pass
    if n_done == 0:
        return None
    return {
        "n_done": n_done,
        "n_pass": n_pass,
        "avg_wall_s": sum(walls) / len(walls) if walls else 0.0,
        "avg_llm": sum(llms) / len(llms) if llms else 0.0,
    }


def parse_run_date(timestamp_name: str) -> str:
    """20260512_101038_909559 → 2026-05-12"""
    try:
        return f"{timestamp_name[:4]}-{timestamp_name[4:6]}-{timestamp_name[6:8]}"
    except Exception:
        return ""


def discover_runs(logs_dir: Path) -> list[dict[str, Any]]:
    """Scan logs/<exp>/<ts>/batch_results.txt for all batches with results.

    No min_tasks filter at discovery time — that's applied post-merge so
    sharded full56 runs (e.g. v1 10 tasks + v1b 46 tasks under one group)
    still aggregate correctly.
    """
    runs: list[dict[str, Any]] = []
    if not logs_dir.is_dir():
        return runs
    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            if not TS_RE.match(ts_dir.name):
                continue
            stats = parse_batch_results(ts_dir / "batch_results.txt")
            if stats is None:
                continue
            run_id = f"{exp_dir.name}/{ts_dir.name}"
            runs.append(
                {
                    "run_id": run_id,
                    "experiment": exp_dir.name,
                    "timestamp": ts_dir.name,
                    "date": parse_run_date(ts_dir.name),
                    **stats,
                }
            )
    return runs


def viewer_link_for(exp_name: str, logs_dir: Path) -> str | None:
    """Relative link from _registry/runs.md to viewer index if present."""
    candidate = logs_dir / exp_name / "analysis" / "viewer" / "index.html"
    if candidate.exists():
        # Path from logs/_registry/runs.md → logs/<exp>/analysis/viewer/index.html
        return f"../{exp_name}/analysis/viewer/index.html"
    return None


def merge_groups(
    runs: list[dict[str, Any]],
    editorial: dict[str, dict[str, Any]],
    logs_dir: Path,
) -> list[dict[str, Any]]:
    """Combine runs that share an editorial `group` key into one logical row.

    Ungrouped runs become their own row.
    """
    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        ed = editorial.get(run["run_id"], {})
        group = ed.get("group") or run["run_id"]
        bucket[group].append(run)

    rows: list[dict[str, Any]] = []
    for group, members in bucket.items():
        members.sort(key=lambda m: m["timestamp"])
        n_done = sum(m["n_done"] for m in members)
        n_pass = sum(m["n_pass"] for m in members)
        # Weighted avg
        weights = [m["n_done"] for m in members]
        avg_wall = (
            sum(m["avg_wall_s"] * w for m, w in zip(members, weights)) / sum(weights)
            if sum(weights)
            else 0.0
        )
        avg_llm = (
            sum(m["avg_llm"] * w for m, w in zip(members, weights)) / sum(weights)
            if sum(weights)
            else 0.0
        )
        # Editorial from the first member that has any
        ed_merged: dict[str, Any] = {}
        for m in members:
            ed_merged.update(editorial.get(m["run_id"], {}))
        # Viewer lookup: explicit override → group dir → first shard's experiment
        viewer_candidates: list[str] = []
        if ed_merged.get("viewer"):
            viewer_candidates.append(ed_merged["viewer"])
        for candidate_name in [group, members[0]["experiment"]]:
            if "/" in candidate_name:
                # group is a run_id, not an experiment name → skip
                continue
            link = viewer_link_for(candidate_name, logs_dir)
            if link:
                viewer_candidates.append(link)
                break
        rows.append(
            {
                "group": group,
                "date": min(m["date"] for m in members),
                "run_ids": [m["run_id"] for m in members],
                "experiment": members[0]["experiment"],
                "n_done": n_done,
                "n_pass": n_pass,
                "avg_wall_s": avg_wall,
                "avg_llm": avg_llm,
                "viewer": viewer_candidates[0] if viewer_candidates else None,
                "headline": ed_merged.get("headline", ""),
                "notes": ed_merged.get("notes", ""),
            }
        )

    rows.sort(key=lambda r: (r["date"], r["group"]))
    return rows


def compute_deltas(rows: list[dict[str, Any]]) -> None:
    """Annotate each row with delta vs prev best PASS and high-water flags.

    `is_high_water_at_time` = the row improved on the prior best when it ran.
    `is_current_high_water` = the chronologically latest row whose PASS
                              count equals the overall max (only one row).
    """
    best_so_far: int | None = None
    for row in rows:
        # Only 56-task (or larger) runs participate in the high-water tally.
        if row["n_done"] < 56:
            row["delta_vs_best"] = "—"
            row["is_high_water_at_time"] = False
            continue
        if best_so_far is None:
            row["delta_vs_best"] = "—"
            best_so_far = row["n_pass"]
            row["is_high_water_at_time"] = True
        else:
            delta = row["n_pass"] - best_so_far
            row["delta_vs_best"] = f"{delta:+d}" if delta else "0"
            if row["n_pass"] > best_so_far:
                row["is_high_water_at_time"] = True
                best_so_far = row["n_pass"]
            else:
                row["is_high_water_at_time"] = False

    # Mark only the chronologically latest row holding the all-time max PASS
    # count among full-size rows.
    full_rows = [r for r in rows if r["n_done"] >= 56]
    if full_rows:
        max_pass = max(r["n_pass"] for r in full_rows)
        latest_winner = max(
            (r for r in full_rows if r["n_pass"] == max_pass),
            key=lambda r: r["date"],
        )
        for r in rows:
            r["is_current_high_water"] = r is latest_winner
    else:
        for r in rows:
            r["is_current_high_water"] = False


def render_md(rows: list[dict[str, Any]]) -> str:
    out: list[str] = []
    out.append("# Full56 run registry (auto-generated)\n")
    out.append(
        "Auto-generated by `scripts/build_run_registry.py`. Editorial fields "
        "live in `logs/_registry/editorial.json` — past rows are preserved "
        "across regenerations. Run dirs themselves are never deleted "
        "(per `feedback_preserve_eval_logs.md`).\n"
    )
    out.append("To refresh after a new full56:\n")
    out.append("```\nuv run python scripts/build_run_registry.py\n```\n")
    out.append(
        "To add a headline / notes / shard grouping, edit "
        "`logs/_registry/editorial.json` and re-run the script.\n"
    )
    out.append("---\n")

    headers = [
        "Date",
        "Run dir(s)",
        "Tasks",
        "Δ vs prev best",
        "Avg wall",
        "Avg LLM",
        "Headline change",
        "Viewer",
        "Notes",
    ]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        run_dirs_md = "<br>".join(f"`{rid}`" for rid in row["run_ids"])
        tasks_cell = f"{row['n_pass']}/{row['n_done']}"
        if row.get("is_current_high_water"):
            tasks_cell = f"**{tasks_cell}**"
        viewer_md = f"[viewer]({row['viewer']})" if row.get("viewer") else "—"
        headline = row["headline"] or "_(set headline in editorial.json)_"
        notes = row["notes"] or ""
        if row.get("is_current_high_water"):
            prefix = "**current high-water mark.** "
            notes = prefix + notes if notes else prefix.rstrip(". ") + ""
        elif row.get("is_high_water_at_time"):
            prefix = "_was high-water at time of run._ "
            notes = prefix + notes if notes else prefix.rstrip(". ") + ""
        out.append(
            "| "
            + " | ".join(
                [
                    row["date"],
                    run_dirs_md,
                    tasks_cell,
                    row["delta_vs_best"],
                    f"{row['avg_wall_s']:.1f}s",
                    f"{row['avg_llm']:.0f}",
                    headline,
                    viewer_md,
                    notes,
                ]
            )
            + " |"
        )

    out.append("\n## Editorial workflow\n")
    out.append(
        "1. New full56 batch completes → re-run "
        "`scripts/build_run_registry.py` to auto-add the row.\n"
        "2. The row appears with `_(set headline in editorial.json)_` — open "
        "`logs/_registry/editorial.json`, add an entry for the new run_id, "
        "and fill in `headline` (≤10 字 of what changed) plus optional `notes` "
        "and `group`.\n"
        "3. Re-run the script. Row updates in place.\n"
    )
    out.append(
        "If a run is sharded (e.g. v1 + v1b because of "
        "the `--limit 10` footgun), set the same `group` string on every "
        "shard's editorial entry. They will combine into one row with "
        "summed `n_done` / `n_pass`.\n"
    )
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=DEFAULT_LOGS_DIR,
        help="Root logs directory to scan (default: repo logs/).",
    )
    parser.add_argument(
        "--editorial",
        type=Path,
        default=DEFAULT_EDITORIAL,
        help="Editorial JSON path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output markdown path.",
    )
    parser.add_argument(
        "--min",
        type=int,
        default=30,
        help="Minimum tasks completed to register a run (default 30).",
    )
    parser.add_argument(
        "--init-empty-editorial",
        action="store_true",
        help="Create empty editorial.json if missing (won't overwrite).",
    )
    args = parser.parse_args()

    editorial = load_editorial(args.editorial)
    if args.init_empty_editorial and not args.editorial.exists():
        save_editorial(args.editorial, {})

    runs = discover_runs(args.logs_dir)
    if not runs:
        print(f"No batches found under {args.logs_dir}", file=sys.stderr)
        sys.exit(0)

    rows = merge_groups(runs, editorial, args.logs_dir)
    # Apply task-count floor after grouping so sharded runs aggregate first.
    rows = [r for r in rows if r["n_done"] >= args.min]
    compute_deltas(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_md(rows))

    print(f"Scanned {len(runs)} run(s) → {len(rows)} registry row(s)")
    print(f"Wrote {args.out}")

    # Report runs missing editorial
    missing = [
        r["run_id"]
        for r in runs
        if r["run_id"] not in editorial or not editorial[r["run_id"]].get("headline")
    ]
    grouped = {
        editorial[k].get("group")
        for k in editorial
        if editorial.get(k, {}).get("group")
    }
    actually_missing = [
        rid
        for rid in missing
        if not (rid in editorial and editorial[rid].get("group") in grouped)
    ]
    if actually_missing:
        print("\nRuns without an editorial headline:")
        for rid in actually_missing:
            print(f"  - {rid}")
        print(f"\nAdd entries to {args.editorial} and re-run.")


if __name__ == "__main__":
    main()
