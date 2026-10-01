#!/usr/bin/env python3
"""Static HTML viewer for cycle-recovery batch failure analysis.

Reads subagent_io.jsonl + task_summary.json from current run dirs
and a baseline run dir, then emits a 3-level navigable site:

    index.html  →  pattern_<NAME>.html  →  task_<TID>.html

Each task page surfaces the per-call rendered_prompt, output envelope,
and (when available) the baseline counterpart's call at the same cycle
so the user can diff inputs side by side.

Usage:
    python scripts/build_analysis_viewer.py \\
        --run logs/cycle_recovery_full56_v1/20260512_101038_909559 \\
        --run logs/cycle_recovery_full56_v1b/20260512_111819_572932 \\
        --baseline logs/cuga_policy_full_v2/20260510_145752_337095 \\
        --pattern-catalog logs/cycle_recovery_full56_v1/analysis/_patterns.json \\
        --out logs/cycle_recovery_full56_v1/analysis/viewer
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Allow `from scripts.lib import translations` whether invoked as
# `scripts/build_analysis_viewer.py` or `python -m scripts.build_analysis_viewer`.
_REPO_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _REPO_SCRIPTS.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from adk_appworld_agent.repo_paths import (  # noqa: E402
    load_skill_config,
    save_skill_config,
)
from adk_appworld_agent.task_sets import TASK_SETS_DIR  # noqa: E402
from scripts.lib import translations as translations_cache  # noqa: E402

# Timestamp dirs look like 20260512_101038_909559
_TS_RE = re.compile(r"^20\d{6}_\d+")
# batch_results.txt rows look like:
#   [9dabbc9_2] PASS 9/9  | 321.4s | 20 llm | 76,210 tok
_BR_RE = re.compile(r"^\[[a-z0-9_]+\]\s+(PASS|FAIL)\b")


def _scan_batches(logs_dir: Path) -> list[dict[str, Any]]:
    """Return per-shard PASS counts for every batch under logs_dir."""
    out: list[dict[str, Any]] = []
    if not logs_dir.is_dir():
        return out
    for exp_dir in sorted(logs_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        for ts_dir in sorted(exp_dir.iterdir()):
            if not ts_dir.is_dir() or not _TS_RE.match(ts_dir.name):
                continue
            br = ts_dir / "batch_results.txt"
            if not br.exists():
                continue
            n_done = n_pass = 0
            for line in br.read_text().splitlines():
                m = _BR_RE.match(line.strip())
                if not m:
                    continue
                n_done += 1
                if m.group(1) == "PASS":
                    n_pass += 1
            if n_done == 0:
                continue
            out.append(
                {
                    "run_id": f"{exp_dir.name}/{ts_dir.name}",
                    "experiment": exp_dir.name,
                    "timestamp": ts_dir.name,
                    "date": f"{ts_dir.name[:4]}-{ts_dir.name[4:6]}-{ts_dir.name[6:8]}",
                    "n_done": n_done,
                    "n_pass": n_pass,
                }
            )
    return out


def scan_all_task_results(logs_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Walk every logs/<exp>/<ts>/<tid>/artifacts/task_summary.json once and
    bucket per-task historical results across the whole logs tree.
    """
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not logs_dir.is_dir():
        return out
    for exp_dir in logs_dir.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        for ts_dir in exp_dir.iterdir():
            if not ts_dir.is_dir() or not _TS_RE.match(ts_dir.name):
                continue
            for task_dir in ts_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                summary_path = task_dir / "artifacts" / "task_summary.json"
                if not summary_path.exists():
                    continue
                try:
                    s = json.loads(summary_path.read_text())
                except Exception:
                    continue
                tid = task_dir.name
                out[tid].append(
                    {
                        "run_id": f"{exp_dir.name}/{ts_dir.name}",
                        "date": f"{ts_dir.name[:4]}-{ts_dir.name[4:6]}-{ts_dir.name[6:8]}",
                        "status": s.get("status"),
                        "eval": s.get("eval") or {},
                        "wall_s": s.get("wall_s"),
                        "aggregate_metrics": s.get("aggregate_metrics") or {},
                        "block_reason": (s.get("overview") or {}).get("block_reason"),
                    }
                )
    return out


def is_pass(summary: dict[str, Any]) -> bool:
    """Canonical task success = AppWorld eval goal completion (all tests pass).

    Matches run_batch.py's batch_results.txt headline. A task whose final
    environment state passes every eval assertion is SOLVED even when the
    controller's status is not "COMPLETED" (e.g. it hit MAX_CYCLES after the
    goal was already reached). Falls back to controller status only when eval
    results are absent (e.g. a missing/blacklisted run).
    """
    ev = summary.get("eval") or {}
    if ev.get("passed_all") is not None:
        return bool(ev.get("passed_all"))
    total = ev.get("total")
    if total:
        return (ev.get("passed") or 0) == total
    return summary.get("status") == "COMPLETED"


def best_result_for(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the best historical result for a task.

    Score priority: COMPLETED > FAILED-with-higher-pass-ratio > TIMED_OUT.
    Ties broken by latest date (re-running an equivalent result is treated
    as the more recent record).
    """
    if not results:
        return None

    def score(r: dict[str, Any]) -> tuple[float, str]:
        st = r["status"]
        ev = r["eval"] or {}
        if st == "COMPLETED" or ev.get("passed_all"):
            base = 1000.0
        elif st == "FAILED":
            passed = ev.get("passed") or 0
            total = ev.get("total") or 1
            base = 100.0 + (passed / max(total, 1)) * 50.0
        else:  # TIMED_OUT, ERROR, unknown
            base = 0.0
        return (base, r["date"])

    return max(results, key=score)


def _milestone_done_criteria_from_io(calls: list[dict[str, Any]]) -> dict[int, str]:
    """For each milestone_index, return the done_criteria from its first
    EXECUTE call's input metadata. EXECUTE calls store milestone_intent +
    milestone_done_criteria as flat keys on io_record.input.
    """
    out: dict[int, str] = {}
    for c in calls:
        if c.get("phase") != "EXECUTE":
            continue
        inp = (c.get("io_record") or {}).get("input") or {}
        idx = inp.get("milestone_index")
        dc = inp.get("milestone_done_criteria")
        if isinstance(idx, int) and isinstance(dc, str) and idx not in out:
            out[idx] = dc
    return out


def build_starter_md(tid: str, task: dict[str, Any]) -> str:
    """Compose a Phase-0 starter .md with sub-tests + milestones pre-filled.

    Frontmatter has placeholders; analyst fills root_layer / pattern /
    flip_vs_baseline / research_scope. Body has Phase 1 sections
    pre-populated from runtime data — analyst writes the verdict +
    Phase 2-4 + Gaps.
    """
    summary = task.get("summary") or {}
    calls = task.get("calls") or []
    instruction = summary.get("instruction", "")
    submission = summary.get("submission") or {}
    sub_tests = submission.get("sub_test_results") or []
    milestones = summary.get("milestones") or []
    done_crits = _milestone_done_criteria_from_io(calls)

    # Sub-tests block — 4-space indented, plain text. Status + assertion
    # quote each on their own line for source-mode readability.
    st_lines: list[str] = []
    for i, st in enumerate(sub_tests):
        if not isinstance(st, dict):
            continue
        status = (st.get("status") or "").upper() or "?"
        req = " ".join((st.get("requirement") or "").split())
        err = (st.get("error") or "").strip()
        st_lines.append(f"    [{i + 1}] {status}  {req}")
        if status != "PASS" and err:
            err_one = err.split("\n")[0]
            st_lines.append(f"              {err_one}")
    st_block = "\n".join(st_lines) if st_lines else "    (沒有 sub-test 紀錄)"

    # Milestones block — one milestone per "M<i> [<app>]  <intent>" line;
    # done_criteria only shown when distinct from intent.
    ms_lines: list[str] = []
    for m in milestones:
        if not isinstance(m, dict):
            continue
        idx = m.get("index", "?")
        app = m.get("app", "?")
        goal = (m.get("goal") or "").strip()
        ms_lines.append(f"    M{idx} [{app}]  {goal}")
        dc = done_crits.get(idx if isinstance(idx, int) else -1)
        if dc and dc.strip() != goal:
            ms_lines.append(f"              done_criteria: {dc}")
    ms_block = "\n".join(ms_lines) if ms_lines else "    (沒有 milestone 紀錄)"

    return f"""---
task_id: {tid}
root_layer: TBD                   # PLAN | VERIFY | FIND | EXECUTE.code_planner | EXECUTE.code_agent | EXECUTE.sandbox | DATA | ENV
pattern: TBD                       # P1-P11 cluster tag, optional
flip_vs_baseline: TBD              # gain | loss | same-fail | same-pass
research_scope: TBD                # RESEARCH-RELEVANT | OUT-OF-RESEARCH-SCOPE
---

# 摘要

<analyst: 3-5 個短段, scannable. 講 (a) task 要什麼, (b) 問題在哪個 layer 的哪個動作,
(c) 最終 eval 失敗的具體 numbers, (d) 是否屬已知 pattern cluster.
不寫 200+ 字的密集段落; 用 bullet 或短句分隔.>


# Phase 1 — milestone 合理性檢查

Task instruction:

    {instruction}

Constraints (從 instruction 抽):

    <analyst: c1, c2, ... 列出 noun qualifiers / 時間 / 條件 / 格式 / 動作方向>

Sub-tests:

{st_block}

Milestones (as worded):

{ms_block}

Sub-test → milestone coverage (assume each milestone executes perfectly):

    <analyst: 對每個 [FAIL] sub-test 列:
        expected = <state>
        M<X> wording 完美執行 = <state>
        gap: <如果有>
        verdict: uncovered | covered

     如果有任何 uncovered → PLAN broken.>

VERDICT: <PLAN sane | PLAN broken>
理由:    <一句話, 引 sub-test 跟 milestone wording 的 gap>


# Phase 2 — 分歧定位

<analyst:
 - 若 VERDICT = PLAN broken: 寫 "Skipped — Phase 1 已歸 PLAN".
 - 若 VERDICT = PLAN sane: 做 sub-stage walk (EXECUTE.sandbox → code_agent → code_planner → FIND).>


# Phase 3 — 結論

[Y] <picked layer>
    <一句 rationale>

其餘 7 層無責 — 證據在 Phase 1/2.

Secondary contributor: <layer> — <一句>     # optional, 留空就刪這行


# Phase 4 — Research-scope

Picked:  <Phase 3 picked layer>
Focus:   API retrieval (FIND layer)
Flag:    <RESEARCH-RELEVANT | OUT-OF-RESEARCH-SCOPE>
理由:    <一句話>


# Gaps

<analyst: 每條 gap 編號條列, 或寫 "None — Phase 1+3+4 都有 evidence。">
"""


def _split_md_body(body: str) -> tuple[str, str]:
    """Split markdown body into (summary, rest).

    summary = content directly under `# 摘要`, stopping at the next `# ` heading.
    rest    = everything else (Phase 1-4 + Gaps), preserved verbatim.

    Both returned without the leading `# 摘要` header itself.
    """
    body = body.lstrip("\n").rstrip()
    lines = body.splitlines()
    # Find `# 摘要` heading (also accept "# Summary" / "# 根因分析（人工）" legacy).
    summary_start = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading.startswith(("摘要", "Summary", "根因分析")):
                summary_start = i
                break
    if summary_start is None:
        # No explicit summary section — treat the whole body as summary.
        return body, ""
    # Find next top-level "# " heading after summary_start
    summary_end = len(lines)
    for j in range(summary_start + 1, len(lines)):
        if lines[j].lstrip().startswith("# "):
            summary_end = j
            break
    summary = "\n".join(lines[summary_start + 1 : summary_end]).strip()
    # rest = everything before summary_start (rare) + everything from summary_end on
    rest_parts: list[str] = []
    if summary_start > 0:
        rest_parts.append("\n".join(lines[:summary_start]).strip())
    if summary_end < len(lines):
        rest_parts.append("\n".join(lines[summary_end:]).strip())
    rest = "\n\n".join(p for p in rest_parts if p).strip()
    return summary, rest


def load_per_task_md(tasks_dir: Path | None) -> dict[str, dict[str, Any]]:
    """Read every `<tid>.md` under tasks_dir, parse YAML frontmatter + body.

    Returns a map task_id → {frontmatter_keys..., "summary": <摘要 only>,
    "body_rest": <Phase 1-4 + Gaps>}. Splitting lets the viewer show only
    the concise summary in the analysis card and the full walk in a
    collapsible block below.
    """
    if tasks_dir is None or not tasks_dir.is_dir():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.stderr.write(
            "[viewer] pyyaml not installed — per-task .md files cannot be parsed\n"
        )
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in tasks_dir.glob("*.md"):
        text = path.read_text()
        if not text.startswith("---"):
            continue
        _, _, rest = text.partition("---\n")
        front, _, body = rest.partition("\n---")
        try:
            fm = yaml.safe_load(front) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(fm, dict):
            continue
        # YAML treats `_` in numeric literals as thousand-separator, so
        # all-digit task_ids like `7847649_2` parse to int(78476492) and
        # silently mismatch the rest of the pipeline. Coerce to string.
        # path.stem is authoritative — fall back if frontmatter is empty.
        tid_raw = fm.get("task_id")
        tid = str(tid_raw) if tid_raw not in (None, "") else path.stem
        if tid != path.stem:
            # Re-canonicalize using the filename so int-parsed IDs realign.
            tid = path.stem
        summary, body_rest = _split_md_body(body)
        # Drop placeholder markers so the viewer treats "no analysis yet" as empty.
        if summary in ("_(尚未撰寫分析。)_", "(尚未撰寫分析。)"):
            summary = ""
        out[tid] = {**fm, "summary": summary, "body_rest": body_rest, "task_id": tid}
    return out


def load_translations(registry_dir: Path | None) -> dict[str, str]:
    """Read the permanent task_translations_zh.json cache at the registry path.

    Translations are immutable per task_id and shared across all batches —
    they live at logs/_registry/task_translations_zh.json, NOT in per-batch
    <tid>.md frontmatter. New task_ids without a cache entry render as the
    original English in the viewer (with a "untranslated" marker) until
    Claude translates them and writes the entry into the JSON.
    """
    if registry_dir is None:
        return {}
    return translations_cache.load_translations(Path(registry_dir))


def load_analyses(per_task: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Derive task_id → analysis dict (layer/pattern/flip/summary/body_rest) from .md.

    summary_zh = `# 摘要` content only (rendered in viewer's analysis card).
    body_rest  = Phase 1-4 + Gaps (rendered in a collapsible block below).
    """
    out: dict[str, dict[str, str]] = {}
    for tid, data in per_task.items():
        entry: dict[str, str] = {}
        for key in ("root_layer", "pattern", "flip_vs_baseline", "research_scope"):
            if isinstance(data.get(key), str) and data[key] and data[key] != "TBD":
                entry[key] = data[key]
        if data.get("summary"):
            entry["summary_zh"] = data["summary"]
        if data.get("body_rest"):
            entry["body_rest"] = data["body_rest"]
        if entry.get("summary_zh") or entry.get("root_layer"):
            out[tid] = entry
    return out


def build_dynamic_pattern_catalog(
    task_outcomes: dict[str, dict],
    per_task_md: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build pattern catalog from .md frontmatter + current/baseline outcomes.

    Returns (flip_catalog, fail_pattern_catalog):
    - flip_catalog: {"GAIN": {...}, "LOSS": {...}} — by binary outcome change
    - fail_pattern_catalog: {pattern_name: {...}} — current-FAIL tasks grouped by
      .md `pattern:` field. PASS tasks excluded so they don't pollute fail
      clusters (the bug in the hardcoded catalog).
    """
    # Bucket tasks by binary flip status
    gains: list[str] = []
    losses: list[str] = []
    same_pass: list[str] = []
    for tid, oc in task_outcomes.items():
        bs = (oc["baseline"] or {}).get("summary", {}) if oc["baseline"] else {}
        cs = oc["current"]["summary"]
        bp = is_pass(bs)
        cp = is_pass(cs)
        if bp and not cp:
            losses.append(tid)
        elif (not bp) and cp:
            gains.append(tid)
        elif bp and cp:
            same_pass.append(tid)

    flip_catalog: dict[str, dict[str, Any]] = {}
    if gains:
        flip_catalog["GAIN"] = {
            "title": "FAIL → PASS (binary gains)",
            "description": (
                "Baseline FAIL → current PASS. Read continuation_planner's "
                "rationale at the cycle that unblocked the milestone — that's "
                "the mechanism the fix targets."
            ),
            "tasks": sorted(gains),
        }
    if losses:
        flip_catalog["LOSS"] = {
            "title": "PASS → FAIL (binary losses)",
            "description": (
                "Baseline PASS → current FAIL/TIMED_OUT. Trajectory-diff to "
                "confirm whether mechanism is stable or stochastic. Block any "
                "merge until you can quote which cycle decision regressed."
            ),
            "tasks": sorted(losses),
        }

    # Bucket currently-failing tasks by .md pattern. Tasks that are now PASS
    # never enter this catalog even if their old .md still tags them (.md is
    # only written for current FAIL anyway, but we double-filter).
    pattern_bucket: dict[str, list[str]] = {}
    for tid, data in per_task_md.items():
        oc = task_outcomes.get(tid)
        if not oc:
            continue
        if is_pass(oc["current"]["summary"]):
            continue  # currently PASS, exclude from fail patterns
        pat = data.get("pattern")
        if not isinstance(pat, str) or not pat or pat == "TBD":
            pat = "UNCATEGORIZED"
        pattern_bucket.setdefault(pat, []).append(tid)

    fail_pattern_catalog: dict[str, dict[str, Any]] = {}
    for pat, tids in pattern_bucket.items():
        fail_pattern_catalog[pat] = {
            "title": pat,
            "description": (
                f"Tasks in this run currently FAIL with pattern '{pat}' "
                "(declared in per-task .md frontmatter)."
            ),
            "tasks": sorted(tids),
        }

    return flip_catalog, fail_pattern_catalog


def summarize_failure(cs: dict[str, Any]) -> dict[str, Any]:
    """Return structured failure summary for multi-line rendering.

    Returns:
        {} for COMPLETED tasks, otherwise a dict with keys:
        - `block`: block_reason / failure_code
        - `failure_point`: layered location e.g. "SUBMIT::COMPLETION_EVAL_FAILED"
        - `one_liner`: runtime-emitted one-liner (or fallback)
        - `result_line`: "3 / 5 sub-tests passed" style line
        - `first_failed_req`: assertion requirement text
        - `first_failed_err`: AssertionError details (multi-line OK)
    """
    if is_pass(cs):
        return {}
    overview = cs.get("overview") or {}
    submission = cs.get("submission") or {}
    eval_d = cs.get("eval") or {}

    one_liner = overview.get("failure_one_liner")
    block = overview.get("block_reason") or overview.get("submit_original_failure_code")
    failure_point = overview.get("failure_point")
    status = cs.get("status")
    if not one_liner and status == "TIMED_OUT":
        wall = cs.get("wall_s") or 0
        one_liner = f"Hit {wall:.1f}s timeout, agent 沒有 finalize 出答案"

    passed = eval_d.get("passed")
    total = eval_d.get("total")
    result_line = ""
    if passed is not None and total is not None:
        result_line = f"{passed} / {total} sub-tests passed"

    sub_tests_raw = submission.get("sub_test_results") or []
    sub_tests: list[dict[str, str]] = []
    for st in sub_tests_raw:
        if not isinstance(st, dict):
            continue
        req = " ".join((st.get("requirement") or "").strip().split())
        sub_tests.append(
            {
                "status": st.get("status") or "",
                "requirement": req,
                "error": (st.get("error") or "").strip(),
            }
        )

    return {
        "block": block or "",
        "failure_point": failure_point or "",
        "one_liner": one_liner or "",
        "result_line": result_line,
        "sub_tests": sub_tests,
    }


def render_failure_block(f: dict[str, Any]) -> str:
    """Render structured failure summary as multi-line HTML block."""
    if not f:
        return ""
    rows: list[str] = ['<div class="failure-card">']
    rows.append(
        '<div style="font-weight:600;margin-bottom:0.4rem;color:#721c24">Eval 失敗訊息</div>'
    )
    bits: list[str] = []
    if f.get("block"):
        bits.append(f"<code>block={esc(f['block'])}</code>")
    if f.get("failure_point"):
        bits.append(f"<code>@ {esc(f['failure_point'])}</code>")
    if f.get("result_line"):
        bits.append(esc(f["result_line"]))
    if bits:
        rows.append('<div style="margin-bottom:0.4rem;">' + " · ".join(bits) + "</div>")
    if f.get("one_liner") and f["one_liner"] not in (f.get("result_line") or ""):
        rows.append(f'<div style="margin-bottom:0.4rem;">{esc(f["one_liner"])}</div>')

    sub_tests = f.get("sub_tests") or []
    if sub_tests:
        rows.append(
            '<div style="margin-top:0.6rem;"><strong>Sub-tests (full list):</strong></div>'
        )
        rows.append('<div style="margin-top:0.3rem;">')
        for i, st in enumerate(sub_tests):
            status = (st.get("status") or "").lower()
            if status == "pass":
                badge = '<span class="badge badge-pass">PASS</span>'
                req_border = "#28a745"
            else:
                badge = '<span class="badge badge-fail">FAIL</span>'
                req_border = "#dc3545"
            req = st.get("requirement") or ""
            err = st.get("error") or ""
            rows.append(
                f'<div style="margin: 0.4rem 0; padding: 0.4rem 0.5rem; '
                f'background: #fff; border-left: 3px solid {req_border}; border-radius: 2px;">'
                f'<div style="margin-bottom: 0.2rem;">{badge} '
                f'<span style="font-family:SFMono-Regular,Consolas,monospace;font-size:0.85em;">'
                f"[{i + 1}] {esc(req)}</span></div>"
            )
            if err and status != "pass":
                rows.append(
                    f'<pre style="font-family:SFMono-Regular,Consolas,monospace;font-size:0.8em;'
                    f"margin:0.2rem 0 0 0;padding:0.3rem 0.4rem;background:#fef5f5;"
                    f"border:1px solid #f5c6cb;border-radius:3px;white-space:pre-wrap;"
                    f'max-height:150px;overflow:auto;">{esc(err)}</pre>'
                )
            rows.append("</div>")
        rows.append("</div>")
    rows.append("</div>")
    return "\n".join(rows)


def find_high_water(
    logs_dir: Path, registry_dir: Path | None = None
) -> dict[str, Any] | None:
    """Find the all-time best full56-class run, applying editorial groupings.

    Reads `<registry_dir>/editorial.json` if present so sharded runs
    (multiple run_ids sharing a `group` key) combine into one logical run
    — matches scripts/build_run_registry.py's logic.
    """
    shards = _scan_batches(logs_dir)
    if not shards:
        return None
    editorial: dict[str, dict[str, Any]] = {}
    if registry_dir is not None:
        ed_path = registry_dir / "editorial.json"
        if ed_path.exists():
            try:
                editorial = json.loads(ed_path.read_text())
            except json.JSONDecodeError:
                editorial = {}

    bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in shards:
        ed = editorial.get(s["run_id"], {})
        group = ed.get("group") or s["run_id"]
        bucket[group].append(s)

    rows: list[dict[str, Any]] = []
    for group, members in bucket.items():
        n_done = sum(m["n_done"] for m in members)
        n_pass = sum(m["n_pass"] for m in members)
        if n_done < 56:
            continue
        rows.append(
            {
                "group": group,
                "run_ids": [m["run_id"] for m in members],
                "date": min(m["date"] for m in members),
                "n_done": n_done,
                "n_pass": n_pass,
            }
        )
    if not rows:
        return None
    max_pass = max(r["n_pass"] for r in rows)
    # Latest by date among rows holding the max.
    winner = max(
        (r for r in rows if r["n_pass"] == max_pass),
        key=lambda r: r["date"],
    )
    return winner


# PATTERN_CATALOG is dynamically built from per-task .md frontmatter at
# build_analysis_viewer.main() time via build_dynamic_pattern_catalog().
# This module-global is overwritten in main() before any pattern lookup
# happens. Empty default = no patterns until main() rebuilds.
PATTERN_CATALOG: dict[str, dict[str, Any]] = {}


CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1400px; margin: 1rem auto; padding: 0 1rem 4rem; line-height: 1.5; color: #222; }
h1 { margin-bottom: 0.2rem; }
h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }
h3 { margin-top: 1.5rem; }
.subtitle { color: #586069; }
.nav { margin-bottom: 1rem; font-size: 0.9em; }
.nav a { margin-right: 1rem; }
a { color: #0366d6; text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9em; }
th, td { border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
th { background: #f4f4f4; }
tr:hover td { background: #fafafa; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; font-weight: bold; white-space: nowrap; }
.badge-pass { background: #d4edda; color: #155724; }
.badge-fail { background: #f8d7da; color: #721c24; }
.badge-timeout { background: #fff3cd; color: #856404; }
.badge-gain { background: #28a745; color: white; }
.badge-loss { background: #dc3545; color: white; }
.badge-same { background: #6c757d; color: white; }
.badge-na { background: #e9ecef; color: #495057; }
.cycle-row td { font-family: SFMono-Regular, Consolas, monospace; font-size: 0.85em; }
.prompt-text { background: #f6f8fa; border: 1px solid #e1e4e8; padding: 0.75rem; font-family: SFMono-Regular, Consolas, monospace; font-size: 0.82em; white-space: pre-wrap; max-height: 500px; overflow: auto; border-radius: 4px; }
.json-block { background: #f6f8fa; border: 1px solid #e1e4e8; padding: 0.5rem; font-family: SFMono-Regular, Consolas, monospace; font-size: 0.8em; max-height: 400px; overflow: auto; white-space: pre-wrap; border-radius: 4px; }
details { margin: 0.5rem 0; border: 1px solid #ddd; border-radius: 4px; padding: 0 0.5rem; }
details[open] { padding-bottom: 0.5rem; }
summary { cursor: pointer; font-weight: 600; padding: 0.5rem 0; user-select: none; }
summary:hover { color: #0366d6; }
.side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.side-by-side > div { border: 1px solid #ddd; padding: 0.5rem; border-radius: 4px; min-width: 0; }
.side-by-side h4 { margin-top: 0; padding-bottom: 0.3rem; border-bottom: 1px solid #eee; }
.label { display: inline-block; min-width: 130px; color: #586069; font-size: 0.9em; }
.summary-card { background: #f6f8fa; padding: 1rem; border-radius: 4px; margin: 1rem 0; }
.kv { margin: 0.2rem 0; }
.kv .key { display: inline-block; min-width: 150px; color: #586069; font-weight: 600; }
.pattern-card { border: 1px solid #ddd; border-radius: 4px; padding: 1rem; margin: 1rem 0; }
.pattern-card h3 { margin-top: 0; }
.divergence-marker { background: #fffbeb; border-left: 4px solid #ffc107; padding-left: 0.5rem; }
.analysis-card { background: #eef6ff; border-left: 4px solid #0366d6; padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 4px; line-height: 1.6; }
.analysis-card .badge { margin-right: 0.3rem; }
.analysis-detail-card { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px; padding: 0.4rem 0.75rem; margin: 0.5rem 0 1rem 0; }
.analysis-detail-card summary { cursor: pointer; color: #586069; font-size: 0.9em; }
.analysis-body-pre { font-family: SFMono-Regular, Consolas, monospace; font-size: 0.85em; line-height: 1.5; background: white; padding: 0.75rem; margin: 0.5rem 0 0 0; white-space: pre-wrap; border-radius: 3px; border: 1px solid #e1e4e8; max-height: 600px; overflow: auto; }
.failure-card { background: #fef5f5; border-left: 4px solid #dc3545; padding: 0.75rem 1rem; margin: 1rem 0; border-radius: 4px; line-height: 1.6; }
.call-step { margin-top: 1.3rem; padding: 0.3rem 0.6rem; background: #f0f4f8; border-left: 3px solid #586069; font-size: 1.0em; }
.call-output-block { margin-top: 1.5rem; padding: 0.6rem 0.8rem; background: #fbf8f0; border: 1px solid #e1d8b8; border-radius: 4px; border-left: 4px solid #b8860b; }
.call-output-block h4 { margin-top: 0.3rem; }
.tabs { display: flex; gap: 2px; margin-top: 1rem; border-bottom: 2px solid #0366d6; }
.tab-btn { padding: 0.5rem 1rem; cursor: pointer; background: #f6f8fa; border: 1px solid #ddd; border-bottom: none; border-radius: 4px 4px 0 0; font-size: 0.95em; font-weight: 600; color: #586069; user-select: none; }
.tab-btn:hover { background: #eef2f5; color: #0366d6; }
.tab-btn.active { background: #0366d6; color: white; border-color: #0366d6; }
.tab-content { display: none; padding-top: 1rem; }
.tab-content.active { display: block; }
.cmp-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
.cmp-col { min-width: 0; border: 1px solid #e1e4e8; border-radius: 6px; padding: 8px 10px; background: #fff; }
.cmp-col.cur { border-top: 4px solid #d73a49; }
.cmp-col.base { border-top: 4px solid #2188ff; }
.cmp-h { position: sticky; top: 0; background: #fff; z-index: 5; margin: 0 0 8px; padding: 6px 0; font-size: 1.05em; border-bottom: 1px solid #eee; }
.cmp-summary-box { border: 2px solid #0366d6; border-radius: 6px; padding: 10px 14px; margin: 14px 0; background: #f7fbff; }
.cmp-summary-h { font-weight: 700; font-size: 1.08em; margin-bottom: 8px; color: #0366d6; }
.cmp-summary-box ol { font-size: 0.95em; }
.ms-goal { font-weight: 600; margin-bottom: 3px; }
.ms-meta { font-size: 0.86em; color: #555; }
.ms-meta + .ms-meta { margin-top: 1px; }
.decision-diff { border: 1px solid #f0b429; background: #fffdf6; border-radius: 6px; padding: 8px 14px; margin-bottom: 16px; }
.decision-diff h3 { margin: 10px 0 6px; font-size: 1em; }
.diff-tbl { width: 100%; border-collapse: collapse; font-size: 0.86em; }
.diff-tbl th, .diff-tbl td { border: 1px solid #e1e4e8; padding: 4px 7px; text-align: left; vertical-align: top; }
.diff-tbl th { background: #f6f8fa; font-weight: 600; }
.diff-row-differ { background: #fff7ec; }
.diff-hl { background: #ffe2b8; font-weight: 600; border-radius: 3px; padding: 0 2px; }
.diff-intent { color: #555; max-width: 320px; }
.diff-kw { font-family: ui-monospace, monospace; font-size: 0.92em; color: #444; max-width: 260px; word-break: break-all; }
.cmp-none { color: #999; }
"""


def esc(s: Any) -> str:
    """HTML-escape an arbitrary value, converting None/non-str to str."""
    if s is None:
        return ""
    return html.escape(str(s))


def fmt_seconds(value: Any) -> str:
    """Render a seconds value rounded to 1 decimal place with 's' suffix.

    Used everywhere wall_s appears so the viewer doesn't surface microsecond
    noise like '1200.379s'.
    """
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    """Render an integer with thousands separators (e.g. 63261 → '63,261')."""
    if value is None or value == "":
        return ""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def fmt_json(obj: Any, max_len: int = 50000) -> str:
    """Pretty JSON, truncated."""
    try:
        s = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + f"\n\n... [truncated, full length = {len(s)} chars]"
    return s


def load_task(task_dir: Path) -> dict[str, Any]:
    """Load summary + per-call IO + sandbox API trace for one task directory."""
    summary_path = task_dir / "artifacts" / "task_summary.json"
    io_path = task_dir / "artifacts" / "subagent_io.jsonl"
    sandbox_path = task_dir / "artifacts" / "sandbox_api_calls.jsonl"

    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            pass

    calls: list[dict[str, Any]] = []
    if io_path.exists():
        for line in io_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except Exception:
                continue

    sandbox_trace: list[dict[str, Any]] = []
    if sandbox_path.exists():
        for line in sandbox_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                sandbox_trace.append(json.loads(line))
            except Exception:
                continue

    return {
        "summary": summary,
        "calls": calls,
        "sandbox_trace": sandbox_trace,
        "dir": task_dir,
    }


def _is_task_dir(p: Path) -> bool:
    return p.is_dir() and (p / "artifacts" / "task_summary.json").exists()


def _expand_run_dir(run_dir: Path) -> list[Path]:
    """Expand <experiment_dir> → list of <timestamp_subdir> that contain tasks.

    Three cases handled:
    1. run_dir IS a timestamp dir (children are task dirs) → return [run_dir].
    2. run_dir IS an experiment dir (children are timestamp dirs, grandchildren
       are task dirs) → return [run_dir/ts1, run_dir/ts2, ...] for all
       timestamp subdirs containing tasks.
    3. run_dir doesn't exist or is empty → return [].
    """
    if not run_dir.is_dir():
        return []
    children = [
        c for c in sorted(run_dir.iterdir()) if c.is_dir() and c.name != "batch_results"
    ]
    if not children:
        return []
    # Case 1: any child is a task dir → run_dir is itself a timestamp dir
    if any(_is_task_dir(c) for c in children):
        return [run_dir]
    # Case 2: children are sub-experiment / timestamp dirs that contain tasks
    expanded = []
    for c in children:
        grandchildren = [
            g for g in c.iterdir() if g.is_dir() and g.name != "batch_results"
        ]
        if any(_is_task_dir(g) for g in grandchildren):
            expanded.append(c)
    return expanded


def discover_tasks(run_dirs: list[Path]) -> dict[str, Path]:
    """Return mapping task_id -> task_dir for all completed tasks across run dirs.

    Each input run_dir may be either a timestamp dir or an experiment dir
    (will be auto-expanded to include all timestamp subdirs). When a task_id
    appears in multiple timestamp subdirs (e.g., resumed batch), the latest
    timestamp's task dir wins.
    """
    found: dict[str, Path] = {}
    for run_dir in run_dirs:
        for ts_dir in _expand_run_dir(run_dir):
            for td in sorted(ts_dir.iterdir()):
                if not _is_task_dir(td):
                    continue
                found[td.name] = td  # Latest sorted wins for resumed batches
    return found


def status_badge(status: str, eval_passed: int | None, eval_total: int | None) -> str:
    # Canonical pass = eval goal completion (all tests pass), even if the
    # controller status is not COMPLETED (e.g. MAX_CYCLES after goal reached).
    passed_all = (status == "COMPLETED") or bool(
        eval_total and eval_passed is not None and eval_passed == eval_total
    )
    if passed_all:
        label = f"PASS {eval_passed}/{eval_total}" if eval_total else "PASS"
        # Surface that the agent solved it but did not finalize cleanly.
        if status and status != "COMPLETED":
            label += f" ({status})"
        return f'<span class="badge badge-pass">{label}</span>'
    if status == "TIMED_OUT":
        return '<span class="badge badge-timeout">TIMED_OUT</span>'
    if status == "FAILED":
        label = f"FAIL {eval_passed}/{eval_total}" if eval_total else "FAIL"
        return f'<span class="badge badge-fail">{label}</span>'
    return f'<span class="badge badge-na">{esc(status or "?")}</span>'


def flip_badge(baseline_pass: bool, current_pass: bool) -> str:
    bp = baseline_pass
    cp = current_pass
    if bp and not cp:
        return '<span class="badge badge-loss">LOSS (PASS → FAIL)</span>'
    if not bp and cp:
        return '<span class="badge badge-gain">GAIN (FAIL → PASS)</span>'
    if bp and cp:
        return '<span class="badge badge-same">PASS / PASS</span>'
    return '<span class="badge badge-same">FAIL / FAIL</span>'


def pattern_for_task(tid: str) -> tuple[str, dict[str, Any]] | None:
    for name, meta in PATTERN_CATALOG.items():
        if tid in meta["tasks"]:
            return name, meta
    return None


def _user_text_from_messages(messages: Any) -> str:
    """Pull the user-facing text from a Gemini-style messages array."""
    if not isinstance(messages, list) or not messages:
        return ""
    # Prefer the first user-role message if role labels are present;
    # else fall back to the first message in the list.
    first = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "user"),
        messages[0] if isinstance(messages[0], dict) else None,
    )
    if not isinstance(first, dict):
        return ""
    parts = first.get("parts") or []
    if parts and isinstance(parts[0], dict):
        text = parts[0].get("text")
        if text:
            return str(text)
    # Direct content field fallback
    content = first.get("content")
    if isinstance(content, str) and content:
        return content
    return ""


def _extract_user_prompt(call_input: dict[str, Any]) -> str:
    """Pull the user-facing prompt out of an input dict regardless of shape.

    Tries in order:
      1. top-level rendered_prompt (rough_planner / continuation single-call)
      2. model_input_raw.prompt / .user_prompt / .rendered_prompt (legacy)
      3. model_input_raw.messages[0].parts[0].text (executor / finder)
      4. model_input.messages[0].parts[0].text (PLAN messages)
    """
    if not isinstance(call_input, dict):
        return ""
    rp = call_input.get("rendered_prompt")
    if rp:
        return str(rp)
    raw = call_input.get("model_input_raw")
    if isinstance(raw, dict):
        for key in ("prompt", "user_prompt", "rendered_prompt"):
            value = raw.get(key)
            if value:
                return str(value)
        text = _user_text_from_messages(raw.get("messages"))
        if text:
            return text
    mi = call_input.get("model_input")
    if isinstance(mi, dict):
        text = _user_text_from_messages(mi.get("messages"))
        if text:
            return text
    return ""


def _extract_system_instruction(call_input: dict[str, Any]) -> str:
    """Pull the system instruction regardless of shape."""
    if not isinstance(call_input, dict):
        return ""
    mi = call_input.get("model_input")
    if isinstance(mi, dict):
        sysi = mi.get("system_instruction")
        if sysi:
            return str(sysi)
    raw = call_input.get("model_input_raw")
    if isinstance(raw, dict):
        for key in ("system_prompt", "system_instruction"):
            value = raw.get(key)
            if value:
                return str(value)
    return ""


def _render_call_step(
    call_input: dict[str, Any], call_output: dict[str, Any]
) -> list[str]:
    """Render one LLM call's INPUT (system_instruction collapsed +
    rendered_prompt) AND the per-step OUTPUT that lives inside this call's
    model_calls[i].out (raw_llm_text, plan_steps for code_plan; code +
    raw_stdout for code_execute; raw_llm_text / parsed_json for single-step
    planners). envelope.payload at the call level still shows the canonical
    aggregated form, but per-step output is shown inline so the eye can
    pair each rendered_prompt with its result without scrolling.
    """
    parts: list[str] = []

    sys_inst = _extract_system_instruction(call_input)
    if sys_inst:
        parts.append("<details><summary>system_instruction</summary>")
        parts.append(f'<pre class="prompt-text">{esc(sys_inst)}</pre>')
        parts.append("</details>")

    user_prompt = _extract_user_prompt(call_input)
    if user_prompt:
        parts.append("<details><summary>rendered_prompt (LLM input)</summary>")
        parts.append(f'<pre class="prompt-text">{esc(user_prompt)}</pre>')
        parts.append("</details>")

    # Per-step output — every step gets a directly-paired output so PLAN /
    # FIND / EXECUTE all read the same way (input above, output below, no
    # accumulation-at-the-end). Show CLEAN parsed forms; raw text only as
    # a last-resort fallback. Every block is collapsible (default closed).
    out = call_output or {}
    step = out.get("step")

    if step == "code_plan":
        plan_steps = None
        parsed = out.get("parsed_json")
        if isinstance(parsed, dict):
            plan_steps = parsed.get("plan_steps")
        if plan_steps:
            parts.append("<details><summary>output plan_steps</summary>")
            parts.append(f'<pre class="json-block">{esc(fmt_json(plan_steps))}</pre>')
            parts.append("</details>")
        elif isinstance(parsed, dict):
            parts.append("<details><summary>output parsed_json</summary>")
            parts.append(f'<pre class="json-block">{esc(fmt_json(parsed))}</pre>')
            parts.append("</details>")

    elif step == "code_execute":
        code = out.get("code")
        if code:
            parts.append("<details><summary>output code (generated Python)</summary>")
            parts.append(f'<pre class="prompt-text">{esc(code)}</pre>')
            parts.append("</details>")
        # Prefer the parsed dict {value, answer, summary} over the raw
        # JSON string that the sandbox printed. Fall back to raw_stdout
        # only if no parsed form exists.
        parsed_stdout = out.get("parsed_json")
        raw_stdout = out.get("raw_stdout")
        if isinstance(parsed_stdout, (dict, list)):
            parts.append("<details><summary>output stdout (parsed)</summary>")
            parts.append(
                f'<pre class="json-block">{esc(fmt_json(parsed_stdout))}</pre>'
            )
            parts.append("</details>")
        elif raw_stdout:
            # No parsed_json — try to pretty-print raw_stdout if it's JSON,
            # else show as-is.
            try:
                obj = json.loads(raw_stdout)
                parts.append(
                    "<details><summary>output stdout (parsed from raw)</summary>"
                )
                parts.append(f'<pre class="json-block">{esc(fmt_json(obj))}</pre>')
                parts.append("</details>")
            except (TypeError, ValueError):
                parts.append("<details><summary>output raw_stdout</summary>")
                parts.append(f'<pre class="prompt-text">{esc(raw_stdout)}</pre>')
                parts.append("</details>")
        parse_err = out.get("parse_error")
        if parse_err:
            parts.append("<details><summary>output parse_error</summary>")
            parts.append(f'<pre class="prompt-text">{esc(parse_err)}</pre>')
            parts.append("</details>")

    else:
        # PLAN single-call AND FIND multi-call rounds use this branch.
        # Pair each input with its parsed output here, same as EXECUTE
        # so the eye doesn't have to scroll to the end to find what each
        # step produced.
        for parsed_key in (
            "parsed_plan",
            "parsed_decision",
            "parsed_api_selection",
            "parsed_json",
        ):
            parsed_val = out.get(parsed_key)
            if parsed_val:
                parts.append(f"<details><summary>output {esc(parsed_key)}</summary>")
                parts.append(
                    f'<pre class="json-block">{esc(fmt_json(parsed_val))}</pre>'
                )
                parts.append("</details>")
                break  # one parsed form is enough per step

    return parts


def _routing_step_summary(step: dict[str, Any]) -> str:
    """One-line compact summary of a routing_trace step (lists → counts)."""
    parts: list[str] = []
    for key, value in step.items():
        if key in ("step", "skipped"):
            continue
        if isinstance(value, list):
            parts.append(f"{key}={len(value)}")
        elif isinstance(value, dict):
            parts.append(f"{key}={{{len(value)}}}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts)


def _render_routing_trace(trace: list[dict[str, Any]]) -> list[str]:
    """Render the finder's routing_trace as a compact, always-open block so the
    retrieval path — and any skipped stage (community_select / dependency
    expansion under ablation) — is visible without expanding the LLM calls."""
    if not trace:
        return []
    out = [
        '<div class="call-output-block">',
        "<details open><summary>routing_trace (finder stages)</summary>",
        '<div class="kv-block">',
    ]
    for step in trace:
        if not isinstance(step, dict):
            continue
        name = step.get("step", "?")
        badge = (
            ' <span class="badge badge-fail">skipped</span>'
            if step.get("skipped") is True
            else ""
        )
        out.append(
            f'<div class="kv"><span class="key">{esc(name)}</span>{badge} '
            f"{esc(_routing_step_summary(step))}</div>"
        )
    out.append("</div>")
    out.append(f'<pre class="json-block">{esc(fmt_json(trace))}</pre>')
    out.append("</details></div>")
    return out


def render_call(call: dict[str, Any], cycle_n: int | None, call_idx: int) -> str:
    """Render one subagent_io.jsonl entry as a collapsible <details> block.

    PLAN / FIND / EXECUTE have different io shapes; render only what each
    actually produces instead of forcing a single template.

    PLAN (rough_planner / continuation):
        input has rendered_prompt + model_input.{system_instruction, messages}
        output has raw_llm_text + parsed_plan / parsed_decision
        continuation also has metadata.history + metadata.milestones

    FIND / EXECUTE:
        input has model_calls list (each call = a separate LLM round)
        output has model_calls (parallel — raw_llm_text per round, code etc)
        Plus milestone context fields at the input root
    """
    phase = call.get("phase", "?")
    ior = call.get("io_record") or {}
    env = call.get("envelope") or {}
    agent = ior.get("agent", "?")
    status = ior.get("status", "?")
    input_block = ior.get("input") or {}
    output_block = ior.get("output") or {}
    si = input_block.get("subagent_input") or {}
    si_md = si.get("metadata") if isinstance(si.get("metadata"), dict) else {}
    output_payload = env.get("payload") or {}
    failure_code = env.get("failure_code")
    env_metrics = env.get("metrics") or {}

    badge_class = "badge-pass" if status == "SUCCEEDED" else "badge-fail"
    fc_html = (
        f' <span class="badge badge-fail">fc={esc(failure_code)}</span>'
        if failure_code
        else ""
    )
    cycle_label = f"c{cycle_n}" if cycle_n is not None else "c?"
    summary_line = (
        f"<code>{cycle_label}</code> <strong>{esc(phase)}</strong> "
        f"<code>{esc(agent)}</code> "
        f'<span class="badge {badge_class}">{esc(status)}</span>{fc_html}'
    )

    parts = [f'<details id="call-{call_idx}"><summary>{summary_line}</summary>']

    # ─── 1. Quick stats (phase-aware) ────────────────────────────────────
    stats: list[tuple[str, Any]] = [
        ("attempt", si.get("attempt") if si else input_block.get("attempt")),
        ("cycle_n", cycle_n),
    ]
    # Continuation planner — has cycle history + milestone list in metadata
    if phase == "PLAN" and si_md:
        if "active_milestone_index" in si_md:
            stats.append(
                ("active_milestone_index", si_md.get("active_milestone_index"))
            )
        history = si_md.get("history") or []
        if history:
            stats.append(("history length", len(history)))
        milestones_in = si_md.get("milestones") or []
        if milestones_in:
            stats.append(("milestones in metadata", len(milestones_in)))
    # FIND / EXECUTE — milestone context lives at input root
    if phase in ("FIND", "EXECUTE"):
        for key in (
            "milestone_index",
            "milestone_total",
            "milestone_intent",
            "milestone_done_criteria",
        ):
            value = input_block.get(key)
            if value not in (None, "", []):
                stats.append((key, value))
        planned = input_block.get("planned_apps")
        if planned:
            stats.append(("planned_apps", ", ".join(str(a) for a in planned)))
    # Token / wall stats from envelope.metrics
    if env_metrics.get("wall_ms") is not None:
        stats.append(("wall_ms", env_metrics["wall_ms"]))
    if (
        env_metrics.get("prompt_tokens") is not None
        or env_metrics.get("completion_tokens") is not None
    ):
        stats.append(
            (
                "tokens (in/out)",
                f"{env_metrics.get('prompt_tokens', '-')}/{env_metrics.get('completion_tokens', '-')}",
            )
        )
    # EXECUTE-specific call-level scalars promoted to the top kv-block so
    # they aren't duplicated as a separate "envelope.payload — summary"
    # block at the bottom. milestone_done / finalize_called / tool_call_count
    # are status flags the analyst wants visible without expanding anything.
    if phase == "EXECUTE" and isinstance(output_payload, dict):
        for key in ("milestone_done", "finalize_called", "tool_call_count"):
            value = output_payload.get(key)
            if value is not None and not isinstance(value, (dict, list)):
                stats.append((key, value))

    parts.append('<div class="kv-block">')
    for key, value in stats:
        parts.append(
            f'<div class="kv"><span class="key">{esc(key)}</span> {esc(value)}</div>'
        )
    parts.append("</div>")

    # ─── 1b. Finder routing trace (FIND only) ────────────────────────────
    # The finder records a step-by-step routing_trace in its payload
    # (planned_app_filter → community_select → seed_candidates → seed_filter →
    # dependency_round, plus ablation markers community_select.skipped /
    # dependency_expansion.skipped). The per-LLM-call sections below only show
    # the LLM rounds, so surface the trace here to make the retrieval mechanism
    # — and which stages were skipped — directly visible.
    if phase == "FIND" and isinstance(output_payload, dict):
        parts.extend(_render_routing_trace(output_payload.get("routing_trace") or []))

    # ─── 2. Per-LLM-call sections — uniform shape across PLAN/FIND/EXECUTE ─
    #
    # Single-call subagents (rough_planner, continuation): one virtual step
    #   built from input.rendered_prompt + input.model_input + output.raw_llm_text.
    # Multi-call subagents (community_finder, code_plan_execute): one step per
    #   input.model_calls[i] paired with output.model_calls[i].
    # Each step renders the same blocks: rendered_prompt → output → system_instruction
    # so the eye can compare across phases without re-orienting.
    model_calls_in = input_block.get("model_calls") or []
    model_calls_out = output_block.get("model_calls") or []
    if model_calls_in or model_calls_out:
        steps = list(
            zip(model_calls_in, model_calls_out + [None] * len(model_calls_in))
        )
        for i, (mc_in, mc_out) in enumerate(steps):
            step_name = (mc_in or {}).get("step") or f"step_{i}"
            parts.append(f'<h3 class="call-step">Step [{i}] {esc(step_name)}</h3>')
            parts.extend(_render_call_step(mc_in or {}, mc_out or {}))
    else:
        parts.extend(_render_call_step(input_block, output_block))

    # ─── 3. envelope.metrics only ───────────────────────────────────────
    # envelope.payload is intentionally NOT rendered here — its fields
    # are derived views of the per-step parsed output already shown above
    # (parsed_plan / parsed_decision / parsed_api_selection / parsed_json /
    # plan_steps / code / stdout). Only metrics (wall_ms, token counts,
    # llm_calls) survives at the call level.
    if env_metrics:
        parts.append('<div class="call-output-block">')
        parts.append("<details><summary>envelope.metrics</summary>")
        parts.append(f'<pre class="json-block">{esc(fmt_json(env_metrics))}</pre>')
        parts.append("</details>")
        parts.append("</div>")

    # ─── 5. Continuation planner extras ──────────────────────────────────
    if phase == "PLAN" and si_md:
        history = si_md.get("history") or []
        milestones_in = si_md.get("milestones") or []
        if history:
            parts.append(
                f"<details><summary>metadata.history ({len(history)} entries)</summary>"
            )
            parts.append(f'<pre class="json-block">{esc(fmt_json(history))}</pre>')
            parts.append("</details>")
        if milestones_in:
            parts.append(
                f"<details><summary>metadata.milestones ({len(milestones_in)} entries)</summary>"
            )
            parts.append(
                f'<pre class="json-block">{esc(fmt_json(milestones_in))}</pre>'
            )
            parts.append("</details>")
    parts.append("</details>")
    return "\n".join(parts)


def extract_cycle_n(call: dict[str, Any]) -> int | None:
    si = call.get("io_record", {}).get("input", {}).get("subagent_input", {}) or {}
    cn = (si.get("metadata") or {}).get("cycle_n")
    return cn if isinstance(cn, int) else None


def _extract_active_mi(call: dict[str, Any]) -> int | None:
    """PLAN input's pre-decision active_milestone_index (None for rough_planner)."""
    si = call.get("io_record", {}).get("input", {}).get("subagent_input", {}) or {}
    ami = (si.get("metadata") or {}).get("active_milestone_index")
    return ami if isinstance(ami, int) else None


def _extract_milestone_idx(call: dict[str, Any]) -> int | None:
    """FIND/EXECUTE input's target milestone_index."""
    si = call.get("io_record", {}).get("input", {}).get("subagent_input", {}) or {}
    mi = (si.get("metadata") or {}).get("milestone_index")
    return mi if isinstance(mi, int) else None


def assign_cycle_ns(calls: list[dict[str, Any]]) -> list[int | None]:
    """Assign cycle_n to every call.

    Preferred path — per-call metadata.cycle_n. PLAN inputs always carry
    cycle_n; FIND/EXECUTE inputs carry it in logs produced after the
    `_build_find_input`/`_build_exec_input` cycle_n injection fix.

    Recovery path (older logs missing cycle_n on FIND/EXECUTE) — derive
    each cycle's "out" active_milestone_index from the PLAN trajectory:
      out_mi(cycle k) := in_active_mi of PLAN(cycle k+1)
                        (or in_active_mi of PLAN(cycle k) if k is final;
                         cycle 0 / rough_planner forced to 0)
    Then:
      - FIND with milestone_index=k → first cycle whose out_mi == k.
      - EXECUTE with milestone_index=k → per-milestone queue popped in
        file order (subagent_io.jsonl writes calls in temporal order
        within each phase group).

    This handles the FIND-skip case (Entry 8): when PLAN count > FIND
    count, the old phase-local pairing (N-th FIND ↔ N-th PLAN) collapses
    all skipped cycles to the wrong cycle. Trajectory recovery places
    each FIND at its actual milestone-transition cycle.

    Last-resort fallback (PLAN cycle_n itself missing — very old log
    format): N-th FIND/EXECUTE ↔ N-th PLAN by phase-local index.
    """
    cn_per_call = [extract_cycle_n(c) for c in calls]

    plan_seq: list[tuple[int, int | None, int | None]] = []
    for i, c in enumerate(calls):
        if c.get("phase") == "PLAN":
            plan_seq.append((i, cn_per_call[i], _extract_active_mi(c)))

    if not plan_seq or any(cn is None for _, cn, _ in plan_seq):
        # Fallback: legacy phase-local pairing
        plan_cycles_legacy = [cn for _, cn, _ in plan_seq]
        out: list[int | None] = []
        plan_idx = find_idx = execute_idx = 0
        for c in calls:
            phase = c.get("phase")
            if phase == "PLAN":
                out.append(
                    plan_cycles_legacy[plan_idx]
                    if plan_idx < len(plan_cycles_legacy)
                    else None
                )
                plan_idx += 1
            elif phase == "FIND":
                out.append(
                    plan_cycles_legacy[find_idx]
                    if find_idx < len(plan_cycles_legacy)
                    else None
                )
                find_idx += 1
            elif phase == "EXECUTE":
                out.append(
                    plan_cycles_legacy[execute_idx]
                    if execute_idx < len(plan_cycles_legacy)
                    else None
                )
                execute_idx += 1
            else:
                out.append(None)
        return out

    # Compute out_mi per cycle from PLAN trajectory
    out_mi_by_cycle: dict[int, int] = {}
    for j, (_, cn, ami) in enumerate(plan_seq):
        if j + 1 < len(plan_seq):
            next_ami = plan_seq[j + 1][2]
            out_mi = (
                next_ami if next_ami is not None else (ami if ami is not None else 0)
            )
        else:
            out_mi = ami if ami is not None else 0
        if cn == 0:
            out_mi = 0  # rough_planner always seeds active_mi=0
        out_mi_by_cycle[cn] = out_mi

    # Per-milestone cycle queue: list of cycles whose out_mi == mi, in cycle
    # order. FIND/EXECUTE calls (in subagent_io.jsonl file order, which is
    # chronological within each phase group) pop from this queue.
    #
    # Works for both pre-Entry-8 logs (FIND every cycle: queue has all cycles
    # for that mi, popped 1-per-call → matches phase-local pairing) and
    # post-Entry-8 logs (FIND only on transition: queue may have more cycles
    # than FIND calls, only first cycle popped → correct transition cycle).
    cycle_queue: dict[int, list[int]] = {}
    for cn in sorted(out_mi_by_cycle.keys()):
        mi = out_mi_by_cycle[cn]
        cycle_queue.setdefault(mi, []).append(cn)

    # Backfill: if last PLAN cycle ADVANCEd to a new milestone (no "next PLAN"
    # to read in_active_mi from), the FIND for that new milestone has a
    # milestone_idx not yet in cycle_queue. Append last cycle to that queue.
    last_cn = plan_seq[-1][1]
    seen_mis = set(cycle_queue.keys())
    for c in calls:
        if c.get("phase") == "FIND":
            mi = _extract_milestone_idx(c)
            if mi is not None and mi not in seen_mis:
                cycle_queue.setdefault(mi, []).append(last_cn)
                seen_mis.add(mi)

    # Separate read-pointers for FIND and EXECUTE (same milestone may have
    # both a FIND and EXECUTE that need to map to the same cycle).
    find_ptr: dict[int, int] = {}
    exec_ptr: dict[int, int] = {}

    out2: list[int | None] = []
    for i, c in enumerate(calls):
        phase = c.get("phase")
        cn = cn_per_call[i]
        if cn is not None:
            out2.append(cn)
            continue
        mi = _extract_milestone_idx(c)
        if mi is None:
            out2.append(None)
            continue
        q = cycle_queue.get(mi, [])
        if phase == "FIND":
            p = find_ptr.get(mi, 0)
            out2.append(q[p] if p < len(q) else None)
            find_ptr[mi] = p + 1
        elif phase == "EXECUTE":
            p = exec_ptr.get(mi, 0)
            out2.append(q[p] if p < len(q) else None)
            exec_ptr[mi] = p + 1
        else:
            out2.append(None)
    return out2


# Backward-compat alias (old name used elsewhere).
inherit_cycle_ns = assign_cycle_ns


PHASE_ORDER = {"PLAN": 0, "FIND": 1, "EXECUTE": 2, "SUBMIT": 3, "VERIFY": 4}


def reorder_calls_by_cycle(
    calls: list[dict[str, Any]],
    cycle_ns: list[int | None],
) -> tuple[list[dict[str, Any]], list[int | None], list[int]]:
    """Stable-sort calls into temporal cycle order so PLAN c0 → FIND c0 →
    EXECUTE c0 → PLAN c1 → … appear adjacent in the timeline.

    Returns (sorted_calls, sorted_cycle_ns, original_indices). The third
    list lets render_call keep its anchor IDs stable across rebuilds.
    """
    indexed = list(zip(range(len(calls)), calls, cycle_ns))

    def sort_key(t: tuple[int, dict[str, Any], int | None]) -> tuple[int, int, int]:
        idx, call, cn = t
        cycle_sort = cn if cn is not None else 10**6
        phase = call.get("phase") or ""
        phase_sort = PHASE_ORDER.get(phase, 10)
        return (cycle_sort, phase_sort, idx)

    indexed.sort(key=sort_key)
    sorted_calls = [t[1] for t in indexed]
    sorted_cycle_ns = [t[2] for t in indexed]
    original_indices = [t[0] for t in indexed]
    return sorted_calls, sorted_cycle_ns, original_indices


def build_cycle_timeline(calls: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    rows.append("<table><thead><tr>")
    rows.append("<th>#</th><th>cycle_n</th><th>phase</th><th>agent</th>")
    rows.append(
        "<th>status</th><th>failure_code</th><th>wall_ms</th><th>tokens (in/out)</th><th>→ details</th>"
    )
    rows.append("</tr></thead><tbody>")
    cycle_ns = assign_cycle_ns(calls)
    sorted_calls, sorted_cns, orig_indices = reorder_calls_by_cycle(calls, cycle_ns)
    for display_i, (call, cn, orig_idx) in enumerate(
        zip(sorted_calls, sorted_cns, orig_indices)
    ):
        ior = call.get("io_record") or {}
        env = call.get("envelope") or {}
        metrics = ior.get("metrics") or {}
        rows.append('<tr class="cycle-row">')
        rows.append(f"<td>{display_i:02d}</td>")
        rows.append(f"<td>c{esc(cn) if cn is not None else '?'}</td>")
        rows.append(f"<td>{esc(call.get('phase'))}</td>")
        rows.append(f"<td>{esc(ior.get('agent'))}</td>")
        st = ior.get("status", "?")
        bc = "badge-pass" if st == "SUCCEEDED" else "badge-fail"
        rows.append(f'<td><span class="badge {bc}">{esc(st)}</span></td>')
        rows.append(f"<td>{esc(env.get('failure_code') or '')}</td>")
        wall_ms_disp = (
            esc(metrics.get("wall_ms"))
            if metrics.get("wall_ms") is not None
            else '<span style="color:#aaa">—</span>'
        )
        tok_in = metrics.get("prompt_tokens")
        tok_out = metrics.get("completion_tokens")
        if tok_in is None and tok_out is None:
            tok_disp = '<span style="color:#aaa">— / —</span>'
        else:
            tok_disp = f"{esc(tok_in if tok_in is not None else '—')}/{esc(tok_out if tok_out is not None else '—')}"
        rows.append(f"<td>{wall_ms_disp}</td>")
        rows.append(f"<td>{tok_disp}</td>")
        rows.append(f'<td><a href="#call-{orig_idx}">jump</a></td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _best_cell(
    best: dict[str, Any] | None,
    current_run_id: str | None,
) -> str:
    """Render a 'Best ever' status badge + run_id (with crown if current is best)."""
    if not best:
        return '<span class="badge badge-na">no data</span>'
    ev = best["eval"] or {}
    badge = status_badge(best["status"], ev.get("passed"), ev.get("total"))
    is_us = best["run_id"] == current_run_id
    crown = "🏆 " if is_us else ""
    suffix = " ← this run" if is_us else ""
    # Timestamp is embedded in run_id (e.g. exp/20260512_101038_909559); no
    # separate date pill needed.
    return (
        f"{crown}{badge}<br>"
        f'<small style="color:#586069">{esc(best["run_id"])}</small>{suffix}'
    )


def render_sandbox_trace(sandbox_trace: list[dict[str, Any]]) -> str:
    """Render `sandbox_api_calls.jsonl` records as collapsible cards.

    Each record = one `execute_python` outer-tool invocation. Inside it lives
    a list of `apis.*` calls captured by `_record_api_call` (kwargs, status,
    result_shape, result_items with the 3-layer size cap). This is the
    canonical analyst evidence for P4 cluster (search over-trust): without it
    we only see `list[N]` summaries; with it we see the actual first N items
    each API returned, so over-broad / ignored-filter behavior is visible.
    """
    if not sandbox_trace:
        return ""
    parts: list[str] = ["<h2>Sandbox API trace</h2>"]
    parts.append(
        '<p class="subtitle">每個 record = 一次 <code>execute_python</code> 外層 tool '
        "呼叫；裡面 <code>api_calls</code> 是 sandbox 真實執行到的 <code>apis.*</code> "
        "呼叫 + 三層 size-cap 的 <code>result_items</code> 樣本。</p>"
    )
    for ri, record in enumerate(sandbox_trace):
        apps = record.get("app_references") or {}
        api_calls = record.get("api_calls") or []
        app_list = ", ".join(sorted(apps.keys())) if apps else "?"
        summary_line = (
            f"<strong>record {ri}</strong> &nbsp; "
            f"<code>{esc(app_list)}</code> &nbsp; "
            f'<span class="badge badge-na">{len(api_calls)} api_calls</span>'
        )
        parts.append(f"<details><summary>{summary_line}</summary>")
        for ci, call in enumerate(api_calls):
            app = call.get("app", "?")
            name = call.get("api_name", "?")
            status = call.get("status", "?")
            shape = call.get("result_shape") or {}
            shape_str = ""
            if isinstance(shape, dict):
                t = shape.get("type", "?")
                if t == "list":
                    shape_str = f"list[{shape.get('count_hint', '?')}]"
                elif t == "object":
                    shape_str = f"object[{len(shape.get('keys', []))}k]"
                else:
                    shape_str = t
            badge = "badge-pass" if status == "ok" else "badge-fail"
            kw = call.get("kwargs") or {}
            kw_repr = ", ".join(
                f"{esc(k)}={esc(v)}" for k, v in kw.items() if k != "access_token"
            )
            call_title = (
                f"<code>[{ci}] {esc(app)}.{esc(name)}</code> "
                f'<span class="badge {badge}">{esc(status)}</span> '
                f'<span style="color:#586069">{esc(shape_str)}</span>'
            )
            parts.append('<div class="kv-block" style="margin:0.4rem 0 0.2rem 1rem;">')
            parts.append(f"<div>{call_title}</div>")
            if kw_repr:
                parts.append(
                    f'<div class="kv"><span class="key">kwargs</span> '
                    f"<code>{kw_repr}</code></div>"
                )
            err = call.get("error")
            if err:
                parts.append(
                    f'<div class="kv"><span class="key">error</span> {esc(err)}</div>'
                )
            parts.append("</div>")
            items_block = call.get("result_items")
            if items_block is not None:
                if isinstance(items_block, dict) and "items" in items_block:
                    total = items_block.get("_list_total")
                    trunc = items_block.get("_truncated")
                    label = f"result_items — first {len(items_block.get('items') or [])} of {total}"
                    if trunc:
                        label += " (truncated)"
                    parts.append(
                        f'<details style="margin-left:1rem;"><summary>{esc(label)}</summary>'
                    )
                    parts.append(
                        f'<pre class="json-block">{esc(fmt_json(items_block["items"]))}</pre>'
                    )
                    parts.append("</details>")
                elif isinstance(items_block, dict) and items_block.get("_oversize"):
                    parts.append(
                        f'<div style="margin-left:1rem;color:#856404;">'
                        f"<em>result oversize — "
                        f"{items_block.get('_size_bytes')} bytes ({items_block.get('_type')})</em></div>"
                    )
                elif isinstance(items_block, dict) and items_block.get(
                    "_global_truncated"
                ):
                    parts.append(
                        '<div style="margin-left:1rem;color:#856404;">'
                        "<em>global trace size cap hit — record dropped</em></div>"
                    )
                else:
                    parts.append(
                        f'<details style="margin-left:1rem;"><summary>result_items</summary>'
                        f'<pre class="json-block">{esc(fmt_json(items_block))}</pre>'
                        f"</details>"
                    )
        parts.append("</details>")
    return "\n".join(parts)


def _find_decisions(run: dict[str, Any]) -> tuple[dict[Any, dict[str, Any]], list[Any]]:
    """Per-milestone candidate_apis (union across that milestone's FIND attempts)."""
    out: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for x in run.get("calls", []) or []:
        if x.get("phase") != "FIND":
            continue
        io = x.get("io_record", {}) or {}
        inp = io.get("input", {}) or {}
        o = io.get("output", {}) or {}
        pas = o.get("parsed_api_selection", {}) or {}
        mi = inp.get("milestone_index")
        cset = {
            f"{a.get('app', '')}.{a.get('name', '')}"
            for a in (pas.get("candidate_apis", []) or [])
            if isinstance(a, dict)
        }
        if mi not in out:
            out[mi] = {
                "intent": (inp.get("milestone_intent") or "")[:80],
                "cands": set(),
            }
            order.append(mi)
        out[mi]["cands"] |= cset
    return out, order


def _sandbox_calls(run: dict[str, Any]) -> tuple[dict[str, int], dict[str, Any]]:
    """Actual API calls executed in the sandbox: name -> count, name -> first kwargs."""
    from collections import Counter

    cnt: Counter = Counter()
    kw: dict[str, Any] = {}
    for rec in run.get("sandbox_trace", []) or []:
        for c in rec.get("api_calls", []) or []:
            n = f"{c.get('app', '')}.{c.get('api_name', '')}"
            cnt[n] += 1
            kw.setdefault(n, c.get("kwargs", {}))
    return dict(cnt), kw


def diff_decision_strip(
    current: dict[str, Any], baseline: dict[str, Any] | None
) -> str:
    """Compact side-by-side of the decision-critical signals (per-milestone
    candidate_apis + sandbox calls) so a flip's mechanism is visible without
    scrolling the full traces. Orange = present on only one side."""
    if not baseline:
        return ""
    cf, corder = _find_decisions(current)
    bf, _ = _find_decisions(baseline)
    csb, ckw = _sandbox_calls(current)
    bsb, bkw = _sandbox_calls(baseline)

    def fmt_set(s: set[str], other: set[str]) -> str:
        if not s:
            return '<span class="cmp-none">∅</span>'
        return ", ".join(
            f'<span class="diff-hl">{esc(a)}</span>' if a not in other else esc(a)
            for a in sorted(s)
        )

    # candidate table aligned by milestone_index
    mids = list(corder) + [m for m in bf if m not in cf]
    cand_rows = []
    for mi in mids:
        c = cf.get(mi, {})
        b = bf.get(mi, {})
        cc = c.get("cands", set())
        bc = b.get("cands", set())
        intent = c.get("intent") or b.get("intent") or ""
        differ = cc != bc
        cand_rows.append(
            f'<tr class="{"diff-row-differ" if differ else ""}">'
            f'<td>{esc(str(mi))}</td><td class="diff-intent">{esc(intent)}</td>'
            f"<td>{fmt_set(cc, bc)}</td><td>{fmt_set(bc, cc)}</td>"
            f"<td>{'★' if differ else ''}</td></tr>"
        )
    cand_tbl = (
        '<table class="diff-tbl"><thead><tr><th>ms</th><th>intent</th>'
        "<th>Current 候選 API</th><th>Baseline 候選 API</th><th></th></tr></thead>"
        f"<tbody>{''.join(cand_rows)}</tbody></table>"
    )

    # sandbox table — union of called APIs
    sb_rows = []
    for n in sorted(set(csb) | set(bsb)):
        cn = csb.get(n, 0)
        bn = bsb.get(n, 0)
        only = (cn == 0) != (bn == 0)
        kwc = esc(json.dumps(ckw.get(n, {}), ensure_ascii=False)[:130]) if cn else ""
        kwb = esc(json.dumps(bkw.get(n, {}), ensure_ascii=False)[:130]) if bn else ""
        sb_rows.append(
            f'<tr class="{"diff-row-differ" if only else ""}">'
            f"<td>{esc(n)}</td><td>{cn or '—'}</td><td>{bn or '—'}</td>"
            f'<td class="diff-kw">{kwc}</td><td class="diff-kw">{kwb}</td></tr>'
        )
    sb_tbl = (
        '<table class="diff-tbl"><thead><tr><th>API</th><th>Cur×</th><th>Base×</th>'
        "<th>Cur kwargs (first call)</th><th>Base kwargs (first call)</th></tr></thead>"
        f"<tbody>{''.join(sb_rows)}</tbody></table>"
    )
    return (
        '<details class="decision-diff">'
        '<summary style="cursor:pointer;font-weight:700;font-size:1.05em">'
        "▸ 逐 milestone 細節（候選 API + sandbox kwargs，點開）</summary>"
        "<h3>逐 milestone 候選 API（橘底列 = 兩邊不同；橘字 = 只在這一邊）</h3>"
        f"{cand_tbl}"
        "<h3>Sandbox 實際呼叫（橘底列 = 只有一邊真的 call 了該 API）</h3>"
        f"{sb_tbl}"
        "</details>"
    )


def summary_compare_table(
    current: dict[str, Any], baseline: dict[str, Any] | None
) -> str:
    """Per-milestone (final executed state) comparison: goal / finder selection /
    execution result, Baseline vs Current. Headline metrics (status, submit
    answer, block, cost) are in the summary-card table above and not repeated.
    """
    if not baseline:
        return ""

    # Headline metrics (status, submit answer, block, cost) live in the
    # summary-card table above — not repeated here. This is only the
    # per-milestone breakdown (the genuinely new info).
    def _ms_list(run: dict[str, Any]) -> list[dict[str, Any]]:
        # Per-milestone FINAL candidate_apis (incl dependency expansion = what the
        # executor actually received), sourced from the io trace — NOT the
        # task_summary find.selected_apis, which is the pre-dep seed_filter raw
        # selection and is often empty in runaway / final-state milestones.
        summary = run.get("summary", {}) or {}
        cand_by_mi: dict[Any, set[str]] = {}
        for x in run.get("calls", []) or []:
            if x.get("phase") != "FIND":
                continue
            io = x.get("io_record", {}) or {}
            mi = (io.get("input", {}) or {}).get("milestone_index")
            pas = (io.get("output", {}) or {}).get("parsed_api_selection", {}) or {}
            bucket = cand_by_mi.setdefault(mi, set())
            for a in pas.get("candidate_apis", []) or []:
                if isinstance(a, dict):
                    bucket.add(f"{a.get('app', '')}.{a.get('name', '')}")
        out: list[dict[str, Any]] = []
        for m in summary.get("milestones") or []:
            if not isinstance(m, dict):
                continue
            ex = m.get("execute") or {}
            out.append(
                {
                    "index": m.get("index"),
                    "goal": str(m.get("goal", "")),
                    "selected": sorted(cand_by_mi.get(m.get("index"), set())),
                    "status": ex.get("status"),
                    "result": ex.get("output_variable_preview"),
                }
            )
        return out

    cm, bm = _ms_list(current), _ms_list(baseline)
    if not cm and not bm:
        return ""
    by_c = {m["index"]: m for m in cm}
    by_b = {m["index"]: m for m in bm}
    order: list[Any] = []
    seen: set[Any] = set()
    for m in bm + cm:
        if m["index"] not in seen:
            seen.add(m["index"])
            order.append(m["index"])

    def _ms_cell(m: dict[str, Any] | None) -> str:
        if not m:
            return '<span class="cmp-none">（這個變體沒有這一步）</span>'
        sel = ", ".join(m["selected"]) if m["selected"] else "—"
        res = m["result"]
        rs = str(res) if res is not None else "—"
        res_short = esc(rs[:160] + ("…" if len(rs) > 160 else ""))
        st = m["status"] or "?"
        stc = "#22863a" if st == "SUCCEEDED" else "#cb2431"
        return (
            f'<div class="ms-goal">{esc(m["goal"])}</div>'
            f'<div class="ms-meta">🔍 檢索：{esc(sel)}　·　'
            f'<span style="color:{stc}">{esc(st)}</span></div>'
            f'<div class="ms-meta" title="{esc(rs)}">▶ 執行結果：{res_short}</div>'
        )

    ms_rows = [
        f"<tr><td>{esc(str(i))}</td><td>{_ms_cell(by_b.get(i))}</td>"
        f"<td>{_ms_cell(by_c.get(i))}</td></tr>"
        for i in order
    ]
    return (
        '<div class="cmp-summary-box">'
        '<div class="cmp-summary-h">逐 milestone（最終執行後）'
        "　—　goal ／ 🔍 檢索選到的 API ／ ▶ 該步執行結果</div>"
        '<table class="diff-tbl"><thead><tr><th>步</th><th>Baseline</th>'
        "<th>Current</th></tr></thead>"
        f"<tbody>{''.join(ms_rows)}</tbody></table></div>"
    )


def task_page(
    tid: str,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    pattern_name: str | None,
    best: dict[str, Any] | None = None,
    best_task: dict[str, Any] | None = None,
    retry_task: dict[str, Any] | None = None,
    current_run_ids: set[str] | None = None,
    instruction_zh: str | None = None,
    analysis: dict[str, str] | None = None,
) -> str:
    cs = current["summary"]
    bs = (baseline or {}).get("summary", {}) if baseline else {}
    inst = cs.get("instruction", "")
    cur_status = cs.get("status", "?")
    cur_eval = cs.get("eval") or {}
    cur_pass = cur_eval.get("passed")
    cur_total = cur_eval.get("total")
    cur_block = (
        cs.get("block_reason")
        or (cs.get("overview") or {}).get("block_reason")
        or cs.get("aggregate_metrics", {}).get("failure_code")
    )

    base_status = bs.get("status", "?") if bs else "?"
    base_eval = (bs.get("eval") or {}) if bs else {}
    base_pass = base_eval.get("passed")
    base_total = base_eval.get("total")

    cur_metrics = cs.get("aggregate_metrics", {})
    base_metrics = bs.get("aggregate_metrics", {}) if bs else {}

    flip = flip_badge(is_pass(bs) if bs else False, is_pass(cs))

    pattern_link = ""
    if pattern_name:
        pattern_link = (
            f'<a href="pattern_{pattern_name}.html">pattern {pattern_name}</a>'
        )

    # Best column derivation
    current_run_id_for_best: str | None = None
    if current_run_ids and best:
        if best["run_id"] in current_run_ids:
            current_run_id_for_best = best["run_id"]
    best_cell_html = _best_cell(best, current_run_id_for_best)

    # Instruction + 中文 + root-cause analysis + raw eval failure
    if instruction_zh:
        zh_html = f'<p class="subtitle" style="color:#222"><strong>中文：</strong>{esc(instruction_zh)}</p>'
    else:
        zh_html = (
            '<p class="subtitle" style="color:#856404">'
            "<strong>中文：</strong>"
            "<em>(尚未翻譯 — 加 entry 到 logs/_registry/task_translations_zh.json)</em>"
            "</p>"
        )

    analysis_html = ""
    analysis_details_html = ""
    if analysis:
        layer = analysis.get("root_layer", "")
        pat = analysis.get("pattern", "")
        flip_kind = analysis.get("flip_vs_baseline", "")
        summary = analysis.get("summary_zh", "")
        body_rest = analysis.get("body_rest", "")
        tag_parts: list[str] = []
        if layer:
            tag_parts.append(f'<span class="badge badge-na">layer: {esc(layer)}</span>')
        if pat:
            tag_parts.append(f'<span class="badge badge-na">pattern: {esc(pat)}</span>')
        if flip_kind:
            tag_parts.append(
                f'<span class="badge badge-na">flip: {esc(flip_kind)}</span>'
            )
        tags_html = " ".join(tag_parts)
        analysis_html = f"""
<div class="analysis-card">
  <div style="margin-bottom:0.4rem;"><strong>分析（人工歸因）：</strong> {tags_html}</div>
  <div style="white-space:pre-wrap;">{esc(summary)}</div>
</div>
"""
        # Full Phase 1-4 + Gaps body collapsed by default — analysts who want
        # the evidence walk can expand; default reading view stays concise.
        if body_rest:
            analysis_details_html = (
                '<details class="analysis-detail-card">'
                "<summary><strong>完整分析 (Phase 1-4 + Gaps)</strong></summary>"
                f'<pre class="analysis-body-pre">{esc(body_rest)}</pre>'
                "</details>"
            )

    why_html = render_failure_block(summarize_failure(cs))

    head = f"""
<div class="nav"><a href="index.html">← index</a> {pattern_link}</div>
<h1>Task {esc(tid)} {flip}</h1>
<p class="subtitle"><strong>Instruction:</strong> {esc(inst)}</p>
{zh_html}
{analysis_html}
{analysis_details_html}
{why_html}

<div class="summary-card">
<table>
<thead><tr><th></th><th>Baseline</th><th>Current</th><th>Best ever</th></tr></thead>
<tbody>
<tr><td>Status</td><td>{status_badge(base_status, base_pass, base_total)}</td><td>{status_badge(cur_status, cur_pass, cur_total)}</td><td>{best_cell_html}</td></tr>
<tr><td>Submit 答案</td><td>{esc((bs.get("submission") or {}).get("submitted_answer") or "—")}</td><td>{esc((cs.get("submission") or {}).get("submitted_answer") or "—")}</td><td>{esc(((best or {}).get("submission") or {}).get("submitted_answer") or "—")}</td></tr>
<tr><td>Block reason</td><td>{esc((bs.get("overview") or {}).get("block_reason") or bs.get("block_reason") or "")}</td><td>{esc(cur_block or "")}</td><td>{esc((best or {}).get("block_reason") or "")}</td></tr>
<tr><td>Wall</td><td>{fmt_seconds(bs.get("wall_s"))}</td><td>{fmt_seconds(cs.get("wall_s"))}</td><td>{fmt_seconds((best or {}).get("wall_s"))}</td></tr>
<tr><td>LLM calls</td><td>{esc(base_metrics.get("llm_calls"))}</td><td>{esc(cur_metrics.get("llm_calls"))}</td><td>{esc(((best or {}).get("aggregate_metrics") or {}).get("llm_calls"))}</td></tr>
<tr><td>Tokens</td><td>{fmt_int(base_metrics.get("total_tokens"))}</td><td>{fmt_int(cur_metrics.get("total_tokens"))}</td><td>{fmt_int(((best or {}).get("aggregate_metrics") or {}).get("total_tokens"))}</td></tr>
</tbody>
</table>
</div>
"""

    def _render_run_body(
        run_data: dict[str, Any] | None, id_offset: int, label: str
    ) -> str:
        """Render cycle timeline + per-call cards + sandbox trace for one run."""
        if not run_data or not run_data.get("calls"):
            return f"<p><em>{esc(label)} run has no subagent_io.jsonl entries.</em></p>"
        cns = assign_cycle_ns(run_data["calls"])
        timeline = build_cycle_timeline(run_data["calls"])
        sorted_calls, sorted_cns, orig_indices = reorder_calls_by_cycle(
            run_data["calls"], cns
        )
        calls_html = "\n".join(
            render_call(c, cn, id_offset + orig_idx)
            for c, cn, orig_idx in zip(sorted_calls, sorted_cns, orig_indices)
        )
        sandbox_html = render_sandbox_trace(run_data.get("sandbox_trace") or [])
        return f"""
<h3>Cycle timeline</h3>
{timeline}
<h3>Per-call details</h3>
<p class="subtitle">每個 row 點開可看完整 rendered_prompt + history + output payload。</p>
{calls_html}
{sandbox_html}
"""

    cur_body = _render_run_body(current, 0, "Current")
    base_body = (
        _render_run_body(baseline, 1000, "Baseline")
        if baseline
        else ("<p><em>No baseline data for this task.</em></p>")
    )

    # Best ever tab — hide if best IS the current run (avoid duplication).
    # Best vs baseline can still be identical (rare), but cheap to render
    # both — keep visible.
    best_tab_visible = False
    best_body = ""
    if best_task and best:
        best_run_id = best.get("run_id")
        is_current = bool(current_run_ids and best_run_id in current_run_ids)
        if not is_current:
            best_body = _render_run_body(best_task, 2000, "Best ever")
            best_tab_visible = True

    # Decision-diff strip is rendered ABOVE the tabs (full width) as the quick
    # at-a-glance comparison; the full traces stay in their own full-width tabs
    # (side-by-side columns were too cramped to read).
    summary_tbl = summary_compare_table(current, baseline)

    tabs_nav = ['<div class="tab-btn active" data-tab="task-current">Current</div>']
    tab_panels = [f'<div id="task-current" class="tab-content active">{cur_body}</div>']
    if baseline:
        tabs_nav.append('<div class="tab-btn" data-tab="task-baseline">Baseline</div>')
        tab_panels.append(
            f'<div id="task-baseline" class="tab-content">{base_body}</div>'
        )
    if best_tab_visible:
        tabs_nav.append('<div class="tab-btn" data-tab="task-best">Best ever</div>')
        tab_panels.append(f'<div id="task-best" class="tab-content">{best_body}</div>')
    if retry_task:
        retry_body = _render_run_body(retry_task, 3000, "Retry")
        tabs_nav.append('<div class="tab-btn" data-tab="task-retry">Retry</div>')
        tab_panels.append(
            f'<div id="task-retry" class="tab-content">{retry_body}</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Task {esc(tid)} — analysis viewer</title>
<style>{CSS}</style></head><body>
{head}

{summary_tbl}

<div class="tabs">
{"".join(tabs_nav)}
</div>

{"".join(tab_panels)}

<script>
function _activateTab(tabId) {{
  const btn = document.querySelector('.tab-btn[data-tab="' + tabId + '"]');
  const panel = document.getElementById(tabId);
  if (!btn || !panel) return;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  panel.classList.add('active');
}}
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => _activateTab(btn.dataset.tab));
}});
// Honor URL fragment on initial load (e.g. task_X.html#tab-retry)
if (window.location.hash.startsWith('#tab-')) {{
  _activateTab(window.location.hash.substring(1));
}}
</script>
</body></html>
"""


def pattern_page(
    name: str, meta: dict[str, Any], task_outcomes: dict[str, dict]
) -> str:
    rows = []
    for tid in meta["tasks"]:
        outcome = task_outcomes.get(tid)
        if outcome is None:
            rows.append(
                f'<tr><td><a href="task_{tid}.html">{tid}</a></td><td colspan="3"><em>not in run</em></td></tr>'
            )
            continue
        cs = outcome["current"]["summary"]
        bs = outcome["baseline"]["summary"] if outcome["baseline"] else {}
        cur_e = cs.get("eval") or {}
        base_e = bs.get("eval") or {}
        rows.append(
            f'<tr><td><a href="task_{tid}.html">{tid}</a></td>'
            f"<td>{status_badge(bs.get('status', '?'), base_e.get('passed'), base_e.get('total'))}</td>"
            f"<td>{status_badge(cs.get('status', '?'), cur_e.get('passed'), cur_e.get('total'))}</td>"
            f"<td>{flip_badge(is_pass(bs) if bs else False, is_pass(cs))}</td>"
            f"<td>{esc(cs.get('block_reason') or cs.get('aggregate_metrics', {}).get('failure_code') or '')}</td></tr>"
        )
    table = (
        "<table><thead><tr><th>Task</th><th>Baseline</th><th>Current</th><th>Flip</th><th>Block reason (current)</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Pattern {esc(name)} — analysis viewer</title>
<style>{CSS}</style></head><body>
<div class="nav"><a href="index.html">← index</a></div>
<h1>{esc(meta["title"])}</h1>
<p class="subtitle">{esc(meta["description"])}</p>
{table}
</body></html>
"""


def _high_water_row(
    hw: dict[str, Any] | None,
    current_run_ids: set[str] | None,
) -> str:
    if not hw:
        return ""
    ids_for_winner = set(hw.get("run_ids", []))
    is_us = bool(current_run_ids and ids_for_winner & current_run_ids)
    crown = "🏆 " if is_us else ""
    suffix = " ← this run" if is_us else ""
    # Sharded runs combine multiple run_ids; surface only the first since
    # the timestamp is embedded in the run_id and additional shards add
    # no useful info to the headline row.
    primary_rid = hw["run_ids"][0] if hw["run_ids"] else ""
    return (
        f'<div class="kv"><span class="key">All-time high-water</span> '
        f"{crown}{hw['n_pass']} / {hw['n_done']} "
        f"<code>{esc(primary_rid)}</code>{suffix}</div>"
    )


def _ts_to_date(ts_name: str) -> str | None:
    """Parse YYYYMMDD_HHMMSS_... → YYYY-MM-DD; return None if not a timestamp dir."""
    if len(ts_name) < 8 or not ts_name[:8].isdigit():
        return None
    return f"{ts_name[:4]}-{ts_name[4:6]}-{ts_name[6:8]}"


def run_label(p: Path) -> str:
    """Render run dir path with date(s).

    - If p is a timestamp dir: '<exp_name>/<ts> (YYYY-MM-DD)'
    - If p is an experiment dir: '<exp_name>/ (YYYY-MM-DD)' if all
      timestamp subdirs share one date, else '<exp_name>/ (YYYY-MM-DD → YYYY-MM-DD)'
    """
    try:
        # Detect whether p is timestamp dir or experiment dir
        ts_subdirs = []
        if p.is_dir():
            for c in sorted(p.iterdir()):
                if c.is_dir() and c.name != "batch_results":
                    d = _ts_to_date(c.name)
                    if d:
                        ts_subdirs.append(d)
        own_date = _ts_to_date(p.name)
        if own_date:
            # p is itself a timestamp dir → exp_name/ts (date)
            return f"{p.parent.name}/{p.name} ({own_date})"
        if ts_subdirs:
            # p is an experiment dir → show date range
            dates = sorted(set(ts_subdirs))
            if len(dates) == 1:
                return f"{p.name}/ ({dates[0]})"
            return f"{p.name}/ ({dates[0]} → {dates[-1]})"
        # Fallback: no timestamp info
        return f"{p.parent.name}/{p.name}"
    except Exception:
        return str(p)


def _resolve_split_total(
    current_task_ids: set[str], fallback: int
) -> tuple[int, str | None]:
    """Detect the tightest tracked project task set containing the run tasks."""
    if not TASK_SETS_DIR.is_dir():
        return fallback, None

    best: tuple[int, int, str] | None = None  # (overlap, total, name)
    for task_set_file in sorted(TASK_SETS_DIR.glob("*.txt")):
        try:
            lines = [
                line.strip()
                for line in task_set_file.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
        except Exception:
            continue
        ids = {ln.split()[0] for ln in lines if ln.split()}
        candidates = [(task_set_file.stem, ids)]
        if task_set_file.stem != "quick_smoke":
            candidates.extend(
                (
                    f"{task_set_file.stem}_{variant}",
                    {task_id for task_id in ids if task_id.endswith(f"_{variant}")},
                )
                for variant in ("1", "2", "3")
            )
        for name, candidate_ids in candidates:
            overlap = len(candidate_ids & current_task_ids)
            total = len(candidate_ids)
            if overlap == 0:
                continue
            if (
                best is None
                or overlap > best[0]
                or (overlap == best[0] and total < best[1])
            ):
                best = (overlap, total, name)
    if best is None:
        return fallback, None
    return best[1], best[2]


def _render_view_body(
    outcomes_for_view: dict[str, dict],
    per_task_md: dict[str, dict[str, Any]],
    best_per_task: dict[str, dict[str, Any]],
    current_run_ids: set[str] | None,
    view_id: str,
    view_note: str = "",
) -> str:
    """Render one tab's body: Run summary metrics + Binary flips + Fail pattern catalog + All tasks.

    Re-computes everything fresh from outcomes_for_view so the Patched tab (which
    substitutes retry results in for timeout tasks) gets accurate aggregates.
    """
    # Build catalogs from THIS view's outcomes
    flip_catalog, fail_pattern_catalog = build_dynamic_pattern_catalog(
        outcomes_for_view, per_task_md
    )

    def _local_pattern(tid: str) -> tuple[str, dict[str, Any]] | None:
        combined = {**flip_catalog, **fail_pattern_catalog}
        for name, meta in combined.items():
            if tid in meta["tasks"]:
                return name, meta
        return None

    def _pattern_row(name: str, meta: dict[str, Any]) -> str:
        in_run = [t for t in meta["tasks"] if t in outcomes_for_view]
        gains = losses = same_pass = same_fail = 0
        for t in in_run:
            cs = outcomes_for_view[t]["current"]["summary"]
            bs = (
                outcomes_for_view[t]["baseline"]["summary"]
                if outcomes_for_view[t]["baseline"]
                else {}
            )
            bp = is_pass(bs)
            cp = is_pass(cs)
            if bp and cp:
                same_pass += 1
            elif not bp and not cp:
                same_fail += 1
            elif bp and not cp:
                losses += 1
            else:
                gains += 1
        flip_summary = []
        if gains:
            flip_summary.append(f'<span class="badge badge-gain">{gains} gain</span>')
        if losses:
            flip_summary.append(f'<span class="badge badge-loss">{losses} loss</span>')
        if same_pass or same_fail:
            flip_summary.append(
                f'<span class="badge badge-same">{same_pass}P/{same_fail}F same</span>'
            )
        return (
            f'<tr><td><a href="pattern_{name}.html">{esc(meta["title"])}</a></td>'
            f"<td>{len(in_run)}</td>"
            f"<td>{' '.join(flip_summary)}</td></tr>"
        )

    flip_rows = [_pattern_row(n, m) for n, m in flip_catalog.items()]
    fail_pattern_rows = [
        _pattern_row(n, m)
        for n, m in sorted(
            fail_pattern_catalog.items(), key=lambda kv: -len(kv[1]["tasks"])
        )
    ]

    total = len(outcomes_for_view)

    def _median(xs: list[float]) -> float | None:
        if not xs:
            return None
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def _fmt_num(v: float | None, kind: str) -> str:
        if v is None:
            return "-"
        if kind == "int":
            return f"{int(round(v)):,}"
        return f"{v:,.1f}"

    def _fmt_delta(cur: float | None, base: float | None, kind: str) -> str:
        if cur is None or base is None or base == 0:
            return "-"
        diff = cur - base
        diff_str = f"{int(round(diff)):+,}" if kind == "int" else f"{diff:+,.1f}"
        pct = (diff / base) * 100
        return f"{diff_str} ({pct:+.1f}%)"

    cur_wall: list[float] = []
    cur_llm: list[float] = []
    cur_tok: list[float] = []
    base_wall: list[float] = []
    base_llm: list[float] = []
    base_tok: list[float] = []
    cur_sub_pass = cur_sub_total = 0
    base_sub_pass = base_sub_total = 0
    cur_pass_subset = base_pass_subset = 0
    subset_n = 0
    for o in outcomes_for_view.values():
        bs_o = (o["baseline"] or {}).get("summary") if o["baseline"] else None
        if not bs_o:
            continue
        cs_o = o["current"]["summary"]
        subset_n += 1
        if is_pass(cs_o):
            cur_pass_subset += 1
        if is_pass(bs_o):
            base_pass_subset += 1
        if cs_o.get("wall_s") is not None and bs_o.get("wall_s") is not None:
            cur_wall.append(float(cs_o["wall_s"]))
            base_wall.append(float(bs_o["wall_s"]))
        cam = cs_o.get("aggregate_metrics", {}) or {}
        bam = bs_o.get("aggregate_metrics", {}) or {}
        if cam.get("llm_calls") is not None and bam.get("llm_calls") is not None:
            cur_llm.append(float(cam["llm_calls"]))
            base_llm.append(float(bam["llm_calls"]))
        if cam.get("total_tokens") is not None and bam.get("total_tokens") is not None:
            cur_tok.append(float(cam["total_tokens"]))
            base_tok.append(float(bam["total_tokens"]))
        ev_c = cs_o.get("eval") or {}
        ev_b = bs_o.get("eval") or {}
        if ev_c.get("passed") is not None and ev_c.get("total"):
            cur_sub_pass += int(ev_c["passed"])
            cur_sub_total += int(ev_c["total"])
        if ev_b.get("passed") is not None and ev_b.get("total"):
            base_sub_pass += int(ev_b["passed"])
            base_sub_total += int(ev_b["total"])

    split_total, split_name = _resolve_split_total(set(outcomes_for_view.keys()), total)

    def _row5(metric: str, cv: str, bv: str, dv: str) -> str:
        return (
            f"<tr><td>{esc(metric)}</td><td>{esc(cv)}</td>"
            f"<td>{esc(bv)}</td><td>{esc(dv)}</td></tr>"
        )

    def _pct_str(p: int, t: int) -> str:
        return f"{p} / {t} ({p / max(t, 1) * 100:.1f}%)"

    def _agg(xs: list[float], stat: str) -> float | None:
        if not xs:
            return None
        if stat == "sum":
            return sum(xs)
        if stat == "avg":
            return sum(xs) / len(xs)
        if stat == "p50":
            return _median(xs)
        return None

    metrics_rows = [
        _row5(
            "Tasks done",
            f"{total} / {split_total}" + (f" [{split_name}]" if split_name else ""),
            f"{subset_n} (with baseline)",
            "-",
        ),
        _row5(
            "Tasks passed",
            _pct_str(cur_pass_subset, subset_n),
            _pct_str(base_pass_subset, subset_n),
            f"{cur_pass_subset - base_pass_subset:+d}",
        ),
        _row5(
            "Test pass (sub-tests)",
            _pct_str(cur_sub_pass, cur_sub_total) if cur_sub_total else "-",
            _pct_str(base_sub_pass, base_sub_total) if base_sub_total else "-",
            f"{cur_sub_pass - base_sub_pass:+d}",
        ),
    ]
    for label, kind, cur_xs, base_xs in [
        ("Wall (s)", "float", cur_wall, base_wall),
        ("LLM calls", "int", cur_llm, base_llm),
        ("Total tokens", "int", cur_tok, base_tok),
    ]:
        for stat in ("sum", "avg", "p50"):
            cv = _agg(cur_xs, stat)
            bv = _agg(base_xs, stat)
            stat_label = {"sum": "total", "avg": "avg", "p50": "P50"}[stat]
            kind_used = kind if stat != "avg" else "float"
            metrics_rows.append(
                _row5(
                    f"{label} ({stat_label})",
                    _fmt_num(cv, kind_used),
                    _fmt_num(bv, kind_used),
                    _fmt_delta(cv, bv, kind_used),
                )
            )

    metrics_table_html = (
        '<h3 style="margin-top:1.2em;margin-bottom:.4em;">Aggregate metrics — current vs baseline (same task subset)</h3>'
        '<table style="margin-bottom:1em;">'
        "<thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Δ</th></tr></thead>"
        f"<tbody>{''.join(metrics_rows)}</tbody></table>"
    )

    rows = []
    for tid in sorted(outcomes_for_view):
        cs = outcomes_for_view[tid]["current"]["summary"]
        bs = (outcomes_for_view[tid]["baseline"] or {}).get("summary", {})
        cur_e = cs.get("eval") or {}
        base_e = bs.get("eval") or {}
        pn = _local_pattern(tid)
        pn_link = f'<a href="pattern_{pn[0]}.html">{esc(pn[0])}</a>' if pn else ""
        best = best_per_task.get(tid)
        if best:
            ev = best["eval"] or {}
            badge = status_badge(best["status"], ev.get("passed"), ev.get("total"))
            is_us = current_run_ids and best["run_id"] in current_run_ids
            crown = "🏆 " if is_us else ""
            run_label_short = best["run_id"].split("/")[0]
            best_cell = f'{crown}{badge}<br><small style="color:#586069">{esc(run_label_short)}</small>'
        else:
            best_cell = '<span class="badge badge-na">no data</span>'
        cur_metrics = cs.get("aggregate_metrics", {}) or {}
        rows.append(
            f'<tr><td><a href="task_{tid}.html">{tid}</a></td>'
            f"<td>{status_badge(bs.get('status', '?'), base_e.get('passed'), base_e.get('total'))}</td>"
            f"<td>{status_badge(cs.get('status', '?'), cur_e.get('passed'), cur_e.get('total'))}</td>"
            f"<td>{best_cell}</td>"
            f"<td>{flip_badge(is_pass(bs) if bs else False, is_pass(cs))}</td>"
            f"<td>{pn_link}</td>"
            f"<td>{fmt_seconds(cs.get('wall_s'))}</td>"
            f"<td>{esc(cur_metrics.get('llm_calls'))}</td>"
            f"<td>{fmt_int(cur_metrics.get('total_tokens'))}</td></tr>"
        )

    note_html = (
        f'<p class="subtitle"><strong>{esc(view_note)}</strong></p>'
        if view_note
        else ""
    )

    return f"""
{note_html}
<div class="summary-card">
{metrics_table_html}
</div>

<h3 style="margin-top:1.5em;">Binary flips</h3>
<table>
<thead><tr><th>Bucket</th><th>Tasks in run</th><th>Status</th></tr></thead>
<tbody>{"".join(flip_rows) if flip_rows else '<tr><td colspan="3"><em>無 binary flip</em></td></tr>'}</tbody>
</table>

<h3 style="margin-top:1.5em;">Fail pattern catalog</h3>
<table>
<thead><tr><th>Pattern</th><th>Tasks in run</th><th>Status</th></tr></thead>
<tbody>{"".join(fail_pattern_rows) if fail_pattern_rows else '<tr><td colspan="3"><em>無 FAIL task</em></td></tr>'}</tbody>
</table>

<h3 style="margin-top:1.5em;">All tasks</h3>
<table>
<thead><tr><th>Task</th><th>Baseline</th><th>Current</th><th>Best ever</th><th>Flip</th><th>Pattern</th><th>Wall</th><th>LLM calls</th><th>Tokens</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
"""


def index_page(
    task_outcomes: dict[str, dict],
    run_dirs: list[Path],
    baseline_dirs: list[Path],
    high_water: dict[str, Any] | None = None,
    current_run_ids: set[str] | None = None,
    best_per_task: dict[str, dict[str, Any]] | None = None,
    flip_catalog: dict[str, dict[str, Any]] | None = None,
    fail_pattern_catalog: dict[str, dict[str, Any]] | None = None,
    retry_outcomes: dict[str, dict] | None = None,
    retry_run_dirs: list[Path] | None = None,
    per_task_md: dict[str, dict[str, Any]] | None = None,
) -> str:
    flip_catalog = flip_catalog or {}
    fail_pattern_catalog = fail_pattern_catalog or {}
    retry_outcomes = retry_outcomes or {}
    per_task_md = per_task_md or {}
    best_per_task = best_per_task or {}

    # Build Full batch view body (raw current data, no retry overrides)
    full_view_body = _render_view_body(
        task_outcomes,
        per_task_md,
        best_per_task,
        current_run_ids,
        view_id="full",
        view_note="原始 batch 結果 — 直接看 17h batch 跑完的數據, 不蓋 retry。",
    )

    # Build Patched view body — substitute retry result for any retry-batch task
    patched_outcomes: dict[str, dict] = {}
    for tid, oc in task_outcomes.items():
        if tid in retry_outcomes and retry_outcomes[tid].get("retry"):
            patched_outcomes[tid] = {
                "current": retry_outcomes[tid]["retry"],
                "baseline": oc["baseline"],
            }
        else:
            patched_outcomes[tid] = oc
    patched_view_body = (
        _render_view_body(
            patched_outcomes,
            per_task_md,
            best_per_task,
            current_run_ids,
            view_id="patched",
            view_note=(
                f"Retry-merged view — 對 {len(retry_outcomes)} 個 TIMED_OUT task 用 retry 結果覆蓋 "
                "(同 code, 無 batch contention), 其他 task 保留 full56 結果。"
                "更接近「fair infra 下的 trace fix 真實能力」估計。"
            ),
        )
        if retry_outcomes
        else ""
    )

    # Build Timeout retry table (diagnostic tab, kept as-is)
    retry_rows: list[str] = []
    retry_flip_counts = {"to_pass": 0, "to_fail": 0, "still_timeout": 0}
    for tid in sorted(retry_outcomes):
        ro = retry_outcomes[tid]
        rs = ro["retry"]["summary"]
        bs = (ro["baseline"] or {}).get("summary", {}) if ro["baseline"] else {}
        fs = (ro["current"] or {}).get("summary", {}) if ro["current"] else {}
        ret_e = rs.get("eval") or {}
        base_e = bs.get("eval") or {}
        full_e = fs.get("eval") or {}
        rs_pf = rs.get("status")
        fs_pf = fs.get("status")
        if fs_pf == "TIMED_OUT" and rs_pf == "COMPLETED":
            retry_flip_counts["to_pass"] += 1
            flip_cell = '<span class="badge badge-gain">FLIP-TO-PASS</span>'
        elif fs_pf == "TIMED_OUT" and rs_pf == "FAILED":
            retry_flip_counts["to_fail"] += 1
            flip_cell = '<span class="badge badge-same">partial</span>'
        elif fs_pf == "TIMED_OUT" and rs_pf == "TIMED_OUT":
            retry_flip_counts["still_timeout"] += 1
            flip_cell = '<span class="badge badge-loss">still-timeout</span>'
        else:
            flip_cell = '<span class="badge badge-na">-</span>'
        pn = pattern_for_task(tid)
        pn_link = f'<a href="pattern_{pn[0]}.html">{esc(pn[0])}</a>' if pn else ""
        ret_metrics = rs.get("aggregate_metrics", {}) or {}
        retry_rows.append(
            # link with #tab-retry so task page opens directly on Retry tab
            f'<tr><td><a href="task_{tid}.html#tab-retry">{tid}</a></td>'
            f"<td>{status_badge(bs.get('status', '?'), base_e.get('passed'), base_e.get('total'))}</td>"
            f"<td>{status_badge(fs.get('status', '?'), full_e.get('passed'), full_e.get('total'))}</td>"
            f"<td>{status_badge(rs.get('status', '?'), ret_e.get('passed'), ret_e.get('total'))}</td>"
            f"<td>{flip_cell}</td>"
            f"<td>{pn_link}</td>"
            f"<td>{fmt_seconds(rs.get('wall_s'))}</td>"
            f"<td>{esc(ret_metrics.get('llm_calls'))}</td>"
            f"<td>{fmt_int(ret_metrics.get('total_tokens'))}</td></tr>"
        )

    retry_view_body = ""
    if retry_rows:
        retry_run_labels = (
            ", ".join(run_label(r) for r in retry_run_dirs) if retry_run_dirs else ""
        )
        rfc = retry_flip_counts
        flip_summary = (
            f'<span class="badge badge-gain">{rfc["to_pass"]} flip-to-pass</span> '
            f'<span class="badge badge-same">{rfc["to_fail"]} partial</span> '
            f'<span class="badge badge-loss">{rfc["still_timeout"]} still-timeout</span>'
        )
        retry_view_body = f"""
<p class="subtitle">Diagnostic re-run of TIMED_OUT tasks: same code, sequential, no batch contention. <strong>不是</strong> 真實 batch 結果; 純 diagnostic 用來分離 infra vs mechanism。</p>
<div class="kv"><span class="key">Retry run</span> <code>{esc(retry_run_labels)}</code></div>
<div class="kv"><span class="key">Tally</span> {flip_summary}</div>
<table style="margin-top:.8em;">
<thead><tr><th>Task</th><th>Baseline</th><th>Full batch</th><th>Retry</th><th>Retry vs full</th><th>Pattern</th><th>Wall</th><th>LLM calls</th><th>Tokens</th></tr></thead>
<tbody>{"".join(retry_rows)}</tbody>
</table>
"""

    # Build header (shared across tabs)
    title_label = run_dirs[0].parent.name if run_dirs else "analysis"
    if run_dirs:
        key_text = "Current run" if len(run_dirs) == 1 else "Current runs"
        run_rows_html = "\n".join(
            f'<div class="kv"><span class="key">{key_text if i == 0 else ""}</span> '
            f"<code>{esc(run_label(rd))}</code></div>"
            for i, rd in enumerate(run_dirs)
        )
    else:
        run_rows_html = (
            '<div class="kv"><span class="key">Current run</span> <em>none</em></div>'
        )
    baseline_labels = ", ".join(run_label(b) for b in baseline_dirs) or "(none)"
    baseline_row_html = (
        f'<div class="kv"><span class="key">Baseline</span> '
        f"<code>{esc(baseline_labels)}</code></div>"
    )
    if retry_run_dirs:
        retry_label_html = (
            f'<div class="kv"><span class="key">Retry run</span> '
            f"<code>{esc(', '.join(run_label(r) for r in retry_run_dirs))}</code></div>"
        )
    else:
        retry_label_html = ""

    # Tab nav
    tabs_nav = ['<div class="tab-btn active" data-tab="tab-full">Full batch</div>']
    tab_bodies = [
        f'<div id="tab-full" class="tab-content active">{full_view_body}</div>'
    ]
    if patched_view_body:
        tabs_nav.append(
            '<div class="tab-btn" data-tab="tab-patched">Patched (retry merged)</div>'
        )
        tab_bodies.append(
            f'<div id="tab-patched" class="tab-content">{patched_view_body}</div>'
        )
    if retry_view_body:
        tabs_nav.append(
            '<div class="tab-btn" data-tab="tab-retry">Timeout retry diagnostic</div>'
        )
        tab_bodies.append(
            f'<div id="tab-retry" class="tab-content">{retry_view_body}</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{esc(title_label)} — analysis viewer</title>
<style>{CSS}</style></head><body>
<h1>{esc(title_label)} — analysis viewer</h1>
<p class="subtitle">Static viewer over the per-call LLM input/output captured in subagent_io.jsonl.</p>

<div class="summary-card">
<h2 style="border:0;margin-top:0;">Run summary</h2>
{run_rows_html}
{baseline_row_html}
{retry_label_html}
{_high_water_row(high_water, current_run_ids)}
</div>

<div class="tabs">
{"".join(tabs_nav)}
</div>

{"".join(tab_bodies)}

<script>
document.querySelectorAll('.tab-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(btn.dataset.tab).classList.add('active');
  }});
}});
</script>
</body></html>
"""


def _read_baseline_config(config_path: Path) -> list[str]:
    """Read baseline list from logs/_registry/config.json. Returns list of
    experiment dir basenames. Multiple entries support sharded baselines
    (e.g., cycle_recovery_full56_v1 + cycle_recovery_full56_v1b)."""
    cfg = load_skill_config(config_path)
    return list(cfg.get("baseline", []))


def _write_baseline_config(config_path: Path, new_baseline: str) -> None:
    """Promote the default future baseline while preserving other config."""
    cfg = load_skill_config(config_path)
    cfg["baseline"] = [new_baseline]
    save_skill_config(cfg, config_path)


_PROMOTION_PREV_RE = re.compile(r"\bprev=(?P<prev>[^ ]+)\s+\(")


def _read_latest_previous_baselines(history_path: Path) -> list[str]:
    """Return the latest promotion's previous baseline list, if available."""
    if not history_path.exists():
        return []
    for line in reversed(history_path.read_text(encoding="utf-8").splitlines()):
        match = _PROMOTION_PREV_RE.search(line)
        if not match:
            continue
        return [name for name in match.group("prev").split(",") if name]
    return []


def _resolve_default_baselines(
    configured_baselines: list[str],
    *,
    current_exp_name: str,
    history_path: Path,
) -> tuple[list[str], str]:
    """Choose the implicit baseline without silently self-comparing.

    `config.json["baseline"]` is the current high-water baseline for future
    runs. When rebuilding that same experiment's viewer, comparing the run to
    itself is useless, so recover the immediately previous baseline from
    `baseline_history.txt` just for this rebuild.
    """
    if configured_baselines != [current_exp_name]:
        return configured_baselines, "config.json"
    previous_baselines = _read_latest_previous_baselines(history_path)
    if previous_baselines:
        return previous_baselines, "baseline_history.txt fallback"
    return configured_baselines, "config.json"


def _append_baseline_promotion(
    history_path: Path,
    *,
    previous_baselines: list[str],
    new_baseline: str,
    base_pass: int,
    cur_pass: int,
) -> None:
    """Append one high-water promotion edge to the history ledger."""
    from datetime import date

    hist_line = (
        f"{date.today().isoformat()} promoted: prev={','.join(previous_baselines)} "
        f"({base_pass}P) → {new_baseline} ({cur_pass}P, +{cur_pass - base_pass})\n"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(hist_line)


def _promote_baseline(
    config_path: Path,
    history_path: Path,
    *,
    previous_baselines: list[str],
    new_baseline: str,
    base_pass: int,
    cur_pass: int,
) -> None:
    """Record A→B in the history ledger only.

    Per the 2026-05-19 fix, auto-promotion no longer mutates
    `config.json["baseline"]`. The history file is the audit trail; the
    operator decides when to promote the next high-water into the
    config. This avoids the self-compare trap that fired on 2026-05-17,
    where `v6v7_universal_full` was rewritten into the config the moment
    it beat the prior baseline, and the next rebuild silently produced
    a `current vs current` diff.

    `config_path` stays in the signature for API symmetry with the
    earlier behavior; it's currently unread.
    """
    del config_path  # retained for backwards compatibility
    _append_baseline_promotion(
        history_path,
        previous_baselines=previous_baselines,
        new_baseline=new_baseline,
        base_pass=base_pass,
        cur_pass=cur_pass,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=Path,
        help="Current-run dir. Accepts either timestamp dir "
        "(logs/<exp>/<ts>/) or experiment dir "
        "(logs/<exp>/) — latter auto-expands to combine "
        "all timestamp subdirs. Repeatable.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        type=Path,
        default=None,
        help="Baseline run dir. Same auto-expansion as --run. "
        "If omitted, reads `baseline` list from "
        "logs/_registry/config.json. Repeatable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir for HTML viewer. Defaults to "
        "logs/<run_exp_name>/analysis/viewer/.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        help="Root logs dir for high-water scan. Defaults to "
        "common parent of --run paths.",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Registry dir. Defaults to <logs-dir>/_registry.",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Dir holding per-task .md editorials. Defaults "
        "to logs/<run_exp_name>/analysis/tasks/.",
    )
    parser.add_argument(
        "--retry-run",
        action="append",
        type=Path,
        default=None,
        help="Optional retry run dir (timeout-retry diagnostic). "
        "Same auto-expansion as --run. Repeatable.",
    )
    parser.add_argument(
        "--init-starter-mds",
        action="store_true",
        help="For each FAIL task without an existing "
        "<tid>.md, write a Phase-0 starter file with "
        "sub-tests + milestones pre-filled.",
    )
    parser.add_argument(
        "--no-promote-baseline",
        action="store_true",
        help="Skip appending a high-water promotion entry even "
        "if current beats baseline on binary PASS.",
    )
    args = parser.parse_args()

    # Derive logs / registry dirs early — needed for config baseline lookup.
    logs_dir = args.logs_dir
    if logs_dir is None and args.run:
        first = args.run[0]
        # If --run is an experiment dir, parent is logs/; if timestamp dir, parent.parent is logs/
        logs_dir = first.parent if _expand_run_dir(first) == [first] else first
        logs_dir = logs_dir.parent if logs_dir.name != "logs" else logs_dir
    registry_dir = args.registry_dir or (logs_dir / "_registry" if logs_dir else None)

    # Auto tasks-dir + out defaulting based on --run's experiment name.
    # If --run is a timestamp dir, exp_name = parent.name; else --run.name.
    first_run = args.run[0]
    exp_name = (
        first_run.parent.name
        if _expand_run_dir(first_run) == [first_run]
        else first_run.name
    )

    # If --baseline not given, fall back to config.json `baseline`. Guard the
    # historical self-compare trap from older auto-promotion behavior.
    baseline_was_implicit = args.baseline is None
    if baseline_was_implicit:
        if registry_dir is None:
            parser.error("cannot infer baseline without a registry directory")
        config_path = registry_dir / "config.json"
        history_path = registry_dir / "baseline_history.txt"
        cfg_baselines = _read_baseline_config(config_path)
        if not cfg_baselines:
            parser.error(
                f"no --baseline given and no `baseline` entry found in {config_path}"
            )
        default_baselines, baseline_source = _resolve_default_baselines(
            cfg_baselines,
            current_exp_name=exp_name,
            history_path=history_path,
        )
        args.baseline = [(logs_dir / name) for name in default_baselines]
        print(f"Baseline from {baseline_source}: {', '.join(default_baselines)}")
    if args.tasks_dir is None:
        args.tasks_dir = logs_dir / exp_name / "analysis" / "tasks"
    if args.out is None:
        args.out = logs_dir / exp_name / "analysis" / "viewer"

    args.out.mkdir(parents=True, exist_ok=True)

    current_tasks = discover_tasks(args.run)
    baseline_tasks = discover_tasks(args.baseline)

    print(f"Current tasks: {len(current_tasks)}")
    print(f"Baseline tasks: {len(baseline_tasks)}")

    task_outcomes: dict[str, dict] = {}
    for tid, td in current_tasks.items():
        cur = load_task(td)
        base = load_task(baseline_tasks[tid]) if tid in baseline_tasks else None
        task_outcomes[tid] = {"current": cur, "baseline": base}

    # Optional: load retry batch data (diagnostic re-run of timeout tasks).
    # Each retry task has corresponding current (full56 timeout) + baseline.
    retry_outcomes: dict[str, dict] = {}
    if args.retry_run:
        retry_tasks = discover_tasks(args.retry_run)
        for tid, td in retry_tasks.items():
            retry_outcomes[tid] = {
                "retry": load_task(td),
                "current": task_outcomes[tid]["current"]
                if tid in task_outcomes
                else None,
                "baseline": task_outcomes[tid]["baseline"]
                if tid in task_outcomes
                else None,
            }
        print(f"Retry tasks: {len(retry_outcomes)}")

    # logs_dir + registry_dir already resolved earlier (before .env baseline
    # lookup); just compute derived artifacts here.
    high_water = find_high_water(logs_dir, registry_dir) if logs_dir else None
    tasks_dir = args.tasks_dir
    per_task = load_per_task_md(tasks_dir)
    translations = load_translations(registry_dir)
    analyses = load_analyses(per_task)
    # Report any task this batch needs but whose translation is missing —
    # Claude (analyst) writes those into task_translations_zh.json by hand.
    instructions_en = {
        tid: oc["current"]["summary"].get("instruction", "")
        for tid, oc in task_outcomes.items()
        if oc["current"]["summary"].get("instruction")
    }
    if registry_dir is not None:
        missing = translations_cache.find_missing(instructions_en, registry_dir)
        if missing:
            sys.stderr.write(
                f"[viewer] {len(missing)} task(s) missing translation in "
                f"{registry_dir}/{translations_cache.REGISTRY_FILENAME}:\n"
            )
            for tid, en in missing:
                sys.stderr.write(f"  - {tid}: {en[:90]}\n")
            sys.stderr.write(
                "[viewer] Translate these and add to the JSON; viewer renders "
                "English fallback until they're added.\n"
            )
    all_task_results = scan_all_task_results(logs_dir) if logs_dir else {}
    best_per_task = {
        tid: best_result_for(results)
        for tid, results in all_task_results.items()
        if best_result_for(results) is not None
    }
    # current_run_ids = "<exp>/<ts>" identifiers, expanded to match the
    # run_id format used by scan_all_task_results (which records each
    # timestamp dir's run_id). When --run is an experiment dir, expand to
    # all its timestamp subdirs; when it's a timestamp dir, take as-is.
    current_run_ids: set[str] = set()
    for rd in args.run:
        for ts_dir in _expand_run_dir(rd):
            current_run_ids.add(f"{ts_dir.parent.name}/{ts_dir.name}")

    # Optionally emit Phase-0 starter .md files for FAIL tasks without one.
    if args.init_starter_mds:
        tasks_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for tid, oc in task_outcomes.items():
            if is_pass(oc["current"]["summary"]):
                continue
            md_path = tasks_dir / f"{tid}.md"
            if md_path.exists():
                continue
            md_path.write_text(build_starter_md(tid, oc["current"]))
            written += 1
        if written:
            sys.stderr.write(
                f"[viewer] init-starter-mds: wrote {written} new starter .md "
                f"file(s) to {tasks_dir}\n"
            )

    # Build dynamic pattern catalog from .md frontmatter (replaces hardcoded).
    # Filters current-PASS tasks out of fail clusters and surfaces gains/losses
    # as their own boxes.
    flip_catalog, fail_pattern_catalog = build_dynamic_pattern_catalog(
        task_outcomes, per_task
    )
    # Module-global override so pattern_for_task() picks up the dynamic catalog.
    # Combined view for per-task lookup (so each task knows its pattern badge).
    global PATTERN_CATALOG
    PATTERN_CATALOG = {**flip_catalog, **fail_pattern_catalog}

    # Write task pages
    for tid, oc in task_outcomes.items():
        pn = pattern_for_task(tid)
        pname = pn[0] if pn else None
        best = best_per_task.get(tid)
        # Load the best run's task data for the Best ever tab.
        best_task = None
        if best and logs_dir:
            best_task_dir = logs_dir / best["run_id"] / tid
            if best_task_dir.is_dir():
                try:
                    best_task = load_task(best_task_dir)
                except Exception:
                    best_task = None
        # Load retry task data for the Retry tab (only for tasks in retry batch)
        retry_task = None
        if tid in retry_outcomes:
            retry_task = retry_outcomes[tid].get("retry")
        html_txt = task_page(
            tid,
            oc["current"],
            oc["baseline"],
            pname,
            best=best,
            best_task=best_task,
            retry_task=retry_task,
            current_run_ids=current_run_ids,
            instruction_zh=translations.get(tid),
            analysis=analyses.get(tid),
        )
        (args.out / f"task_{tid}.html").write_text(html_txt)

    # Write pattern pages — one per dynamic catalog entry (flip + fail combined)
    for name, meta in PATTERN_CATALOG.items():
        html_txt = pattern_page(name, meta, task_outcomes)
        (args.out / f"pattern_{name}.html").write_text(html_txt)

    # Write index
    (args.out / "index.html").write_text(
        index_page(
            task_outcomes,
            args.run,
            args.baseline,
            high_water=high_water,
            current_run_ids=current_run_ids,
            best_per_task=best_per_task,
            flip_catalog=flip_catalog,
            fail_pattern_catalog=fail_pattern_catalog,
            retry_outcomes=retry_outcomes,
            retry_run_dirs=args.retry_run,
            per_task_md=per_task,
        )
    )

    # Baseline promotion: strict binary PASS improvement against the implicit
    # current high-water baseline → append A→B to the history ledger.
    # `config.json` is no longer auto-mutated (operator promotes manually).
    # The current_is_configured_baseline guard still applies in case an
    # operator manually set the config to the run being rebuilt — prevents
    # appending a B→B line.
    cur_pass = sum(
        1 for o in task_outcomes.values() if is_pass(o["current"]["summary"])
    )
    base_pass = sum(
        1
        for o in task_outcomes.values()
        if o["baseline"] and is_pass(o["baseline"]["summary"])
    )
    # `base_total` MUST be the actual baseline run's task count (e.g. 56 for
    # full56), NOT the intersection with current. Otherwise a 10-task smoke
    # passes `cur_total >= base_total` (10 >= 10) and wrongly promotes over a
    # 56-task baseline. The criterion per /view-build SKILL.md is "current task
    # count >= baseline task count" — guards against short-batch 篡位.
    cur_total = len(task_outcomes)
    base_total = len(baseline_tasks)
    config_path = registry_dir / "config.json" if registry_dir is not None else None
    configured_baselines = (
        _read_baseline_config(config_path) if config_path is not None else []
    )
    current_is_configured_baseline = configured_baselines == [exp_name]
    if (
        not args.no_promote_baseline
        and baseline_was_implicit
        and registry_dir is not None
        and not current_is_configured_baseline
        and cur_pass > base_pass
        and cur_total >= base_total > 0
    ):
        new_baseline = exp_name
        hist_path = registry_dir / "baseline_history.txt"
        _promote_baseline(
            config_path,
            hist_path,
            previous_baselines=configured_baselines,
            new_baseline=new_baseline,
            base_pass=base_pass,
            cur_pass=cur_pass,
        )
        sys.stderr.write(
            f"\n🏆 New high-water detected: {base_pass}P → {cur_pass}P (+{cur_pass - base_pass}).\n"
            f"   Appended to history: {hist_path}\n"
            f'   `config.json["baseline"]` NOT auto-updated — promote manually when ready.\n'
        )
    else:
        # Status report only — we did NOT promote (took the else branch).
        # Reason can be: delta<=0, --no-promote-baseline, or task-count guard.
        delta = cur_pass - base_pass
        sign = "+" if delta >= 0 else ""
        if delta <= 0:
            reason = "No promotion (strict > required)."
        elif args.no_promote_baseline:
            reason = "Auto-promotion disabled (--no-promote-baseline)."
        elif not baseline_was_implicit:
            reason = "No promotion: explicit --baseline does not mutate registry."
        elif current_is_configured_baseline:
            reason = "No promotion: current run is already the configured baseline."
        elif cur_total < base_total:
            reason = (
                f"No promotion: current task count ({cur_total}) < baseline "
                f"task count ({base_total}). Short-batch can't promote over "
                "a wider baseline."
            )
        else:
            reason = "No promotion (criterion not met)."
        sys.stderr.write(
            f"\nBaseline check: current {cur_pass}P / baseline {base_pass}P "
            f"(Δ {sign}{delta}). {reason}\n"
        )

    print(f"Viewer written to {args.out}/index.html")


if __name__ == "__main__":
    main()
