from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

from adk_appworld_agent.observability.markdown import render_task_summary_markdown

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_RUN_DIR_RE = re.compile(r"^(?P<ts>\d{8}_\d{6}(?:_\d{6})?)$")
LOG_DETAIL_ENV = "APPWORLD_LOG_DETAIL"
COMPACT_LOG_DETAIL = "compact"
RAW_LOG_DETAIL = "raw"


def log_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def format_seconds(ms: int | None) -> str:
    if ms is None:
        return "NA"
    return f"{ms / 1000:.1f}s"


def clean_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


def raw_log_detail_enabled() -> bool:
    return os.getenv(LOG_DETAIL_ENV, COMPACT_LOG_DETAIL).strip().lower() in {
        RAW_LOG_DETAIL,
        "debug",
        "full",
    }


def extract_report_text(report: object) -> str:
    if isinstance(report, dict):
        for key in ("result", "report", "message"):
            value = report.get(key)
            if value is not None:
                return str(value)
    return str(report or "")


def write_evaluation_report(
    log_dir: Path, timestamp: str, report: object
) -> Path | None:
    report_text = clean_ansi(extract_report_text(report)).strip()
    if not report_text:
        return None
    report_path = log_dir / "evaluation.txt"
    report_path.write_text(report_text + "\n", encoding="utf-8")
    return report_path


def write_json(path: Path, payload: object) -> Path:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def write_jsonl(path: Path, records: list[object]) -> Path:
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_subagent_artifacts(
    log_dir: Path,
    *,
    subagent_outputs: dict[str, list[dict]],
    ledger: list[dict],
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    for phase_name, items in subagent_outputs.items():
        phase_records = [dict(item) for item in (items or []) if isinstance(item, dict)]
        if not phase_records:
            continue
        phase_path = write_jsonl(
            log_dir / f"{phase_name.lower()}.subagent.jsonl", phase_records
        )
        artifact_paths[f"{phase_name.lower()}_subagent_path"] = str(phase_path)
    if ledger:
        ledger_path = write_jsonl(log_dir / "ledger.jsonl", ledger)
        artifact_paths["ledger_path"] = str(ledger_path)
    return artifact_paths


def write_task_summary(log_dir: Path, summary: dict) -> Path:
    path = log_dir / "task_summary.txt"
    path.write_text(render_task_summary_markdown(summary), encoding="utf-8")
    return path


def summarize_subagent_outputs(
    subagent_outputs: dict[str, list[dict]],
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for phase_name, items in subagent_outputs.items():
        latest = next(
            (dict(item) for item in reversed(items or []) if isinstance(item, dict)),
            None,
        )
        if latest is None:
            continue
        payload = latest.get("payload") or {}
        entry = {
            "subagent_name": latest.get("subagent_name"),
            "status": latest.get("status"),
            "failure_code": latest.get("failure_code"),
            "metrics": latest.get("metrics") or {},
        }
        if phase_name == "FIND":
            entry["payload_summary"] = {
                "matched_apps": payload.get("matched_apps"),
                "selected_communities": payload.get("selected_communities"),
                "candidate_count": payload.get("candidate_count"),
            }
        elif phase_name == "EXECUTE":
            candidate = payload.get("submission_candidate")
            candidate_summary = {}
            if isinstance(candidate, dict):
                candidate_summary = {
                    "task_type_hint": candidate.get("task_type_hint"),
                    "answer_type": candidate.get("answer_type"),
                    "source": candidate.get("source"),
                }
            entry["payload_summary"] = {
                "tool_call_count": payload.get("tool_call_count"),
                "final_response_chars": len(str(payload.get("final_response") or "")),
                "submission_candidate": candidate_summary,
                "llm_raised": payload.get("llm_raised"),
            }
        elif phase_name == "SUBMIT":
            entry["payload_summary"] = {
                "status": payload.get("status"),
                "block_reason": payload.get("block_reason"),
                "extra": payload.get("extra"),
            }
        else:
            entry["payload_summary"] = payload
        summary[phase_name.lower()] = entry
    return summary


def _parse_timestamp(raw: str) -> datetime | None:
    for fmt in ("%Y%m%d_%H%M%S_%f", "%Y%m%d_%H%M%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _prune_grouped_files(
    files: list[Path],
    *,
    matcher: re.Pattern[str],
    keep_last: int,
) -> None:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        match = matcher.match(path.name)
        if match is None:
            continue
        timestamp = match.group("ts")
        grouped.setdefault(timestamp, []).append(path)
    if len(grouped) <= keep_last:
        return
    ordered = sorted(
        grouped,
        key=lambda raw: (_parse_timestamp(raw) or datetime.min, raw),
        reverse=True,
    )
    to_remove = ordered[keep_last:]
    for timestamp in to_remove:
        for path in grouped[timestamp]:
            try:
                path.unlink()
            except FileNotFoundError:
                continue


def prune_experiment_runs(experiment_dir: Path, *, keep_last: int = 5) -> None:
    run_dirs = [
        path
        for path in experiment_dir.iterdir()
        if path.is_dir() and _RUN_DIR_RE.match(path.name)
    ]
    if len(run_dirs) <= keep_last:
        return
    ordered = sorted(
        run_dirs,
        key=lambda path: (_parse_timestamp(path.name) or datetime.min, path.name),
        reverse=True,
    )
    for path in ordered[keep_last:]:
        shutil.rmtree(path, ignore_errors=True)
