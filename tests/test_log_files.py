from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.observability.log_files import (
    clean_ansi,
    extract_report_text,
    format_seconds,
    prune_experiment_runs,
    summarize_subagent_outputs,
    write_evaluation_report,
    write_subagent_artifacts,
    write_task_summary,
)


def _touch(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def test_prune_experiment_runs_keeps_latest_five_run_dirs(tmp_path):
    for idx in range(1, 8):
        timestamp = f"20260420_11010{idx}_00000{idx}"
        run_dir = tmp_path / timestamp / "task_a"
        run_dir.mkdir(parents=True)
        _touch(run_dir / "controller.jsonl")
        _touch(run_dir / "executor.jsonl")
        _touch(run_dir / "evaluation.txt")

    prune_experiment_runs(tmp_path, keep_last=5)

    remaining = sorted(path.name for path in tmp_path.iterdir() if path.is_dir())
    assert remaining == [
        "20260420_110103_000003",
        "20260420_110104_000004",
        "20260420_110105_000005",
        "20260420_110106_000006",
        "20260420_110107_000007",
    ]


def test_format_seconds_uses_seconds_not_milliseconds():
    assert format_seconds(12345) == "12.3s"
    assert format_seconds(None) == "NA"


def test_write_evaluation_report_cleans_ansi_and_extracts_result(tmp_path):
    report_path = write_evaluation_report(
        tmp_path,
        "20260420_120000_000001",
        {"result": "\u001b[92mNum Failed Tests : 1\u001b[0m"},
    )

    assert report_path is not None
    assert report_path.read_text(encoding="utf-8") == "Num Failed Tests : 1\n"
    assert clean_ansi("\u001b[92mhello\u001b[0m") == "hello"
    assert extract_report_text({"message": "fallback"}) == "fallback"


def test_write_subagent_artifacts_writes_phase_files_and_ledger(tmp_path):
    artifact_paths = write_subagent_artifacts(
        tmp_path,
        subagent_outputs={
            "FIND": [{"phase": "FIND", "subagent_name": "finder", "attempt": 1}],
            "EXECUTE": [
                {"phase": "EXECUTE", "subagent_name": "executor", "attempt": 1}
            ],
        },
        ledger=[{"phase_decision": "PLAN succeeded -> FIND"}],
    )

    assert (tmp_path / "find.subagent.jsonl").exists()
    assert (tmp_path / "execute.subagent.jsonl").exists()
    assert (tmp_path / "ledger.jsonl").exists()
    assert "subagents_path" not in artifact_paths
    assert "final_state_path" not in artifact_paths


def test_summarize_subagent_outputs_extracts_phase_specific_fields():
    summary = summarize_subagent_outputs(
        {
            "FIND": [
                {
                    "subagent_name": "finder",
                    "status": "SUCCEEDED",
                    "payload": {"matched_apps": ["venmo"], "candidate_count": 3},
                    "metrics": {"llm_calls": 2},
                }
            ],
            "EXECUTE": [
                {
                    "subagent_name": "executor",
                    "status": "SUCCEEDED",
                    "payload": {
                        "tool_call_count": 4,
                        "final_response": "raw final text",
                        "submission_candidate": {
                            "answer": "null",
                            "task_type_hint": "action",
                            "answer_type": "null",
                            "source": "executor",
                        },
                    },
                }
            ],
        }
    )

    assert summary["find"]["payload_summary"]["candidate_count"] == 3
    assert summary["execute"]["payload_summary"]["tool_call_count"] == 4
    assert summary["execute"]["payload_summary"]["final_response_chars"] == len(
        "raw final text"
    )
    assert summary["execute"]["payload_summary"]["submission_candidate"] == {
        "task_type_hint": "action",
        "answer_type": "null",
        "source": "executor",
    }
    assert "answer" not in summary["execute"]["payload_summary"]["submission_candidate"]


def test_write_task_summary_writes_task_summary_plain_text(tmp_path):
    path = write_task_summary(tmp_path, {"task_id": "abc", "status": "FAILED"})
    assert path.name == "task_summary.txt"
    content = path.read_text(encoding="utf-8")
    assert "Task abc" in content
    assert "Status        FAILED" in content
