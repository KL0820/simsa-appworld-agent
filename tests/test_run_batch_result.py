from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from unittest.mock import patch


def _load_run_batch_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_batch.py"
    spec = importlib.util.spec_from_file_location("run_batch_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_config_record_keeps_provider_default_marker():
    run_batch = _load_run_batch_module()
    args = type(
        "Args",
        (),
        {
            "model_name": "gemini-test",
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "seed": 0,
            "max_output_tokens": 0,
        },
    )()

    assert run_batch._model_config_record(args) == {
        "name": "gemini-test",
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 0,
        "max_output_tokens": "provider_default",
    }


def test_batch_header_uses_authoritative_json_config(tmp_path):
    run_batch = _load_run_batch_module()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {"name": "configured-model", "temperature": 0.25},
                "impls": {
                    "PLAN": "rough_skill",
                    "FIND": "community",
                    "EXECUTE": "code_plan_execute_skill",
                },
                "cache": {"read": [], "dir": None},
            }
        )
    )
    args = _controller_args(config=str(config))
    assert run_batch._model_config_record(args)["name"] == "configured-model"
    assert run_batch._model_config_record(args)["temperature"] == 0.25
    header = run_batch._batch_config_record(args)
    assert header["plan"] == "rough_skill"
    assert header["execute"] == "code_plan_execute_skill"
    assert header["verify"] == "registry-default"
    assert header["cache_read"] == ""


def test_batch_header_describes_cli_skill_implementations():
    run_batch = _load_run_batch_module()
    assert run_batch._batch_config_record(_controller_args())["plan"] == "rough_skill"
    assert (
        run_batch._batch_config_record(_controller_args(skills="off"))["plan"]
        == "rough"
    )
    header = run_batch._batch_config_record(_controller_args(skills="native_best"))
    assert header["execute"] == "code_plan_execute_skill_native"


def _controller_args(**overrides):
    base = {
        "execute_impl": "stub",
        "experiment_name": "find_milestone",
        "rpc_url": "tcp://127.0.0.1:4242",
        "log_root": "logs",
        "find_impl": "community",
        "plan_impl": "rough",
        "verify_impl": "stub",
        "timeout": 30,
        "model_name": "gemini-test",
        "temperature": 0,
        "top_p": 1,
        "top_k": 1,
        "seed": 0,
        "max_output_tokens": 0,
        "show_raw_eval_report": False,
        "keep_debug": False,
    }
    base.update(overrides)
    return type("Args", (), base)()


def test_controller_command_passes_run_name_and_execute_impl():
    run_batch = _load_run_batch_module()
    args = _controller_args()

    cmd = run_batch._controller_command(
        "task_1", args, run_name="20260423_020000_000000"
    )

    assert "--run_name" in cmd
    assert cmd[cmd.index("--run_name") + 1] == "20260423_020000_000000"
    assert cmd[cmd.index("--execute") + 1] == "stub"
    # Cache CLI flags removed in M9; cache control is env-based now.
    for stale in (
        "--rough_plan_cache",
        "--subagent_cache",
        "--cache_read",
        "--cache_write",
        "--cache_version",
        "--cache",
    ):
        assert stale not in cmd
    assert "--keep-debug" not in cmd


def test_controller_command_passes_keep_debug_flag():
    run_batch = _load_run_batch_module()
    args = _controller_args(keep_debug=True)

    cmd = run_batch._controller_command(
        "task_1", args, run_name="20260423_020000_000000"
    )

    assert "--keep-debug" in cmd


def test_wrapper_command_file_lives_under_artifacts(tmp_path):
    run_batch = _load_run_batch_module()
    path = run_batch._write_command_file(tmp_path, ["python", "scripts/run_task.py"])

    assert path == tmp_path / "artifacts" / "command.txt"
    assert path.read_text(encoding="utf-8").strip() == "python scripts/run_task.py"


def test_load_fresh_task_summary_ignores_stale_json(tmp_path):
    run_batch = _load_run_batch_module()
    path = tmp_path / "task_summary.json"
    path.write_text(json.dumps({"task_id": "old"}), encoding="utf-8")
    old_mtime = time.time() - 60
    os.utime(path, (old_mtime, old_mtime))

    assert (
        run_batch._load_fresh_task_summary(path, started_at_epoch=time.time()) is None
    )

    now = time.time()
    os.utime(path, (now, now))
    assert run_batch._load_fresh_task_summary(path, started_at_epoch=now - 1) == {
        "task_id": "old"
    }


def test_parse_split_line_handles_skip_marker_and_comments():
    run_batch = _load_run_batch_module()
    parse = run_batch._parse_split_line

    assert parse("3d9a636_2") == ("3d9a636_2", False)
    assert parse("  3d9a636_2  ") == ("3d9a636_2", False)
    assert parse("986aa4e_2 SKIP") == ("986aa4e_2", True)
    assert parse("986aa4e_2 SKIP never-pass-1200s-timeout") == ("986aa4e_2", True)
    assert parse("") is None
    assert parse("   ") is None
    assert parse("# comment-only line") is None
    # SKIP must be a separate whitespace-bounded token; a task id that
    # happens to contain those letters does not get blacklisted.
    assert parse("SKIPPER_task_2") == ("SKIPPER_task_2", False)


def test_load_task_ids_tracked_dataset_parses_skip(tmp_path):
    run_batch = _load_run_batch_module()
    task_sets_dir = tmp_path / "task_sets"
    task_sets_dir.mkdir()
    (task_sets_dir / "fake.txt").write_text(
        "\n".join(
            [
                "alpha_2",
                "beta_2 SKIP never-pass",
                "",
                "# comment line",
                "gamma_2",
                "delta_2 SKIP",
            ]
        ),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "task_ids": None,
            "dataset": "fake",
            "variant": "full",
            "extra_task_ids": None,
            "limit": 0,
            "offset": 0,
        },
    )()
    with patch.object(
        run_batch,
        "task_set_path",
        side_effect=lambda dataset: task_sets_dir / f"{dataset}.txt",
    ):
        items = run_batch._load_task_ids(args)
    assert items == [
        ("alpha_2", False),
        ("beta_2", True),
        ("gamma_2", False),
        ("delta_2", True),
    ]


def test_load_task_ids_explicit_task_ids_never_blacklisted():
    """`--task_ids` is the explicit-override path; SKIP semantics do not
    apply because the CLI list is the user asserting intent."""
    run_batch = _load_run_batch_module()
    args = type(
        "Args",
        (),
        {
            "task_ids": "986aa4e_2,beta_2",
            "dataset": None,
            "variant": "full",
            "extra_task_ids": None,
            "limit": 0,
            "offset": 0,
        },
    )()
    items = run_batch._load_task_ids(args)
    assert items == [("986aa4e_2", False), ("beta_2", False)]


def test_load_task_ids_filters_fixed_dataset_variant_and_appends_unique_extras(
    tmp_path,
):
    run_batch = _load_run_batch_module()
    task_sets_dir = tmp_path / "task_sets"
    task_sets_dir.mkdir()
    (task_sets_dir / "dev.txt").write_text(
        "alpha_1\nalpha_2\nalpha_3\nbeta_2 SKIP reason\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "dataset": "dev",
            "variant": "2",
            "extra_task_ids": "alpha_2,target_3",
            "task_ids": None,
            "limit": 0,
            "offset": 0,
        },
    )()

    with patch.object(
        run_batch,
        "task_set_path",
        side_effect=lambda dataset: task_sets_dir / f"{dataset}.txt",
    ):
        assert run_batch._load_task_ids(args) == [
            ("alpha_2", False),
            ("beta_2", True),
            ("target_3", False),
        ]


def test_load_quick_smoke_uses_tracked_task_set(tmp_path):
    run_batch = _load_run_batch_module()
    task_sets_dir = tmp_path / "task_sets"
    task_sets_dir.mkdir()
    (task_sets_dir / "quick_smoke.txt").write_text(
        "# stable smoke\none_2\ntwo_2\nthree_2\n",
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "dataset": "quick_smoke",
            "variant": "full",
            "extra_task_ids": "target_1",
            "task_ids": None,
            "limit": 0,
            "offset": 0,
        },
    )()

    with patch.object(
        run_batch,
        "task_set_path",
        side_effect=lambda dataset: task_sets_dir / f"{dataset}.txt",
    ):
        assert run_batch._load_task_ids(args) == [
            ("one_2", False),
            ("two_2", False),
            ("three_2", False),
            ("target_1", False),
        ]


def test_tracked_quick_smoke_has_three_unique_tasks():
    run_batch = _load_run_batch_module()
    lines = (
        run_batch.task_set_path("quick_smoke").read_text(encoding="utf-8").splitlines()
    )
    task_ids = [
        line.strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert task_ids == ["13547f5_2", "024c982_2", "fd1f8fa_2"]
    assert len(task_ids) == len(set(task_ids)) == 3


def test_tracked_task_sets_match_provenance_manifest():
    run_batch = _load_run_batch_module()
    task_sets_dir = run_batch.task_set_path("train").parent
    manifest = json.loads((task_sets_dir / "SOURCE.json").read_text(encoding="utf-8"))

    for filename, expected in manifest["files"].items():
        path = task_sets_dir / filename
        payload = path.read_bytes()
        task_ids = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert len(task_ids) == expected["task_count"]
        assert len(task_ids) == len(set(task_ids))
        assert hashlib.sha256(payload).hexdigest() == expected["tracked_sha256"]
        suffix_counts = {
            suffix: sum(task_id.endswith(f"_{suffix}") for task_id in task_ids)
            for suffix in ("1", "2", "3")
        }
        assert len(set(suffix_counts.values())) == 1


def test_write_blacklisted_summary_schema(tmp_path):
    run_batch = _load_run_batch_module()
    from adk_appworld_agent.observability.subagent_logs import TaskLogContext

    task = TaskLogContext(
        task_id="986aa4e_2",
        instruction="some instruction text",
        task_datetime="2023-05-18T12:00:00",
        index=1,
        total=1,
        is_blacklisted=True,
    )
    path = tmp_path / "artifacts" / "task_summary.json"
    summary = run_batch._write_blacklisted_summary(path, task)

    assert path.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == summary

    assert summary["task_id"] == "986aa4e_2"
    assert summary["status"] == "FAILED"
    assert summary["phase"] == "BLACKLISTED"
    assert summary["wall_s"] == 0.0
    assert summary["eval"] is None
    assert summary["overview"]["block_reason"] == "BLACKLISTED"
    for key in (
        "wall_ms",
        "llm_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "retry_count",
    ):
        assert summary["aggregate_metrics"][key] == 0
    assert summary["failures"][0]["code"] == "BLACKLISTED"
