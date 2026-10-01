from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from adk_appworld_agent.orchestration.run_config import RunConfig


def _launcher_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "view_config.py"
    spec = importlib.util.spec_from_file_location("config_launcher_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload() -> dict:
    return {
        "config": RunConfig().model_dump(mode="json"),
        "batch": {
            "experiment_name": "launcher_smoke",
            "dataset": "test_normal",
            "variant": "2",
            "extra_task_ids": "target_3",
            "rpc_url": "tcp://127.0.0.1:4242",
            "log_root": "custom-logs",
            "offset": 1,
            "limit": 2,
        },
    }


def test_prepare_launch_uses_generated_config_as_single_runtime_source(tmp_path: Path):
    launcher = _launcher_module()

    result = launcher.prepare_launch(
        _payload(),
        project_root=tmp_path,
        write_config=True,
    )

    assert result["config_path"].exists()
    assert result["config"].log.log_root == Path("custom-logs")
    command = result["command"]
    assert command[command.index("--config") + 1] == str(result["config_path"])
    assert command[command.index("--dataset") + 1] == "test_normal"
    assert command[command.index("--variant") + 1] == "2"
    assert command[command.index("--extra_task_ids") + 1] == "target_3"
    assert command[command.index("--limit") + 1] == "2"


def test_launch_rejects_missing_task_selection():
    launcher = _launcher_module()
    payload = _payload()
    payload["batch"]["dataset"] = "extra_only"
    payload["batch"]["variant"] = "full"
    payload["batch"]["extra_task_ids"] = ""

    with pytest.raises(ValueError, match="Extra Task ID"):
        launcher.validate_launch_payload(payload)


def test_launch_rejects_unknown_subagent_implementation():
    launcher = _launcher_module()
    payload = _payload()
    payload["config"]["impls"]["FIND"] = "not_registered"

    with pytest.raises(ValueError, match="Unknown FIND implementation"):
        launcher.validate_launch_payload(payload)


def test_launcher_prevents_duplicate_running_batch(tmp_path: Path):
    launcher = _launcher_module()
    process = type("Process", (), {"pid": 321, "poll": lambda self: None})()
    state = launcher.LauncherState(project_root=tmp_path)

    with patch.object(launcher.subprocess, "Popen", return_value=process):
        first = state.launch(_payload())
        assert first["pid"] == 321
        with pytest.raises(RuntimeError, match="already running"):
            state.launch(_payload())


def test_launcher_cancels_running_batch_process_group(tmp_path: Path):
    launcher = _launcher_module()
    process = Mock(pid=321)
    process.poll.return_value = None
    state = launcher.LauncherState(project_root=tmp_path)
    state.process = process

    with patch.object(launcher, "_stop_process_group", return_value=-15) as stop:
        result = state.cancel()

    stop.assert_called_once_with(process)
    assert result == {"state": "cancelled", "pid": 321, "returncode": -15}
    process.poll.return_value = -15
    assert state.status()["state"] == "cancelled"


def test_launcher_rejects_cancel_when_no_batch_is_running(tmp_path: Path):
    launcher = _launcher_module()
    state = launcher.LauncherState(project_root=tmp_path)

    with pytest.raises(RuntimeError, match="No batch"):
        state.cancel()


def test_stop_process_group_escalates_after_grace_period():
    launcher = _launcher_module()
    process = Mock(pid=321)
    process.wait.side_effect = [
        launcher.subprocess.TimeoutExpired(cmd="batch", timeout=5.0),
        -9,
    ]

    with (
        patch.object(launcher.os, "getpgid", return_value=321),
        patch.object(launcher.os, "killpg") as killpg,
    ):
        returncode = launcher._stop_process_group(process)

    assert returncode == -9
    assert killpg.call_args_list == [
        ((321, launcher.signal.SIGTERM),),
        ((321, launcher.signal.SIGKILL),),
    ]
