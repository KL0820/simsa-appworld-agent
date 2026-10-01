from __future__ import annotations

import json

from adk_appworld_agent.appworld.bootstrap import load_task


class _FakeRpcClient:
    def __init__(self) -> None:
        self.init_calls: list[tuple[str, str]] = []

    def init_world(self, task_id: str, experiment_name: str) -> None:
        self.init_calls.append((task_id, experiment_name))

    def get_task_instruction(self) -> str:
        return "Do the task."


def test_load_task_reads_task_datetime_from_specs(monkeypatch, tmp_path):
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "demo_1"
    task_dir.mkdir(parents=True)
    (task_dir / "specs.json").write_text(
        json.dumps(
            {
                "instruction": "ignored by bootstrap client",
                "datetime": "2023-05-18T12:00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("APPWORLD_TASKS_DIR", str(tasks_dir))
    client = _FakeRpcClient()

    result = load_task("demo_1", "exp", client=client)

    assert client.init_calls == [("demo_1", "exp")]
    assert result.task_instruction == "Do the task."
    assert result.task_datetime == "2023-05-18T12:00:00"


def test_load_task_defaults_task_datetime_to_empty_when_specs_missing(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("APPWORLD_TASKS_DIR", str(tmp_path / "tasks"))
    client = _FakeRpcClient()

    result = load_task("missing_1", "exp", client=client)

    assert result.task_datetime == ""


def test_load_task_uses_appworld_root_helper_when_no_env(monkeypatch, tmp_path):
    # When APPWORLD_TASKS_DIR is unset, bootstrap must walk up to a sibling
    # appworld/data instead of falling silently to "" — prior bug surfaced as
    # task_datetime_dt.date() raising AttributeError inside the sandbox.
    monkeypatch.delenv("APPWORLD_TASKS_DIR", raising=False)
    appworld_root = tmp_path / "appworld"
    (appworld_root / "data" / "tasks" / "demo_1").mkdir(parents=True)
    (appworld_root / "data" / "tasks" / "demo_1" / "specs.json").write_text(
        json.dumps({"instruction": "x", "datetime": "2024-01-01T00:00:00"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("APPWORLD_ROOT", str(appworld_root))
    client = _FakeRpcClient()

    result = load_task("demo_1", "exp", client=client)

    assert result.task_datetime == "2024-01-01T00:00:00"
