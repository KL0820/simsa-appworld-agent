"""AppWorld zerorpc server.

Migrated from `adk-api-agent/eval/benchmark/appworld/server.py` (2026-05-19)
so the server lives alongside its data + code under `agents/appworld/`.
Clients connect via tcp://127.0.0.1:4244 by default.

Launch:
    cd agents/appworld
    .venv/bin/python /path/to/simsa/scripts/appworld_rpc_server.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import zerorpc
from appworld import AppWorld
from appworld.common.path_store import path_store
from dotenv import load_dotenv

load_dotenv()
APPWORLD_ROOT = Path(path_store.root)


def _output_task_dir(experiment_name: str, task_id: str) -> Path:
    return (
        APPWORLD_ROOT / "experiments" / "outputs" / experiment_name / "tasks" / task_id
    )


class WorldNotInitializedError(RuntimeError):
    """Raised when a world-dependent RPC arrives while self.world is None.

    Surfaced as a clear, actionable error instead of a silent
    `AttributeError: 'NoneType' object has no attribute 'execute'`. The
    client should respond by re-calling init_world for the current task.
    """


class AppWorldService:
    def __init__(self):
        self.world = None
        self.current_task_id: str | None = None

    def _require_world(self):
        if self.world is None:
            raise WorldNotInitializedError(
                "AppWorld world is not initialized — call init_world(task_id, "
                "experiment_name) first. This usually means: (a) the server "
                "just started with no init, or (b) a previous task's "
                "close_world() arrived out of order on the zerorpc queue. "
                "Client: re-issue init_world for the current task and retry."
            )
        return self.world

    def init_world(self, task_id: str, experiment_name: str):
        log_dir = _output_task_dir(experiment_name, task_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # Self-heal: close any leftover world before creating new one.
        # Without this, a stale world from a prior task (whose close_world
        # was lost / arrived late on the zerorpc queue) leaks into the new
        # task and the next operation either touches the wrong sandbox or
        # gets nulled mid-flight by the late close.
        if self.world is not None:
            try:
                self.world.close()
            except Exception:
                pass
            self.world = None
        self.world = AppWorld(
            task_id=task_id, experiment_name=experiment_name, max_interactions=300
        )
        self.current_task_id = task_id

    def close_world(self, expected_task_id: str | None = None):
        # Idempotent + task-scoped. If the caller knows which task it
        # belongs to and that doesn't match current_task_id, drop the call —
        # this is the cross-task race fix. Old clients that omit the
        # argument still get the original unconditional close (back-compat).
        if expected_task_id is not None and self.current_task_id != expected_task_id:
            return
        if self.world is not None:
            try:
                self.world.close()
            except Exception:
                pass
            self.world = None
            self.current_task_id = None

    def get_task_instruction(self) -> str:
        return self._require_world().task.instruction

    def submit_answer(self, answer: str):
        world = self._require_world()
        world.execute(f"apis.supervisor.complete_task(answer='{answer}')")
        return world.evaluate().report(print_it=False)

    def execute_function(self, service_name: str, function_name: str, data: dict):
        world = self._require_world()
        code = f"print(apis.{service_name}.{function_name}("
        for key, value in data.items():
            if isinstance(value, str):
                value = f'"""{value}"""'
            code += f"{key}={value}, "
        if not data:
            code += "))"
        else:
            code = code[:-2] + "))"
        execute_result = world.execute(code)
        execute_result = execute_result.strip()
        try:
            result = json.loads(execute_result)
            return result
        except json.JSONDecodeError:
            exception_index = execute_result.find("Exception:")
            if exception_index != -1:
                result = execute_result[exception_index:]
            else:
                result = execute_result
            return result

    def execute_python(self, python_code: str):
        return self._require_world().execute(python_code)


if __name__ == "__main__":
    try:
        port = os.getenv("RPC_PORT", "4244")
        server = zerorpc.Server(AppWorldService())
        server.bind(f"tcp://127.0.0.1:{port}")
        print(f"AppWorld RPC Server running on port {port}...")
        server.run()
    except Exception as e:
        print(f"Error starting server: {str(e)}")
