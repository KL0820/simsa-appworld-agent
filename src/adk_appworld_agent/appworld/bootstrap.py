from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

from adk_appworld_agent.appworld.client import AppWorldRpcClient
from adk_appworld_agent.appworld_paths import find_appworld_root


class AppWorldBootstrapResult(BaseModel):
    task_id: str
    experiment_name: str
    task_instruction: str
    task_datetime: str
    rpc_url: str


def _task_specs_path(task_id: str) -> Path:
    raw = os.getenv("APPWORLD_TASKS_DIR")
    base_dir = Path(raw) if raw else find_appworld_root() / "data" / "tasks"
    return base_dir / task_id / "specs.json"


def _load_task_datetime(task_id: str) -> str:
    specs_path = _task_specs_path(task_id)
    try:
        payload = json.loads(specs_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        # Silent empty-string fallback used to mask path-resolution bugs and
        # surface only as downstream NoneType errors at execute_python time.
        sys.stderr.write(
            f"[bootstrap] WARNING: could not read task specs for "
            f"{task_id!r} at {specs_path} ({type(exc).__name__}: {exc}); "
            f"task_datetime will be empty.\n"
        )
        return ""
    task_datetime = payload.get("datetime")
    return task_datetime if isinstance(task_datetime, str) else ""


def load_task(
    task_id: str,
    experiment_name: str,
    *,
    rpc_url: str = "tcp://127.0.0.1:4242",
    client: AppWorldRpcClient | None = None,
) -> AppWorldBootstrapResult:
    """Initialize AppWorld for one task and return the instruction.

    Minimum v1 bootstrap: init_world + get_task_instruction. Memory loading
    and pre-login are deferred to the AuthManager seam.
    """
    rpc = client or AppWorldRpcClient(addr=rpc_url)
    rpc.init_world(task_id, experiment_name)
    instruction = rpc.get_task_instruction()
    return AppWorldBootstrapResult(
        task_id=task_id,
        experiment_name=experiment_name,
        task_instruction=instruction,
        task_datetime=_load_task_datetime(task_id),
        rpc_url=rpc_url,
    )
