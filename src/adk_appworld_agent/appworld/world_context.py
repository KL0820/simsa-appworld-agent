from __future__ import annotations

import re

from adk_appworld_agent.appworld.client import AppWorldRpcClient

_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_world_name_part(value: str) -> str:
    cleaned = _SAFE_PART_RE.sub("_", value.strip())
    return cleaned.strip("_") or "run"


def build_world_run_name(*, experiment_name: str, run_name: str, task_id: str) -> str:
    parts = (
        _sanitize_world_name_part(experiment_name),
        _sanitize_world_name_part(run_name),
        _sanitize_world_name_part(task_id),
    )
    return "__".join(parts)


def close_world_safely(client: AppWorldRpcClient | None) -> None:
    if client is None:
        return
    try:
        client.close_world()
    except Exception:
        return


__all__ = [
    "build_world_run_name",
    "close_world_safely",
]
