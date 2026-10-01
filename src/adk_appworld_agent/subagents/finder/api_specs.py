from __future__ import annotations

import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from adk_appworld_agent.appworld_paths import find_appworld_root


def _api_docs_dir() -> Path:
    raw = os.getenv("APPWORLD_API_DOCS_DIR")
    if raw:
        return Path(raw)
    return find_appworld_root() / "data" / "api_docs" / "standard"


def _load_apis_from_disk() -> list[dict]:
    docs_dir = _api_docs_dir()
    if not docs_dir.exists():
        raise FileNotFoundError(f"AppWorld API docs directory not found: {docs_dir}")

    apis: list[dict] = []
    for json_file in sorted(docs_dir.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        for api_name, api_def in payload.items():
            if not isinstance(api_def, dict):
                continue
            apis.append(
                {
                    "app_name": api_def.get("app_name"),
                    "api_name": api_name,
                    "method": api_def.get("method", ""),
                    "description": api_def.get("description", ""),
                    "parameters": api_def.get("parameters", []),
                    "response_schemas": api_def.get("response_schemas", {}),
                }
            )
    return apis


@lru_cache(maxsize=1)
def _api_spec_index() -> dict[tuple[str, str], dict]:
    return {
        (str(api.get("app_name", "")), str(api.get("api_name", ""))): api
        for api in _load_apis_from_disk()
        if api.get("app_name") and api.get("api_name")
    }


def get_api_spec(app_name: str, api_name: str) -> dict | None:
    if not app_name or not api_name:
        return None
    spec = _api_spec_index().get((app_name, api_name))
    if spec is None:
        return None
    return deepcopy(spec)


__all__ = ["get_api_spec"]
