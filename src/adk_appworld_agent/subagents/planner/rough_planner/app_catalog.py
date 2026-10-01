"""Load RoughPlanner app descriptions from AppWorld's app-description source."""

from __future__ import annotations

import os
import sqlite3
import tomllib
from dataclasses import dataclass
from pathlib import Path

from adk_appworld_agent.appworld_paths import find_appworld_root

APPWORLD_ROOT_ENV = "APPWORLD_ROOT"
APPWORLD_SOURCE_DIR_ENV = "APPWORLD_SOURCE_DIR"
APP_DESCRIPTIONS_SOURCE = (
    "AppWorld api_docs.show_app_descriptions (/api_docs/app_descriptions)"
)


@dataclass(frozen=True)
class AppCatalogEntry:
    name: str
    description: str


def load_app_catalog(app_names: tuple[str, ...]) -> tuple[AppCatalogEntry, ...]:
    appworld_root = _find_appworld_root()
    descriptions = _load_app_descriptions(appworld_root)
    missing = [app_name for app_name in app_names if app_name not in descriptions]
    if missing:
        raise ValueError(
            "AppWorld app-description output is missing: " + ", ".join(missing)
        )

    return tuple(
        AppCatalogEntry(name=app_name, description=descriptions[app_name])
        for app_name in app_names
    )


def app_descriptions_source_label() -> str:
    return APP_DESCRIPTIONS_SOURCE


def _find_appworld_root() -> Path:
    for candidate in _candidate_roots():
        if _is_appworld_root(candidate):
            return candidate
    searched = ", ".join(str(path) for path in _candidate_roots())
    raise FileNotFoundError(f"AppWorld source/data not found. Searched: {searched}")


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    root_env = os.getenv(APPWORLD_ROOT_ENV)
    source_env = os.getenv(APPWORLD_SOURCE_DIR_ENV)
    if root_env:
        candidates.append(Path(root_env).expanduser())
    if source_env:
        candidates.append(Path(source_env).expanduser().parent)
    try:
        candidates.append(find_appworld_root())
    except FileNotFoundError:
        pass
    return candidates


def _is_appworld_root(path: Path) -> bool:
    return _apps_dir(path).is_dir() and _api_docs_db_path(path).is_file()


def _load_app_descriptions(appworld_root: Path) -> dict[str, str]:
    # Mirrors api_docs.show_app_descriptions: documented app names from ApiDoc,
    # descriptions from appworld.apps.get_app_to_description()/info.toml.
    return {
        app_name: _read_app_description(appworld_root, app_name)
        for app_name in _documented_app_names(appworld_root)
        if _app_info_path(appworld_root, app_name).is_file()
    }


def _documented_app_names(appworld_root: Path) -> list[str]:
    query = (
        "select app_name_, min(id) from api_docs group by app_name_ order by min(id)"
    )
    with sqlite3.connect(_api_docs_db_path(appworld_root)) as connection:
        rows = connection.execute(query).fetchall()
    return [str(app_name).strip() for app_name, _ in rows if str(app_name).strip()]


def _read_app_description(appworld_root: Path, app_name: str) -> str:
    info = tomllib.loads(
        _app_info_path(appworld_root, app_name).read_text(encoding="utf-8")
    )
    if info.get("name") != app_name:
        raise ValueError(f"AppWorld info.toml name mismatch for {app_name!r}.")
    description = info.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError(f"AppWorld info.toml for {app_name!r} has no description.")
    return description


def _apps_dir(appworld_root: Path) -> Path:
    return appworld_root / "src" / "appworld" / "apps"


def _app_info_path(appworld_root: Path, app_name: str) -> Path:
    return _apps_dir(appworld_root) / app_name / "info.toml"


def _api_docs_db_path(appworld_root: Path) -> Path:
    return appworld_root / "data" / "base_dbs" / "api_docs.db"


__all__ = ["AppCatalogEntry", "app_descriptions_source_label", "load_app_catalog"]
