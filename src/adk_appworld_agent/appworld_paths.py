"""Locate the AppWorld checkout regardless of repo / worktree layout.

`__file__`-relative defaults broke when this codebase ran from a git worktree
under `.claude/worktrees/<name>/`, because the parent-walk landed on
`.claude/worktrees/appworld/...` instead of `agents/appworld/...`. Walking up
until we find a sibling `appworld/data/` works for both the main checkout
and any worktree depth.
"""

from __future__ import annotations

import os
from pathlib import Path

_APPWORLD_ROOT_ENV = "APPWORLD_ROOT"


def find_appworld_root(start: Path | None = None) -> Path:
    """Return the AppWorld checkout root (the directory containing `data/`).

    Resolution order: ``$APPWORLD_ROOT`` if set, otherwise walk up from
    ``start`` (default: this file) and return the first ancestor that has a
    sibling ``appworld/data/`` directory.
    """
    raw = os.getenv(_APPWORLD_ROOT_ENV)
    if raw:
        return Path(raw)

    search_starts = [start] if start is not None else [Path(__file__), Path.cwd()]
    searched_from: list[Path] = []
    for search_start in search_starts:
        here = search_start.resolve()
        searched_from.append(here)
        for ancestor in here.parents:
            candidate = ancestor / "appworld" / "data"
            if candidate.is_dir():
                return ancestor / "appworld"

    raise FileNotFoundError(
        f"Could not locate AppWorld checkout from {searched_from}; "
        f"set ${_APPWORLD_ROOT_ENV} or run from a path with appworld/data/ as a sibling."
    )


_REPO_ROOT_ENV = "REPO_ROOT"


def find_repo_root(start: Path | None = None) -> Path:
    """Return this repo's root (the directory containing ``pyproject.toml``).

    Resolution order: ``$REPO_ROOT`` if set, otherwise walk up from ``start``
    (default: this file) and return the first ancestor that contains
    ``pyproject.toml``. Used to locate the top-level ``data/`` artefact tree
    (api_graph / mind_skill skills + gold) independent of module nesting, so
    data lives outside ``src/``. Works from a git worktree too (each worktree
    has its own ``pyproject.toml``).
    """
    raw = os.getenv(_REPO_ROOT_ENV)
    if raw:
        return Path(raw)

    here = (start or Path(__file__)).resolve()
    for ancestor in here.parents:
        if (ancestor / "pyproject.toml").is_file():
            return ancestor

    raise FileNotFoundError(
        f"Could not locate repo root (pyproject.toml) from {here}; "
        f"set ${_REPO_ROOT_ENV}."
    )
