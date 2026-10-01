"""Resolve checkout-local vs shared repo paths.

`git rev-parse --git-common-dir` returns the main repo's `.git` directory
even when called from inside a `git worktree`. That is useful for files that
should be shared across worktrees, but wrong for tracked files that belong to
the current checkout.

The two artifacts below intentionally use different roots:

- `repo_env_path()` — the gitignored `.env` holding secrets
  (`GOOGLE_API_KEY`, etc.). Only the main repo has it; worktrees read
  the same file via this helper instead of needing a per-worktree copy.
- `repo_config_path()` — the tracked-in-git `logs/_registry/config.json`
  holding skill workflow state (baseline, timeout blacklist, retry limits).
  It follows the current checkout so each worktree sees the version it checked
  out instead of mutating the main repo behind the user's back.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_RELPATH = Path("logs/_registry/config.json")
ENV_RELPATH = Path(".env")


@lru_cache(maxsize=1)
def main_repo_root() -> Path:
    """Main repo root, even when called from a worktree.

    Uses `git rev-parse --git-common-dir` which returns the main
    repo's `.git` (worktrees share a single common-dir). Falls back to
    walking up from this file's location if git is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        git_dir = Path(out)
        if not git_dir.is_absolute():
            git_dir = (Path(__file__).resolve().parent / git_dir).resolve()
        return git_dir.parent
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def checkout_root() -> Path:
    """Root of the checkout that owns this source file."""
    return Path(__file__).resolve().parents[2]


def repo_env_path() -> Path:
    return main_repo_root() / ENV_RELPATH


def load_repository_env() -> None:
    """Load runtime settings before importing modules that resolve data paths."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path=repo_env_path())


def repo_config_path() -> Path:
    return checkout_root() / CONFIG_RELPATH


def load_skill_config(path: Path | None = None) -> dict[str, Any]:
    """Load skill config JSON. Returns defaults if file is missing."""
    path = path or repo_config_path()
    if not path.exists():
        return {
            "baseline": [],
            "timeout_blacklist": [],
            "retry_limits": {"max_llm_calls": 30, "max_tokens": 100000},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_skill_config(cfg: dict[str, Any], path: Path | None = None) -> None:
    """Write skill config JSON atomically (preserves key order from cfg)."""
    path = path or repo_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
