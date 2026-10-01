"""Resolve the AppWorld checkout from any depth, including worktrees."""

from __future__ import annotations

import pytest

from adk_appworld_agent import appworld_paths
from adk_appworld_agent.appworld_paths import find_appworld_root


def test_find_appworld_root_walks_up_to_sibling(tmp_path, monkeypatch):
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)

    # Simulate any nesting depth (mirrors a worktree at .claude/worktrees/<name>/src/...).
    appworld = tmp_path / "appworld"
    (appworld / "data").mkdir(parents=True)
    deep_caller = (
        tmp_path
        / "agent"
        / ".claude"
        / "worktrees"
        / "wt"
        / "src"
        / "pkg"
        / "module.py"
    )
    deep_caller.parent.mkdir(parents=True)
    deep_caller.write_text("")

    assert find_appworld_root(start=deep_caller) == appworld


def test_find_appworld_root_honours_env_override(tmp_path, monkeypatch):
    forced = tmp_path / "elsewhere" / "appworld"
    forced.mkdir(parents=True)
    monkeypatch.setenv("APPWORLD_ROOT", str(forced))

    # Even with no sibling on disk, env wins.
    assert find_appworld_root(start=tmp_path / "nope.py") == forced


def test_find_appworld_root_falls_back_to_working_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)
    package_file = tmp_path / "isolated" / "src" / "module.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("")
    checkout = tmp_path / "workspace" / "agent"
    checkout.mkdir(parents=True)
    appworld = tmp_path / "workspace" / "appworld"
    (appworld / "data").mkdir(parents=True)

    monkeypatch.setattr(appworld_paths, "__file__", str(package_file))
    monkeypatch.chdir(checkout)

    assert appworld_paths.find_appworld_root() == appworld


def test_find_appworld_root_raises_when_unresolvable(tmp_path, monkeypatch):
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)
    isolated = tmp_path / "lonely" / "deep" / "module.py"
    isolated.parent.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        find_appworld_root(start=isolated)
