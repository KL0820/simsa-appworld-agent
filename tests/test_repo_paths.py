from pathlib import Path

from adk_appworld_agent import repo_paths


def test_shared_env_and_checkout_local_config_use_different_roots(monkeypatch):
    main_root = Path("/tmp/main-repo")
    checkout_root = Path("/tmp/worktree")

    monkeypatch.setattr(repo_paths, "main_repo_root", lambda: main_root)
    monkeypatch.setattr(repo_paths, "checkout_root", lambda: checkout_root)

    assert repo_paths.repo_env_path() == main_root / ".env"
    assert (
        repo_paths.repo_config_path()
        == checkout_root / "logs" / "_registry" / "config.json"
    )
