from __future__ import annotations

from pathlib import Path

import pytest

from scripts.demo import _required_runtime


def test_demo_requires_appworld_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPWORLD_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="APPWORLD_ROOT"):
        _required_runtime()


def test_demo_accepts_installed_root_and_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "appworld"
    (root / "src" / "appworld").mkdir(parents=True)
    (root / "data").mkdir()
    python = tmp_path / "appworld-python"
    python.touch()
    monkeypatch.setenv("APPWORLD_ROOT", str(root))
    monkeypatch.setenv("APPWORLD_PYTHON", str(python))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-not-used")

    assert _required_runtime() == (root, python)
