"""continuation prompts — production text loaded verbatim from prompts/full.md."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


CONTINUATION_SYSTEM_PROMPT = _load_prompt("full")

__all__ = ["CONTINUATION_SYSTEM_PROMPT"]
