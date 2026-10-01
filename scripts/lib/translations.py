"""Permanent Chinese-translation cache for AppWorld task instructions.

Task instructions are immutable per task_id; once translated, the result is
shared across every batch. The single source of truth is:

    logs/_registry/task_translations_zh.json

**Translation is performed by Claude (interactively), not by an LLM API.**
This module only manages the cache + reports which task_ids are missing
a translation. New task_ids that don't yet have an entry render as the
original English in the viewer until Claude translates them and writes
the entry into the JSON.

Public entry points:

    load_translations(registry_dir)
        → dict[task_id, instruction_zh]

    save_translations(registry_dir, data)
        Writes the dict back to disk, sorted, with _meta header restored.

    get_translation(task_id, registry_dir)
        Cache-only lookup. Returns None if missing.

    find_missing(task_id_to_instruction, registry_dir)
        → list[(task_id, instruction_en)] for tids that don't yet have
        a cached translation. Use this to print a worklist for Claude.

Translation rules (when Claude does the translation):
- Preserve noun qualifiers verbatim ("in my phone" → "電話聯絡簿裡的",
  "this year" → "今年", "from my siblings" → "我兄弟姊妹的"). These are
  the modifiers the planner most often drops; the Chinese version must
  carry them so the analyst spots qualifier-loss bugs at a glance.
- Keep app names untranslated: Venmo, Spotify, SimpleNote, Splitwise,
  Todoist, Gmail, phone, file_system, Amazon.
- Concise but complete. No markdown / explanatory wrapping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REGISTRY_FILENAME = "task_translations_zh.json"

_META_KEY = "_meta"
_META_BODY: dict[str, str] = {
    "description": (
        "Permanent Chinese translation cache for AppWorld task instructions, "
        "shared across all batches. Translation is performed by Claude "
        "interactively, not by an LLM API."
    ),
    "rule": (
        "Task instructions are immutable per task_id; once translated, the entry "
        "is reused forever. Add new entries only when scripts/lib/translations.py:"
        "find_missing() reports them as untranslated."
    ),
    "translation_rules": (
        "1) Preserve noun qualifiers verbatim ('in my phone' → '電話聯絡簿裡的', "
        "'this year' → '今年', 'from my siblings' → '我兄弟姊妹的') — these are the "
        "modifiers the planner most often drops. "
        "2) Keep app names untranslated (Venmo / Spotify / SimpleNote / Splitwise / "
        "Todoist / Gmail / phone / file_system / Amazon). "
        "3) Concise + complete. No markdown / explanatory wrapping."
    ),
}


def _registry_path(registry_dir: Path) -> Path:
    return Path(registry_dir) / REGISTRY_FILENAME


def load_translations(registry_dir: Path) -> dict[str, str]:
    """Load the cached translations dict, skipping `_meta` + non-string entries."""
    path = _registry_path(registry_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.stderr.write(f"[translations] cache parse error: {e}\n")
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        k: v for k, v in data.items() if k != _META_KEY and isinstance(v, str) and v
    }


def save_translations(registry_dir: Path, data: dict[str, str]) -> None:
    """Write the cache back, sorted, with _meta header restored."""
    path = _registry_path(registry_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {_META_KEY: dict(_META_BODY)}
    for tid in sorted(data):
        out[tid] = data[tid]
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")


def get_translation(task_id: str, registry_dir: Path) -> str | None:
    """Cache-only lookup. Returns None if the task_id isn't translated yet."""
    return load_translations(registry_dir).get(task_id)


def find_missing(
    task_id_to_instruction: dict[str, str],
    registry_dir: Path,
) -> list[tuple[str, str]]:
    """Return the (task_id, instruction_en) pairs not yet in the cache.

    Use this to print a worklist for Claude — translate each, then call
    save_translations() with the updated dict.
    """
    cache = load_translations(registry_dir)
    return [(tid, en) for tid, en in task_id_to_instruction.items() if tid not in cache]


__all__ = [
    "REGISTRY_FILENAME",
    "find_missing",
    "get_translation",
    "load_translations",
    "save_translations",
]
