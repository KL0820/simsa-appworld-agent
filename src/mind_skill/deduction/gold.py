"""Shared gold-trajectory access for all deduction harnesses.

The corpus and trajectory-pick rules live here; each per-component deduction
module (code_executor / rough_planner / code_planner) builds its own
teacher-forcing context on top of these.
"""

from __future__ import annotations

import json
from pathlib import Path


def pick_gold_trajectory(runs_dir: Path, task_id: str) -> Path:
    """Choose the PASS trajectory: blind, else open-book, else blind_fwd
    (e3d6c94_2's forward-ON run — disclosed in spec §2)."""
    task_dir = Path(runs_dir) / task_id
    for rel in (
        "trajectory.json",
        "openbook/trajectory.json",
        "blind_fwd/trajectory.json",
    ):
        path = task_dir / rel
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data.get("result") or {}).get("cleared"):
            return path
    raise FileNotFoundError(f"no cleared trajectory for {task_id} under {task_dir}")


def gold_corpus_task_ids(runs_dir: Path, suffix: str = "_1") -> list[str]:
    """The 30-trajectory induction corpus: every `<suffix>` task with a cleared
    gold trajectory. Default `_1` = the current canonical corpus (the bottom-up
    `_1` training); pass suffix='_2' for the original `_2` set. The gold dir now
    holds BOTH corpora, so the suffix is what disambiguates which one `--tasks
    all` means — picking the wrong one silently trains on the wrong tasks."""
    ids: list[str] = []
    for task_dir in sorted(Path(runs_dir).iterdir()):
        if not task_dir.is_dir() or not task_dir.name.endswith(suffix):
            continue
        try:
            pick_gold_trajectory(runs_dir, task_dir.name)
        except (FileNotFoundError, ValueError):
            continue
        ids.append(task_dir.name)
    return ids


def step_user_text(step: dict) -> str:
    """Verbatim user text of a trajectory step (input.context first, shim turns
    as fallback)."""
    inp = step.get("input") or {}
    context = inp.get("context") or []
    if context and context[0].get("text"):
        return context[0]["text"]
    turns = step.get("turns") or []
    if turns:
        request = turns[0].get("request") or {}
        ctx = request.get("context") or [{}]
        return (ctx[0] or {}).get("text", "") or ""
    return ""


def strip_role_prefix(text: str) -> str:
    return text.split("<<user>>\n", 1)[1] if text.startswith("<<user>>") else text


def payload_from_step_text(text: str) -> dict:
    """code_plan user prompt = '<<user>>\\n{json payload}\\n\\n## ...' -> payload."""
    start = text.index("{")
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(payload, dict):
        raise ValueError("step payload is not a JSON object")
    return payload


__all__ = [
    "gold_corpus_task_ids",
    "payload_from_step_text",
    "pick_gold_trajectory",
    "step_user_text",
    "strip_role_prefix",
]
