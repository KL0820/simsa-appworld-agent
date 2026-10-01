"""Offline tests for the closed-loop artifact assembly (no LLM, no RPC)."""

from __future__ import annotations

import json
from pathlib import Path

from mind_skill.loop import assemble_libraries

SKILL_TEMPLATE = """---
name: {name}
description: generic retrieval key
---
## Overview
Generic.
"""


def _seed_task(out_dir: Path, task_id: str, *, best_q: int | None, qs: int = 3) -> None:
    task_dir = out_dir / task_id
    for q in range(qs):
        path = task_dir / f"q{q}" / "skill.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            SKILL_TEMPLATE.format(name=f"skill-{task_id}-q{q}"), encoding="utf-8"
        )
    (task_dir / "result.json").write_text(
        json.dumps({"task_id": task_id, "best_q": best_q, "iterations": []}),
        encoding="utf-8",
    )


def test_assemble_builds_per_q_and_best_libraries(tmp_path: Path):
    out_dir, skills_root = tmp_path / "run", tmp_path / "skills"
    _seed_task(out_dir, "aaa_2", best_q=1)
    _seed_task(out_dir, "bbb_2", best_q=None)  # no best iteration -> excluded from best

    summary = assemble_libraries(out_dir, skills_root, q_iterations=3)

    # per-q libraries keep EVERY task's iteration (q0/q1/q2 comparison data)
    for q in range(3):
        assert (
            skills_root
            / "code_executor"
            / f"q{q}"
            / "aaa_2"
            / f"skill-aaa_2-q{q}"
            / "SKILL.md"
        ).exists()
        assert (
            skills_root
            / "code_executor"
            / f"q{q}"
            / "bbb_2"
            / f"skill-bbb_2-q{q}"
            / "SKILL.md"
        ).exists()
    assert summary["libraries"]["q0"] == 2

    # best library: tasks with a best_q, pointing at that skill
    best_aaa = (
        skills_root / "code_executor" / "best" / "aaa_2" / "skill-aaa_2-q1" / "SKILL.md"
    )
    assert best_aaa.exists()
    assert not (skills_root / "code_executor" / "best" / "bbb_2").exists()
    assert summary["quarantined"] == ["bbb_2"]

    # leaf dir name == frontmatter name (native ADK loadability invariant)
    text = best_aaa.read_text(encoding="utf-8")
    assert text.splitlines()[1] == f"name: {best_aaa.parent.name}"
