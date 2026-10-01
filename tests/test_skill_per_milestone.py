"""Per-milestone skill resolution (skills_per_milestone): the executor
re-resolves per active milestone, querying the milestone intent; per-task mode
(default) resolves once per task; train pinned tasks are unaffected.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.utils import skills_source as ss_mod
from adk_appworld_agent.subagents.utils.skills_source import SkillsSource


def _held_out_lib(root: Path) -> None:
    libdir = root / "code_executor" / "best"
    for src, name in [
        ("t1", "alpha"),
        ("t2", "beta"),
        ("t3", "gamma"),
        ("t4", "delta"),
    ]:
        d = libdir / src / name
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            f"name: {name}\ndescription: d\n\n# {name}\n", encoding="utf-8"
        )


def _fake_retrieve(calls: list):
    async def fn(
        *, instruction, library_dir, model_cfg, k, selector="llm", embedding_model=""
    ):
        calls.append(instruction)
        return [{"text": f"# {instruction[-12:]}", "name": instruction[-12:]}]

    return fn


def test_per_milestone_queries_intent_and_caches_per_milestone(tmp_path, monkeypatch):
    _held_out_lib(tmp_path)
    calls: list = []
    monkeypatch.setattr(ss_mod, "retrieve_skills_async", _fake_retrieve(calls))
    src = SkillsSource(
        component="code_executor",
        root=tmp_path,
        library="best",
        model_cfg=ModelConfig(),
        k=3,
        per_milestone=True,
    )
    asyncio.run(
        src.resolve(
            "held1", "do the task", milestone_intent="read the notes", milestone_index=0
        )
    )
    asyncio.run(
        src.resolve(
            "held1", "do the task", milestone_intent="send a text", milestone_index=1
        )
    )
    # distinct milestones -> distinct queries -> two retrievals
    assert len(calls) == 2
    assert "read the notes" in calls[0] and "send a text" in calls[1]
    assert "do the task" in calls[0]  # task instruction kept for context
    # same milestone again -> cache hit, no new retrieval
    asyncio.run(
        src.resolve(
            "held1", "do the task", milestone_intent="read the notes", milestone_index=0
        )
    )
    assert len(calls) == 2


def test_per_task_default_ignores_milestone(tmp_path, monkeypatch):
    _held_out_lib(tmp_path)
    calls: list = []
    monkeypatch.setattr(ss_mod, "retrieve_skills_async", _fake_retrieve(calls))
    src = SkillsSource(
        component="code_executor",
        root=tmp_path,
        library="best",
        model_cfg=ModelConfig(),
        k=3,
        per_milestone=False,
    )
    asyncio.run(
        src.resolve(
            "held1", "do the task", milestone_intent="read the notes", milestone_index=0
        )
    )
    asyncio.run(
        src.resolve(
            "held1", "do the task", milestone_intent="send a text", milestone_index=1
        )
    )
    # per-task: one retrieval, cached by task_id, milestone ignored, query = instruction
    assert len(calls) == 1
    assert calls[0] == "do the task"


def test_per_milestone_train_task_stays_pinned(tmp_path, monkeypatch):
    pin = tmp_path / "code_executor" / "best" / "traintask" / "the-pinned-skill"
    pin.mkdir(parents=True)
    (pin / "SKILL.md").write_text(
        "name: the-pinned-skill\ndescription: d\n\n# x\n", encoding="utf-8"
    )
    calls: list = []
    monkeypatch.setattr(ss_mod, "retrieve_skills_async", _fake_retrieve(calls))
    src = SkillsSource(
        component="code_executor",
        root=tmp_path,
        library="best",
        model_cfg=ModelConfig(),
        k=3,
        per_milestone=True,
    )
    r = asyncio.run(
        src.resolve(
            "traintask", "do it", milestone_intent="m0 intent", milestone_index=0
        )
    )
    assert r.source == "per_task"
    assert r.names == ["the-pinned-skill"]
    assert len(calls) == 0  # pinned train task never hits retrieval
