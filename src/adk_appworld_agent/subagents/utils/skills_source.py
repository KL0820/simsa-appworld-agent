"""Per-task skill resolution for the skill-enabled subagent impls.

A SkillsSource is owned by a skill impl (rough_skill / code_plan_execute_skill)
and resolves, INSIDE the subagent at run time, which SKILL.md texts this task
gets:

  1. train tasks: the library's per-task directory
     (<root>/<component>/<library>/<task_id>/<name>/SKILL.md)
  2. held-out tasks: LLM retrieval top-K over the library's (name, description)
     catalog (spec §12 / paper B.3), once per task (composite-per-task), cached.

The resolution outcome is returned alongside the texts so the subagent can put
it in its standard io payload — no side-channel artifacts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.utils.skills_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_K,
    load_library,
    retrieve_skills_async,
)


class MutableInstruction:
    """Instruction provider whose text a skill impl swaps per task.

    Callable => ADK skips {identifier} state templating; `.text` feeds the io
    logs (instruction_text()).
    """

    def __init__(self, text: str) -> None:
        self.current = text

    def __call__(self, _ctx=None) -> str:
        return self.current

    @property
    def text(self) -> str:
        return self.current


@dataclass
class ResolvedSkills:
    texts: list[str]
    names: list[str]
    source: str  # "per_task" | "retrieval" | "none"


@dataclass
class SkillsSource:
    component: str  # code_executor | code_planner | rough_planner
    root: Path
    library: str  # best | q0 | q1 | q2
    model_cfg: ModelConfig
    k: int = DEFAULT_K
    selector: str = "llm"  # "llm" | "embedding" (held-out ranker)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    per_milestone: bool = False  # re-resolve per active milestone (executor)
    _cache: dict[str, ResolvedSkills] = field(default_factory=dict)

    @property
    def library_dir(self) -> Path:
        return Path(self.root) / self.component / self.library

    async def resolve(
        self,
        task_id: str,
        instruction: str,
        *,
        milestone_intent: str | None = None,
        milestone_index: int | None = None,
    ) -> ResolvedSkills:
        # Per-milestone: focus the held-out query on the active milestone and
        # cache per (task, milestone, intent) so a continuation-revised intent
        # re-selects. The per-task DIR lookup below still keys on task_id, so
        # train tasks stay pinned regardless of milestone.
        if self.per_milestone and milestone_intent:
            query = f"{instruction}\n\nActive milestone: {milestone_intent}"
            cache_key = f"{task_id}#m{milestone_index}#{milestone_intent}"
        else:
            query = instruction
            cache_key = task_id
        if cache_key in self._cache:
            return self._cache[cache_key]
        resolved = await self._resolve_uncached(task_id, query)
        self._cache[cache_key] = resolved
        return resolved

    def _warn_none(self, task_id: str, why: str) -> None:
        # A no-skill resolution is almost always a misconfiguration (wrong
        # skills_root / library / bottom-up order), NOT an intended baseline —
        # surface it instead of silently degrading to fat/no-skill.
        sys.stderr.write(
            f"[skills] {self.component}/{self.library} task={task_id}: source=none "
            f"({why}); running WITHOUT skill — check skills_root/library/order "
            f"[{self.library_dir}]\n"
        )

    async def _resolve_uncached(self, task_id: str, query: str) -> ResolvedSkills:
        per_task = self.library_dir / task_id
        if per_task.is_dir():
            hits = sorted(per_task.glob("*/SKILL.md"))
            if hits:
                texts = [h.read_text(encoding="utf-8") for h in hits[:1]]
                return ResolvedSkills(
                    texts=texts, names=[hits[0].parent.name], source="per_task"
                )
            # Per-task dir EXISTS but holds no SKILL.md = broken/partial build. Do
            # NOT silently substitute library-wide retrieval for the missing pinned
            # skill — surface it and return empty.
            self._warn_none(task_id, f"per-task dir present but empty: {per_task}")
            return ResolvedSkills(texts=[], names=[], source="none")
        if not query or not self.library_dir.is_dir():
            self._warn_none(
                task_id,
                "empty query"
                if not query
                else f"library_dir missing: {self.library_dir}",
            )
            return ResolvedSkills(texts=[], names=[], source="none")
        chosen = await retrieve_skills_async(
            instruction=query,
            library_dir=self.library_dir,
            model_cfg=self.model_cfg,
            k=self.k,
            selector=self.selector,
            embedding_model=self.embedding_model,
        )
        if not chosen:
            self._warn_none(task_id, "retrieval returned nothing")
        return ResolvedSkills(
            texts=[c["text"] for c in chosen],
            names=[c["name"] for c in chosen],
            source="retrieval" if chosen else "none",
        )

    def catalog_size(self) -> int:
        return len(load_library(self.library_dir))


__all__ = ["MutableInstruction", "ResolvedSkills", "SkillsSource"]
