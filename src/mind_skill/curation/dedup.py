"""Post-assembly name-collision dedup for assembled skill libraries.

`assemble_libraries` writes
    <skills_root>/<component>/<library>/<task_id>/<skill-name>/SKILL.md
Two skills induced from DIFFERENT tasks can converge on the same frontmatter
`name` (the induction model picks a generic name). On disk they live under
different task_id dirs so they do not clobber, but at LOAD time both
`load_library` (skills_retrieval) and ADK flat discovery key by frontmatter
name -> one silently SHADOWS the other, and the retrieval catalog shows two
same-named lines with different descriptions.

This module runs RIGHT AFTER assembly: it detects each within-library name
collision and resolves it with a gemini-2.5-flash judge that reads the FULL
bodies and decides per group:
  - "merge"    : same skill, only cosmetic/wording/emphasis differences ->
                 combine into ONE focused skill (union of useful knowledge).
  - "distinct" : genuinely different/orthogonal knowledge -> rename each to a
                 unique, non-overlapping name + sharpened description; bodies
                 are kept verbatim.
After dedup, frontmatter names are unique within the library, so it is both
ADK-flat-deployable and unambiguous for top-K retrieval.

Provenance (AGENTS.md): operates ONLY on already-induced skill content as
general library hygiene. It hand-writes nothing task-specific and reads no
fail logs; the merge/rename judgment is the model's, on the skill bodies.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Literal

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field

from mind_skill.runtime import (
    run_meta_agent_validated,
)


# ── data ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SkillFile:
    """One assembled skill: its on-disk location plus parsed frontmatter/body."""

    task_id: str
    name: str
    description: str
    body: str
    path: Path  # the SKILL.md path

    @property
    def skill_dir(self) -> Path:
        return self.path.parent


def parse_skill_md(text: str) -> tuple[str, str, str]:
    """(name, description, body) from a SKILL.md string. Mirrors ADK parsing."""
    if not text.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md frontmatter not properly closed with ---")
    front, body = parts[1], parts[2].lstrip("\n")
    name = description = ""
    for line in front.splitlines():
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("description:"):
            description = line.split(":", 1)[1].strip()
    if not name:
        raise ValueError("SKILL.md frontmatter missing name")
    return name, description, body


def render_skill_md(name: str, description: str, body: str) -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n{body.lstrip(chr(10))}"


def load_skill_files(library_dir: Path) -> list[SkillFile]:
    """Every SKILL.md under <library_dir>/<task_id>/<skill-name>/SKILL.md."""
    out: list[SkillFile] = []
    for p in sorted(library_dir.glob("*/*/SKILL.md")):
        name, description, body = parse_skill_md(p.read_text(encoding="utf-8"))
        out.append(
            SkillFile(
                task_id=p.parent.parent.name,
                name=name,
                description=description,
                body=body,
                path=p,
            )
        )
    return out


def find_collisions(library_dir: Path) -> dict[str, list[SkillFile]]:
    """{name: [SkillFile, ...]} for every frontmatter name shared by >1 skill."""
    groups: dict[str, list[SkillFile]] = {}
    for sf in load_skill_files(library_dir):
        groups.setdefault(sf.name, []).append(sf)
    return {name: fs for name, fs in groups.items() if len(fs) > 1}


# ── judge contract ────────────────────────────────────────────────────────────
class Rename(BaseModel):
    """One renamed skill (distinct verdict), positionally matched to its input."""

    new_name: str = Field(
        description="unique lowercase kebab-case name (a-z, 0-9, hyphens)"
    )
    new_description: str = Field(
        description="one-line description, SHARPENED so a retriever seeing only "
        "name+description can tell it apart from its same-named siblings"
    )


class CollisionResolution(BaseModel):
    """gemini judge output for one same-name collision group."""

    verdict: Literal["merge", "distinct"]
    reason: str = Field(description="one sentence: why merge or why distinct")
    merged_name: str = Field(
        default="", description="merge only: unique kebab-case name"
    )
    merged_description: str = Field(
        default="", description="merge only: one-line description"
    )
    merged_body: str = Field(
        default="",
        description="merge only: full markdown body (no frontmatter); union of the "
        "genuinely-useful knowledge, kept FOCUSED (no grab-bag)",
    )
    renames: list[Rename] = Field(
        default_factory=list,
        description="distinct only: exactly one Rename per input skill, SAME ORDER as given",
    )


# NOTE: the induction-time LLM dedup judge (was DEDUP_SYSTEM / build_dedup_judge)
# has been removed. Same-name collisions in an assembled library are now resolved
# by the deterministic `uniquify_collisions` below (rename only; no LLM, no merge).
# Semantic merging of similar skills is deliberately the CURATION stage's job.
# CollisionResolution / _llm_resolver / apply_resolution below are kept because the
# curation merge judge (curate.py) reuses them.


def build_dedup_input(
    name: str, skills: list[SkillFile], avoid_names: list[str]
) -> str:
    blocks = []
    for i, sf in enumerate(skills, 1):
        blocks.append(
            f"=== Skill {i} (from task {sf.task_id}) ===\n"
            f"name: {sf.name}\ndescription: {sf.description}\n\n{sf.body}"
        )
    avoid = ", ".join(sorted(avoid_names)) or "(none)"
    return (
        f"COLLIDING_NAME: {name}\n"
        f"AVOID_NAMES (do not reuse these existing names): {avoid}\n\n"
        + "\n\n".join(blocks)
    )


# resolver signature: (name, skills, avoid_names) -> CollisionResolution
Resolver = Callable[[str, list[SkillFile], list[str]], Awaitable[CollisionResolution]]


def _llm_resolver(judge: LlmAgent) -> Resolver:
    async def resolve(name, skills, avoid_names):
        return await run_meta_agent_validated(
            judge, build_dedup_input(name, skills, avoid_names), CollisionResolution
        )

    return resolve


# ── apply (deterministic file ops) ─────────────────────────────────────────────
def _unique(name: str, taken: set[str]) -> str:
    if name not in taken:
        return name
    i = 2
    while f"{name}-{i}" in taken:
        i += 1
    return f"{name}-{i}"


def apply_resolution(
    skills: list[SkillFile], res: CollisionResolution, taken: set[str]
) -> list[dict]:
    """Rewrite files for one resolved collision group. `taken` = all OTHER names
    currently in the library (the colliding name(s) excluded). Mutates `taken`
    to reserve the new names. Returns per-action log entries."""
    actions: list[dict] = []
    # remove the colliding dirs first; we re-materialize below
    for sf in skills:
        shutil.rmtree(sf.skill_dir)

    if res.verdict == "merge":
        if not (res.merged_name and res.merged_body):
            raise ValueError("merge verdict missing merged_name/merged_body")
        home = min(skills, key=lambda s: s.task_id)  # deterministic home dir
        final = _unique(res.merged_name, taken)
        taken.add(final)
        dest = home.skill_dir.parent / final / "SKILL.md"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            render_skill_md(final, res.merged_description, res.merged_body),
            encoding="utf-8",
        )
        actions.append(
            {
                "verdict": "merge",
                "from": [{"task_id": s.task_id, "name": s.name} for s in skills],
                "to": {"task_id": home.task_id, "name": final},
                "reason": res.reason,
            }
        )
    else:  # distinct
        if len(res.renames) != len(skills):
            raise ValueError(
                f"distinct verdict needs {len(skills)} renames, got {len(res.renames)}"
            )
        for sf, rn in zip(skills, res.renames):
            final = _unique(rn.new_name, taken)
            taken.add(final)
            dest = sf.skill_dir.parent / final / "SKILL.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            # keep body verbatim; only frontmatter name+description change
            dest.write_text(
                render_skill_md(final, rn.new_description, sf.body), encoding="utf-8"
            )
            actions.append(
                {
                    "verdict": "distinct",
                    "from": {"task_id": sf.task_id, "name": sf.name},
                    "to": {"task_id": sf.task_id, "name": final},
                    "reason": res.reason,
                }
            )
    return actions


# ── orchestration (deterministic name-uniquify — NO LLM, NO merge) ──────────────
def uniquify_collisions(library_dir: Path) -> list[dict]:
    """Resolve every same-name collision in one library by RENAMING (name-2, name-3,
    ...; the first by sorted task_id keeps the name) so each skill is uniquely
    addressable. Purely deterministic: no LLM, no merging. Semantic merging of
    similar skills is intentionally the curation stage's job, not induction's."""
    collisions = find_collisions(library_dir)
    actions: list[dict] = []
    if not collisions:
        return actions
    taken = {sf.name for sf in load_skill_files(library_dir)}
    for name in sorted(collisions):
        for sf in sorted(collisions[name], key=lambda s: s.task_id)[1:]:
            new = _unique(name, taken)
            taken.add(new)
            skill_dir = sf.path.parent  # <task_id>/<name>/
            dest = skill_dir.parent / new  # <task_id>/<new>/
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "SKILL.md").write_text(
                render_skill_md(new, sf.description, sf.body), encoding="utf-8"
            )
            shutil.rmtree(skill_dir)
            actions.append({"task_id": sf.task_id, "from": name, "to": new})
    return actions


def uniquify_component(component_root: Path) -> dict:
    """Deterministic same-name collision resolution on every assembled library
    (q0.., best) under a component root. No LLM; no merge."""
    out: dict = {"component_root": str(component_root), "libraries": {}}
    for lib in sorted(component_root.iterdir()):
        if lib.is_dir():
            out["libraries"][lib.name] = uniquify_collisions(lib)
    return out


__all__ = [
    "CollisionResolution",
    "Rename",
    "SkillFile",
    "apply_resolution",
    "build_dedup_input",
    "find_collisions",
    "load_skill_files",
    "parse_skill_md",
    "render_skill_md",
    "uniquify_collisions",
    "uniquify_component",
]
