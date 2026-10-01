"""Structured output schema for the induction agents + renderer to a native SKILL.md.

Each *_induction_agent (gemini-2.5-flash LlmAgent) is given `output_schema=InducedSkill`,
so the model emits validated JSON instead of free-form markdown. render_skill_md() turns
the object into the exact SKILL.md format the native ADK loader expects (kebab `name` +
`description` frontmatter, then the five sections). This guarantees a loadable skill and
makes the TextGrad optimizer's "keep format" constraint automatic (it edits P_I, not this).

ADK compatibility note: output_schema goes through Gemini controlled generation. Nested
KeyPattern + lists are supported; `min_length` / `max_length` may not be enforced by the
provider, so re-validate post-hoc (`InducedSkill(**json)`). If nesting misbehaves, fall back
to `key_patterns: list[str]` ("Name: insight").
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class KeyPattern(BaseModel):
    name: str = Field(
        description="Short label for the structural insight, e.g. 'Pagination Loop'. Generic, no task nouns."
    )
    insight: str = Field(
        description="The insight itself, stated generically — no concrete API/field/entity names."
    )


class InducedSkill(BaseModel):
    # ── YAML frontmatter ──
    name: str = Field(
        description="kebab-case slug; MUST equal the skill's directory name; no task-specific nouns."
    )
    description: str = Field(
        max_length=1024,
        description="ONE sentence: when this skill applies (the retrieval key). No task specifics.",
    )
    # ── body sections ──
    overview: str = Field(
        description="2-3 sentences: the structural problem this task family poses."
    )
    when_to_apply: list[str] = Field(
        min_length=1,
        description="Instruction-level signals that this pattern applies; each stated generically.",
    )
    procedure: list[str] = Field(
        min_length=1,
        description="Ordered generic steps; reference roles ('the credential', 'the target'), not concrete APIs/fields.",
    )
    key_patterns: list[KeyPattern] = Field(
        min_length=1,
        description="Non-obvious structural insights (join-key choice, pagination exhaustion, verbatim pass-through, ...).",
    )
    common_pitfalls: list[str] = Field(
        min_length=1,
        description="Mistakes a naive agent makes here; each stated generically.",
    )

    @field_validator("name")
    @classmethod
    def _kebab(cls, v: str) -> str:
        if not _KEBAB.match(v):
            raise ValueError(f"name must be kebab-case: {v!r}")  # caller → quarantine
        return v


def render_skill_md(s: InducedSkill) -> str:
    """InducedSkill -> native SKILL.md text (YAML frontmatter + five sections)."""
    out = [
        "---",
        f"name: {s.name}",
        f"description: {s.description}",
        "---",
        "## Overview",
        s.overview,
        "## When to Apply",
        *(f"- {x}" for x in s.when_to_apply),
        "## Procedure",
        *(f"{i}. {step}" for i, step in enumerate(s.procedure, 1)),
        "## Key Patterns",
        *(f"- **{p.name}:** {p.insight}" for p in s.key_patterns),
        "## Common Pitfalls",
        *(f"- {x}" for x in s.common_pitfalls),
    ]
    return "\n".join(out) + "\n"
