from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.subagents.planner.rough_planner.app_catalog import (
    app_descriptions_source_label,
    load_app_catalog,
)

ALLOWED_APP_NAMES: tuple[str, ...] = (
    "gmail",
    "spotify",
    "venmo",
    "splitwise",
    "amazon",
    "todoist",
    "simple_note",
    "phone",
    "file_system",
)

APP_CATALOG = load_app_catalog(ALLOWED_APP_NAMES)
APP_CATALOG_SOURCE = app_descriptions_source_label()
APP_DESCRIPTIONS = {entry.name: entry.description for entry in APP_CATALOG}

_APP_CONTEXT_PATH = (
    Path(__file__).resolve().parents[1] / "community" / "data" / "app_context.md"
)


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def render_app_catalog() -> str:
    lines: list[str] = []
    for entry in APP_CATALOG:
        lines.append(f"- {entry.name}: {entry.description}")
    return "\n".join(lines)


APP_SELECTION_CONTEXT = _load_text(_APP_CONTEXT_PATH)


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


# Production (fat) prompt — prompts/full.md, app catalog interpolated.
ROUGH_PLANNER_INSTRUCTION = _load_prompt("full").replace(
    "{app_catalog}", render_app_catalog()
)


# ── Frozen minimal rough-planner prompt (MIND-Skill deduction / skill-eval) ──
#
# FROZEN: the rough_planner deduction prompt of the MIND-Skill pipeline
# (mind_skill/docs/induction_training_spec.md §6, paper Fig 8). Must stay
# byte-stable across a training + eval cycle. Skills enter ONLY through the
# `### SKILLS BEGIN/END` slot.
# Kept: role framing, the official-spec app catalog, structural output rules,
# JSON contract, one demonstration.
# Stripped (skills carry these): constraint-preservation rule block,
# named-target rule, funding/method-resource rule, completion heuristics.
ROUGH_PLANNER_MINIMAL_INSTRUCTION = _load_prompt("minimal").replace(
    "{app_catalog}", render_app_catalog()
)

_PLANNER_SKILLS_SECTION_TEMPLATE = """You are also provided with a curated set of skills to help you solve the task effectively.
Read the skills first, then execute the task by explicitly leveraging each relevant section:
### SKILLS BEGIN
{skills}
### SKILLS END
"""


def render_rough_planner_instruction(
    *, variant: str = "full", skill_texts: list[str] | None = None
) -> str:
    """`full` -> fat prompt (skills appended on top when present); `minimal` ->
    frozen thin prompt, skills fill the Fig-8 slot."""
    cleaned = [t.strip() for t in (skill_texts or []) if t and t.strip()]
    if variant == "full":
        if cleaned:
            return "\n".join(
                [
                    ROUGH_PLANNER_INSTRUCTION,
                    _PLANNER_SKILLS_SECTION_TEMPLATE.format(
                        skills="\n\n".join(cleaned)
                    ),
                ]
            )
        return ROUGH_PLANNER_INSTRUCTION
    if variant == "minimal":
        sections = [ROUGH_PLANNER_MINIMAL_INSTRUCTION]
        if cleaned:
            sections.append(
                _PLANNER_SKILLS_SECTION_TEMPLATE.format(skills="\n\n".join(cleaned))
            )
        return "\n".join(sections)
    raise ValueError(f"unknown rough planner prompt variant: {variant!r}")


def render_rough_planner_prompt(
    *,
    task_id: str,
    instruction: str,
    task_datetime: str,
) -> str:
    lines = [
        f"Task ID: {task_id}",
    ]
    if task_datetime:
        lines.append(f"Current datetime: {task_datetime}")
    lines.extend(
        [
            "Task instruction:",
            instruction,
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ALLOWED_APP_NAMES",
    "APP_CATALOG",
    "APP_CATALOG_SOURCE",
    "APP_DESCRIPTIONS",
    "APP_SELECTION_CONTEXT",
    "ROUGH_PLANNER_INSTRUCTION",
    "ROUGH_PLANNER_MINIMAL_INSTRUCTION",
    "render_rough_planner_instruction",
    "render_rough_planner_prompt",
    "render_app_catalog",
]
