"""code_plan_execute prompts — production (full) and frozen deduction
(minimal) variants for BOTH stages, loaded verbatim from prompts/*.md.

FROZEN: the *_minimal prompts are the MIND-Skill deduction prompts
(induction_training_spec §6, paper Fig 8); skills enter ONLY through the
`### SKILLS BEGIN/END` slot. Editing them mid-cycle invalidates every skill
library trained against them.
"""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


CODE_PLANNER_SYSTEM_PROMPT = _load_prompt("code_planner_full")
CODE_PLANNER_MINIMAL_SYSTEM_PROMPT = _load_prompt("code_planner_minimal")
CODE_EXECUTOR_SYSTEM_PROMPT = _load_prompt("executor_full")
CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT = _load_prompt("executor_minimal")

_EXECUTOR_SKILLS_SECTION_TEMPLATE = """You are also provided with a curated set of skills to help you solve the task effectively.
Read the skills first, then execute the task by explicitly leveraging each relevant section:
### SKILLS BEGIN
{skills}
### SKILLS END
"""


from adk_appworld_agent.subagents.utils.instructions import (  # noqa: F401
    instruction_text,
    static_instruction_provider,
)


def render_executor_skills_section(skill_texts: list[str]) -> str:
    """Render the Fig-8 skill slot. No non-empty skill -> empty string (thin-no-skill)."""
    cleaned = [text.strip() for text in skill_texts if text and text.strip()]
    if not cleaned:
        return ""
    return _EXECUTOR_SKILLS_SECTION_TEMPLATE.format(skills="\n\n".join(cleaned))


def render_executor_instruction(
    *, variant: str = "full", skill_texts: list[str] | None = None
) -> str:
    """Resolve the executor system prompt for a prompt variant.

    `full` -> CODE_EXECUTOR_SYSTEM_PROMPT (skills appended on top when present).
    `minimal` -> frozen thin prompt; `skill_texts` fill the SKILLS slot.
    """
    skills_section = render_executor_skills_section(skill_texts or [])
    if variant == "full":
        if skills_section:
            return "\n".join([CODE_EXECUTOR_SYSTEM_PROMPT, skills_section])
        return CODE_EXECUTOR_SYSTEM_PROMPT
    if variant == "minimal":
        sections = [CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT]
        if skills_section:
            sections.append(skills_section)
        return "\n".join(sections)
    raise ValueError(f"unknown executor prompt variant: {variant!r}")


def render_code_planner_instruction(
    *, variant: str = "full", skill_texts: list[str] | None = None
) -> str:
    """`full` -> CODE_PLANNER_SYSTEM_PROMPT (skills appended on top when present);
    `minimal` -> frozen thin prompt, skills fill the Fig-8 slot."""
    skills_section = render_executor_skills_section(skill_texts or [])
    if variant == "full":
        if skills_section:
            return "\n".join([CODE_PLANNER_SYSTEM_PROMPT, skills_section])
        return CODE_PLANNER_SYSTEM_PROMPT
    if variant == "minimal":
        sections = [CODE_PLANNER_MINIMAL_SYSTEM_PROMPT]
        if skills_section:
            sections.append(skills_section)
        return "\n".join(sections)
    raise ValueError(f"unknown code planner prompt variant: {variant!r}")


__all__ = [
    "CODE_EXECUTOR_MINIMAL_SYSTEM_PROMPT",
    "CODE_EXECUTOR_SYSTEM_PROMPT",
    "CODE_PLANNER_MINIMAL_SYSTEM_PROMPT",
    "CODE_PLANNER_SYSTEM_PROMPT",
    "instruction_text",
    "render_code_planner_instruction",
    "render_executor_instruction",
    "render_executor_skills_section",
    "static_instruction_provider",
]
