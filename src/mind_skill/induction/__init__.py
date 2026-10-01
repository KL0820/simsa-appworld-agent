"""Induction side of MIND-Skill: A_I (the only TextGrad-variable agent),
its P_I^(0) prompts, and the InducedSkill output schema + SKILL.md renderer.

Layout (role-based): deduction/ judges/ textgrad/ are sibling packages;
shared utilities (runtime / render / losses / loop) live at the package top.
"""

from mind_skill.induction.agent import PromptHolder, build_induction_agent, induce_skill
from mind_skill.induction.prompts import INDUCTION_PROMPTS
from mind_skill.induction.schemas import InducedSkill, KeyPattern, render_skill_md

__all__ = [
    "INDUCTION_PROMPTS",
    "InducedSkill",
    "KeyPattern",
    "PromptHolder",
    "build_induction_agent",
    "induce_skill",
    "render_skill_md",
]
