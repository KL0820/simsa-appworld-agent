"""Adapt immutable research skill files to ADK's native skill toolset."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml


def _native_name(name: str) -> str:
    """Keep valid names; give unsupported names a stable, collision-resistant alias."""
    if len(name) <= 64 and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        return name
    stem = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "skill"
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:53].rstrip('-')}-{digest}"


def _load_native_skill(path: Path):
    from google.adk.skills.models import Frontmatter, Skill

    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError(f"Missing YAML frontmatter: {path}")
    frontmatter = yaml.safe_load(parts[1])
    original_name = frontmatter["name"]
    frontmatter["name"] = _native_name(original_name)
    metadata = dict(frontmatter.get("metadata") or {})
    metadata["research_original_name"] = original_name
    frontmatter["metadata"] = metadata
    return Skill(
        frontmatter=Frontmatter.model_validate(frontmatter),
        instructions=parts[2].lstrip("\n"),
    )


def build_skill_toolset(skills_dir: Path | None):
    """Load a text-only library without rewriting its published names or bodies.

    Native tool names have stricter limits than the research corpus. Aliases
    apply only in memory; duplicate original names retain first-file precedence,
    matching the previous native loader.
    """
    if skills_dir is None:
        return None
    paths = sorted(Path(skills_dir).glob("*/*/SKILL.md"))
    if not paths:
        return None
    from google.adk.tools.skill_toolset import SkillToolset

    skills = {}
    for path in paths:
        skill = _load_native_skill(path)
        skills.setdefault(skill.name, skill)
    return SkillToolset(skills=list(skills.values()))
