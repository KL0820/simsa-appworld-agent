"""ADK compatibility must not mutate the published skill corpus."""

from adk_appworld_agent.subagents.utils.native_skills import (
    _load_native_skill,
    _native_name,
)


def test_valid_names_are_preserved():
    assert _native_name("filter-then-update") == "filter-then-update"


def test_long_names_are_stable_and_distinct():
    prefix = "retrieve-and-process-" * 5
    first = _native_name(prefix + "first")
    assert len(first) <= 64
    assert first == _native_name(prefix + "first")
    assert first != _native_name(prefix + "second")


def test_native_loading_preserves_source_and_body(tmp_path):
    name = "retrieve-and-process-" * 5 + "results"
    body = "Keep every matching item.\n"
    source = f"---\nname: {name}\ndescription: Process matching results.\n---\n{body}"
    path = tmp_path / "SKILL.md"
    path.write_text(source)

    skill = _load_native_skill(path)

    assert path.read_text() == source
    assert skill.instructions == body
    assert skill.name == _native_name(name)
    assert skill.frontmatter.metadata["research_original_name"] == name
