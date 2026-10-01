from __future__ import annotations

from adk_appworld_agent.contracts.executor_result import MemoryVariable
from adk_appworld_agent.orchestration.state import RunState


def test_variable_store_commits_enriched_variable_metadata():
    state = RunState()
    committed = state.variable_store.commit(
        MemoryVariable(
            name="contacts",
            value_json='[{"email":"a@example.com"},{"email":"b@example.com"}]',
            description="Phone contacts.",
            source_milestone_id="m1",
        )
    )

    assert committed.type_name == "list"
    assert committed.count_items == 2
    assert committed.created_at
    assert state.variable_creation_order == ["contacts"]
    assert state.named_variables["contacts"]["description"] == "Phone contacts."


def test_variable_store_summary_renders_preview_without_storing_preview():
    state = RunState()
    state.variable_store.commit(
        MemoryVariable(
            name="wife_email",
            value_json='"sarah@ex.com"',
            description="Email address for the user's wife.",
            source_milestone_id="m1",
        )
    )

    summary = state.variable_store.summary()

    assert "### wife_email" in summary
    assert "Email address for the user's wife." in summary
    assert "sarah@ex.com" in summary
    assert "preview" not in state.named_variables["wife_email"]
    # created_at is wall-clock noise; leaking it into the prompt makes
    # cross-session smoke comparisons drift on Gemini Flash temp=0.
    assert "created_at" not in summary
    import re

    assert not re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", summary)


def test_variable_store_replaces_existing_name_without_reordering():
    state = RunState()
    state.variable_store.commit(MemoryVariable(name="value", value_json="1"))
    state.variable_store.commit(MemoryVariable(name="value", value_json="2"))

    assert state.variable_creation_order == ["value"]
    assert state.variable_store.preview("value") == "2"


def test_summary_exposes_top_level_fields_for_dict_value():
    """Fix 3: variable_store.summary() should surface top-level dict keys
    so the next milestone's executor doesn't have to parse the preview JSON
    to know accessible fields. Observed 986aa4e_2 m3 error 'task_id is
    missing' — M2 committed a task object but M3 stage-2 prompt only saw
    a JSON preview, not an explicit field index."""
    state = RunState()
    state.variable_store.commit(
        MemoryVariable(
            name="task_obj",
            value_json='{"id": 42, "title": "X", "completed": false, "due_date": "2023-05-18"}',
            description="A task object",
        )
    )
    summary = state.variable_store.summary(["task_obj"])
    assert "accessible_fields" in summary
    assert "id" in summary
    assert "title" in summary
    assert "completed" in summary


def test_summary_exposes_top_level_fields_for_list_of_dicts():
    state = RunState()
    state.variable_store.commit(
        MemoryVariable(
            name="users",
            value_json='[{"first_name": "Alice", "email": "a@x"}, {"first_name": "Bob", "email": "b@y"}]',
            description="List of users",
        )
    )
    summary = state.variable_store.summary(["users"])
    assert "accessible_fields" in summary
    assert "first_name" in summary
    assert "email" in summary


def test_summary_omits_accessible_fields_for_primitive_value():
    state = RunState()
    state.variable_store.commit(
        MemoryVariable(
            name="count",
            value_json="42",
            description="A primitive number",
        )
    )
    summary = state.variable_store.summary(["count"])
    assert "accessible_fields" not in summary
