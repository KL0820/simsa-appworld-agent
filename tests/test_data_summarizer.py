"""Unit tests for the generic data summarizer helper.

Covers each classifier branch (constant / id / categorical / numeric_range /
datetime_range / free_form / nested_dict / nested_list / default fallback)
and the top-level entry point's shape dispatch (list[dict] / dict / scalar).
"""

from __future__ import annotations

from adk_appworld_agent.subagents.utils.data_summarizer import (
    summarize_data_for_llm,
    summarize_list_of_dicts,
)


def test_constant_field_collapses_into_constants_section():
    items = [
        {"id": 1, "status": "pending"},
        {"id": 2, "status": "pending"},
        {"id": 3, "status": "pending"},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "constants (all-same):" in out
    assert "status = 'pending' (all 3)" in out


def test_id_field_skipped_from_distribution():
    items = [
        {"payment_request_id": 1001, "amount": 100.0},
        {"payment_request_id": 1002, "amount": 200.0},
        {"payment_request_id": 1003, "amount": 300.0},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "ids (skipped distribution):" in out
    assert "payment_request_id: 3 unique values" in out
    # ID values themselves should NOT appear in a distribution line
    assert "payment_request_id (3 unique):" not in out


def test_datetime_range_renders_min_max():
    items = [
        {"created_at": "2023-05-10T08:00:00"},
        {"created_at": "2023-05-20T09:00:00"},
        {"created_at": "2023-05-15T10:00:00"},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "ranges:" in out
    assert "created_at: [2023-05-10T08:00:00 .. 2023-05-20T09:00:00]" in out


def test_numeric_range_when_high_unique():
    items = [{"amount": v} for v in range(10)]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "amount: [0, 9], median 5" in out


def test_categorical_distribution_top_k_plus_more():
    items = [{"description": f"Bill for Housing {i % 6}"} for i in range(10)]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "distributions:" in out
    assert "description (6 unique):" in out


def test_free_form_field_skips_distribution():
    # avg_len > 60 → free_form
    items = [
        {"content": "x" * 100},
        {"content": "y" * 100},
        {"content": "z" * 100},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "free-form fields:" in out
    assert "content:" in out
    # The actual values should not be enumerated under distributions
    assert "distributions:" not in out


def test_short_string_stays_categorical_even_when_all_unique():
    # All 100% unique BUT short avg → categorical (Entry 4 design decision)
    items = [{"title": f"Title {i}"} for i in range(5)]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "distributions:" in out
    assert "title (5 unique):" in out


def test_nested_dict_goes_to_raw_only():
    items = [
        {"sender": {"name": "Alice", "email": "a@x.com"}, "amount": 10.0},
        {"sender": {"name": "Bob", "email": "b@x.com"}, "amount": 20.0},
        {"sender": {"name": "Carol", "email": "c@x.com"}, "amount": 30.0},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "raw-only fields" in out
    assert "sender: dict" in out
    # Sample should still show full nested
    assert "Alice" in out


def test_nested_list_with_values_goes_to_raw_only_with_lengths():
    items = [
        {"tags": ["a", "b"], "id": 1},
        {"tags": ["a"], "id": 2},
        {"tags": ["a", "b", "c"], "id": 3},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "raw-only fields" in out
    assert "tags: list, lengths [1..3]" in out


def test_empty_nested_list_collapses_to_constants():
    items = [
        {"tags": [], "id": 1},
        {"tags": [], "id": 2},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    assert "constants (all-same):" in out
    assert "tags = [] (all 2)" in out


def test_single_dict_returns_schema_plus_value():
    out = summarize_data_for_llm({"album_id": 12, "genre": "rock", "rating": 3.7})
    assert "result: dict (3 fields)" in out
    assert "schema:" in out
    assert "genre: " in out
    assert "rating: 3.7" in out


def test_empty_list_returns_empty_label():
    out = summarize_data_for_llm([])
    assert "list[0]" in out


def test_list_of_primitives_renders_samples():
    out = summarize_data_for_llm(["alpha", "beta", "gamma", "delta", "epsilon"])
    assert "list[5]" in out
    assert "non-dict items" in out


def test_samples_render_all_fields_no_field_omission():
    items = [
        {"id": 1, "name": "Alice", "amount": 100, "status": "pending"},
        {"id": 2, "name": "Bob", "amount": 200, "status": "paid"},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    # Even fields that appear in constants/etc. should still be visible per sample
    # (per Entry 4: per-field truncation, no field hiding)
    assert "samples (diverse by" in out
    # Each sample row should contain all fields
    sample_lines = [line for line in out.splitlines() if line.startswith("  {")]
    assert len(sample_lines) >= 1
    for sample in sample_lines:
        assert "id=" in sample
        assert "name=" in sample
        assert "amount=" in sample
        assert "status=" in sample


def test_long_field_value_truncated_in_sample_but_field_still_shown():
    items = [
        {"name": "A", "content": "x" * 500},
        {"name": "B", "content": "y" * 500},
    ]
    out = "\n".join(summarize_list_of_dicts(items))
    sample_lines = [line for line in out.splitlines() if line.startswith("  {")]
    assert any("content=" in s for s in sample_lines)
    # value should be truncated (visible "…" marker)
    assert "…" in out


def test_nested_list_field_with_none_value_does_not_crash():
    """Regression: when a nested-list field exists in some rows with value
    None (not missing), `it.get(field, [])` returns None and the previous
    `len(it.get(field, []))` crashed with TypeError. Surfaced 2026-05-17
    when per-API aggregator started rendering dict-pool items per-group
    (prior path dropped that pool entirely, masking the bug).

    Layer-level: any API whose response_schema has a list field that can
    be null on some rows hits this. E.g. phone.update_alarm returns
    `{"days": null, ...}` for non-recurring alarms — that null leaked
    through.
    """
    items = [
        {"id": 1, "tags": ["a", "b"]},
        {"id": 2, "tags": None},  # ← the trigger
        {"id": 3, "tags": ["c"]},
    ]
    # Must not raise
    out = "\n".join(summarize_list_of_dicts(items))
    # And must still produce some characterization of the field
    assert "tags" in out
