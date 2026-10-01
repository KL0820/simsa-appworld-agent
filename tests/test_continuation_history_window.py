"""Tests for continuation history two-tier sliding window (Entry 6).

Older cycles condense to one-liner; recent MAX_HISTORY_CYCLES cycles
render in full. Verify both rendering modes + the boundary.
"""

from __future__ import annotations

from adk_appworld_agent.contracts.limits import MAX_HISTORY_CYCLES
from adk_appworld_agent.subagents.continuer.continuation import (
    _build_history_block,
    _render_cycle_oneliner,
    _summarize_value_for_oneliner,
)


def _exec_entry(cycle, milestone_index, success=True, value=None, failure_code=None):
    return {
        "cycle_n": cycle,
        "phase_completed": "EXECUTE",
        "milestone_index": milestone_index,
        "success": success,
        "failure_code": failure_code,
        "agent_output": {
            "summary": "ok" if success else "fail",
            "stdout_json": {"value": value} if success else None,
            "code": None if success else "print('x')",
        },
    }


def _plan_entry(cycle, decision="RETRY", revised=False):
    return {
        "cycle_n": cycle,
        "phase_completed": "PLAN",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "next_action": decision,
            "revised_milestones_applied": revised,
            "rationale": "some rationale",
        },
    }


# ── _summarize_value_for_oneliner ───────────────────────────────────────


def test_summarize_value_none():
    assert _summarize_value_for_oneliner(None) == "null"


def test_summarize_value_dict():
    assert _summarize_value_for_oneliner({"a": 1, "b": 2}) == "dict(2 keys)"


def test_summarize_value_list():
    assert _summarize_value_for_oneliner([1, 2, 3, 4]) == "list[4]"


def test_summarize_value_scalar():
    assert _summarize_value_for_oneliner(42) == "int=42"
    assert _summarize_value_for_oneliner(3.14) == "float=3.14"


# ── _render_cycle_oneliner ──────────────────────────────────────────────


def test_oneliner_with_exec_and_plan():
    entries = [_exec_entry(3, 0, value=None), _plan_entry(3)]
    line = _render_cycle_oneliner(3, entries)
    assert "cycle 3" in line
    assert "EXEC m0 null" in line
    assert "PLAN RETRY" in line


def test_oneliner_with_revised_milestones():
    entries = [_exec_entry(4, 0, value=None), _plan_entry(4, revised=True)]
    line = _render_cycle_oneliner(4, entries)
    assert "✱revised milestones" in line


def test_oneliner_with_only_plan():
    entries = [_plan_entry(5, decision="ADVANCE")]
    line = _render_cycle_oneliner(5, entries)
    assert "cycle 5" in line
    assert "PLAN ADVANCE" in line
    assert "EXEC" not in line


def test_oneliner_with_failed_exec():
    entries = [
        _exec_entry(2, 0, success=False, failure_code="TIMED_OUT"),
        _plan_entry(2),
    ]
    line = _render_cycle_oneliner(2, entries)
    assert "EXEC m0 TIMED_OUT" in line


# ── _build_history_block sliding window ─────────────────────────────────


def test_short_history_renders_all_full():
    # ≤ MAX_HISTORY_CYCLES cycles → all in full mode (no condensed section)
    history = [
        _exec_entry(0, 0, value=None),
        _plan_entry(0),
        _exec_entry(1, 0, value=None),
        _plan_entry(1),
    ]
    block = _build_history_block(history)
    # Should NOT contain one-liner format (no "cycle N  EXEC ... + PLAN ...")
    assert "  cycle 0  EXEC" not in block
    # Should contain full-detail header
    assert "cycle=0 EXECUTE" in block or "cycle=1 EXECUTE" in block


def test_long_history_condenses_older_cycles():
    # 5 cycles, MAX_HISTORY_CYCLES=2 → cycles 0-2 condensed, cycles 3-4 full
    history = []
    for c in range(5):
        history.append(_exec_entry(c, 0, value=None))
        history.append(_plan_entry(c))
    block = _build_history_block(history)
    # Cycles 0-2 should be in condensed one-liner format
    assert "cycle 0  EXEC m0 null + PLAN RETRY" in block
    assert "cycle 1  EXEC m0 null + PLAN RETRY" in block
    assert "cycle 2  EXEC m0 null + PLAN RETRY" in block
    # Cycles 3-4 should be in full detail format
    assert "cycle=3 EXECUTE" in block or "cycle=4 EXECUTE" in block


def test_history_boundary_at_max_history_cycles():
    # Verify exactly the last MAX_HISTORY_CYCLES cycles are full
    cycles_total = MAX_HISTORY_CYCLES + 3
    history = []
    for c in range(cycles_total):
        history.append(_exec_entry(c, 0, value=None))
        history.append(_plan_entry(c))
    block = _build_history_block(history)
    max_cycle = cycles_total - 1
    recent_threshold = max_cycle - MAX_HISTORY_CYCLES + 1
    # Older cycles (< recent_threshold) in condensed
    for c in range(recent_threshold):
        assert f"cycle {c}  EXEC" in block, f"cycle {c} should be condensed"
    # Recent cycles in full
    for c in range(recent_threshold, cycles_total):
        assert f"cycle={c} EXECUTE" in block, f"cycle {c} should be full"


def test_empty_history_returns_empty_label():
    assert _build_history_block([]) == "(empty)"


# ── api_trace integration via render history entry ──────────────────────


def test_history_entry_renders_helper_aggregated_section_for_api_trace():
    """End-to-end: a successful EXECUTE history entry with api_trace
    should render the helper's aggregated section (schema /
    distributions / ranges / samples), not just the call headers.

    Regression: an earlier version of _render_api_trace_lines read
    `call["result_items"]` while summarize_trace_for_prompt was
    producing `call["items_sample"]`. The mismatch silently skipped
    aggregation, leaving the prompt with only call headers.
    """
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 1,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "summary": "found something",
            "stdout_json": {"value": {"x": 1}},
            "variables": [{"name": "x"}],
            "api_trace": [
                {
                    "app": "venmo",
                    "api_name": "show_received_payment_requests",
                    "kwargs": {"status": "pending", "page_index": 0},
                    "result_shape": {
                        "type": "list",
                        "count_hint": 5,
                        "keys": ["amount", "description"],
                    },
                    "result_items": {
                        "_list_total": 5,
                        "items": [
                            {"amount": 600.0, "description": "Bill for Housing"},
                            {"amount": 39.0, "description": "For Phone Bill"},
                            {"amount": 18.0, "description": "Art Supplies"},
                            {"amount": 50.0, "description": "Cooking Class"},
                            {"amount": 25.0, "description": "Gift Fund"},
                        ],
                    },
                },
                {
                    "app": "venmo",
                    "api_name": "show_received_payment_requests",
                    "kwargs": {"status": "pending", "page_index": 1},
                    "result_shape": {
                        "type": "list",
                        "count_hint": 4,
                        "keys": ["amount", "description"],
                    },
                    "result_items": {
                        "_list_total": 4,
                        "items": [
                            {"amount": 1000.0, "description": "Bill for Housing"},
                            {"amount": 22.0, "description": "Gaming Session"},
                            {"amount": 41.0, "description": "Art Supplies"},
                            {"amount": 65.0, "description": "Wedding Gift"},
                        ],
                    },
                },
            ],
        },
    }
    rendered = _render_history_entry(entry)
    # Must have the call headers
    assert "venmo.show_received_payment_requests" in rendered
    # Must have the aggregated section (the regression target). Per-API
    # grouping renames the header to "aggregated by {app}.{api_name}"
    # so the same API's schema can be matched to its calls.
    assert "aggregated by venmo.show_received_payment_requests" in rendered
    # Must have helper-rendered sections
    assert "schema:" in rendered
    assert "distributions:" in rendered
    # Target value 'Bill for Housing' should appear in the distribution
    assert "Bill for Housing" in rendered


def test_api_trace_call_headers_cap_at_three_with_varying_kwarg_collapse():
    """When the same API is invoked >3 times with only one kwarg varying,
    the renderer shows the first 3 calls then collapses the rest into
    '... and N more (kwarg in [...])' — not one line per call."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 3,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": None,
                "summary": "ok",
                "description": "",
                "answer": "null",
            },
            "api_trace": [
                {
                    "app": "venmo",
                    "api_name": "show_received_payment_requests",
                    "kwargs": {"status": "pending", "page_index": i},
                    "result_shape": {
                        "type": "list",
                        "count_hint": 5,
                        "keys": ["amount"],
                    },
                    "result_items": {
                        "_list_total": 5,
                        "items": [{"amount": float(i * 10 + j)} for j in range(5)],
                    },
                }
                for i in range(7)
            ],
        },
    }
    rendered = _render_history_entry(entry)
    # First 3 calls fully shown
    assert "[0] venmo.show_received_payment_requests" in rendered
    assert "[1] venmo.show_received_payment_requests" in rendered
    assert "[2] venmo.show_received_payment_requests" in rendered
    # Trailing 4 collapsed
    assert "[3] venmo.show_received_payment_requests" not in rendered
    assert "and 4 more" in rendered
    assert "page_index in" in rendered


def test_api_trace_per_api_cap_does_not_hide_secondary_api():
    """425a494-style trap: when one API is invoked many times (e.g.
    show_liked_songs × 4 paginated) AND a second API is invoked many
    more times (show_album × 11), a GLOBAL cap of 3 lets the first API
    fill all 3 slots and collapses the second API into
    `... and N more (mixed apps/apis)` — hiding its name from the
    prompt entirely. continuation_planner then concludes "show_album
    was not called" from evidence that simply doesn't show it.

    Per-API cap of HEADER_DISPLAY_CAP must keep both APIs visible:
    each gets its own first-3 + group-local collapse line."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 1,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": "hip-hop",
                "summary": "ok",
                "description": "",
                "answer": "hip-hop",
            },
            "api_trace": [
                # 4 paginated show_liked_songs calls
                *[
                    {
                        "app": "spotify",
                        "api_name": "show_liked_songs",
                        "kwargs": {"page_index": str(i), "page_limit": "20"},
                        "result_shape": {
                            "type": "list",
                            "count_hint": 20,
                            "keys": ["song_id", "album_id"],
                        },
                        "result_items": {
                            "_list_total": 20,
                            "items": [
                                {"song_id": j, "album_id": j % 11} for j in range(20)
                            ],
                        },
                    }
                    for i in range(4)
                ],
                # 11 show_album calls — would vanish under a global cap
                *[
                    {
                        "app": "spotify",
                        "api_name": "show_album",
                        "kwargs": {"album_id": aid},
                        "result_shape": {
                            "type": "object",
                            "keys": ["album_id", "genre"],
                        },
                        "result_items": {"album_id": aid, "genre": "EDM"},
                    }
                    for aid in (2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14)
                ],
            ],
        },
    }
    rendered = _render_history_entry(entry)
    # Both APIs must be present in the header section
    assert "spotify.show_liked_songs" in rendered
    assert "spotify.show_album" in rendered
    # First 3 of EACH API rendered explicitly
    assert rendered.count("spotify.show_liked_songs(") >= 3
    assert rendered.count("spotify.show_album(") >= 3
    # Trailing collapse line per group, NOT a global "mixed apps/apis" hider
    assert "mixed apps/apis" not in rendered
    # show_album group has 8 calls collapsed (11 - 3 shown)
    assert "and 8 more spotify.show_album" in rendered
    # show_liked_songs group has 1 call collapsed (4 - 3 shown)
    assert "and 1 more spotify.show_liked_songs" in rendered


def test_api_trace_aggregator_renders_each_api_pool_independently():
    """425a494-style trap (aggregator half): when one API returns lists
    and a second API returns single dicts, the old aggregator picked the
    bigger pool and dropped the smaller. show_album's 11 dict returns
    (with the `genre` field) vanished, so continuation_planner couldn't
    see that genre data had been retrieved.

    New behavior: render one aggregated block per (app, api_name) so
    every API's schema + distributions are surfaced independently."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 1,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": "hip-hop",
                "summary": "ok",
                "description": "",
                "answer": "hip-hop",
            },
            "api_trace": [
                # Large list pool: 2× show_liked_songs × 20 items each = 40
                *[
                    {
                        "app": "spotify",
                        "api_name": "show_liked_songs",
                        "kwargs": {"page_index": str(i), "page_limit": "20"},
                        "result_shape": {
                            "type": "list",
                            "count_hint": 20,
                            "keys": ["song_id", "album_id"],
                        },
                        "result_items": {
                            "_list_total": 20,
                            "items": [
                                {"song_id": j, "album_id": (j % 7) + 2}
                                for j in range(20)
                            ],
                        },
                    }
                    for i in range(2)
                ],
                # Small dict pool: 5× show_album each returning {album_id, genre}
                *[
                    {
                        "app": "spotify",
                        "api_name": "show_album",
                        "kwargs": {"album_id": aid},
                        "result_shape": {
                            "type": "object",
                            "keys": ["album_id", "genre"],
                        },
                        "result_items": {"album_id": aid, "genre": g},
                    }
                    for aid, g in [
                        (2, "R&B"),
                        (3, "indie"),
                        (4, "EDM"),
                        (5, "jazz"),
                        (6, "rock"),
                    ]
                ],
            ],
        },
    }
    rendered = _render_history_entry(entry)
    # Both per-API aggregated blocks present
    assert "aggregated by spotify.show_liked_songs" in rendered
    assert "aggregated by spotify.show_album" in rendered
    # show_album's schema must surface — including the `genre` field
    # that the old aggregator dropped together with its parent pool.
    sa_block = rendered.split("aggregated by spotify.show_album", 1)[1]
    assert "genre" in sa_block
    # Genre distribution must be visible (sanity: 5 unique genres)
    assert "R&B" in sa_block or "EDM" in sa_block


def test_execute_success_renders_all_four_stdout_fields_including_null():
    """EXECUTE success render must show stdout (4-field): with all of
    value / summary / description / answer present, even when value is
    null. Old render dropped value when None and never showed
    description / answer."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 3,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": None,
                "summary": "Could not find the matching request",
                "description": "value is the matching list[dict]; null when no match",
                "answer": "null",
            },
            "variables": [{"name": "venmo_housing_bill_request"}],
        },
    }
    rendered = _render_history_entry(entry)
    assert "stdout (4-field):" in rendered
    assert "value: null" in rendered
    assert "summary: Could not find" in rendered
    assert "description: value is the matching" in rendered
    assert "answer: null" in rendered
    # self-claim line removed (not in design)
    assert "executor self-claim" not in rendered


def test_execute_value_routes_through_helper_for_nested_list():
    """When stdout.value is a list of dicts, it must render through
    summarize_data_for_llm (schema + distributions + samples), not be
    first-N truncated like the old json.dumps[:500] approach."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 4,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": [{"note_id": i, "title": f"note-{i}"} for i in range(10)],
                "summary": "found 10 notes",
                "description": "list of all notes",
                "answer": "null",
            },
            "variables": [{"name": "all_notes"}],
        },
    }
    rendered = _render_history_entry(entry)
    # Helper output markers
    assert "result: list[10]" in rendered
    assert "schema:" in rendered
    assert "distributions:" in rendered or "ids" in rendered  # note_id detected as id


def test_execute_value_scalar_shown_raw():
    """Scalars (int / float / bool / short str) bypass the helper and show
    raw — helper boilerplate would be noisy for trivial values."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 4,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": 42,
                "summary": "answer is 42",
                "description": "scalar",
                "answer": "42",
            },
            "variables": [],
        },
    }
    rendered = _render_history_entry(entry)
    assert "value: 42" in rendered
    assert "result: scalar" not in rendered


# ── M1 (P1): self_assess wired into the EXECUTE-success render ───────────
# A CLAIMED-DONE milestone (success=True) can still be flagged ok=False by
# the executor's own grounded self-check. Before M1 this branch dropped
# self_assess, so the continuation never saw the problem -> ADVANCEd on a
# false positive. These tests pin the physical-disconnection fix.


def test_execute_success_with_self_assess_not_ok_renders_prominent_warning():
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 5,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": [],
                "summary": "committed an empty list",
                "description": "matching items",
                "answer": "null",
            },
            "variables": [{"name": "matches"}],
            "self_assess": {
                "ok": False,
                "summary": "no items matched",
                "problem": "the candidate APIs lack a search capability for this field",
            },
        },
    }
    rendered = _render_history_entry(entry)
    # The high-priority NOT-OK warning must surface on a SUCCESS entry...
    assert "executor self-assess: NOT OK" in rendered
    assert "HIGH-PRIORITY" in rendered
    # ...carrying the executor's actual problem text (not just summary).
    assert "lack a search capability" in rendered


def test_execute_success_with_self_assess_not_ok_falls_back_to_summary():
    """ok=False with no `problem` still warns, using `summary` as the detail."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 5,
        "phase_completed": "EXECUTE",
        "milestone_index": 1,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": None,
                "summary": "x",
                "description": "",
                "answer": "",
            },
            "variables": [],
            "self_assess": {
                "ok": False,
                "summary": "result looks incomplete",
                "problem": "",
            },
        },
    }
    rendered = _render_history_entry(entry)
    assert "executor self-assess: NOT OK" in rendered
    assert "result looks incomplete" in rendered


def test_execute_success_with_clean_self_assess_omits_hint():
    """A plain OK self-assess (ok=True, no problem) adds nothing -> not rendered."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 5,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": 7,
                "summary": "ok",
                "description": "",
                "answer": "7",
            },
            "variables": [],
            "self_assess": {"ok": True, "summary": "looks correct", "problem": ""},
        },
    }
    rendered = _render_history_entry(entry)
    assert "self-assess" not in rendered


def test_api_trace_single_call_single_dict_uses_dict_path():
    """When the only API call returns a single dict, the helper must go
    through summarize_dict (schema + value) — NOT summarize_list_of_dicts
    with N=1, which degenerates because every field becomes 'constant'."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 11,
        "phase_completed": "EXECUTE",
        "milestone_index": 2,
        "success": True,
        "agent_output": {
            "stdout_json": {
                "value": None,
                "summary": "ok",
                "description": "",
                "answer": "null",
            },
            "api_trace": [
                {
                    "app": "spotify",
                    "api_name": "show_playlist",
                    "kwargs": {"playlist_id": 654},
                    "result_shape": {
                        "type": "object",
                        "keys": ["playlist_id", "title", "songs"],
                    },
                    "result_items": {
                        "playlist_id": 654,
                        "title": "Road Trip",
                        "songs": [{"id": i, "title": f"song-{i}"} for i in range(22)],
                    },
                }
            ],
        },
    }
    rendered = _render_history_entry(entry)
    # dict-path markers
    assert "single-object return" in rendered
    assert "result: dict (3 fields)" in rendered
    # list-of-dicts noise markers must NOT appear
    assert "constants (all-same)" not in rendered
    assert "diverse by first-N" not in rendered
    assert "result: list[1]" not in rendered


def test_constants_section_truncates_long_nested_value():
    """In summarize_list_of_dicts, an all-same field with a large nested
    value (e.g. 22-item list shared across rows) used to be dumped at
    full length in the constants section. Cap with
    HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH = 200."""
    from adk_appworld_agent.subagents.utils.data_summarizer import (
        summarize_list_of_dicts,
    )

    big_list = [
        {"id": i, "title": f"song-{i}", "artist_ids": [3, 26, 6]} for i in range(22)
    ]
    items = [
        {"playlist_id": 654, "title": "Road Trip", "songs": big_list},
        {"playlist_id": 655, "title": "Road Trip B", "songs": big_list},
    ]
    rendered = "\n".join(summarize_list_of_dicts(items))
    # 'songs' is identical across both rows → all-same → constants
    assert "constants (all-same):" in rendered
    songs_line = next(line for line in rendered.splitlines() if "songs =" in line)
    # 200 char cap + "…" + "(all 2)" suffix — total well below the 1000+
    # char untruncated repr of the 22-song list.
    assert "…" in songs_line
    assert len(songs_line) < 350


def test_skipped_find_entry_is_annotated_in_render():
    """FIND entries with agent_output.reused_from_prior_cycle should
    render with a '(skipped, reused from prior cycle)' suffix so the
    LLM (and the viewer reading the same data) can tell skipped FINDs
    apart from real ones."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    skipped_entry = {
        "cycle_n": 2,
        "phase_completed": "FIND",
        "milestone_index": 0,
        "success": True,
        "agent_output": {"reused_from_prior_cycle": True},
    }
    rendered = _render_history_entry(skipped_entry)
    assert "cycle=2 FIND m0 success" in rendered
    assert "(skipped, reused from prior cycle)" in rendered


def test_real_find_entry_has_no_skip_annotation():
    """Real (non-skipped) FIND entries must NOT carry the skip suffix."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    real_entry = {
        "cycle_n": 0,
        "phase_completed": "FIND",
        "milestone_index": 0,
        "success": True,
        "agent_output": None,
    }
    rendered = _render_history_entry(real_entry)
    assert "cycle=0 FIND m0 success" in rendered
    assert "skipped" not in rendered


def test_history_entry_renders_helper_for_single_object_return():
    """Single-dict API returns (e.g. show_current_song) should also
    flow through the helper — each call's dict becomes one item in
    the aggregated view."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        _render_history_entry,
    )

    entry = {
        "cycle_n": 1,
        "phase_completed": "EXECUTE",
        "milestone_index": 0,
        "success": True,
        "agent_output": {
            "summary": "got album",
            "stdout_json": {"value": {"genre": "rock"}},
            "variables": [{"name": "album"}],
            "api_trace": [
                {
                    "app": "spotify",
                    "api_name": "show_album",
                    "kwargs": {"album_id": 12},
                    "result_shape": {
                        "type": "object",
                        "keys": ["album_id", "genre", "rating"],
                    },
                    "result_items": {
                        "album_id": 12,
                        "genre": "rock",
                        "rating": 3.7,
                    },
                },
            ],
        },
    }
    rendered = _render_history_entry(entry)
    assert "spotify.show_album" in rendered
    assert "object(3 keys)" in rendered or "object(" in rendered
    # Single-dict path: helper sees the dict as one item, shows schema/value
    assert "rock" in rendered
