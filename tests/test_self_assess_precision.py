"""Self-assess is ADVISORY (does not gate the milestone; continuation §3 owns the
done judgement). These cover what still matters under advisory: the grounded
value/observed rendering (so the hint is accurate, not a truncation artifact) and
that the hint propagates to the consumers. The milestone_done-flipping backstop
was removed (it was patch-prone — a carve-out per false-flag); continuation §3
catches the cases it used to flip (mutation value=null §3b, etc.).
"""

from __future__ import annotations

from adk_appworld_agent.subagents.executor.code_plan_execute.agent import (
    SELF_ASSESS_SYS_PROMPT,
    _format_prior_attempts_block,
    _render_prior_returns,
    _render_value_for_self_assess,
)

# ── prior-attempts propagation (executor's own retry sees the hint) ──────────


def test_prior_attempts_block_renders_self_assess_hint():
    prior = [
        {
            "success": False,
            "failure_code": "EXECUTOR_MILESTONE_NOT_DONE",
            "agent_output": {
                "self_assess": {
                    "problem": "filter 'electricity bill' won't match observed 'Bill for Electricity'",
                },
                "code_plan": {"plan_steps": ["read transactions", "filter"]},
            },
        }
    ]
    block = _format_prior_attempts_block(prior)
    assert block is not None
    assert "self-assess" in block
    assert "Bill for Electricity" in block


def test_prior_attempts_block_no_self_assess_section_when_absent():
    prior = [
        {
            "success": False,
            "failure_code": "EXECUTOR_DID_NOT_FINALIZE",
            "agent_output": {"code_plan": {"plan_steps": ["x"]}},
        }
    ]
    block = _format_prior_attempts_block(prior)
    assert block is not None
    assert "self-assess" not in block


# ── value rendering (structured summary, never a raw char-cut) ───────────────


def test_render_value_large_list_shows_count_not_raw_cut():
    value = [
        {
            "id": 1000 + i,
            "name": f"Person {i}",
            "address": "123 Main St\nSeattle\nWA\nUSA\n18461",
        }
        for i in range(50)
    ]
    out = _render_value_for_self_assess(value)
    assert "list[50]" in out  # count is explicit, not a mid-object cut
    assert not out.rstrip().endswith(",")
    assert len(out) < 4000


def test_render_value_scalar_shows_full_value():
    assert _render_value_for_self_assess(144.0) == "144.0"
    assert _render_value_for_self_assess(0) == "0"


def test_render_value_null():
    assert _render_value_for_self_assess(None) == "null"


# ── observed view counts by true total (_list_total), not the display cap ────


def _trunc_call(app, api, list_total, shown_n):
    return {
        "app": app,
        "api_name": api,
        "status": "ok",
        "result_items": {
            "_list_total": list_total,
            "_truncated": list_total > shown_n,
            "items": [{"id": i} for i in range(shown_n)],
        },
    }


def test_render_prior_returns_shows_true_total_not_truncated_len():
    # _list_total=20 but only 10 captured -> the reviewer must see 20, not 10
    out = _render_prior_returns([_trunc_call("phone", "search_contacts", 20, 10)])
    assert out is not None
    assert "returned 20 item(s)" in out
    assert "display shows 10" in out


# ── prompt is advisory + guards against display-artifact misreads ────────────


def test_self_assess_prompt_is_advisory_not_a_gate():
    p = SELF_ASSESS_SYS_PROMPT.lower()
    assert "advisory" in p
    assert "does not gate" in p


def test_self_assess_prompt_forbids_inferring_truncation_from_display():
    p = SELF_ASSESS_SYS_PROMPT.lower()
    assert "structured summaries" in p
    assert "never infer" in p
    assert "list[n]" in p


def test_self_assess_prompt_has_mutation_null_exception():
    # the blanket "value null -> flag" rule (which false-flagged mutations) is gone
    p = SELF_ASSESS_SYS_PROMPT.lower()
    assert "side effect" in p or "state-changing action that correctly commits" in p


def test_self_assess_prompt_empty_is_evidence_based_not_blanket():
    """FIX-B: empty/null is flagged ok=false ONLY with positive evidence it's a
    miss (observed data contains items it should have captured) — emptiness alone
    is not a deviation (a correctly-empty result must not be flagged → would loop)."""
    p = " ".join(SELF_ASSESS_SYS_PROMPT.lower().split())  # collapse line wraps
    assert "emptiness alone is not a deviation" in p
    assert "may be the correct answer" in p
    assert "observed api returns clearly contain" in p
