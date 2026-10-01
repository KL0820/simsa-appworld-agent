"""M2 (P2): the continuation prompt must HONOR the executor self_assess.

The audit (`wgn420qt5`) found the self_assess content was accurate but the
continuation overrode it with phase-level `success=true`. P2 flips that:
`self_assess.ok=False` is a high-priority override of `success=true` — the
continuation must not ADVANCE / SUBMIT, must route on `problem`, and must
still respect the documented false-positive classes (§3b mutation, §3d
under-fetch, pagination). These pin the prompt contract.
"""

from __future__ import annotations

from adk_appworld_agent.subagents.continuer.continuation.prompts import (
    CONTINUATION_SYSTEM_PROMPT,
)

PROMPT = CONTINUATION_SYSTEM_PROMPT
LOW = PROMPT.lower()


def test_ok_false_is_high_priority_override_of_success():
    # The §3g section exists and frames ok=False as overriding success=true.
    assert "§3g" in PROMPT
    assert "high-priority override of success=true" in LOW
    # Must explicitly forbid ADVANCE/SUBMIT on an ok=False milestone.
    assert "do not advance or submit" in LOW


def test_success_true_not_equal_milestone_done_is_stated():
    # §3 intro must decouple success=true from "milestone accomplished".
    assert 'is not the same as "milestone accomplished' in LOW


def test_routes_on_problem_three_classes():
    # The three routing arms must be present (capability / upstream state / shape).
    assert "needed capability the candidate apis lack" in LOW
    assert "missing upstream value / state" in LOW
    assert "data-shape mismatch" in LOW


def test_mutation_false_positive_exception_preserved():
    # §3b mutation-null must remain a documented false positive (do NOT block ADVANCE).
    assert "documented false positives" in LOW
    assert "mutation milestone" in LOW
    assert "value=null" in LOW
    # under-fetch + pagination false positives also preserved.
    assert "under-fetch" in LOW
    assert "pagination" in LOW


def test_never_declares_impossible():
    # Keystone: keep re-routing toward the solvable path; never give up.
    assert "never declare the task impossible" in LOW


def test_empty_value_is_evidence_based_not_blanket_failure():
    """FIX-B: §3b must not treat emptiness ALONE as failure — only when api_trace
    shows the source data did contain items the milestone should have captured.
    A correctly-empty result must not trigger RETRY (else it loops forever)."""
    assert "emptiness alone is not failure" in LOW
    assert "may be the correct answer" in LOW


def test_no_task_specific_api_or_app_names_in_override_rule():
    # AGENTS.md: the override rule stays layer-level — it routes by reworording
    # the OPERATION and explicitly defers API choice to the finder (§1a).
    assert "never name the api" in LOW or "do not name an api" in LOW
