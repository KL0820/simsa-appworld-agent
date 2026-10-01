"""Tests for build_analysis_viewer.assign_cycle_ns.

Two recovery paths the viewer needs to handle:

1. **New logs (post-fix)**: every PLAN/FIND/EXECUTE input carries
   metadata.cycle_n directly. Pass-through.

2. **Old logs with FIND skip (Entry 8) but no per-call cycle_n on
   FIND/EXECUTE**: PLAN trajectory recovery — derive each cycle's
   active_milestone_index out-state from the next PLAN's in-state,
   then match FIND/EXECUTE by milestone_index.

3. **Legacy fallback (PLAN cycle_n itself missing)**: phase-local
   pairing (N-th FIND ↔ N-th PLAN).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_VIEWER_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_analysis_viewer.py"
)


def _load_viewer_module():
    spec = importlib.util.spec_from_file_location(
        "build_analysis_viewer_under_test", _VIEWER_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_analysis_viewer_under_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


viewer = _load_viewer_module()
assign_cycle_ns = viewer.assign_cycle_ns


def _call(phase: str, *, cycle_n=None, active_mi=None, milestone_idx=None) -> dict:
    """Build a synthetic subagent_io.jsonl record."""
    meta: dict = {}
    if cycle_n is not None:
        meta["cycle_n"] = cycle_n
    if active_mi is not None:
        meta["active_milestone_index"] = active_mi
    if milestone_idx is not None:
        meta["milestone_index"] = milestone_idx
    return {
        "phase": phase,
        "io_record": {
            "input": {"subagent_input": {"metadata": meta}},
        },
    }


def test_new_log_all_calls_carry_cycle_n():
    calls = [
        _call("PLAN", cycle_n=0),
        _call("PLAN", cycle_n=1, active_mi=0),
        _call("FIND", cycle_n=0, milestone_idx=0),
        _call("EXECUTE", cycle_n=0, milestone_idx=0),
        _call("EXECUTE", cycle_n=1, milestone_idx=0),
    ]
    assert assign_cycle_ns(calls) == [0, 1, 0, 0, 1]


def test_find_skip_no_per_call_cycle_n_on_find_or_exec():
    """The actual 9dabbc9_2 shape from input_hygiene_validation_v2:
    9 PLANs (cycles 0-8), 4 FINDs (M0/M1/M2/M3), 8 EXECUTEs (M0×5 + M1 + M2 + M3).
    M0 RETRYed cycles 0-4, ADVANCEd at cycle 5/6/7. cycle 8 aborted before
    FIND/EXECUTE.

    Expected mapping:
      FIND m0 → cycle 0   FIND m1 → cycle 5   FIND m2 → cycle 6   FIND m3 → cycle 7
      EXECUTE m0 (x5) → cycles 0/1/2/3/4
      EXECUTE m1 → cycle 5   EXECUTE m2 → cycle 6   EXECUTE m3 → cycle 7
    """
    calls = [
        _call("PLAN", cycle_n=0),  # rough_planner, no active_mi
        _call("PLAN", cycle_n=1, active_mi=0),
        _call("PLAN", cycle_n=2, active_mi=0),
        _call("PLAN", cycle_n=3, active_mi=0),
        _call("PLAN", cycle_n=4, active_mi=0),
        _call("PLAN", cycle_n=5, active_mi=0),  # decides ADVANCE → cycle 6 sees M1
        _call("PLAN", cycle_n=6, active_mi=1),  # decides ADVANCE → cycle 7 sees M2
        _call("PLAN", cycle_n=7, active_mi=2),  # decides ADVANCE → cycle 8 sees M3
        _call("PLAN", cycle_n=8, active_mi=3),  # final cycle, no FIND/EXEC
        # FIND group (file-order: M0, M1, M2, M3) — no per-call cycle_n
        _call("FIND", milestone_idx=0),
        _call("FIND", milestone_idx=1),
        _call("FIND", milestone_idx=2),
        _call("FIND", milestone_idx=3),
        # EXECUTE group (file-order chronological per phase)
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=1),
        _call("EXECUTE", milestone_idx=2),
        _call("EXECUTE", milestone_idx=3),
    ]
    expected = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,  # PLAN cycles
        0,
        5,
        6,
        7,  # FIND cycles
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,  # EXECUTE cycles
    ]
    assert assign_cycle_ns(calls) == expected


def test_legacy_fallback_when_plan_cycle_n_missing():
    """If even PLAN inputs lack cycle_n (very old log format), fall back to
    phase-local pairing (N-th FIND ↔ N-th PLAN). This is the pre-Entry-8
    behavior."""
    calls = [
        _call("PLAN"),  # cycle_n=None
        _call("PLAN"),
        _call("FIND"),
        _call("FIND"),
        _call("EXECUTE"),
        _call("EXECUTE"),
    ]
    out = assign_cycle_ns(calls)
    # All None — but the structure should still iterate correctly without
    # IndexError. The legacy path returns None entries since plan_cycles is
    # all None.
    assert out == [None] * 6


def test_pre_entry8_log_multiple_finds_on_same_milestone():
    """Pre-Entry-8 logs: FIND ran every cycle. 10 cycles all RETRYing on M0
    means 10 FIND m0 calls. Cycle queue for M0 should be [0..9]; each FIND
    pops sequentially → cycles 0,1,2,...,9 (matches old phase-local pairing).
    """
    calls = [_call("PLAN", cycle_n=0)]
    for k in range(1, 10):
        calls.append(_call("PLAN", cycle_n=k, active_mi=0))
    for _ in range(10):
        calls.append(_call("FIND", milestone_idx=0))
    for _ in range(10):
        calls.append(_call("EXECUTE", milestone_idx=0))

    out = assign_cycle_ns(calls)
    assert out[:10] == list(range(10))  # PLAN
    assert out[10:20] == list(range(10))  # FIND
    assert out[20:30] == list(range(10))  # EXECUTE


def test_pure_no_skip_case_recovery_matches_phase_local():
    """No FIND skipped → trajectory recovery should produce same answer as
    legacy phase-local pairing."""
    calls = [
        _call("PLAN", cycle_n=0),
        _call("PLAN", cycle_n=1, active_mi=0),  # ADVANCE
        _call("PLAN", cycle_n=2, active_mi=1),  # ADVANCE
        _call("FIND", milestone_idx=0),
        _call("FIND", milestone_idx=1),
        _call("FIND", milestone_idx=2),
        _call("EXECUTE", milestone_idx=0),
        _call("EXECUTE", milestone_idx=1),
        _call("EXECUTE", milestone_idx=2),
    ]
    # Each cycle had its own FIND — first_cycle_for_mi: {0:0, 1:1, 2:2}
    expected = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert assign_cycle_ns(calls) == expected
