"""Part B: structural no-progress fingerprint + bounded give-up.

Covers the controller's no-progress detector (the trigger Guard 4/5 miss because
they ignore structural content) and the clean give-up that replaces "never-
succeeded last milestone -> RETRY -> burn to max_cycles + the 429 storm".
"""

from __future__ import annotations

from adk_appworld_agent._controller_helpers import (
    _ensure_milestone_ids,
    _interpret_plan_envelope,
    _milestones_from_plan_payload,
)
from adk_appworld_agent.agent import _execute_fingerprint, _update_no_progress
from adk_appworld_agent.contracts.plan_ir import Milestone
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import (
    CycleHistoryEntry,
    Phase,
    RunState,
    RunStatus,
)
from adk_appworld_agent.orchestration.state_repo import RunStateRepository
from tests.helpers import run_agent
from tests.test_cycle_loop import (
    _build_test_agent,
    _MultiMilestonePlannerStub,
    _run_with_scripted_continuation,
)

# ── stable milestone id ownership (§4.1) ─────────────────────────────────────


def test_ensure_milestone_ids_assigns_monotonic_stable_ids():
    state = RunState()
    state.milestones = _milestones_from_plan_payload(
        {"milestones": [{"intent": "a"}, {"intent": "b"}, {"intent": "c"}]}
    )
    _ensure_milestone_ids(state)
    assert [m.id for m in state.milestones] == ["ms0", "ms1", "ms2"]
    assert state.next_milestone_uid == 3


def test_cold_start_milestones_strip_planner_emitted_id():
    # A planner that leaks an id must not be able to set milestone identity.
    ms = _milestones_from_plan_payload(
        {"milestones": [{"intent": "a", "id": "planner-leak"}]}
    )
    assert ms[0].id is None


def test_interpret_plan_envelope_strips_llm_emitted_id_from_revise():
    state = RunState()
    env = SubagentEnvelope(
        phase=Phase.PLAN,
        subagent_name="continuation",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "next_action": "RETRY",
            "revised_milestones": [{"intent": "reworded", "id": "llm-made-this-up"}],
            "rationale": "r",
        },
    )
    _decision, revised = _interpret_plan_envelope(env, state, cycle_n=2)
    assert revised is not None
    assert revised[0].id is None  # controller owns ids; LLM id ignored


def test_revise_loop_milestone_ids_are_controller_minted_not_llm():
    """End-to-end through the real cycle loop: even when the continuation emits
    its own milestone ids in revised_milestones, the controller strips them and
    mints/preserves its own stable ids — so the attempt count stays keyed on a
    controller-owned id the LLM cannot perturb (§4.1)."""
    revise = {
        "next_action": "RETRY",
        "revised_milestones": [
            {"id": "llm-a", "intent": "stuck (revised)"},
            {"id": "llm-b", "intent": "downstream"},
        ],
        "rationale": "revise",
    }
    state = _run_with_scripted_continuation([revise] * 8)
    ids = [m.id for m in state.milestones]
    assert ids and all(i and i.startswith("ms") for i in ids), ids
    assert "llm-a" not in ids and "llm-b" not in ids


# ── §4.3 bounded prerequisite insertion ──────────────────────────────────────


def test_prereq_insertion_allows_plus_one_growth_and_preserves_active_id():
    """A RETRY revise of the form [prerequisite, active] — exactly +1 over the
    writable region — is PERMITTED: the prerequisite is inserted before the active
    milestone (fresh id) and the active milestone is preserved (shifted by one,
    keeping its stable id so its attempt count survives)."""
    insertion = {
        "next_action": "RETRY",
        "revised_milestones": [
            {"intent": "prerequisite: establish upstream state"},
            {"intent": "step 2 active preserved"},
        ],
        "rationale": "missing prerequisite before the active milestone",
    }
    decisions = [
        {
            "next_action": "ADVANCE",
            "revised_milestones": None,
            "rationale": "fwd",
        },  # m0->m1
        {
            "next_action": "ADVANCE",
            "revised_milestones": None,
            "rationale": "fwd",
        },  # m1->m2 (last)
        insertion,  # insert before m2
        {
            "next_action": "ADVANCE",
            "revised_milestones": None,
            "rationale": "fwd",
        },  # prereq->m2
        {
            "next_action": "SUBMIT",
            "revised_milestones": None,
            "rationale": "done",
        },  # submit m2
    ]
    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    assert len(state.milestones) == 4, intents
    assert "prerequisite" in intents[2]
    assert intents[3] == "step 2 active preserved"
    # The preserved active kept its original controller id (ms2); prereq got a fresh one.
    assert state.milestones[3].id == "ms2"
    assert state.milestones[2].id != "ms2" and state.milestones[2].id.startswith("ms")
    assert state.prereq_insertions.get("ms2") == 1
    assert state.status == RunStatus.COMPLETED


def test_prereq_insertion_capped_per_milestone():
    """The +1 prerequisite insertion is capped per stable milestone id; once the
    cap is spent, a further +1 insertion before the SAME milestone is rejected as
    growth (prevents a slow one-per-cycle unroll dressed up as prerequisites)."""
    set_active_config(RunConfig(prereq_insertion_cap=1))
    try:
        insertion = {
            "next_action": "RETRY",
            "revised_milestones": [
                {"intent": "prereq step"},
                {"intent": "active preserved"},
            ],
            "rationale": "insert prereq",
        }
        decisions = [
            {
                "next_action": "ADVANCE",
                "revised_milestones": None,
                "rationale": "fwd",
            },  # m0->m1
            {
                "next_action": "ADVANCE",
                "revised_milestones": None,
                "rationale": "fwd",
            },  # m1->m2 (last)
            insertion,  # 1st: permitted (cap=1)
            {
                "next_action": "ADVANCE",
                "revised_milestones": None,
                "rationale": "fwd",
            },  # prereq->m2
            insertion,  # 2nd: cap spent -> rejected
            {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
            {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
        ]
        state = _run_with_scripted_continuation(decisions)
    finally:
        set_active_config(None)
    # Only ONE prerequisite inserted (plan 3 -> 4, never 5).
    assert len(state.milestones) == 4, [m.intent for m in state.milestones]
    assert state.prereq_insertions.get("ms2") == 1


def test_prereq_insertion_disabled_when_cap_zero():
    """prereq_insertion_cap=0 disables insertion: a +1 [prereq, active] revise is
    then rejected as growth (pure in-place-refine mode, ablation control)."""
    set_active_config(RunConfig(prereq_insertion_cap=0))
    try:
        decisions = [
            {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
            {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
            {
                "next_action": "RETRY",
                "revised_milestones": [
                    {"intent": "prereq step"},
                    {"intent": "active preserved"},
                ],
                "rationale": "insert prereq",
            },
            {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
        ]
        state = _run_with_scripted_continuation(decisions)
    finally:
        set_active_config(None)
    assert len(state.milestones) == 3, [m.intent for m in state.milestones]


# ── fingerprint ────────────────────────────────────────────────────────────


def test_fingerprint_identical_for_same_structural_inputs():
    a = _execute_fingerprint(
        ["app.x", "app.y"], [{"app": "app", "api_name": "x"}], [1, 2]
    )
    b = _execute_fingerprint(
        ["app.y", "app.x"], [{"app": "app", "api_name": "x"}], [1, 2]
    )
    assert a == b  # candidate order does not matter


def test_fingerprint_differs_on_distinct_value_552869a_canary():
    """552869a converged via DISTINCT committed values 7134.0 -> 0.0 -> correct.
    Exact-value hashing must keep them apart so no-progress never fires on a
    genuinely-progressing task."""
    apis = ["venmo.show_transactions"]
    trace = [{"app": "venmo", "api_name": "show_transactions"}]
    fps = {
        _execute_fingerprint(apis, trace, 7134.0),
        _execute_fingerprint(apis, trace, 0.0),
        _execute_fingerprint(apis, trace, 14.32),
    }
    assert len(fps) == 3


def test_fingerprint_differs_when_different_apis_called():
    a = _execute_fingerprint(["app.x"], [{"app": "app", "api_name": "x"}], None)
    b = _execute_fingerprint(["app.x"], [{"app": "app", "api_name": "y"}], None)
    assert a != b


# ── no-progress streak ─────────────────────────────────────────────────────


def test_update_no_progress_fires_at_window():
    state = RunState()
    fp = _execute_fingerprint(["a"], [], None)
    fired = [_update_no_progress(state, "ms0", fp, window=4) for _ in range(4)]
    assert fired == [False, False, False, True]
    assert state.no_progress_streak == 4


def test_update_no_progress_does_not_fire_when_attempts_differ():
    """The 552869a shape: distinct values each attempt -> streak never builds."""
    state = RunState()
    apis, trace = ["a"], [{"app": "a", "api_name": "x"}]
    fired = [
        _update_no_progress(
            state, "ms0", _execute_fingerprint(apis, trace, v), window=4
        )
        for v in (1, 2, 3, 4, 5)
    ]
    assert not any(fired)
    assert state.no_progress_streak == 1


def test_update_no_progress_fires_at_attempt_budget_despite_changing_fingerprints():
    """The continuation-revise loop (2c544f9): each revise re-routes the finder
    so the fingerprint changes every cycle -> the identical streak never builds
    and the window trigger never fires. The attempt_budget fires on the Nth
    attempt regardless of fingerprint, bounding the revise loop instead of
    revising to MAX_CYCLES (which burned the project token quota). Keying on the
    STABLE milestone id (not index) is what makes this count survive the
    re-routes — the id is unchanged by a re-word."""
    state = RunState()
    apis, trace = ["a"], [{"app": "a", "api_name": "x"}]
    fired = [
        _update_no_progress(
            state,
            "ms0",
            _execute_fingerprint(apis, trace, v),
            window=4,
            attempt_budget=6,
        )
        for v in range(10)
    ]
    assert fired[:5] == [False] * 5  # attempts 1-5: under the budget
    assert fired[5] is True  # 6th attempt: attempt-budget trigger fires
    assert state.no_progress_streak == 1  # fingerprints all differ -> no streak


def test_update_no_progress_attempt_budget_zero_disables_attempt_trigger():
    """attempt_budget=0 disables the total-attempt trigger: differing
    fingerprints then never give up via the budget (window-only behavior)."""
    state = RunState()
    apis, trace = ["a"], [{"app": "a", "api_name": "x"}]
    fired = [
        _update_no_progress(
            state,
            "ms0",
            _execute_fingerprint(apis, trace, v),
            window=4,
            attempt_budget=0,
        )
        for v in range(10)
    ]
    assert not any(fired)


def test_update_no_progress_resets_on_stable_id_change_not_index():
    """The count is keyed by the STABLE milestone id, so it resets only when the
    active milestone's IDENTITY changes — NOT merely its index. A re-word keeps
    the id (count accumulates); a genuinely-different milestone (e.g. an inserted
    prerequisite) has a new id (count resets), which is what keeps a stuck
    milestone's budget from being falsely reset by an insertion (§4.1)."""
    state = RunState()
    fp = _execute_fingerprint(["a"], [], None)
    _update_no_progress(state, "ms0", fp, window=4)
    _update_no_progress(state, "ms0", fp, window=4)
    assert state.no_progress_streak == 2
    # A milestone with a DIFFERENT stable id -> history resets.
    fired = _update_no_progress(state, "ms1", fp, window=4)
    assert fired is False
    assert state.fingerprint_milestone_id == "ms1"
    assert state.active_milestone_fingerprints == [fp]
    assert state.no_progress_streak == 1


def test_update_no_progress_same_id_survives_reword_index_unchanged():
    """A re-word changes the fingerprint (re-route) but NOT the stable id, so the
    attempt count keeps accumulating across re-words toward the budget."""
    state = RunState()
    apis, trace = ["a"], [{"app": "a", "api_name": "x"}]
    # 6 distinct fingerprints (re-routes) all under the SAME stable id "ms0".
    fired = [
        _update_no_progress(
            state,
            "ms0",
            _execute_fingerprint(apis, trace, v),
            window=4,
            attempt_budget=6,
        )
        for v in range(6)
    ]
    assert fired[-1] is True
    assert len(state.active_milestone_fingerprints) == 6


# ── give-up FSM ─────────────────────────────────────────────────────────────


def test_cycle_loop_no_progress_give_up_last_milestone_clean_failed():
    """Failing executor on a single milestone: the structurally-identical loop
    gives up cleanly at the window instead of burning to max_cycles."""
    set_active_config(RunConfig(give_up_enabled=True, no_progress_fingerprint_window=4))
    try:
        agent = _build_test_agent(fail_phase=Phase.EXECUTE, max_cycles=15)
        session, _ = run_agent(agent)
    finally:
        set_active_config(None)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.FAILED
    assert state.phase == Phase.FAILED
    assert state.economic_stop_reason == "no_progress"
    # completion_block_reason keeps the real executor failure for attribution;
    # the economic stop is recorded separately in economic_stop_reason.
    assert state.completion_block_reason == "EXECUTE_STUB_FAILURE"
    # Fired at the window (4), well before max_cycles (15).
    assert state.cycle_n == 4


def test_cycle_loop_give_up_disabled_falls_back_to_max_cycles():
    """With give_up_enabled=False the old loop-to-wall behaviour holds (ablation
    control)."""
    set_active_config(RunConfig(give_up_enabled=False))
    try:
        agent = _build_test_agent(fail_phase=Phase.EXECUTE, max_cycles=3)
        session, _ = run_agent(agent)
    finally:
        set_active_config(None)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.FAILED
    assert state.economic_stop_reason is None
    assert state.completion_block_reason == "EXECUTE_STUB_FAILURE"
    assert state.cycle_n >= 3


def test_cycle_loop_no_progress_advances_downstream_milestone():
    """A no-progress give-up on a non-last milestone ADVANCEs (reallocates budget
    downstream) rather than terminating."""
    set_active_config(RunConfig(give_up_enabled=True, no_progress_fingerprint_window=4))
    try:
        agent = _build_test_agent(
            planner=_MultiMilestonePlannerStub(name="mm_planner"),
            fail_phase=Phase.EXECUTE,
            max_cycles=30,
        )
        session, _ = run_agent(agent)
    finally:
        set_active_config(None)

    state = RunStateRepository().load(session.state, default_instruction="")
    # It must have ADVANCED past at least m0 (active reached a later milestone or
    # the run terminated on the LAST milestone via the economic stop).
    assert state.economic_stop_reason == "no_progress"
    assert state.active_milestone_index >= 1


# ── per-task total-work backstop (decompose-immune) ──────────────────────────


def test_task_attempt_budget_fires_before_per_milestone_and_max_cycles():
    """The whole-task EXECUTE-attempt budget triggers economic-stop independently
    of the per-milestone streak/budget — the decompose-immune backstop. Set
    task_attempt_budget=3 below the streak window(4) so it fires first."""
    set_active_config(
        RunConfig(
            give_up_enabled=True,
            no_progress_fingerprint_window=4,
            attempt_budget=6,
            task_attempt_budget=3,
        )
    )
    try:
        agent = _build_test_agent(fail_phase=Phase.EXECUTE, max_cycles=30)
        session, _ = run_agent(agent)
    finally:
        set_active_config(None)
    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.economic_stop_reason == "task_attempt_budget"
    assert state.total_execute_attempts == 3
    assert state.cycle_n < 4  # fired before the streak window / max_cycles


def test_task_attempt_budget_zero_disables_backstop():
    """task_attempt_budget=0 disables the per-task backstop: the per-milestone
    streak (window) is then what terminates (control)."""
    set_active_config(
        RunConfig(
            give_up_enabled=True,
            no_progress_fingerprint_window=4,
            attempt_budget=6,
            task_attempt_budget=0,
        )
    )
    try:
        agent = _build_test_agent(fail_phase=Phase.EXECUTE, max_cycles=30)
        session, _ = run_agent(agent)
    finally:
        set_active_config(None)
    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.economic_stop_reason == "no_progress"  # streak fired, not task-budget


# ── FIX-A: prior_attempts keyed by stable milestone_id (survives index shift) ─


def test_prior_attempts_keyed_by_stable_id_survives_index_shift():
    """A prerequisite insertion shifts the active milestone's index; the executor's
    prior attempts (incl. self_assess) must still be gathered by stable id, NOT
    index — else the executor cold-starts and repeats the bug (3d9a636)."""
    agent = _build_test_agent()
    state = RunState()
    # "ms_target" is the stuck step, now at index 1 (a prereq was inserted at 0).
    state.milestones = [
        Milestone(id="ms_prereq", intent="inserted prerequisite"),
        Milestone(id="ms_target", intent="the stuck step"),
    ]
    state.active_milestone_index = 1
    # It was attempted earlier at index 0 (PRE-shift) AND at index 1 (post-shift).
    state.history = [
        CycleHistoryEntry(
            cycle_n=1,
            phase_completed="EXECUTE",
            milestone_index=0,
            milestone_id="ms_target",
            success=False,
            agent_output={"self_assess": {"problem": "loop-ordering bug"}},
        ),
        CycleHistoryEntry(
            cycle_n=2,
            phase_completed="EXECUTE",
            milestone_index=1,
            milestone_id="ms_target",
            success=False,
            agent_output={"self_assess": {"problem": "same loop bug again"}},
        ),
    ]
    inp = agent._build_exec_input(state, {})
    prior = inp.metadata.get("prior_attempts_for_this_milestone") or []
    # BOTH attempts present (index-keying would have dropped the index-0 one).
    assert len(prior) == 2
    assert "loop-ordering bug" in str(prior)
