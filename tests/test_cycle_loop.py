from __future__ import annotations

from typing import AsyncIterator, ClassVar

import pytest

from adk_appworld_agent.agent import (
    AppWorldAgent,
    _interpret_plan_envelope,
    build_appworld_controller_agent,
)
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.orchestrator import Orchestrator
from adk_appworld_agent.orchestration.state import (
    Phase,
    RunState,
    RunStatus,
    TaskContext,
)
from adk_appworld_agent.orchestration.state_repo import RunStateRepository
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
from adk_appworld_agent.subagents.base import BaseSubagent
from adk_appworld_agent.subagents.continuer.continuation import (
    build_continuation_stub_subagent,
)
from adk_appworld_agent.subagents.stubs import build_stub_subagent
from tests.helpers import run_agent


@pytest.fixture(autouse=True)
def _reset_active_config():
    yield
    set_active_config(None)


def _make_state(cycle_n: int = 0, milestones: list | None = None) -> RunState:
    return RunState(
        cycle_n=cycle_n,
        milestones=milestones or [],
        task_context=TaskContext(instruction="x"),
    )


def _envelope(phase: Phase, payload: dict) -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=phase,
        subagent_name="stub",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload=payload,
    )


def test_interpret_cycle0_milestones_payload():
    env = _envelope(
        Phase.PLAN,
        {"milestones": [{"id": "m1", "intent": "do x"}]},
    )
    decision, milestones = _interpret_plan_envelope(env, _make_state(), cycle_n=0)
    # Cold-start uses RETRY so the routing block keeps active=0 (entering m0);
    # ADVANCE would increment active to 1 and skip m0.
    assert decision.next_action == "RETRY"
    assert milestones is not None
    assert milestones[0].intent == "do x"


def test_interpret_cycle0_tasks_payload_falls_back_to_rough_planner_format():
    env = _envelope(
        Phase.PLAN,
        {"tasks": [{"task": "find email", "app": "gmail"}]},
    )
    decision, milestones = _interpret_plan_envelope(env, _make_state(), cycle_n=0)
    assert decision.next_action == "RETRY"
    assert milestones is not None
    assert milestones[0].intent == "find email"
    assert milestones[0].app == "gmail"


def test_interpret_cycle0_empty_payload_aborts():
    env = _envelope(Phase.PLAN, {})
    decision, milestones = _interpret_plan_envelope(env, _make_state(), cycle_n=0)
    assert decision.next_action == "ABORT"
    assert milestones is None


def test_interpret_continuation_advance():
    env = _envelope(
        Phase.PLAN,
        {
            "next_action": "ADVANCE",
            "revised_milestones": None,
            "rationale": "next",
        },
    )
    decision, milestones = _interpret_plan_envelope(
        env, _make_state(cycle_n=1), cycle_n=1
    )
    assert decision.next_action == "ADVANCE"
    assert milestones is None


def test_interpret_continuation_submit():
    env = _envelope(
        Phase.PLAN,
        {"next_action": "SUBMIT", "rationale": ""},
    )
    decision, _ = _interpret_plan_envelope(env, _make_state(cycle_n=2), cycle_n=2)
    assert decision.next_action == "SUBMIT"


def test_interpret_continuation_retry_with_revised_milestones():
    env = _envelope(
        Phase.PLAN,
        {
            "next_action": "RETRY",
            "revised_milestones": [{"id": "m1", "intent": "new step"}],
            "rationale": "revise active",
        },
    )
    decision, milestones = _interpret_plan_envelope(
        env, _make_state(cycle_n=1), cycle_n=1
    )
    assert decision.next_action == "RETRY"
    assert milestones is not None
    assert milestones[0].intent == "new step"


def test_interpret_unknown_verb_degrades_to_retry():
    """Unknown / malformed next_action values are degraded to RETRY (not
    ABORT) on 2026-05-17 onward. ABORT was removed from the continuation
    action space — every AppWorld task should run to natural wall-budget
    timeout rather than self-terminate via LLM. A parse failure shouldn't
    silently terminate the task either."""
    env = _envelope(
        Phase.PLAN,
        {"next_action": "ADVANCE_TO_FIND", "rationale": "legacy"},
    )
    decision, _ = _interpret_plan_envelope(env, _make_state(cycle_n=1), cycle_n=1)
    assert decision.next_action == "RETRY"


def test_interpret_abort_decision_is_redirected_to_retry():
    """If a continuation_planner LLM hallucinates ABORT despite the
    prompt no longer offering it, the framework MUST redirect to RETRY.
    The wall budget is the only legitimate task-stop signal; LLM-driven
    early termination is structurally disabled."""
    env = _envelope(
        Phase.PLAN,
        {"next_action": "ABORT", "rationale": "I think this task is impossible"},
    )
    decision, _ = _interpret_plan_envelope(env, _make_state(cycle_n=1), cycle_n=1)
    assert decision.next_action == "RETRY", (
        "ABORT must be redirected to RETRY by _interpret_plan_envelope so "
        "the wall budget — not the LLM — decides task termination"
    )


class _MultiMilestonePlannerStub(BaseSubagent):
    """Stub cycle-0 planner that emits a multi-milestone plan."""

    phase_: ClassVar[Phase] = Phase.PLAN

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        payload = {
            "milestones": [
                {
                    "id": f"m{i}",
                    "intent": f"step {i}",
                }
                for i in range(3)
            ]
        }
        yield self.succeeded(attempt=subagent_input.attempt, payload=payload)


class _FakeSubmitter:
    def submit_answer(self, answer: str):
        return {
            "result": "Num Passed Tests : 1\nNum Failed Tests : 0\nNum Total Tests : 1"
        }


def _build_test_agent(
    *,
    planner: BaseSubagent | None = None,
    fail_phase: Phase | None = None,
    max_cycles: int = 15,
) -> AppWorldAgent:
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    planner = planner or build_stub_subagent(
        name="planner_subagent_stub", phase=Phase.PLAN
    )
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = (
        build_stub_subagent(
            name="executor_subagent_stub",
            phase=Phase.EXECUTE,
            stub_status=SubagentStatus.FAILED,
        )
        if fail_phase == Phase.EXECUTE
        else build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)
    )
    continuation = build_continuation_stub_subagent()
    return AppWorldAgent(
        name="appworld_controller_agent",
        description="cycle-loop test agent",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=max_cycles,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )


def test_cycle_loop_completes_three_milestones_with_stub_continuation():
    agent = _build_test_agent(planner=_MultiMilestonePlannerStub(name="mm_planner"))
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.COMPLETED
    assert state.phase == Phase.COMPLETE
    # cycle 0 plan + 3 EXECUTE successes (one per milestone) + 1 final cycle for SUBMIT_NOW.
    assert state.cycle_n == 4
    # history: cycle 0 PLAN/FIND/EXECUTE, cycle 1 PLAN/FIND/EXECUTE, cycle 2 PLAN/FIND/EXECUTE,
    # cycle 3 PLAN(SUBMIT_NOW)/SUBMIT
    exec_entries = [
        h for h in state.history if h.phase_completed == "EXECUTE" and h.success
    ]
    assert len(exec_entries) == 3
    assert [h.milestone_index for h in exec_entries] == [0, 1, 2]
    submit_entries = [h for h in state.history if h.phase_completed == "SUBMIT"]
    assert len(submit_entries) == 1
    assert submit_entries[0].success is True


def test_cycle_loop_envelope_attempt_monotonic_across_cycles():
    """Per-phase `attempt` must increment with each invocation across cycles.

    Without this, `(phase, subagent_name, attempt)` dedup in the stdout
    progress printer collapses every cycle's worker summary into one line.
    """

    agent = _build_test_agent(planner=_MultiMilestonePlannerStub(name="mm_planner"))
    session, _ = run_agent(agent)

    raw_outputs = session.state.get("subagent_outputs", {})
    for phase_name in ("PLAN", "FIND", "EXECUTE"):
        attempts = [item.get("attempt") for item in raw_outputs.get(phase_name, [])]
        assert attempts == list(range(1, len(attempts) + 1)), (
            f"{phase_name} attempts not monotonic: {attempts}"
        )
        # Ensure dedup key (phase, name, attempt) is unique per envelope.
        keys = {
            (phase_name, item.get("subagent_name"), item.get("attempt"))
            for item in raw_outputs.get(phase_name, [])
        }
        assert len(keys) == len(raw_outputs.get(phase_name, []))


def test_cycle_loop_single_milestone_happy_path():
    agent = _build_test_agent()  # default stub planner -> 1 milestone
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.COMPLETED
    assert state.phase == Phase.COMPLETE
    # cycle 0 (PLAN m_stub + FIND + EXECUTE) + cycle 1 (continuation -> SUBMIT_NOW)
    assert state.cycle_n == 2


def test_cycle_loop_max_cycles_caps_runaway_loop():
    # Executor always fails -> continuation always advances/retries -> max_cycles fires.
    agent = _build_test_agent(fail_phase=Phase.EXECUTE, max_cycles=3)
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.FAILED
    assert state.phase == Phase.FAILED
    # cycle_n stopped at max_cycles
    assert state.cycle_n >= 3
    # The recent cycle's executor failure is preserved over the MAX_CYCLES marker.
    assert state.completion_block_reason == "EXECUTE_STUB_FAILURE"


def test_cycle_loop_planner_failure_terminates_run():
    agent = build_appworld_controller_agent(
        impls={Phase.FIND: "stub", Phase.PLAN: "stub", Phase.EXECUTE: "stub"},
        fail_phase=Phase.PLAN,
        submitter_provider=_FakeSubmitter,
    )
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    assert state.status == RunStatus.FAILED
    assert state.completion_block_reason == "PLAN_STUB_FAILURE"
    # cycle 0 PLAN failure should land in history.
    assert any(
        h.phase_completed == "PLAN" and h.success is False for h in state.history
    )


def test_cycle_loop_continuation_invoked_every_cycle_ge_1():
    """Cycle 0 calls rough planner; cycle >= 1 ALWAYS invokes continuation
    (no more deterministic short-circuit). Under cuga-aligned routing,
    success status doesn't guarantee milestone done — continuation must
    read the summary text to judge whether to advance / retry / submit."""

    class _SpyPlanner(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.PLAN
        invocation_count: ClassVar[list[int]] = [0]

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            type(self).invocation_count[0] += 1
            yield self.succeeded(
                attempt=subagent_input.attempt,
                payload={
                    "milestones": [
                        {"id": "m1", "intent": "x"},
                        {"id": "m2", "intent": "y"},
                    ]
                },
            )

    class _SpyContinuation(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.PLAN
        invocation_count: ClassVar[list[int]] = [0]

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            type(self).invocation_count[0] += 1
            metadata = subagent_input.metadata or {}
            active_idx = metadata.get("active_milestone_index", 0)
            milestones = metadata.get("milestones") or []
            history = metadata.get("history") or []
            active_succeeded = False
            for h in reversed(history):
                if (
                    h.get("milestone_index") == active_idx
                    and h.get("phase_completed") == "EXECUTE"
                ):
                    active_succeeded = bool(h.get("success"))
                    break
            last_idx = max(0, len(milestones) - 1)
            if active_succeeded and active_idx >= last_idx:
                action = "SUBMIT"
            elif active_succeeded:
                action = "ADVANCE"
            else:
                action = "RETRY"
            yield self.succeeded(
                attempt=subagent_input.attempt,
                payload={
                    "next_action": action,
                    "revised_milestones": None,
                    "rationale": "",
                },
            )

    _SpyPlanner.invocation_count[0] = 0
    _SpyContinuation.invocation_count[0] = 0

    planner = _SpyPlanner(name="spy_planner")
    continuation = _SpyContinuation(name="spy_continuation")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)

    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="spy test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    run_agent(agent)

    # Exactly one call to the cold-start planner.
    assert _SpyPlanner.invocation_count[0] == 1
    # Continuation LLM runs at the start of every cycle >= 1. For a
    # 2-milestone happy path: cycle 1 (advance m0 -> m1), cycle 2
    # (advance from m1 -> SUBMIT_NOW) = 2 continuation invocations.
    assert _SpyContinuation.invocation_count[0] == 2


def test_executor_failure_diagnostics_extract_code_and_stdout():
    """Failure history entries must carry the executor's code + stdout + parse_error
    so the continuation LLM can diagnose the bug instead of looping."""
    from adk_appworld_agent.agent import _executor_failure_diagnostics

    envelope = SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={
            "finalize_called": False,
            "code_execute": {
                "code": "import io\nx = io.BytesIO()\nprint(x)",
                "parse_error": "no valid JSON stdout with 'value' key",
                "raw_stdout": "Execution failed. Traceback:\nUsage of the following module is not allowed: io.",
                "llm_raised": None,
                "tool_call_count": 1,
                "repair_attempts": [
                    {"attempt": 1, "code": "..."},
                    {"attempt": 2, "code": "..."},
                ],
            },
        },
    )

    diag = _executor_failure_diagnostics(envelope)
    assert diag is not None
    assert "io.BytesIO" in diag["code"]
    assert "no valid JSON stdout" in diag["parse_error"]
    assert (
        "io is not allowed" in diag["stdout_excerpt"]
        or "module is not allowed" in diag["stdout_excerpt"]
    )
    assert diag["repair_attempt_count"] == 2
    assert diag["tool_call_count"] == 1


def test_executor_failure_diagnostics_handles_missing_code_execute():
    """Synthetic-failure envelopes (e.g. SUBAGENT_OUTPUT_MISSING) lack code_execute —
    helper must return None rather than crash."""
    from adk_appworld_agent.agent import _executor_failure_diagnostics

    envelope = SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.FAILED,
        payload={},
        failure_code="SUBAGENT_OUTPUT_MISSING",
    )
    assert _executor_failure_diagnostics(envelope) is None


def test_cycle_loop_failure_history_carries_code():
    """End-to-end: when EXECUTE fails, the appended CycleHistoryEntry's
    agent_output dict carries the failed code + stdout for the next cycle's
    continuation LLM to inspect."""

    class _FailingExecutorStub(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.EXECUTE

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            yield self.succeeded(
                attempt=subagent_input.attempt,
                payload={
                    "finalize_called": False,
                    "code_execute": {
                        "code": "import io\nprint(io)",
                        "parse_error": "no valid JSON stdout",
                        "raw_stdout": "Usage of the following module is not allowed: io.",
                        "tool_call_count": 1,
                    },
                },
            )

    planner = _MultiMilestonePlannerStub(name="mm")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = _FailingExecutorStub(name="failing_executor")
    continuation = build_continuation_stub_subagent()
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="failure-history test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=3,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    session, _ = run_agent(agent)

    state = RunStateRepository().load(session.state, default_instruction="")
    fail_entries = [
        h for h in state.history if h.phase_completed == "EXECUTE" and not h.success
    ]
    assert fail_entries, "expected at least one EXECUTE failure entry"
    diag = fail_entries[0].agent_output
    assert isinstance(diag, dict)
    assert "io" in diag.get("code", "")
    assert "not allowed" in diag.get("stdout_excerpt", "")


def test_premature_submit_forces_retry_with_framework_override_in_next_cycle():
    """Framework guard: if continuation says SUBMIT but active is not the last
    milestone (or active hasn't succeeded), reject and force RETRY. Next
    cycle's continuation input must carry the framework_override message so
    the LLM knows what rule it violated."""

    class _SubmitProneContinuation(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.PLAN
        seen_overrides: ClassVar[list[str | None]] = []

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            metadata = subagent_input.metadata or {}
            type(self).seen_overrides.append(metadata.get("last_framework_override"))
            # First few cycles: try SUBMIT prematurely. After the framework
            # rejection signal arrives, switch to ADVANCE to walk through.
            if metadata.get("last_framework_override"):
                action = "ADVANCE"
            else:
                action = "SUBMIT"
            yield self.succeeded(
                attempt=subagent_input.attempt,
                payload={
                    "next_action": action,
                    "revised_milestones": None,
                    "rationale": "scripted",
                },
            )

    _SubmitProneContinuation.seen_overrides = []
    planner = _MultiMilestonePlannerStub(name="mm_planner_3ms")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)
    continuation = _SubmitProneContinuation(name="submit_prone")
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="premature-submit guard test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=15,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    session, _ = run_agent(agent)
    state = RunStateRepository().load(session.state, default_instruction="")

    # All 3 milestones should have EXECUTE success entries before final submit.
    exec_successes = [
        h for h in state.history if h.phase_completed == "EXECUTE" and h.success
    ]
    executed_indices = {h.milestone_index for h in exec_successes}
    assert executed_indices == {0, 1, 2}, (
        f"All 3 milestones must execute before submit, got {executed_indices}"
    )
    # At least one PLAN entry must carry framework_override citing SUBMIT rejection.
    overrides = [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and isinstance(h.agent_output, dict)
        and "SUBMIT rejected" in str(h.agent_output.get("framework_override", "") or "")
    ]
    assert overrides, "expected at least one premature-SUBMIT framework override"
    # Continuation must have observed last_framework_override at least once
    # (so it could react). The very first invocation has None; later ones
    # should see the override surfaced.
    assert any(o for o in _SubmitProneContinuation.seen_overrides), (
        "continuation never saw last_framework_override — feedback channel broken"
    )


class _ScriptedContinuation(BaseSubagent):
    """Continuation stub that returns pre-scripted decisions in order.

    decisions is a list of payload dicts; each call pops the first one.
    When only one remains, that decision is returned for all subsequent calls
    (lets us script a finite scenario and then settle into "done").
    """

    phase_: ClassVar[Phase] = Phase.PLAN
    decisions: list = []

    async def run_subagent(
        self, subagent_input: SubagentInput, ctx
    ) -> AsyncIterator[SubagentEnvelope]:
        decision = (
            self.decisions[0] if len(self.decisions) == 1 else self.decisions.pop(0)
        )
        yield self.succeeded(attempt=subagent_input.attempt, payload=decision)


def _run_with_scripted_continuation(decisions: list[dict]):
    planner = _MultiMilestonePlannerStub(name="mm_planner_regression")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)
    continuation = _ScriptedContinuation(
        name="scripted_cont", decisions=list(decisions)
    )
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="regression-guard test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=15,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    session, _ = run_agent(agent)
    return RunStateRepository().load(session.state, default_instruction="")


def test_per_milestone_budget_forces_advance_on_revise_loop():
    """Guard 5 — a continuation that REVISES the same active milestone every
    cycle slips past Guard 4 (whose streak resets on revised_milestones_applied),
    so only the per-milestone cycle budget can stop it. After
    PER_MILESTONE_CYCLE_LIMIT cycles on the milestone the framework must force
    ADVANCE (accept best-so-far, move on) instead of looping to the wall clock.
    This is the 042a9fc_2 mechanism (12 revise cycles on one milestone → wall
    timeout, downstream milestones never ran)."""
    from adk_appworld_agent.contracts.limits import PER_MILESTONE_CYCLE_LIMIT

    # RETRY WITH a 2-milestone revise every cycle → Guard 4 never fires; the
    # revise keeps active=m0 with a real m1 to advance into.
    revise = {
        "next_action": "RETRY",
        "revised_milestones": [
            {"id": "m0", "intent": "stuck step (revised)"},
            {"id": "m1", "intent": "downstream step"},
        ],
        "rationale": "revise m0 again",
    }
    decisions = [revise] * (PER_MILESTONE_CYCLE_LIMIT + 3)
    state = _run_with_scripted_continuation(decisions)

    plan_entries = [
        h
        for h in state.history
        if h.phase_completed == "PLAN" and isinstance(h.agent_output, dict)
    ]
    budget_overrides = [
        h
        for h in plan_entries
        if "per-milestone budget" in str(h.agent_output.get("framework_override") or "")
    ]
    assert budget_overrides, (
        "Guard 5 should force progress after PER_MILESTONE_CYCLE_LIMIT cycles on the "
        f"same milestone; plan entries={[(h.cycle_n, h.milestone_index, h.agent_output.get('next_action')) for h in plan_entries]}"
    )
    first = budget_overrides[0]
    # m0 is not the last milestone → force ADVANCE (not SUBMIT)
    assert first.agent_output.get("next_action") == "ADVANCE"
    # cycle 0 cold-start on m0 is excluded from the count; cycles 1..LIMIT
    # accumulate, the (LIMIT+1)-th RETRY is the one overridden to ADVANCE.
    assert first.cycle_n == PER_MILESTONE_CYCLE_LIMIT + 1, (
        f"expected first budget-guard ADVANCE at cycle {PER_MILESTONE_CYCLE_LIMIT + 1}, "
        f"got cycle {first.cycle_n}"
    )


def test_forward_advance_unchanged_by_framework_guard():
    """Sanity: normal forward progress (RETRY on cycle 0 cold-start, then
    ADVANCE through each milestone) must not trigger framework overrides."""

    decisions = [
        # cycle 1: m0 succeeded last cycle → ADVANCE to m1
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # cycle 2: m1 succeeded → ADVANCE to m2
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # cycle 3+: m2 succeeded, active is last → SUBMIT
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)

    overrides = [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and isinstance(h.agent_output, dict)
        and h.agent_output.get("framework_override")
    ]
    assert not overrides, (
        f"Forward progress must not trigger framework override; got {overrides}"
    )
    assert state.status == RunStatus.COMPLETED


def test_retry_with_revised_milestones_replaces_active_and_tail():
    """RETRY + revised_milestones writes into state.milestones[active:],
    keeping the prior immutable history milestones (indices < active).

    A full 1:1 tail replace (len(revised) == writable_len) is allowed even with
    the default-on growth guard — the guard only blocks NET growth
    (len > writable_len). Covered alongside test_growth_guard_rejects_net_growth."""

    decisions = [
        # cycle 1: m0 succeeded → ADVANCE to m1
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # cycle 2: at m1 → RETRY with revise that rewrites [m1, new_m2]
        {
            "next_action": "RETRY",
            "revised_milestones": [
                {"id": "m1_new", "intent": "revised m1"},
                {"id": "m2_new", "intent": "revised m2"},
            ],
            "rationale": "revise active and tail",
        },
        # cycle 3+: walk forward → SUBMIT
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)

    intents = [m.intent for m in state.milestones]
    # m0 (index 0) is below active when revise fired → preserved as "step 0".
    # m1, m2 (indices 1, 2) were replaced.
    assert intents[0] == "step 0", f"m0 should be preserved, got {intents}"
    assert intents[1] == "revised m1", f"m1 should be revised, got {intents}"
    assert intents[2] == "revised m2", f"m2 should be revised, got {intents}"


def test_advance_with_revised_milestones_replaces_tail_starting_at_next():
    """ADVANCE + revised_milestones writes into state.milestones[active+1:];
    the active milestone (which just succeeded) stays put. A full 1:1 tail
    replace (len == writable_len) is allowed by the default-on growth guard."""

    decisions = [
        # cycle 1: m0 succeeded → ADVANCE with revise that replaces tail
        # ([m1', m2']). active becomes 1 (m1').
        {
            "next_action": "ADVANCE",
            "revised_milestones": [
                {"id": "m1_alt", "intent": "alt m1"},
                {"id": "m2_alt", "intent": "alt m2"},
            ],
            "rationale": "advance + revise tail",
        },
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    assert intents[0] == "step 0", "m0 preserved (was successful)"
    assert intents[1] == "alt m1"
    assert intents[2] == "alt m2"


def _framework_overrides(state):
    return [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and isinstance(h.agent_output, dict)
        and h.agent_output.get("framework_override")
    ]


def test_growth_guard_allows_in_place_refine_at_last_milestone():
    """Regression for the 325d6ec bug: with the default-on growth guard, a
    1-item in-place refine of the active milestone when the writable region is
    just 1 (active == last, or a single-milestone plan) must be APPLIED, not
    rejected. An earlier `>= writable_len` guard wrongly treated this legitimate
    refine as growth (1 >= 1), so the continuation could never re-word the lone
    milestone (e.g. to add 'downloaded-status' wording) — it spun. The guard now
    blocks only NET growth (len > writable_len)."""
    decisions = [
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # at m2 (last), writable=[m2] (len 1); 1-item revise == writable -> must apply.
        {
            "next_action": "RETRY",
            "revised_milestones": [{"id": "m2_ref", "intent": "refined m2 in place"}],
            "rationale": "in-place refine of the lone writable milestone",
        },
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]
    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    assert intents[2] == "refined m2 in place", (
        f"single-milestone in-place refine must be applied, got {intents}"
    )


def test_growth_guard_rejects_net_growth_retry():
    """Default-on guard: a RETRY revise that grows the plan by MORE than the one
    allowed prerequisite (the 325d6ec 1->15 loop-unrolling pathology) is rejected;
    plan size is unchanged. NB: a +1 revise ([prereq, active, ...]) is now the
    permitted §4.3 prerequisite insertion — see
    test_prereq_insertion_allows_plus_one_growth — so this rejection case must
    grow by +2 to stay an unroll."""
    decisions = [
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # at m1, writable=[m1,m2] (len 2); 4 items = +2 growth > the one allowed
        # prerequisite -> loop-unroll -> rejected.
        {
            "next_action": "RETRY",
            "revised_milestones": [
                {"id": "g1", "intent": "loop iter 1"},
                {"id": "g2", "intent": "loop iter 2"},
                {"id": "g3", "intent": "loop iter 3"},
                {"id": "g4", "intent": "loop iter 4"},
            ],
            "rationale": "loop-unroll attempt",
        },
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]
    state = _run_with_scripted_continuation(decisions)
    assert len(state.milestones) == 3, (
        f"net-growth revise must not lengthen the plan, got {len(state.milestones)}"
    )
    assert _framework_overrides(state), (
        "rejected net-growth must record a framework_override"
    )


def test_bounded_forward_growth_allows_advance_add_missing_step():
    """§4.4 — the initial plan under-decomposed (a lone read milestone for a
    read-then-act task). ADVANCE + a 1-item revise that GROWS the plan past the
    last milestone is now APPLIED (bounded forward growth), not rejected — this is
    the a30375d / 0a9d82a / b9c5c9a / b6d1104 'add the missing compute/act step'
    recovery. forward_growth_count increments; cap (default 3) not yet spent."""
    add_step = {
        "next_action": "ADVANCE",
        "revised_milestones": [{"id": "g_act", "intent": "added compute/act step"}],
        "rationale": "plan under-decomposed; ADVANCE to add the missing next step",
    }
    decisions = [
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # at m2 (last), writable_len == 0; a 1-item revise GROWS by +1 → applied.
        add_step,
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]
    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    assert "added compute/act step" in intents, (
        f"bounded forward growth must add the missing step, got {intents}"
    )
    assert len(state.milestones) == 4, (
        f"plan should grow 3→4, got {len(state.milestones)}"
    )
    assert state.forward_growth_count == 1


def test_forward_growth_is_capped_per_task():
    """§4.4 — bounded forward growth is bounded: once forward_growth_count reaches
    continuation_forward_growth_cap (default 3) further ADVANCE-grows are rejected,
    so a genuine loop (advance-add one iteration per cycle) cannot unroll to 1→15.
    Start plan = 3; allow exactly 3 forward-grows (→6), reject the 4th."""

    def add(i):
        return {
            "next_action": "ADVANCE",
            "revised_milestones": [{"id": f"g{i}", "intent": f"grow {i}"}],
            "rationale": "add step",
        }

    decisions = [
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        add(1),
        add(2),
        add(3),  # 3 grows → cap spent, plan 3→6
        add(4),  # 4th grow → rejected (cap), no further growth
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]
    state = _run_with_scripted_continuation(decisions)
    assert state.forward_growth_count == 3, (
        f"forward growth must cap at 3, got {state.forward_growth_count}"
    )
    assert len(state.milestones) == 6, (
        f"plan must grow by at most cap (3→6), got {len(state.milestones)}"
    )


def test_retry_with_short_revised_milestones_preserves_unchanged_tail():
    """RETRY + revised_milestones SHORTER than writable region must NOT
    silently truncate downstream milestones — they're preserved at their
    original positions.

    This is the 59fae45_2 2026-05-31 footgun: the continuation_planner sent
    a 1-item revise at active=3 intending to revise only that milestone,
    but the OLD contract truncated tail (m4 = "Update playlist titles"
    mutation milestone deleted), causing P_MUTATION_SKIPPED with zero
    spotify.update_playlist calls.

    NEW contract: partial replacement preserves the unchanged tail.

    Stub planner emits 3 milestones (m0, m1, m2). After cycle 1 ADVANCE
    active=1 (m1). Cycle 2 RETRY with 1-item revise should overwrite m1,
    PRESERVE m2 (this is the regression assertion)."""
    decisions = [
        # cycle 1: m0 success → ADVANCE to m1
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        # cycle 2: at m1, RETRY with 1-item revise. Writable region = [m1, m2].
        # Expected: m1 overwritten, m2 preserved.
        {
            "next_action": "RETRY",
            "revised_milestones": [
                {"id": "m1_new", "intent": "revised m1"},
            ],
            "rationale": "revise only the active milestone",
        },
        # cycle 3+: m1 success → ADVANCE to m2 → m2 success → SUBMIT
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    # m0: untouched history (below active when revise fired).
    assert intents[0] == "step 0", f"m0 history preserved: {intents}"
    # m1: revised by the 1-item list.
    assert intents[1] == "revised m1", f"m1 revised: {intents}"
    # m2: tail preserved (NOT deleted) — this is the critical assertion.
    assert intents[2] == "step 2", (
        f"m2 must be preserved when revised_milestones is shorter than tail; "
        f"got {intents}"
    )


def test_advance_with_short_revised_milestones_preserves_unchanged_tail():
    """ADVANCE + short revise: same tail-preservation as RETRY case,
    just offset by +1 (writable region starts at active+1).

    Stub planner: m0, m1, m2. Cycle 1 ADVANCE with 1-item revise.
    Writable region after ADVANCE = [m1, m2]. 1-item revise should
    overwrite m1, preserve m2."""
    decisions = [
        # cycle 1: m0 success → ADVANCE with 1-item revise (revising only m1).
        {
            "next_action": "ADVANCE",
            "revised_milestones": [
                {"id": "m1_alt", "intent": "alt m1"},
            ],
            "rationale": "advance + revise only next",
        },
        # cycle 2+: m1 success → ADVANCE to m2 → SUBMIT
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)
    intents = [m.intent for m in state.milestones]
    assert intents[0] == "step 0"
    assert intents[1] == "alt m1", f"m1 overwritten: {intents}"
    assert intents[2] == "step 2", (
        f"m2 tail preserved on short ADVANCE revise; got {intents}"
    )


def test_advance_with_empty_revised_milestones_rejected():
    """ADVANCE + [] would attempt to advance past the end of the plan; framework
    must reject and force RETRY with a framework_override message."""

    decisions = [
        # cycle 1: m0 success → ADVANCE with empty revise (illegal)
        {
            "next_action": "ADVANCE",
            "revised_milestones": [],
            "rationale": "collapse tail",
        },
        # cycle 2+: legitimate ADVANCE to walk through
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "ADVANCE", "revised_milestones": None, "rationale": "fwd"},
        {"next_action": "SUBMIT", "revised_milestones": None, "rationale": "done"},
    ]

    state = _run_with_scripted_continuation(decisions)

    overrides = [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and isinstance(h.agent_output, dict)
        and "empty revised_milestones"
        in str(h.agent_output.get("framework_override", "") or "")
    ]
    assert overrides, "expected framework override for ADVANCE + []"


def test_framework_override_cleared_after_one_cycle():
    """last_framework_override must be visible to the NEXT continuation call,
    then cleared (not propagated for multiple cycles)."""

    seen: list[str | None] = []

    class _RecordingContinuation(BaseSubagent):
        phase_: ClassVar[Phase] = Phase.PLAN
        call_n: ClassVar[list[int]] = [0]

        async def run_subagent(
            self, subagent_input: SubagentInput, ctx
        ) -> AsyncIterator[SubagentEnvelope]:
            metadata = subagent_input.metadata or {}
            seen.append(metadata.get("last_framework_override"))
            type(self).call_n[0] += 1
            # Call 1: trigger override by submitting before reaching last.
            # Call 2: should see override surfaced; ADVANCE properly.
            # Call 3+: should see override CLEARED (None); ADVANCE/SUBMIT.
            if type(self).call_n[0] == 1:
                action = "SUBMIT"
            else:
                # walk through remaining milestones
                action = "ADVANCE"
            yield self.succeeded(
                attempt=subagent_input.attempt,
                payload={
                    "next_action": action,
                    "revised_milestones": None,
                    "rationale": "scripted",
                },
            )

    _RecordingContinuation.call_n = [0]
    seen.clear()

    planner = _MultiMilestonePlannerStub(name="mm_planner_clear")
    finder = build_stub_subagent(name="finder_subagent_stub", phase=Phase.FIND)
    executor = build_stub_subagent(name="executor_subagent_stub", phase=Phase.EXECUTE)
    continuation = _RecordingContinuation(name="recording")
    store = SubagentOutputStore()
    runner = SubagentRunner(subagent_output_store=store)
    controller = Orchestrator(
        subagent_output_store=store,
        subagent_runner=runner,
        submitter_provider=_FakeSubmitter,
    )
    agent = AppWorldAgent(
        name="appworld_controller_agent",
        description="override-clear test",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation,
        executor_subagent=executor,
        max_cycles=15,
        sub_agents=[
            planner.build_agent(),
            continuation.build_agent(),
            finder.build_agent(),
            executor.build_agent(),
        ],
    )
    run_agent(agent)

    # First call has no prior override.
    assert seen[0] is None
    # Second call should have observed the SUBMIT-rejected override.
    assert seen[1] and "SUBMIT rejected" in seen[1]
    # By the third call (after a clean ADVANCE on call 2), the override must
    # have been cleared.
    if len(seen) >= 3:
        assert seen[2] is None, (
            f"override should be cleared after one cycle, got {seen[2]}"
        )


def test_continuation_prompt_carries_axis_c_mutation_null_exception():
    """axis C Phase 1 — continuation prompt §3 evidence source 2 MUST carry
    an exception for mutation / state-changing milestones: `value=null` is
    the expected commit shape, NOT failure. Without this, planner-side
    fix (which encourages `value=null` for action milestones) gets undone
    when continuation flags null as failure and triggers retry loops.

    Assert layer-level phrases only per `feedback_no_task_specific_in_prompts`."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    assert "Exception" in prompt and "mutation milestones" in prompt
    assert "value=null" in prompt
    assert "expected commit shape" in prompt or "EXPECTED commit shape" in prompt
    assert "side effect" in prompt
    # api_trace cross-check for mutation milestones
    assert "non-read API call" in prompt


def test_retry_without_revise_loop_triggers_framework_override_after_limit():
    """Guard 4 — when continuation picks RETRY without revised_milestones
    on the SAME active milestone for RETRY_WITHOUT_REVISE_LIMIT consecutive
    cycles, the framework must inject a `last_framework_override` on the
    next cycle telling the LLM to revise / ADVANCE / ABORT.

    Surfacing tasks: 9dabbc9_2 (14 RETRY × no-revise), 425a494_2 (13),
    042a9fc_2 (9). All wasted wall budget on identical-approach loops the
    LLM couldn't break out of."""
    from adk_appworld_agent.contracts.limits import RETRY_WITHOUT_REVISE_LIMIT

    # Script: LIMIT+1 RETRY no-revise on m0 → ABORT. The (LIMIT+1)-th RETRY
    # is the one that should carry the framework_override (because by then,
    # LIMIT prior real-retry entries are in history on m0).
    decisions: list[dict] = [
        {"next_action": "RETRY", "revised_milestones": None, "rationale": f"r{i}"}
        for i in range(RETRY_WITHOUT_REVISE_LIMIT + 1)
    ]
    decisions.append(
        {"next_action": "ABORT", "revised_milestones": None, "rationale": "stop"}
    )

    state = _run_with_scripted_continuation(decisions)

    plan_entries_on_m0 = [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and h.milestone_index == 0
        and isinstance(h.agent_output, dict)
    ]
    # cold-start cycle 0 + (LIMIT+1) real continuation RETRY decisions, then
    # ABORT (ABORT also writes a PLAN entry on the active milestone).
    overrides = [
        h
        for h in plan_entries_on_m0
        if "RETRY ×" in str(h.agent_output.get("framework_override", "") or "")
        and "without revised_milestones"
        in str(h.agent_output.get("framework_override", "") or "")
    ]
    assert overrides, (
        "expected Guard 4 to inject a RETRY-loop framework_override; "
        f"got plan_entries={[(h.cycle_n, h.agent_output.get('next_action'), h.agent_output.get('framework_override')) for h in plan_entries_on_m0]}"
    )

    # The first override must fire on the (LIMIT+1)-th real RETRY, not earlier.
    # cycle_n of the first override entry == LIMIT+1 (cycle 0 cold-start is
    # excluded from the streak count, so cycles 1..LIMIT add up to LIMIT
    # before the (LIMIT+1)-th cycle hits the check).
    first_override = overrides[0]
    assert first_override.cycle_n == RETRY_WITHOUT_REVISE_LIMIT + 1, (
        f"first override expected at cycle {RETRY_WITHOUT_REVISE_LIMIT + 1}, "
        f"got cycle {first_override.cycle_n}"
    )

    # Earlier real-retry cycles (1..LIMIT) must NOT carry the override —
    # the LLM gets LIMIT free attempts before framework steps in.
    early_overrides = [
        h
        for h in plan_entries_on_m0
        if 1 <= h.cycle_n <= RETRY_WITHOUT_REVISE_LIMIT
        and h.agent_output.get("framework_override")
    ]
    assert not early_overrides, (
        f"Guard 4 fired too early — got overrides on cycles "
        f"{[h.cycle_n for h in early_overrides]}, expected only cycle ≥ {RETRY_WITHOUT_REVISE_LIMIT + 1}"
    )


def test_retry_with_revise_resets_loop_counter():
    """Guard 4 — when continuation picks RETRY WITH revised_milestones, the
    streak resets. A subsequent RETRY-no-revise immediately after should not
    trigger the override even if there were prior no-revise retries.

    Post-2026-05-17 note: ABORT was removed, so the script's final decision
    no longer terminates the agent. _ScriptedContinuation replays the last
    decision indefinitely (up to max_cycles=15). With the script ending in
    RETRY-no-revise, Guard 4 WILL eventually fire — that's correct
    behavior. The reset-correctness check is bounded to the window
    immediately after the cycle-3 revise."""
    from adk_appworld_agent.contracts.limits import RETRY_WITHOUT_REVISE_LIMIT

    decisions: list[dict] = [
        {"next_action": "RETRY", "revised_milestones": None, "rationale": "r1"},
        {"next_action": "RETRY", "revised_milestones": None, "rationale": "r2"},
        # cycle 3: revise — resets streak. 1-item partial revise of the active
        # milestone (in place; < writable region so the tail is preserved and
        # the growth guard allows it). A full-tail-replace would now be rejected.
        {
            "next_action": "RETRY",
            "revised_milestones": [
                {"id": "m0_new", "intent": "revised m0"},
            ],
            "rationale": "revise to break loop",
        },
        # cycle 4+: RETRY no-revise. Streak counts from 1 (cycle 3 reset it
        # to 0). Guard 4 should NOT fire until streak crosses
        # RETRY_WITHOUT_REVISE_LIMIT — i.e. not before cycle 3 + LIMIT + 1.
        {"next_action": "RETRY", "revised_milestones": None, "rationale": "r4"},
    ]

    state = _run_with_scripted_continuation(decisions)

    # The reset-correctness window: cycles immediately after the cycle-3
    # revise, up to RETRY_WITHOUT_REVISE_LIMIT cycles of no-revise should
    # NOT trigger Guard 4. Beyond that window, the override CAN fire
    # legitimately (the script keeps replaying RETRY-no-revise).
    early_overrides = [
        h
        for h in state.history
        if h.phase_completed == "PLAN"
        and isinstance(h.agent_output, dict)
        and 4 <= h.cycle_n <= 3 + RETRY_WITHOUT_REVISE_LIMIT
        and "RETRY ×" in str(h.agent_output.get("framework_override", "") or "")
    ]
    assert not early_overrides, (
        "revise on cycle 3 must reset the streak; no override should fire "
        f"on cycles 4..{3 + RETRY_WITHOUT_REVISE_LIMIT}. Got "
        f"{[(h.cycle_n, h.agent_output.get('framework_override')) for h in early_overrides]}"
    )


def test_continuation_prompt_warns_about_truncated_distribution():
    """Entry 4b — system prompt §3 must teach the LLM that the `distributions:`
    block is top-K truncated. Without this, the LLM reads a `... N more`
    sentinel as 'X is not in the data' when X may simply have been collapsed
    into the hidden tail. 425a494_2 RETRY loop class.

    Layer-level phrasing only (no specific app/entity/operation per
    `feedback_no_task_specific_in_prompts`)."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    # core directive — must reference the top-K truncation + the sentinel
    assert "top" in prompt.lower() and "more" in prompt
    assert "tail" in prompt.lower() or "hidden" in prompt.lower()
    # must state the rule that absence-from-visible ≠ absence-from-data
    assert (
        "INDETERMINATE" in prompt
        or "indeterminate" in prompt
        or ("cannot conclude" in prompt.lower() or "cannot cite" in prompt.lower())
    )
    # cross-link: the rule must apply to BOTH distributions and the per-call
    # header collapse line (entry 4 helper renders both with `... N more`)
    assert "distribution" in prompt.lower()
    assert "header" in prompt.lower() or "per-call" in prompt.lower()


def test_continuation_prompt_states_rationale_target_length():
    """Entry 2 — producer-side rationale ≤300 char target must appear in
    §5b so the LLM knows the desired brevity. Reader-side cap stays at
    RATIONALE_MAX_LENGTH=1000 (entry 1) as a safety buffer."""
    from adk_appworld_agent.contracts.limits import RATIONALE_TARGET_LENGTH
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    assert str(RATIONALE_TARGET_LENGTH) in prompt, (
        f"prompt must cite the target length {RATIONALE_TARGET_LENGTH}"
    )
    # phrasing must signal it's a *length* target — not e.g. a count or
    # ID. "characters" or "char" within a few lines of the target number.
    target_idx = prompt.find(str(RATIONALE_TARGET_LENGTH))
    window = prompt[max(0, target_idx - 50) : target_idx + 100]
    assert "char" in window.lower(), (
        f"target {RATIONALE_TARGET_LENGTH} must be qualified as characters, got window={window!r}"
    )


def test_continuation_prompt_describes_retry_loop_guard():
    """Entry 9 — §1 RETRY description must warn about consecutive
    RETRY-without-revise loops and tell the LLM to emit revised_milestones
    or ADVANCE after N attempts. The action space was reduced to
    RETRY / ADVANCE / SUBMIT on 2026-05-17 (ABORT removed) so the wall
    budget — not the LLM — decides task termination. Framework-side
    enforcement (Guard 4 in agent.py) backs this with a
    `last_framework_override` when LLM ignores the hint."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    # §1 must mention the loop guard concept
    assert "Loop guard" in prompt or "loop guard" in prompt
    # must offer the remaining escape paths (RETRY-with-revise, ADVANCE)
    assert "revised_milestones" in prompt
    assert "ADVANCE" in prompt
    # must reference framework_override channel as the enforcement layer
    assert "last_framework_override" in prompt or "framework_override" in prompt


def test_continuation_prompt_does_not_offer_abort_action():
    """Removed ABORT from continuation_planner's action space on 2026-05-17.
    Every AppWorld task is solvable with the inputs already provided
    (task_instruction + candidate_apis + prior trace); LLM-driven early
    termination produces zero PASS/FAIL benefit and only loses the
    partial-progress trace. The prompt should frame this positively —
    "every task is solvable" — rather than negatively as "no give-up
    option", so the LLM treats each cycle as constructive rather than
    rationing toward an unavailable exit.

    Universal benefit: every API task benefits from full-budget revise-
    and-retry over early surrender."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    # Action table should NOT list ABORT
    action_table = prompt.split("## §1. Actions and schema", 1)[1].split("## §2.", 1)[0]
    assert "ABORT" not in action_table, (
        "ABORT must be removed from §1 action table — every AppWorld task "
        "is solvable with the inputs, no LLM-self-termination action"
    )
    # The whole prompt should not recommend ABORT as a possible action
    # (a passing mention in historical-note context would also be wrong)
    assert "ABORT" not in prompt
    # Positive framing: explicitly states every task is solvable with
    # available inputs, so the LLM doesn't search for an "I quit" exit
    p_lower = prompt.lower()
    assert "solvable" in p_lower
    # The solution-locator framing: revise milestone wording to surface
    # what's already in the inputs
    assert (
        "solvable path" in p_lower
        or "the solution exists" in p_lower
        or "every task" in p_lower
    )


def test_continuation_prompt_allows_multiple_api_paths():
    """Continuation_planner must NOT reject an executor attempt solely
    because a specific API was not called. Many retrievals have multiple
    valid paths (per-item-detail API × N vs lookup-table API + local
    join). Reject only when stdout violates §3a/§3b OR api_trace shows no
    candidate API call at all."""
    from adk_appworld_agent.subagents.continuer.continuation import (
        CONTINUATION_SYSTEM_PROMPT,
    )

    prompt = CONTINUATION_SYSTEM_PROMPT
    # must acknowledge multiple valid paths
    assert "multiple" in prompt.lower() or "either" in prompt.lower()
    # must teach the per-item vs bulk-then-join framing (or equivalent)
    assert (
        "per-item" in prompt.lower()
        and ("bulk" in prompt.lower() or "lookup" in prompt.lower())
    ) or "join" in prompt.lower()
    # explicit disallow rule
    assert (
        "Do NOT reject solely because" in prompt
        or "do not reject solely because" in prompt.lower()
    )


def test_infra_abort_codes_contract():
    """Infra failure codes (provider-side: quota / stream stall) abort the run
    instead of feeding continuation a re-decomposition (0a9d82a: one
    EXECUTOR_LLM_STALLED cascaded into a 103-call doom loop). Ambiguous codes
    that may be real codegen faults stay OUT (they still go to continuation)."""
    from adk_appworld_agent.agent import _INFRA_ABORT_CODES
    from adk_appworld_agent.subagents.failure_codes import (
        EXECUTOR_LLM_RAISED,
        EXECUTOR_LLM_STALLED,
        EXECUTOR_TIMEOUT,
        PROVIDER_RATE_LIMITED,
    )

    assert PROVIDER_RATE_LIMITED in _INFRA_ABORT_CODES
    assert EXECUTOR_LLM_STALLED in _INFRA_ABORT_CODES
    # ambiguous (could be genuine codegen/sandbox faults) → NOT infra-abort
    assert EXECUTOR_LLM_RAISED not in _INFRA_ABORT_CODES
    assert EXECUTOR_TIMEOUT not in _INFRA_ABORT_CODES


def test_self_assess_reason_flows_to_continuation_diagnostics():
    """Whole-workflow propagation: the executor self-assess NOT-DONE reason must
    reach continuation's diagnostics (not stay buried in io), so continuation
    revises on the real cause instead of rewording blind."""
    from adk_appworld_agent.agent import _executor_failure_diagnostics
    from adk_appworld_agent.contracts.subagent_output import (
        SubagentEnvelope,
        SubagentStatus,
    )

    env = SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="exec",
        attempt=1,
        status=SubagentStatus.FAILED,
        payload={
            "code_execute": {
                "code": "x",
                "self_assess": {
                    "ok": False,
                    "summary": "committed 0",
                    "problem": "filter 'electricity bill' won't match observed 'Bill for Electricity'",
                },
            }
        },
        failure_code="EXECUTOR_MILESTONE_NOT_DONE",
    )
    diag = _executor_failure_diagnostics(env)
    assert diag is not None and "self_assess" in diag
    assert "Bill for Electricity" in diag["self_assess"]["problem"]
