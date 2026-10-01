from __future__ import annotations

from collections.abc import Callable

from google.adk.agents import BaseAgent
from pydantic import ConfigDict, Field

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.cache.policy import CacheLayer
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.orchestrator import (
    Orchestrator,
    _executor_completion_failure,
    _filter_controller_candidate_apis,
)
from adk_appworld_agent.orchestration.progress_policy import apply_progress_decision
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.orchestration.state import (
    CycleHistoryEntry,
    Phase,
    RunState,
    RunStatus,
)
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
from adk_appworld_agent.subagents.base import Subagent
from adk_appworld_agent.subagents.failure_codes import (
    EXECUTOR_LLM_STALLED,
    PROVIDER_RATE_LIMITED,
)

# Infra failure codes: provider-side problems (quota / stream hang), NOT agent
# planning errors. The controller aborts the run on these and does NOT feed them
# into continuation — revising the plan off an infra artifact is wrong and
# (observed on 0a9d82a) a single EXECUTOR_LLM_STALLED cascaded into a 103-call
# re-decomposition doom loop. The task is rerun later, like a 429. Only
# unambiguous infra codes belong here (EXECUTOR_LLM_RAISED / EXECUTOR_TIMEOUT
# are ambiguous — could be real codegen faults — so they still go to continuation).
_INFRA_ABORT_CODES = frozenset({PROVIDER_RATE_LIMITED, EXECUTOR_LLM_STALLED})
from adk_appworld_agent._controller_helpers import (
    DEFAULT_MAX_CYCLES,
    _capture_sandbox_trace_for_execute,
    _collect_prior_milestone_returns,
    _env_max_cycles,
    _execute_fingerprint,
    _executor_failure_diagnostics,
    _update_no_progress,
)
from adk_appworld_agent._controller_helpers import (
    _interpret_plan_envelope as _interpret_plan_envelope,
)
from adk_appworld_agent.subagents.continuer.continuation import (
    build_continuation_planner_subagent,
    build_continuation_stub_subagent,
)
from adk_appworld_agent.subagents.executor.appworld_tools import (
    sandbox_trace_byte_offset,
)
from adk_appworld_agent.subagents.registry import build_subagent
from adk_appworld_agent.subagents.stubs import build_stub_subagent


class AppWorldAgent(BaseAgent):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    controller: Orchestrator
    finder_subagent: Subagent
    planner_subagent: Subagent
    continuation_subagent: Subagent = Field(
        default_factory=build_continuation_stub_subagent
    )
    executor_subagent: Subagent
    max_cycles: int = DEFAULT_MAX_CYCLES

    async def _run_async_impl(self, ctx):
        async for event in self._cycle_loop(ctx):
            yield event

    async def _cycle_loop(self, ctx):
        ctrl = self.controller
        instruction = content_to_text(ctx.user_content)
        state = ctrl.load_state(ctx.session.state, default_instruction=instruction)

        if state.phase == Phase.BOOTSTRAP and state.status == RunStatus.RUNNING:
            state_before = state.model_copy(deep=True)
            state = state.model_copy(deep=True)
            state.phase = Phase.PLAN
            delta: dict = {}
            delta.update(ctrl.save_state(state))
            delta.update(
                ctrl.append_ledger(
                    ctx.session.state,
                    state_before=state_before,
                    state_after=state,
                    phase_decision="bootstrap_complete -> PLAN",
                    failure_code=None,
                )
            )
            yield ctrl._state_update_event(ctx, self.name, delta)

        while state.cycle_n < self.max_cycles and state.status == RunStatus.RUNNING:
            done = False
            async for event, next_state, finished in self._run_one_cycle(ctx, state):
                if event is not None:
                    yield event
                state = next_state
                done = finished

            if done:
                break

        if state.status == RunStatus.RUNNING and state.cycle_n >= self.max_cycles:
            state_before = state.model_copy(deep=True)
            state = state.model_copy(deep=True)
            state.status = RunStatus.FAILED
            state.phase = Phase.FAILED
            # Preserve the most recent cycle failure_code (if any). MAX_CYCLES_EXCEEDED
            # only fires when the loop ran out of cycles without surfacing any other
            # error — typically an infinite SUBMIT-deferring loop, not a repeated
            # subagent failure.
            ledger_failure_code = state.completion_block_reason or "MAX_CYCLES_EXCEEDED"
            if state.completion_block_reason is None:
                state.completion_block_reason = "MAX_CYCLES_EXCEEDED"
            delta = {}
            delta.update(ctrl.save_state(state))
            delta.update(
                ctrl.append_ledger(
                    ctx.session.state,
                    state_before=state_before,
                    state_after=state,
                    phase_decision="max_cycles_exceeded -> FAILED",
                    failure_code=ledger_failure_code,
                )
            )
            yield ctrl._state_update_event(ctx, self.name, delta)

        if state.status == RunStatus.FAILED:
            existing_submit = ctrl.subagent_output_store.read(
                ctx.session.state, Phase.SUBMIT
            )
            if existing_submit is None:
                forced_envelope = ctrl._run_forced_null_submission(
                    ctx.session.state,
                    original_failure_code=state.completion_block_reason,
                )
                forced_delta = ctrl.subagent_output_store.append_delta(
                    ctx.session.state, forced_envelope
                )
                yield ctrl._state_update_event(ctx, self.name, forced_delta)

        yield ctrl._final_response_event(ctx, self.name, state)

    async def _run_one_cycle(self, ctx, state: RunState):
        """Plan or review progress, retrieve APIs, then execute one milestone."""
        phase_input = state
        async for event, state, finished in self._run_plan_phase(ctx, phase_input):
            yield event, state, finished
        if finished or state.phase != Phase.FIND:
            return

        phase_input = state
        async for event, state, finished in self._run_find_phase(ctx, phase_input):
            yield event, state, finished
        if finished or state.phase != Phase.EXECUTE:
            return

        async for result in self._run_execute_phase(ctx, state):
            yield result

    async def _run_plan_phase(self, ctx, state: RunState):
        ctrl = self.controller
        cycle_idx = state.cycle_n
        # ---- PHASE 1: PLAN ----
        # Cycle 0 invokes the cold-start planner; cycle >= 1 always invokes
        # ContinuationAgent. The earlier deterministic short-circuit on EXECUTE
        # success was retired: under cuga-aligned routing, EXECUTE status=SUCCEEDED
        # only means the executor produced output, not that the milestone's
        # intent was actually accomplished. Short-circuiting bypassed
        # continuation's semantic judgement and made the framework advance even
        # when the executor's summary said "Failed to ..." or "Deleted 0 files"
        # (observed regression on 5a83b05_2 in cycle_recovery_18_v4 — agent
        # reported 0 deletions but short-circuit advanced and SUBMIT failed).
        # The cost is one extra continuation LLM call per cycle on the happy
        # path; the win is correctness on milestone-success-but-work-failed cases.
        if state.cycle_n == 0:
            planner = self.planner_subagent
        else:
            planner = self.continuation_subagent
        state_before = state.model_copy(deep=True)

        plan_delta = await ctrl.subagent_runner.invoke(
            ctx, planner, self._build_plan_input(state, ctx.session.state)
        )
        session_after_plan = ctrl._overlay_state_delta(ctx.session.state, plan_delta)
        plan_envelope = ctrl.subagent_output_store.read(session_after_plan, Phase.PLAN)

        if plan_envelope is None or plan_envelope.status != SubagentStatus.SUCCEEDED:
            failure_code = (
                plan_envelope.failure_code
                if plan_envelope is not None
                else "PLANNER_OUTPUT_MISSING"
            )
            state = state.model_copy(deep=True)
            state.history.append(
                CycleHistoryEntry(
                    cycle_n=state.cycle_n,
                    phase_completed="PLAN",
                    success=False,
                    failure_code=failure_code,
                )
            )
            state.cycle_n += 1
            state.status = RunStatus.FAILED
            state.phase = Phase.FAILED
            state.completion_block_reason = failure_code
            event = self._emit_event(
                ctx,
                base_delta=plan_delta,
                state_before=state_before,
                state_after=state,
                session_state=session_after_plan,
                phase_decision=f"PLAN failed (cycle={cycle_idx}) -> FAILED",
                failure_code=failure_code,
            )
            yield event, state, True
            return

        state = state.model_copy(deep=True)
        decision = apply_progress_decision(state, plan_envelope)

        if decision.next_action == "ABORT" or not state.milestones:
            state.cycle_n += 1
            state.status = RunStatus.FAILED
            state.phase = Phase.FAILED
            state.completion_block_reason = (
                "PLANNER_ABORTED" if decision.next_action == "ABORT" else "PLAN_EMPTY"
            )
            event = self._emit_event(
                ctx,
                base_delta=plan_delta,
                state_before=state_before,
                state_after=state,
                session_state=session_after_plan,
                phase_decision=f"PLAN -> {decision.next_action} -> FAILED",
                failure_code=state.completion_block_reason,
            )
            yield event, state, True
            return

        if decision.next_action == "SUBMIT":
            # Emit PLAN event first so completion_gate's read of EXECUTE
            # sees the canonical session state.
            state.phase = Phase.SUBMIT
            plan_event = self._emit_event(
                ctx,
                base_delta=plan_delta,
                state_before=state_before,
                state_after=state,
                session_state=session_after_plan,
                phase_decision=f"cycle {cycle_idx} PLAN -> SUBMIT",
                failure_code=None,
            )
            yield plan_event, state, False

            submit_before = state.model_copy(deep=True)
            gate_envelope = await ctrl._run_completion_gate(ctx.session.state)
            gate_delta = ctrl.subagent_output_store.append_delta(
                ctx.session.state, gate_envelope
            )
            session_after_gate = ctrl._overlay_state_delta(
                ctx.session.state, gate_delta
            )

            state = state.model_copy(deep=True)
            state.cycle_n += 1
            if gate_envelope.status == SubagentStatus.SUCCEEDED:
                state.status = RunStatus.COMPLETED
                state.phase = Phase.COMPLETE
                state.completion_block_reason = None
                phase_decision = "SUBMIT succeeded -> COMPLETE"
                failure_code = None
            else:
                state.status = RunStatus.FAILED
                state.phase = Phase.FAILED
                state.completion_block_reason = gate_envelope.failure_code
                phase_decision = "SUBMIT failed -> FAILED"
                failure_code = gate_envelope.failure_code

            state.history.append(
                CycleHistoryEntry(
                    cycle_n=state.cycle_n,
                    phase_completed="SUBMIT",
                    success=gate_envelope.status == SubagentStatus.SUCCEEDED,
                    failure_code=gate_envelope.failure_code,
                )
            )
            gate_event = self._emit_event(
                ctx,
                base_delta=gate_delta,
                state_before=submit_before,
                state_after=state,
                session_state=session_after_gate,
                phase_decision=phase_decision,
                failure_code=failure_code,
            )
            yield gate_event, state, True
            return

        # RETRY / ADVANCE — persist PLAN event so subsequent invokes see it.
        # active_milestone_index was already updated above based on the action.
        state.active_milestone_index = max(
            0, min(state.active_milestone_index, len(state.milestones) - 1)
        )
        state.phase = Phase.FIND
        plan_event = self._emit_event(
            ctx,
            base_delta=plan_delta,
            state_before=state_before,
            state_after=state,
            session_state=session_after_plan,
            phase_decision=f"cycle {cycle_idx} PLAN -> FIND m{state.active_milestone_index}",
            failure_code=None,
        )
        yield plan_event, state, False

    async def _run_find_phase(self, ctx, state: RunState):
        ctrl = self.controller
        cycle_idx = state.cycle_n
        # ---- PHASE 2: FIND ----
        find_before = state.model_copy(deep=True)

        # FIND skip — when the active milestone already has candidate_apis
        # from a previous cycle's FIND, reuse them instead of re-invoking the
        # finder LLM. When continuation applies `revised_milestones`, the
        # replacement Milestone has empty candidate_apis (the field is
        # runtime-populated, not in the continuation contract), so this
        # naturally re-runs FIND only for newly-revised or first-time
        # milestones. Per docs/notes/2026-05-16_input_fixes.md entry 8.
        skip_idx = state.active_milestone_index
        skip_find = 0 <= skip_idx < len(state.milestones) and bool(
            state.milestones[skip_idx].candidate_apis
        )
        if skip_find:
            find_delta = {}
            reused = state.milestones[skip_idx].candidate_apis
            synthesized_payload = {
                "candidate_apis": reused,
                "reused_from_prior_cycle": True,
            }
            find_envelope = SubagentEnvelope(
                phase=Phase.FIND,
                subagent_name="find_skip_reuse",
                attempt=1,
                status=SubagentStatus.SUCCEEDED,
                payload=synthesized_payload,
            )
            session_after_find = ctx.session.state
        else:
            find_delta = await ctrl.subagent_runner.invoke(
                ctx,
                self.finder_subagent,
                self._build_find_input(state, ctx.session.state),
            )
            session_after_find = ctrl._overlay_state_delta(
                ctx.session.state, find_delta
            )
            find_envelope = ctrl.subagent_output_store.read(
                session_after_find, Phase.FIND
            )

        if find_envelope is None or find_envelope.status != SubagentStatus.SUCCEEDED:
            failure_code = (
                find_envelope.failure_code
                if find_envelope is not None
                else "FINDER_OUTPUT_MISSING"
            )
            state = state.model_copy(deep=True)
            state.history.append(
                CycleHistoryEntry(
                    cycle_n=state.cycle_n,
                    phase_completed="FIND",
                    milestone_index=state.active_milestone_index,
                    success=False,
                    failure_code=failure_code,
                )
            )
            state.cycle_n += 1
            state.completion_block_reason = failure_code
            if failure_code == PROVIDER_RATE_LIMITED:
                # Unified 429 contract: provider quota exhaustion is not a
                # planning problem — do NOT loop into continuation (that
                # would revise the plan off an infra artifact). Abort the
                # run; forced null submission + RATE_LIMITED status follow.
                state.status = RunStatus.FAILED
                state.phase = Phase.FAILED
                event = self._emit_event(
                    ctx,
                    base_delta=find_delta,
                    state_before=find_before,
                    state_after=state,
                    session_state=session_after_find,
                    phase_decision=f"FIND rate-limited (cycle={cycle_idx}) -> FAILED",
                    failure_code=failure_code,
                )
                yield event, state, True
                return
            state.phase = Phase.PLAN  # next cycle starts at PLAN again
            event = self._emit_event(
                ctx,
                base_delta=find_delta,
                state_before=find_before,
                state_after=state,
                session_state=session_after_find,
                phase_decision=f"FIND failed (cycle={cycle_idx}) -> loop",
                failure_code=failure_code,
            )
            yield event, state, False
            return

        # Commit FIND candidate_apis to active milestone
        state = state.model_copy(deep=True)
        find_payload = find_envelope.payload or {}
        candidate_apis = find_payload.get("candidate_apis")
        idx = state.active_milestone_index
        if isinstance(candidate_apis, list) and idx < len(state.milestones):
            controller_candidate_apis = _filter_controller_candidate_apis(
                candidate_apis
            )
            current_m = state.milestones[idx]
            state.milestones[idx] = current_m.model_copy(
                update={"candidate_apis": controller_candidate_apis}
            )

        state.history.append(
            CycleHistoryEntry(
                cycle_n=state.cycle_n,
                phase_completed="FIND",
                milestone_index=idx,
                success=True,
                agent_output=({"reused_from_prior_cycle": True} if skip_find else None),
            )
        )
        state.phase = Phase.EXECUTE
        find_event = self._emit_event(
            ctx,
            base_delta=find_delta,
            state_before=find_before,
            state_after=state,
            session_state=session_after_find,
            phase_decision=f"cycle {cycle_idx} FIND m{idx} -> EXECUTE",
            failure_code=None,
        )
        yield find_event, state, False

    async def _run_execute_phase(self, ctx, state: RunState):
        ctrl = self.controller
        cycle_idx = state.cycle_n
        idx = state.active_milestone_index
        # ---- PHASE 3: EXECUTE ----
        exec_before = state.model_copy(deep=True)
        # Snapshot sandbox trace file size BEFORE EXECUTE so we can read the
        # records appended DURING this phase only (multi-cycle / multi-task
        # share the same file).
        sandbox_offset_before_exec = sandbox_trace_byte_offset()
        exec_delta = await ctrl.subagent_runner.invoke(
            ctx,
            self.executor_subagent,
            self._build_exec_input(state, ctx.session.state),
        )
        session_after_exec = ctrl._overlay_state_delta(ctx.session.state, exec_delta)
        exec_envelope = ctrl.subagent_output_store.read(
            session_after_exec, Phase.EXECUTE
        )

        executor_failure: str | None
        if exec_envelope is None:
            executor_failure = "EXECUTOR_OUTPUT_MISSING"
        elif exec_envelope.status != SubagentStatus.SUCCEEDED:
            executor_failure = exec_envelope.failure_code or "EXECUTOR_FAILED"
        else:
            executor_failure = _executor_completion_failure(exec_envelope)

        state = state.model_copy(deep=True)
        # Per-task total-work counter (decompose-immune, §4.1b): every EXECUTE
        # attempt across the whole task, regardless of which milestone / how the
        # plan was reshaped. Bounds the re-decompose churn the per-milestone
        # attempt_budget can't (3d9a636).
        state.total_execute_attempts += 1
        # Part B: capture this execute's api_trace ONCE (reused for diagnostics,
        # the success history entry, AND the structural no-progress fingerprint).
        exec_api_trace = _capture_sandbox_trace_for_execute(sandbox_offset_before_exec)
        cfg = active_config()
        task_budget_exhausted = (
            cfg.task_attempt_budget >= 1
            and state.total_execute_attempts >= cfg.task_attempt_budget
        )
        no_progress = False
        if cfg.give_up_enabled:
            fp_value = None
            if exec_envelope is not None:
                fp_value = (
                    ((exec_envelope.payload or {}).get("code_execute") or {}).get(
                        "stdout_json"
                    )
                    or {}
                ).get("value")
            active_milestone = (
                state.milestones[idx] if idx < len(state.milestones) else None
            )
            fp_candidate_apis = (
                active_milestone.candidate_apis if active_milestone else []
            )
            # Key the attempt count on the STABLE milestone id (§4.1), not idx, so
            # it survives re-routes and prerequisite insertions. Fall back to the
            # index only if a milestone somehow lacks an id (should not happen —
            # ids are stabilized each cycle above).
            milestone_key = (
                active_milestone.id
                if active_milestone and active_milestone.id
                else f"idx{idx}"
            )
            no_progress = _update_no_progress(
                state,
                milestone_key,
                _execute_fingerprint(fp_candidate_apis, exec_api_trace, fp_value),
                cfg.no_progress_fingerprint_window,
                cfg.attempt_budget,
            )
        if executor_failure is not None:
            failure_output = _executor_failure_diagnostics(exec_envelope)
            # Even on failure, attach the api_trace — for P_LLM_SURRENDER /
            # P_PREREQUISITE_BLOCK type bugs, the trace will be `[]` (executor
            # didn't actually call any API), which is itself diagnostic.
            if isinstance(failure_output, dict):
                failure_output["api_trace"] = exec_api_trace
            state.history.append(
                CycleHistoryEntry(
                    cycle_n=state.cycle_n,
                    phase_completed="EXECUTE",
                    milestone_index=idx,
                    milestone_id=(
                        state.milestones[idx].id
                        if idx < len(state.milestones)
                        else None
                    ),
                    milestone_intent=(
                        state.milestones[idx].intent
                        if idx < len(state.milestones)
                        else None
                    ),
                    success=False,
                    agent_output=failure_output,
                    failure_code=executor_failure,
                )
            )
            state.cycle_n += 1
            state.completion_block_reason = executor_failure
            if executor_failure in _INFRA_ABORT_CODES:
                # Infra (429 quota OR Gemini stream stall): provider-side, not a
                # planning problem — abort, do NOT feed back into continuation
                # (revising off an infra artifact triggered a 103-call
                # re-decomposition doom loop on 0a9d82a). Rerun the task later.
                state.status = RunStatus.FAILED
                state.phase = Phase.FAILED
                event = self._emit_event(
                    ctx,
                    base_delta=exec_delta,
                    state_before=exec_before,
                    state_after=state,
                    session_state=session_after_exec,
                    phase_decision=f"EXECUTE infra-abort {executor_failure} (cycle={cycle_idx}) -> FAILED",
                    failure_code=executor_failure,
                )
                yield event, state, True
                return
            # §4.1/4.2 attempt-budget terminal (ECONOMIC STOP, never "give up").
            # The per-milestone budget is spent: the executor reproduced
            # structurally-identical work `window` times, OR ran the full
            # attempt_budget without an accepted result. Stop spending on this
            # milestone instead of looping to max_cycles + the provider
            # rate-limit storm — accept best-so-far. The task is still solvable;
            # we have simply exhausted the budget allotted to this step.
            if no_progress or task_budget_exhausted:
                last_idx = len(state.milestones) - 1
                # Distinguish the two terminals: per-milestone no-progress
                # (this step is stuck) vs whole-task budget (the task as a whole,
                # incl. any re-decomposition, has spent its total work). The
                # latter is decompose-immune — it's what catches 3d9a636-style
                # churn the per-milestone budget slips past.
                if task_budget_exhausted and not no_progress:
                    state.economic_stop_reason = "task_attempt_budget"
                    _why = f"task attempt-budget spent (total {state.total_execute_attempts} EXECUTEs)"
                else:
                    state.economic_stop_reason = "no_progress"
                    _why = f"attempt-budget spent m{idx} (streak {state.no_progress_streak})"
                if idx < last_idx:
                    # Reallocate the remaining budget downstream: accept
                    # best-so-far and ADVANCE rather than spinning in place.
                    state.active_milestone_index = idx + 1
                    state.phase = Phase.PLAN
                    event = self._emit_event(
                        ctx,
                        base_delta=exec_delta,
                        state_before=exec_before,
                        state_after=state,
                        session_state=session_after_exec,
                        phase_decision=f"EXECUTE {_why} -> ADVANCE m{idx + 1} (best-so-far)",
                        failure_code=executor_failure,
                    )
                    yield event, state, False
                    return
                # Last milestone, budget spent without an accepted result ->
                # economic stop. completion_block_reason keeps the real executor
                # failure (set above) for attribution; economic_stop_reason records
                # the terminal class. The post-loop forced-null floor still records
                # real eval counts (best-so-far), never "task impossible".
                state.status = RunStatus.FAILED
                state.phase = Phase.FAILED
                event = self._emit_event(
                    ctx,
                    base_delta=exec_delta,
                    state_before=exec_before,
                    state_after=state,
                    session_state=session_after_exec,
                    phase_decision=f"EXECUTE {_why} (last milestone) -> economic stop [{executor_failure}]",
                    failure_code=executor_failure,
                )
                yield event, state, True
                return
            state.phase = Phase.PLAN
            event = self._emit_event(
                ctx,
                base_delta=exec_delta,
                state_before=exec_before,
                state_after=state,
                session_state=session_after_exec,
                phase_decision=f"EXECUTE failed (cycle={cycle_idx}) -> loop",
                failure_code=executor_failure,
            )
            yield event, state, False
            return

        # EXECUTE delivered output. NOTE: "success" here means the executor
        # produced a finalizable result — NOT that the milestone's intent
        # was actually accomplished. The continuation LLM reads agent_output.summary
        # to judge the latter (cuga-aligned: framework does not gate routing on
        # the executor's self-reported milestone_done).
        ctrl._commit_executor_variables(state, exec_envelope)
        state.cycle_n += 1
        exec_payload = exec_envelope.payload or {}
        code_execute = exec_payload.get("code_execute") or {}
        code_plan = exec_payload.get("code_plan") or {}
        exec_result = exec_payload.get("executor_result") or {}
        agent_output = {
            "summary": exec_result.get("summary") or "",
            "milestone_done_self_claim": exec_result.get("milestone_done"),
            # ADVISORY self-assess hint (executor no longer flips milestone_done;
            # this is a grounded summary + optional `problem` for continuation §3
            # to weigh — not authoritative). Surfaced on SUCCESS too, not just on
            # the failure diag, so the hint reaches the planner on claimed-done
            # milestones.
            "self_assess": code_execute.get("self_assess"),
            "stdout_json": code_execute.get("stdout_json"),
            "variables": exec_result.get("variables") or [],
            # `code` lets next-cycle hint pipe show executor what it wrote
            # last time — needed for "semantic empty" retry cluster where
            # success=True + value=[] is really a failure and executor must
            # diff its own prior code to pick a different filter / approach.
            "code": code_execute.get("code") or "",
            # `code_plan` is the planner's structured prior output (plan_steps +
            # construct_step + print_step + output_variable). Surfaced so the
            # next-cycle code_planner can diff its own prior plan, not just
            # the executor's Python. Useful for "success but empty value"
            # retries where the plan itself was wrong.
            "code_plan": (
                {
                    k: code_plan.get(k)
                    for k in (
                        "plan_steps",
                        "construct_step",
                        "print_step",
                        "output_variable",
                    )
                    if code_plan.get(k) is not None
                }
                if isinstance(code_plan, dict) and code_plan
                else None
            ),
            # api_trace = bounded summary of THIS milestone's sandbox API calls,
            # consumed by continuation_planner to cross-check milestone wording
            # vs API return shapes (e.g. milestone wants singular target but
            # API returned list[20] — refuse done).
            "api_trace": exec_api_trace,
        }
        state.history.append(
            CycleHistoryEntry(
                cycle_n=state.cycle_n,
                phase_completed="EXECUTE",
                milestone_index=idx,
                milestone_id=(
                    state.milestones[idx].id if idx < len(state.milestones) else None
                ),
                milestone_intent=(
                    state.milestones[idx].intent
                    if idx < len(state.milestones)
                    else None
                ),
                success=True,
                agent_output=agent_output,
            )
        )
        state.completion_block_reason = None
        state.phase = Phase.PLAN  # next cycle starts at PLAN

        exec_event = self._emit_event(
            ctx,
            base_delta=exec_delta,
            state_before=exec_before,
            state_after=state,
            session_state=session_after_exec,
            phase_decision=f"cycle {cycle_idx} EXECUTE m{idx} succeeded -> loop",
            failure_code=None,
        )
        yield exec_event, state, False

    def _emit_event(
        self,
        ctx,
        *,
        base_delta: dict,
        state_before: RunState,
        state_after: RunState,
        session_state: dict,
        phase_decision: str,
        failure_code: str | None,
    ):
        ctrl = self.controller
        delta = dict(base_delta)
        delta.update(ctrl.save_state(state_after))
        delta.update(
            ctrl.append_ledger(
                session_state,
                state_before=state_before,
                state_after=state_after,
                phase_decision=phase_decision,
                failure_code=failure_code,
            )
        )
        return ctrl._state_update_event(ctx, self.name, delta)

    def _build_plan_input(self, state: RunState, session_state: dict) -> SubagentInput:
        ctrl = self.controller
        attempt = ctrl.subagent_output_store.next_attempt(session_state, Phase.PLAN)
        metadata: dict = {"capture_io": True, "cycle_n": state.cycle_n}
        if state.cycle_n >= 1:
            metadata.update(
                {
                    "active_milestone_index": state.active_milestone_index,
                    "milestones": [m.model_dump() for m in state.milestones],
                    "history": [h.model_dump(mode="json") for h in state.history],
                    "prior_variables_preview": state.variable_store.summary(),
                    "last_failure_code": state.completion_block_reason,
                    "last_framework_override": state.last_framework_override,
                    # How many times in a row the executor produced structurally
                    # identical work on the active milestone. The framework gives
                    # up when this reaches its window, so a non-zero streak means
                    # rewording is NOT working — change the structural approach.
                    "no_progress_streak": state.no_progress_streak,
                }
            )
        return SubagentInput(
            phase=Phase.PLAN,
            attempt=attempt,
            task_context=state.task_context,
            metadata=metadata,
        )

    def _build_find_input(self, state: RunState, session_state: dict) -> SubagentInput:
        ctrl = self.controller
        metadata: dict = {"capture_io": True, "cycle_n": state.cycle_n}
        milestone = (
            state.milestones[state.active_milestone_index]
            if state.active_milestone_index < len(state.milestones)
            else None
        )
        if milestone is not None and milestone.app:
            metadata["planned_apps"] = [milestone.app.strip()]
        metadata.update(ctrl._milestone_metadata(state))
        metadata.update(ctrl._prior_variables_metadata(state))
        return SubagentInput(
            phase=Phase.FIND,
            attempt=ctrl.subagent_output_store.next_attempt(session_state, Phase.FIND),
            task_context=state.task_context,
            metadata=metadata,
        )

    def _build_exec_input(self, state: RunState, session_state: dict) -> SubagentInput:
        ctrl = self.controller
        idx = state.active_milestone_index
        metadata: dict = {"capture_io": True, "cycle_n": state.cycle_n}
        if idx < len(state.milestones):
            milestone = state.milestones[idx]
            if milestone.candidate_apis:
                metadata["candidate_apis"] = milestone.candidate_apis
            metadata.update(ctrl._milestone_metadata(state))
        metadata.update(ctrl._prior_variables_metadata(state))

        # Prior EXECUTE attempts on THIS logical milestone — keyed by the STABLE
        # milestone id, NOT the index (FIX-A). The index shifts when the plan
        # grows (a prerequisite insertion), so index-keying drops the executor's
        # own earlier attempts + self_assess diagnosis after a shift → it
        # cold-starts and repeats the same bug (3d9a636: pagination loop bug that
        # self_assess had already named). Fall back to index only when no id is
        # available (cold-start / legacy entries).
        active_id = state.milestones[idx].id if idx < len(state.milestones) else None
        if active_id:
            prior_attempts = [
                h.model_dump(mode="json")
                for h in state.history
                if h.phase_completed == "EXECUTE" and h.milestone_id == active_id
            ]
        else:
            prior_attempts = [
                h.model_dump(mode="json")
                for h in state.history
                if h.phase_completed == "EXECUTE" and h.milestone_index == idx
            ]
        if prior_attempts:
            metadata["prior_attempts_for_this_milestone"] = prior_attempts
            metadata["retry_reason"] = (
                state.completion_block_reason or "prior_attempt_failed"
            )

        # Intermediate-visibility bridge: surface the REAL api returns that
        # EARLIER completed milestones observed, so this milestone's code_planner
        # can recover a raw field an earlier step saw but did not commit into a
        # named variable (9016950: last_name). Gated OFF by default; renders via
        # the existing _render_prior_returns path in prompt_builders.
        _cfg = active_config()
        if _cfg.executor_sees_prior_milestone_returns:
            prior_ms_returns = _collect_prior_milestone_returns(
                state, active_id, cap=_cfg.executor_prior_milestone_returns_cap
            )
            if prior_ms_returns:
                metadata["prior_milestone_returns"] = prior_ms_returns

        # Pull the most recent continuation_planner rationale (i.e. the PLAN
        # entry that decided to (re)try this milestone). Without this the
        # executor is a cold-start on every retry and re-emits the same code.
        for h in reversed(state.history):
            if h.phase_completed == "PLAN" and isinstance(h.agent_output, dict):
                rationale = (h.agent_output.get("rationale") or "").strip()
                if rationale:
                    metadata["latest_continuation_rationale"] = rationale
                break

        return SubagentInput(
            phase=Phase.EXECUTE,
            attempt=ctrl.subagent_output_store.next_attempt(
                session_state, Phase.EXECUTE
            ),
            task_context=state.task_context,
            metadata=metadata,
        )


def build_appworld_controller_agent(
    *,
    impls: dict[Phase, str] | None = None,
    fail_phase: Phase | None = None,
    invalid_phase: Phase | None = None,
    missing_phase: Phase | None = None,
    submitter_provider: Callable[[], object] | None = None,
    cache_layer: CacheLayer | None = None,
    invocation_source: str = "live",
    run_config: RunConfig | None = None,
    continuation_subagent: Subagent | None = None,
    planner_subagent: Subagent | None = None,
) -> AppWorldAgent:
    """Build the cycle-based controller agent.

    Subagent selection precedence:
    - For each phase, if a stub edge-case flag matches (fail/invalid/missing),
      the stub for that phase is forced into that mode (test-only path).
    - Otherwise, look up ``impls[phase]`` in the subagent registry.
    The continuation subagent is built from the LLM impl by default; pass
    ``continuation_subagent`` to substitute a stub or alternate impl. Symmetric
    ``planner_subagent`` substitutes the PLAN subagent (e.g. a stub that replays
    a fixed cycle-0 plan, as the rough-planner outcome harness does); when given,
    it does NOT change continuation selection (that still keys on impls[PLAN]).
    """

    def _stub_edge_case(phase: Phase, name: str) -> Subagent | None:
        if invalid_phase == phase:
            return build_stub_subagent(name=name, phase=phase, emit_mode="invalid")
        if missing_phase == phase:
            return build_stub_subagent(name=name, phase=phase, emit_mode="missing")
        if fail_phase == phase:
            return build_stub_subagent(
                name=name, phase=phase, stub_status=SubagentStatus.FAILED
            )
        return None

    resolved_impls: dict[Phase, str] = {
        Phase.FIND: "community",
        Phase.PLAN: "rough",
        Phase.EXECUTE: "code_plan_execute",
    }
    if run_config is not None:
        resolved_impls.update(run_config.impls)
    if impls:
        resolved_impls.update(impls)
    if run_config is None:
        run_config = RunConfig(impls=resolved_impls)
    else:
        run_config = run_config.model_copy(update={"impls": resolved_impls})

    def _subagent_for(phase: Phase, stub_name: str) -> Subagent:
        edge = _stub_edge_case(phase, stub_name)
        if edge is not None:
            return edge
        return build_subagent(phase, resolved_impls[phase], run_config=run_config)

    finder = _subagent_for(Phase.FIND, "finder_subagent_stub")
    planner = planner_subagent or _subagent_for(Phase.PLAN, "planner_subagent_stub")
    executor = _subagent_for(Phase.EXECUTE, "executor_subagent_stub")

    if continuation_subagent is None:
        if resolved_impls.get(Phase.PLAN) == "stub":
            continuation_subagent = build_continuation_stub_subagent()
        else:
            continuation_subagent = build_continuation_planner_subagent(
                run_config=run_config
            )

    subagent_output_store = SubagentOutputStore()
    subagent_runner = SubagentRunner(
        subagent_output_store=subagent_output_store,
        cache_layer=cache_layer,
        invocation_source=invocation_source,
    )
    controller = Orchestrator(
        subagent_output_store=subagent_output_store,
        subagent_runner=subagent_runner,
        submitter_provider=submitter_provider,
    )
    return AppWorldAgent(
        name="appworld_controller_agent",
        description="Cycle-based ADK custom agent shell.",
        controller=controller,
        finder_subagent=finder,
        planner_subagent=planner,
        continuation_subagent=continuation_subagent,
        executor_subagent=executor,
        max_cycles=_env_max_cycles(),
        sub_agents=[
            finder.build_agent(),
            planner.build_agent(),
            continuation_subagent.build_agent(),
            executor.build_agent(),
        ],
    )
