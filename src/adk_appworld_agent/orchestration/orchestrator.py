from __future__ import annotations

from collections.abc import Callable

from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from adk_appworld_agent.appworld.holder import AppWorldClientHolder
from adk_appworld_agent.contracts.executor_result import MemoryVariable
from adk_appworld_agent.contracts.plan_ir import Milestone
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.contracts.submission import (
    CompletionStatus,
    SubmissionCandidate,
)
from adk_appworld_agent.observability.ledger import (
    ExecutionLedgerEntry,
    ExecutionLedgerStore,
)
from adk_appworld_agent.orchestration.completion_gate import CompletionGate
from adk_appworld_agent.orchestration.run_config import model_config_from_env
from adk_appworld_agent.orchestration.state import Phase, RunState, RunStatus
from adk_appworld_agent.orchestration.state_repo import RunStateRepository
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.orchestration.subagent_runner import SubagentRunner
from adk_appworld_agent.subagents.failure_codes import (
    EXECUTOR_DID_NOT_FINALIZE,
    PROVIDER_RATE_LIMITED,
)
from adk_appworld_agent.subagents.utils.retry import RateLimitExhausted
from adk_appworld_agent.submission.final_answer_extractor import (
    extract_appworld_final_answer,
)


def _extract_submission_candidate(
    exec_output: SubagentEnvelope | None,
) -> SubmissionCandidate | None:
    if exec_output is None or exec_output.status != SubagentStatus.SUCCEEDED:
        return None
    raw = (exec_output.payload or {}).get("submission_candidate")
    if raw is None:
        return None
    try:
        return SubmissionCandidate.model_validate(raw)
    except Exception:
        return None


def _executor_completion_failure(output: SubagentEnvelope) -> str | None:
    """Decide if an executor envelope represents a true execution failure.

    Cuga-aligned policy: only "the executor produced no usable output" counts
    as an EXECUTE-phase failure. The executor's self-reported `milestone_done`
    is NOT a routing gate — that judgement belongs to the continuation LLM
    reading the milestone's summary text against its intent. An executor
    that ran code, hit an API error, and printed a "Failed: ..." summary is
    a successful delivery from the cycle-loop's perspective; continuation
    decides whether to retry with a different approach.
    """
    payload = output.payload or {}
    if payload.get("finalize_called") is not True:
        return EXECUTOR_DID_NOT_FINALIZE
    return None


_CONTROLLER_EXCLUDED_LOGIN_APIS = frozenset(
    {
        ("gmail", "login"),
        ("spotify", "login"),
        ("venmo", "login"),
        ("splitwise", "login"),
        ("amazon", "login"),
        ("todoist", "login"),
        ("simple_note", "login"),
        ("phone", "login"),
        ("file_system", "login"),
    }
)


def _filter_controller_candidate_apis(candidate_apis: list) -> list:
    filtered: list = []
    for api in candidate_apis:
        if not isinstance(api, dict):
            filtered.append(api)
            continue
        app_name = str(api.get("app") or api.get("app_name") or "")
        api_name = str(api.get("name") or api.get("api_name") or "")
        if (app_name, api_name) in _CONTROLLER_EXCLUDED_LOGIN_APIS:
            continue
        filtered.append(api)
    return filtered


class Orchestrator:
    """Service bag of helpers wired into AppWorldAgent's cycle loop.

    Routing has moved into `AppWorldAgent._run_one_cycle`. What's left here is
    the per-cycle plumbing: state load/save, ledger append, completion gate,
    forced-null fallback, and static helpers (variable commit, milestone
    metadata, event factories). Originally an FSM dispatcher; trimmed to a
    plain helper container during step 4 cull.
    """

    def __init__(
        self,
        *,
        state_repo: RunStateRepository | None = None,
        subagent_output_store: SubagentOutputStore | None = None,
        ledger_store: ExecutionLedgerStore | None = None,
        subagent_runner: SubagentRunner | None = None,
        completion_gate: CompletionGate | None = None,
        submitter_provider: Callable[[], object] | None = None,
    ) -> None:
        self.state_repo = state_repo or RunStateRepository()
        self.subagent_output_store = subagent_output_store or SubagentOutputStore()
        self.ledger_store = ledger_store or ExecutionLedgerStore()
        self.subagent_runner = subagent_runner or SubagentRunner(
            subagent_output_store=self.subagent_output_store
        )
        self.completion_gate = completion_gate or CompletionGate()
        self._submitter_provider = submitter_provider or AppWorldClientHolder.get_client

    def load_state(self, session_state: dict, *, default_instruction: str) -> RunState:
        return self.state_repo.load(
            session_state, default_instruction=default_instruction
        )

    def save_state(self, state: RunState) -> dict[str, dict]:
        return self.state_repo.save_delta(state)

    @staticmethod
    def _commit_executor_variables(state: RunState, output: SubagentEnvelope) -> None:
        payload = output.payload or {}
        result = payload.get("executor_result")
        if not isinstance(result, dict):
            return
        variables = result.get("variables")
        if not isinstance(variables, list):
            return
        for var in variables:
            if not isinstance(var, dict):
                continue
            try:
                typed_var = MemoryVariable.model_validate(var)
            except Exception:
                continue
            state.variable_store.commit(typed_var)

    @staticmethod
    def _active_milestone(state: RunState) -> Milestone | None:
        if state.active_milestone_index >= len(state.milestones):
            return None
        return state.milestones[state.active_milestone_index]

    @staticmethod
    def _planned_apps_for_milestone(milestone: Milestone) -> list[str]:
        app = milestone.app
        return [app.strip()] if isinstance(app, str) and app.strip() else []

    @staticmethod
    def _milestone_metadata(state: RunState) -> dict:
        milestone = Orchestrator._active_milestone(state)
        if milestone is None:
            return {}
        idx = state.active_milestone_index
        return {
            "milestone_id": milestone.id or "",
            "milestone_intent": milestone.intent or "",
            "milestone_index": idx,
            "milestone_total": len(state.milestones),
        }

    @staticmethod
    def _prior_variables_metadata(state: RunState) -> dict:
        if not state.named_variables:
            return {}
        return {
            "prior_variables": {
                name: dict(var)
                for name, var in state.named_variables.items()
                if isinstance(var, dict)
            },
            "prior_variables_preview": state.variable_store.summary(),
        }

    def append_ledger(
        self,
        session_state: dict,
        *,
        state_before: RunState,
        state_after: RunState,
        phase_decision: str,
        failure_code: str | None,
        policy_source: str | None = None,
    ) -> dict[str, list[dict]]:
        entry = ExecutionLedgerEntry(
            state_before=state_before,
            state_after=state_after,
            phase_decision=phase_decision,
            failure_code=failure_code,
            policy_source=policy_source,
        )
        return self.ledger_store.append_delta(session_state, entry)

    async def _run_completion_gate(self, session_state: dict) -> SubagentEnvelope:
        exec_output = self.subagent_output_store.read(session_state, Phase.EXECUTE)
        candidate = _extract_submission_candidate(exec_output)
        attempt = self.subagent_output_store.next_attempt(session_state, Phase.SUBMIT)

        try:
            candidate, extractor_payload = await self._apply_final_answer_extractor(
                session_state=session_state,
                exec_output=exec_output,
                candidate=candidate,
            )
        except RateLimitExhausted as exc:
            return SubagentEnvelope(
                phase=Phase.SUBMIT,
                subagent_name="completion_gate",
                attempt=attempt,
                status=SubagentStatus.FAILED,
                payload={
                    "block_reason": PROVIDER_RATE_LIMITED,
                    "error": str(exc),
                },
                failure_code=PROVIDER_RATE_LIMITED,
            )

        try:
            submitter = self._submitter_provider()
        except Exception as exc:
            return SubagentEnvelope(
                phase=Phase.SUBMIT,
                subagent_name="completion_gate",
                attempt=attempt,
                status=SubagentStatus.FAILED,
                payload={
                    "block_reason": "COMPLETION_CLIENT_UNAVAILABLE",
                    "error": str(exc),
                    "final_answer_extractor": extractor_payload,
                },
                failure_code="COMPLETION_CLIENT_UNAVAILABLE",
            )

        decision = self.completion_gate.verify_and_submit(candidate, submitter)
        payload = decision.model_dump(mode="json")
        if extractor_payload is not None:
            payload["final_answer_extractor"] = extractor_payload

        if decision.status == CompletionStatus.SUBMITTED:
            return SubagentEnvelope(
                phase=Phase.SUBMIT,
                subagent_name="completion_gate",
                attempt=attempt,
                status=SubagentStatus.SUCCEEDED,
                payload=payload,
            )

        return SubagentEnvelope(
            phase=Phase.SUBMIT,
            subagent_name="completion_gate",
            attempt=attempt,
            status=SubagentStatus.FAILED,
            payload=payload,
            failure_code=decision.block_reason or "COMPLETION_BLOCKED",
        )

    def _run_forced_null_submission(
        self,
        session_state: dict,
        *,
        original_failure_code: str | None,
    ) -> SubagentEnvelope:
        """Submit a synthetic null answer when the FSM aborts before SUBMIT.

        The AppWorld evaluator only records per-task test outcomes if the agent
        calls submit_answer at least once. When the executor crashes (e.g.
        EXECUTOR_DID_NOT_FINALIZE, EXECUTOR_LLM_RAISED, timeout) the run aborts
        without ever entering Phase.SUBMIT, leaving the task with NA in the
        aggregate report. This forced-fallback path runs exactly once at the
        end of the FSM loop to push a null answer through the same
        CompletionGate so the evaluator emits real test counts.

        Idempotency is the caller's responsibility — only invoke when no
        SUBMIT envelope already exists for this run.
        """
        attempt = self.subagent_output_store.next_attempt(session_state, Phase.SUBMIT)
        forced_marker = {
            "forced_fallback": True,
            "original_failure_code": original_failure_code,
        }

        try:
            submitter = self._submitter_provider()
        except Exception as exc:
            return SubagentEnvelope(
                phase=Phase.SUBMIT,
                subagent_name="completion_gate",
                attempt=attempt,
                status=SubagentStatus.FAILED,
                payload={
                    **forced_marker,
                    "block_reason": "COMPLETION_CLIENT_UNAVAILABLE",
                    "error": str(exc),
                },
                failure_code="COMPLETION_CLIENT_UNAVAILABLE",
            )

        candidate = SubmissionCandidate(
            answer="null",
            task_type_hint="action",
            answer_type="null",
            source="forced_fallback",
        )
        decision = self.completion_gate.verify_and_submit(candidate, submitter)
        payload = {**forced_marker, **decision.model_dump(mode="json")}

        if decision.status == CompletionStatus.SUBMITTED:
            return SubagentEnvelope(
                phase=Phase.SUBMIT,
                subagent_name="completion_gate",
                attempt=attempt,
                status=SubagentStatus.SUCCEEDED,
                payload=payload,
            )
        return SubagentEnvelope(
            phase=Phase.SUBMIT,
            subagent_name="completion_gate",
            attempt=attempt,
            status=SubagentStatus.FAILED,
            payload=payload,
            failure_code=decision.block_reason or "COMPLETION_BLOCKED",
        )

    @staticmethod
    def _overlay_state_delta(session_state: dict, state_delta: dict) -> dict:
        next_state = dict(session_state)
        next_state.update(state_delta)
        return next_state

    async def _apply_final_answer_extractor(
        self,
        *,
        session_state: dict,
        exec_output: SubagentEnvelope | None,
        candidate: SubmissionCandidate | None,
    ) -> tuple[SubmissionCandidate | None, dict | None]:
        """Run cuga-style extractor on the last EXECUTE output.

        On success, override candidate.answer + answer_type with the
        extractor's bare value. On any failure (LLM raise, schema
        mismatch, missing inputs), return the original candidate
        untouched so we fall back to the executor's own answer.
        """
        if candidate is None or exec_output is None:
            return candidate, None
        payload = exec_output.payload or {}
        stdout_json = (payload.get("code_execute") or {}).get("stdout_json")
        if not isinstance(stdout_json, dict) or not stdout_json:
            return candidate, None

        try:
            run_state = self.state_repo.load(session_state, default_instruction="")
        except Exception:
            return candidate, None
        task_instruction = run_state.task_context.instruction or ""
        if not task_instruction:
            return candidate, None

        try:
            model_cfg = model_config_from_env()
        except Exception:
            return candidate, None

        final = await extract_appworld_final_answer(
            task_instruction=task_instruction,
            last_executor_output=stdout_json,
            model_cfg=model_cfg,
        )
        if final is None:
            return candidate, {
                "status": "fallback",
                "reason": "extractor_returned_none",
            }

        # cuga's "N/A" maps to action / null; everything else is a query answer.
        bare = final.final_answer
        if bare.strip().upper() in {"N/A", "NULL", ""}:
            new_candidate = candidate.model_copy(
                update={
                    "answer": "null",
                    "answer_type": "null",
                    "task_type_hint": "action",
                    "source": "final_answer_extractor",
                }
            )
        else:
            new_candidate = candidate.model_copy(
                update={
                    "answer": bare,
                    "answer_type": final.final_answer_type,
                    "task_type_hint": "query",
                    "source": "final_answer_extractor",
                }
            )

        return new_candidate, {
            "status": "applied",
            "thoughts": final.thoughts,
            "final_answer": final.final_answer,
            "final_answer_type": final.final_answer_type,
            "previous_candidate_answer": candidate.answer,
            "previous_candidate_answer_type": candidate.answer_type,
        }

    @staticmethod
    def _state_update_event(
        ctx: InvocationContext, agent_name: str, state_delta: dict
    ) -> Event:
        return Event(
            invocation_id=ctx.invocation_id,
            author=agent_name,
            branch=ctx.branch,
            actions=EventActions(state_delta=state_delta),
        )

    @staticmethod
    def _final_response_event(
        ctx: InvocationContext, agent_name: str, state: RunState
    ) -> Event:
        if state.status == RunStatus.COMPLETED:
            message = "Skeleton workflow completed."
        else:
            message = f"Skeleton workflow failed: {state.completion_block_reason or 'unknown'}"

        return Event(
            invocation_id=ctx.invocation_id,
            author=agent_name,
            branch=ctx.branch,
            content=types.Content(role="model", parts=[types.Part(text=message)]),
        )
