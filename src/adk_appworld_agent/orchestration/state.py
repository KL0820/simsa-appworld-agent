from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from adk_appworld_agent.contracts.plan_ir import Milestone


class Phase(str, Enum):
    BOOTSTRAP = "BOOTSTRAP"
    PLAN = "PLAN"
    VERIFY = "VERIFY"
    FIND = "FIND"
    EXECUTE = "EXECUTE"
    SUBMIT = "SUBMIT"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RunStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TaskContext(BaseModel):
    task_id: str = ""
    instruction: str = ""
    task_datetime: str = ""


class BudgetSnapshot(BaseModel):
    llm_calls: int = 0
    total_tokens: int = 0
    subagent_calls: int = 0


class CycleHistoryEntry(BaseModel):
    cycle_n: int
    phase_completed: Literal["PLAN", "FIND", "EXECUTE", "SUBMIT"]
    milestone_index: int | None = None
    # STABLE milestone identity (§4.1). milestone_index is positional and SHIFTS
    # when the plan grows (a prerequisite insertion), so anything that needs to
    # find "prior attempts on the same logical milestone" must key on this id,
    # NOT the index (else the executor loses its own prior attempts + self_assess
    # diagnosis on an index shift → repeats the same bug; see _build_exec_input).
    milestone_id: str | None = None
    milestone_intent: str | None = None
    success: bool
    agent_output: dict | str | None = None
    failure_code: str | None = None


class CurrentStateSnapshot(BaseModel):
    cycle_n: int = 0
    active_milestone_index: int = 0
    milestone_count: int = 0
    last_failure_code: str | None = None


class RunState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    phase: Phase = Phase.BOOTSTRAP
    status: RunStatus = RunStatus.RUNNING
    task_context: TaskContext = Field(default_factory=TaskContext)
    budget_snapshot: BudgetSnapshot = Field(default_factory=BudgetSnapshot)
    replan_count: int = 0
    retry_count: int = 0
    completion_block_reason: str | None = None
    milestones: list[Milestone] = Field(default_factory=list)
    active_milestone_index: int = 0
    named_variables: dict[str, dict] = Field(default_factory=dict)
    variable_creation_order: list[str] = Field(default_factory=list)
    cycle_n: int = 0
    history: list[CycleHistoryEntry] = Field(default_factory=list)
    last_framework_override: str | None = None
    # Monotonic counter that mints a STABLE id for each milestone (§4.1). A
    # milestone keeps its id when re-worded in place (so its attempt count
    # accumulates across re-routes); an inserted prerequisite gets a fresh id.
    next_milestone_uid: int = 0
    # How many bounded prerequisite milestones have been inserted BEFORE each
    # stable milestone id (§4.3). Capped per id so a "+1 prerequisite" recovery
    # cannot degenerate into a slow one-per-cycle unroll.
    prereq_insertions: dict[str, int] = Field(default_factory=dict)
    # How many bounded FORWARD-growth ADVANCEs have grown the plan this task
    # (§4.4). When the initial plan under-decomposed (a lone "read X" milestone
    # for a "read X THEN act/compute" task), the continuation legitimately
    # ADVANCE-adds the missing next step. Capped per TASK by
    # continuation_forward_growth_cap so a genuine loop (advance-add one iteration
    # per cycle) is still bounded — the 1→15 unroll the growth guard exists to
    # stop. Distinct from prereq_insertions (which inserts UPSTREAM, per id).
    forward_growth_count: int = 0
    # ── Attempt-budget / no-progress tracking, owned by RunController ─────
    # Structural fingerprints of EXECUTE attempts on the active milestone, keyed
    # by its STABLE id (`fingerprint_milestone_id`) — NOT its index, so a
    # re-word (which re-routes the finder and shifts no index) still accumulates,
    # and an inserted prerequisite (new id) does not falsely reset the stuck
    # milestone. The list resets only when the active milestone's id changes.
    # `len(active_milestone_fingerprints)` is therefore the re-route-immune
    # per-milestone attempt count that the attempt-budget bounds.
    # `no_progress_streak` = trailing count of IDENTICAL fingerprints (the
    # structurally-stuck signal, separate from the total-attempt budget).
    # `economic_stop_reason` records the typed terminal class for the ledger /
    # viewer when the run stops on a spent budget — framed as an ECONOMIC STOP
    # ("budget spent, best-so-far"), never as "task impossible / give up" (every
    # task is solvable; we stop because the per-milestone budget is exhausted).
    active_milestone_fingerprints: list[str] = Field(default_factory=list)
    fingerprint_milestone_id: str | None = None
    no_progress_streak: int = 0
    economic_stop_reason: str | None = None
    # Total EXECUTE attempts across the WHOLE task (every milestone, every
    # attempt). The per-milestone attempt_budget is escapable by re-decomposition
    # (each new milestone id gets a fresh per-id count), so a continuation that
    # keeps splitting a stuck step churns to MAX_CYCLES (3d9a636: plan grew 4->6,
    # 96 LLM calls). This counter is decompose-IMMUNE — it accumulates regardless
    # of how the plan is reshaped — and bounds the whole-task work via
    # `task_attempt_budget` (economic stop, never "give up").
    total_execute_attempts: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.status in {RunStatus.COMPLETED, RunStatus.FAILED}

    @property
    def variable_store(self):
        from adk_appworld_agent.orchestration.variable_store import VariableStore

        return VariableStore(self)
