"""Teacher-forced executor deduction harness (spec §7).

Replays one gold trajectory with the upstream pinned and ONLY the executor live:

  1. gold rough plan / finder candidates / code plans are replayed verbatim
     (extracted from the trajectory's code_plan steps — never re-run);
  2. the candidate skill is deterministically injected into the FROZEN minimal
     executor prompt's SKILLS slot (no load_skill non-determinism in training);
  3. the real `code_executor_llm` (gemini + skill) rewrites each milestone's
     Python; within the component the chain feeds its OWN committed variables
     to the next milestone (not the gold ones);
  4. code runs in the real sandbox on the DEDICATED RPC server (default 4243 —
     never the ablation server on 4242);
  5. after the last milestone the answer goes through the real CompletionGate
     -> world evaluator => outcome.

No controller, no continuation, no replan: a milestone that fails stays failed
(beyond the executor's own built-in repair loop) and the outcome loss carries it.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from adk_appworld_agent.appworld.auth import AppWorldAuthManager
from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder
from adk_appworld_agent.appworld.bootstrap import load_task
from adk_appworld_agent.appworld.client import AppWorldRpcClient
from adk_appworld_agent.appworld.holder import AppWorldClientHolder
from adk_appworld_agent.appworld.world_context import (
    build_world_run_name,
    close_world_safely,
)
from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.contracts.executor_result import MemoryVariable
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.submission import SubmissionCandidate
from adk_appworld_agent.orchestration.completion_gate import CompletionGate
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.orchestration.state import Phase, RunState, TaskContext
from adk_appworld_agent.subagents.executor.appworld_tools import (
    read_sandbox_trace_since,
    sandbox_trace_byte_offset,
)
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    CodePlanExecuteSubagent,
    _build_executor_result,
    _build_inner_code_executor_agent,
    render_executor_instruction,
)
from mind_skill.deduction.gold import (
    payload_from_step_text,
    pick_gold_trajectory,
    step_user_text,
)
from mind_skill.trajectory.render import render_executor_trajectory

DEFAULT_RPC_URL = "tcp://127.0.0.1:4243"

_PASS_RE = re.compile(r"Num Passed Tests\s*:\s*(\d+)")
_FAIL_RE = re.compile(r"Num Failed Tests\s*:\s*(\d+)")
_TOTAL_RE = re.compile(r"Num Total\s*Tests\s*:\s*(\d+)")


# ── Gold context (teacher-forced upstream) ────────────────────────────────────


@dataclass
class GoldMilestone:
    index: int
    milestone_id: str
    intent: str
    done_criteria: str
    total: int
    candidate_apis: list[dict]  # FULL specs from the gold code_plan payload
    plan: CodePlanOutput  # gold code plan, replayed verbatim


@dataclass
class GoldTask:
    task_id: str
    instruction: str
    task_datetime: str
    milestones: list[GoldMilestone]
    trajectory_path: Path


def load_gold_task(runs_dir: Path, task_id: str) -> GoldTask:
    path = pick_gold_trajectory(runs_dir, task_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", [])

    # Last code_plan step per milestone_index = the FINAL (successful) plan.
    plan_steps_by_index: dict[int, dict] = {}
    for step in steps:
        if step.get("agent", "").endswith("code_plan"):
            plan_steps_by_index[int(step.get("milestone_index") or 0)] = step

    milestones: list[GoldMilestone] = []
    total = len(plan_steps_by_index)
    for index in sorted(plan_steps_by_index):
        step = plan_steps_by_index[index]
        payload = payload_from_step_text(step_user_text(step))
        parsed = (step.get("output") or {}).get("parsed")
        if not isinstance(parsed, dict):
            raise ValueError(f"{task_id} m{index}: code_plan step has no parsed plan")
        ms = payload.get("milestone") or {}
        milestones.append(
            GoldMilestone(
                index=index,
                milestone_id=str(ms.get("id") or f"m{index}"),
                intent=str(ms.get("intent") or ""),
                done_criteria=str(ms.get("done_criteria") or ""),
                total=int(ms.get("total") or total),
                candidate_apis=list(payload.get("candidate_apis") or []),
                plan=CodePlanOutput.model_validate(parsed),
            )
        )
    if not milestones:
        raise ValueError(f"{task_id}: no code_plan steps in {path}")

    task = data.get("task") or {}
    return GoldTask(
        task_id=task.get("task_id") or task_id,
        instruction=task.get("instruction") or "",
        task_datetime=task.get("datetime") or "",
        milestones=milestones,
        trajectory_path=path,
    )


# ── Deduction records ─────────────────────────────────────────────────────────


@dataclass
class MilestoneRecord:
    index: int
    intent: str
    apis: list[str]
    code_plan: dict
    code: str | None
    result_summary: str
    api_calls: list[str]
    finalize_called: bool
    milestone_done: bool
    turns: int
    repair_count: int
    error: str | None
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class DeductionResult:
    task_id: str
    records: list[MilestoneRecord]
    passed: int
    failed: int
    total: int
    cleared: bool
    block_reason: str | None
    submitted_answer: str | None
    wall_s: float
    usage: dict[str, int]
    # Per-requirement failed sub-tests from the OFFICIAL evaluator report
    # (provenance-clean): each {"requirement": <docstring>, "error": <assertion
    # text incl. "In left but not right: [...]">}. The surgical evidence of what
    # the rollout dropped/over-added — fed into the outcome gradient (spec §3a).
    failed_requirements: list[dict] = field(default_factory=list)

    def reconstruction_entries(self) -> list[dict]:
        """Structured reconstructed trajectory tau-hat (persisted as
        reconstructed_trajectory.json; also the source rows for the render)."""
        return [
            {
                "milestone": r.intent,
                "apis": r.apis,
                "code_plan": r.code_plan,
                "code": r.code,
                "result_summary": r.result_summary,
                "api_calls": r.api_calls,
                "turns": r.turns,
                "repair_count": r.repair_count,
            }
            for r in self.records
        ]

    def reconstruction_render(self) -> str:
        return render_executor_trajectory(
            self.reconstruction_entries(), label="EXECUTOR_TRAJECTORY (reconstructed)"
        )

    def outcome_feedback(self) -> str:
        lines = [
            f"evaluator: passed={self.passed} failed={self.failed} total={self.total} "
            f"cleared={self.cleared}"
        ]
        if self.block_reason:
            lines.append(f"submission blocked: {self.block_reason}")
        # Per-requirement evaluator detail (spec §3a): the surgical "what was
        # dropped / over-added" signal. e.g. expected [11,15,95] but the rollout
        # produced [11,15,36,70,95,99,105] -> "In left but not right: [...]". This
        # is what lets the gradient credit a missing scope/qualifier to the right
        # layer instead of only seeing "milestone not done".
        if self.failed_requirements:
            lines.append("failed requirements (official evaluator):")
            for fr in self.failed_requirements:
                req = str(fr.get("requirement") or "").strip()
                err = str(fr.get("error") or "").strip()
                lines.append(f"  - {req}" + (f"\n      {err}" if err else ""))
        for r in self.records:
            if r.error or not r.milestone_done:
                lines.append(
                    f"milestone {r.index} ({r.intent[:60]}): "
                    f"{'error=' + str(r.error) if r.error else 'not done'}"
                )
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "failed": self.failed,
            "total": self.total,
            "cleared": self.cleared,
            "block_reason": self.block_reason,
            "submitted_answer": self.submitted_answer,
            "wall_s": round(self.wall_s, 1),
            "usage": self.usage,
            "failed_requirements": self.failed_requirements,
            "milestones": [
                {
                    "index": r.index,
                    "intent": r.intent,
                    "code": r.code,
                    "result_summary": r.result_summary,
                    "api_calls": r.api_calls,
                    "finalize_called": r.finalize_called,
                    "milestone_done": r.milestone_done,
                    "turns": r.turns,
                    "repair_count": r.repair_count,
                    "error": r.error,
                }
                for r in self.records
            ],
        }


def _api_names(candidate_apis: list[dict]) -> list[str]:
    names = []
    for spec in candidate_apis:
        if isinstance(spec, dict):
            app = spec.get("app_name") or spec.get("app") or ""
            api = spec.get("api_name") or spec.get("name") or ""
            if api:
                names.append(f"{app}.{api}" if app else str(api))
    return names


def _trace_api_calls(records: list[dict]) -> list[str]:
    calls: list[str] = []
    for record in records:
        for call in record.get("api_calls") or []:
            if isinstance(call, dict):
                app = call.get("app_name") or ""
                api = call.get("api_name") or ""
                status = call.get("status") or ""
                calls.append(f"{app}.{api}:{status}" if status else f"{app}.{api}")
    return calls


def _extract_counts(decision_extra: dict, report: object) -> tuple[int, int, int]:
    passed = decision_extra.get("passed")
    failed = decision_extra.get("failed")
    total = decision_extra.get("total")
    if isinstance(passed, int) and isinstance(failed, int) and isinstance(total, int):
        return passed, failed, total
    text = str(report or "")
    pm, fm, tm = _PASS_RE.search(text), _FAIL_RE.search(text), _TOTAL_RE.search(text)
    if pm and fm and tm:
        return int(pm.group(1)), int(fm.group(1)), int(tm.group(1))
    return 0, 0, 0


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# A failed sub-test block runs from ">> Failed Requirement" to the next ">>"
# marker, a box-drawing section banner ("──── ... ────"), or end of report.
_FAIL_BLOCK_RE = re.compile(
    r">>\s*Failed Requirement\s*\n(.*?)(?=\n>>|\n[─=]{5,}|\Z)", re.DOTALL
)
_FAIL_DETAIL_CAP = 600  # keep the full diff (incl. left/right tail), drop noise


def _failed_requirements(report: object) -> list[dict]:
    """Per-requirement FAIL detail from the official evaluator report (spec §3a).

    Provenance-clean: parses the world evaluator's own report. For each failed
    requirement returns {"requirement": <docstring>, "error": <assertion +
    diff>} where `error` keeps the SURGICAL tail the deployment viewer's parser
    drops — e.g. expected [11,15,95] but produced [11,15,36,70,95,99,105] ->
    "In left but not right: [36, 70, 99, 105]". That over-added/under-included
    diff (which lives AFTER a blank line, past the deployment parser's blank-line
    cutoff and 240-char cap) is exactly what lets the gradient credit a dropped
    scope/qualifier to the right layer. The verbose ```python``` test source is
    skipped (it's evaluator internals, not the mismatch signal).
    """
    text = _ANSI_RE.sub("", CompletionGate._extract_report_text(report))
    out: list[dict] = []
    for block in _FAIL_BLOCK_RE.findall(text):
        requirement = _fail_requirement_text(block)
        detail = _fail_detail_text(block)
        if requirement or detail:
            out.append({"requirement": requirement, "error": detail or None})
    return out


def _fail_requirement_text(block: str) -> str:
    """The requirement docstring: lines before the test-source fence / divider."""
    parts: list[str] = []
    for line in block.splitlines():
        s = line.strip()
        if (
            s.startswith("```")
            or s.startswith("----")
            or s.startswith("AssertionError")
        ):
            break
        if s:
            parts.append(s)
    return " ".join(parts).strip()


def _fail_detail_text(block: str) -> str | None:
    """Assertion + mismatch diff, KEEPING the left/right tail past the blank line.

    Starts at the AssertionError line (falls back to the post-`----` divider)
    and collapses blank runs to a single newline so the diff stays readable for
    the gradient without the surrounding test source."""
    lines = block.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("AssertionError")),
        None,
    )
    if start is None:
        start = next(
            (i + 1 for i, ln in enumerate(lines) if ln.strip().startswith("----")),
            None,
        )
    if start is None:
        return None
    detail_lines, blank = [], False
    for ln in lines[start:]:
        s = ln.strip()
        if s.startswith("```"):
            break
        if not s:
            blank = True
            continue
        if blank and detail_lines:
            detail_lines.append("")  # one separator before the left/right tail
        blank = False
        detail_lines.append(s)
    detail = "\n".join(detail_lines).strip()
    if len(detail) > _FAIL_DETAIL_CAP:
        detail = detail[: _FAIL_DETAIL_CAP - 3].rstrip() + "..."
    return detail or None


# ── The harness ───────────────────────────────────────────────────────────────


async def run_deduction(
    gold: GoldTask,
    *,
    skill_texts: list[str],
    model_cfg: ModelConfig,
    rpc_url: str = DEFAULT_RPC_URL,
    work_dir: Path,
    experiment_name: str = "mind_skill_deduction",
    executor_instruction: str | None = None,
) -> DeductionResult:
    """Teacher-forced rollout of one gold task with `skill_texts` in the slot.

    `executor_instruction` overrides the executor system prompt — used by the
    code_planner OUTCOME loop, which runs the executor as the deployment
    baseline (fat prompt, no skill) over GENERATED plans while only the
    planner carries the candidate skill (spec §10 credit assignment).

    `work_dir` receives the sandbox API trace (per-run isolation). The caller
    owns retry policy; this function raises on infra errors (RPC down).
    """
    t0 = time.monotonic()
    work_dir.mkdir(parents=True, exist_ok=True)
    trace_path = work_dir / f"sandbox_api_calls_{uuid.uuid4().hex[:8]}.jsonl"

    instruction_text = (
        executor_instruction
        if executor_instruction is not None
        else render_executor_instruction(variant="minimal", skill_texts=skill_texts)
    )
    subagent = CodePlanExecuteSubagent(
        name="deduction_executor",
        description="Teacher-forced MIND-Skill deduction executor.",
        inner_agent=None,
        inner_executor_agent=_build_inner_code_executor_agent(
            model_cfg, instruction=instruction_text
        ),
        model_cfg=model_cfg,
        use_llm_code_plan=False,
        use_llm_code_execute=True,
    )

    client = AppWorldRpcClient(addr=rpc_url)
    run_name = f"dedup_{uuid.uuid4().hex[:6]}"
    world_run_name = build_world_run_name(
        experiment_name=experiment_name, run_name=run_name, task_id=gold.task_id
    )
    prev_trace_env = os.environ.get("EXECUTOR_SANDBOX_LOG_PATH")
    bootstrap = load_task(gold.task_id, world_run_name, rpc_url=rpc_url, client=client)
    AppWorldClientHolder.set_client(client)
    AppWorldAuthHolder.set_auth_manager(AppWorldAuthManager(client))
    os.environ["EXECUTOR_SANDBOX_LOG_PATH"] = str(trace_path)

    state = RunState(
        task_context=TaskContext(
            task_id=gold.task_id,
            instruction=bootstrap.task_instruction or gold.instruction,
            task_datetime=bootstrap.task_datetime or gold.task_datetime,
        )
    )

    records: list[MilestoneRecord] = []
    usage_total = {
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
    }
    last_candidate: SubmissionCandidate | None = None

    try:
        for gm in gold.milestones:
            metadata: dict = {
                "milestone_id": gm.milestone_id,
                "milestone_index": gm.index,
                "milestone_total": gm.total,
                "milestone_intent": gm.intent,
                "milestone_done_criteria": gm.done_criteria,
                "candidate_apis": gm.candidate_apis,
            }
            if state.named_variables:
                metadata["prior_variables"] = {
                    name: dict(var)
                    for name, var in state.named_variables.items()
                    if isinstance(var, dict)
                }
                metadata["prior_variables_preview"] = state.variable_store.summary()

            subagent_input = SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=state.task_context,
                metadata=metadata,
            )

            offset = sandbox_trace_byte_offset()
            exec_run = await subagent._run_code_executor(subagent_input, gm.plan)
            api_calls = _trace_api_calls(read_sandbox_trace_since(offset))

            executor_result, candidate, finalize_called = _build_executor_result(
                exec_run=exec_run, plan=gm.plan, milestone_id=gm.milestone_id
            )
            if finalize_called:
                last_candidate = candidate

            # Chain OWN outputs (not gold): commit this milestone's variables.
            for var in executor_result.variables:
                try:
                    state.variable_store.commit(MemoryVariable.model_validate(var))
                except Exception:
                    continue

            for key in usage_total:
                usage_total[key] += int(exec_run.usage.get(key, 0))

            records.append(
                MilestoneRecord(
                    index=gm.index,
                    intent=gm.intent,
                    apis=_api_names(gm.candidate_apis),
                    code_plan=gm.plan.model_dump(mode="json"),
                    code=exec_run.code,
                    result_summary=executor_result.summary or "",
                    api_calls=api_calls,
                    finalize_called=finalize_called,
                    milestone_done=executor_result.milestone_done,
                    turns=exec_run.tool_call_count,
                    repair_count=max(0, len(exec_run.repair_attempts) - 1),
                    error=exec_run.llm_raised or exec_run.parse_error,
                    usage=dict(exec_run.usage),
                )
            )

        decision = CompletionGate().verify_and_submit(last_candidate, client)
        passed, failed, total = _extract_counts(
            decision.extra or {}, decision.evaluation_report
        )
        return DeductionResult(
            task_id=gold.task_id,
            records=records,
            passed=passed,
            failed=failed,
            total=total,
            cleared=(total > 0 and failed == 0),
            block_reason=decision.block_reason,
            submitted_answer=decision.submitted_answer,
            wall_s=time.monotonic() - t0,
            usage=usage_total,
            failed_requirements=_failed_requirements(decision.evaluation_report),
        )
    finally:
        if prev_trace_env is None:
            os.environ.pop("EXECUTOR_SANDBOX_LOG_PATH", None)
        else:
            os.environ["EXECUTOR_SANDBOX_LOG_PATH"] = prev_trace_env
        close_world_safely(client)
        AppWorldAuthHolder.reset()
        AppWorldClientHolder.reset()


__all__ = [
    "DEFAULT_RPC_URL",
    "DeductionResult",
    "GoldMilestone",
    "GoldTask",
    "MilestoneRecord",
    "load_gold_task",
    "run_deduction",
]
