"""rough_planner deduction-lite: gold task prompt -> live thin+skill planner -> Plan.

One LLM call per deduction. The reconstruction is judged against the gold
milestone decomposition by the rough_planner recon judge (PLAN_JUDGE_SYSTEM).
"""

from __future__ import annotations

import json
from pathlib import Path

from google.adk.agents import LlmAgent

from adk_appworld_agent.contracts.plan import Plan
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    static_instruction_provider,
)
from adk_appworld_agent.subagents.failure_codes import PROVIDER_RATE_LIMITED
from adk_appworld_agent.subagents.planner.rough_planner.prompts import (
    render_rough_planner_instruction,
)
from mind_skill.deduction.gold import (
    pick_gold_trajectory,
    step_user_text,
    strip_role_prefix,
)
from mind_skill.deduction.planner_common import PlannerDeductionResult, PlannerGold
from mind_skill.runtime import meta_generate_config, run_meta_agent_validated
from mind_skill.trajectory.render import render_rough_planner_output

COMPONENT = "rough_planner"


class RoughOutcomeRateLimited(RuntimeError):
    """The full-system rollout aborted on provider 429 exhaustion (infra, not the
    rough skill). Raised so the loop does NOT fold an infra failure into the
    outcome / best-of-Q (which would attribute API quota to skill quality); the
    task is rerun later instead — deployment's RATE_LIMITED discipline. grace
    extension handles transient 429s; this fires only on full ladder exhaustion."""


def _rollout_rate_limited(
    final_state: dict | None, submit_payload: dict | None
) -> bool:
    """True if the rollout aborted on provider 429 exhaustion. The controller
    sets completion_block_reason=PROVIDER_RATE_LIMITED and aborts (it must NOT
    revise off an infra artifact); a forced-null SUBMIT may follow, so the
    SUBMIT block_reason is checked too."""
    return PROVIDER_RATE_LIMITED in {
        (final_state or {}).get("completion_block_reason"),
        (submit_payload or {}).get("block_reason"),
    }


def load_gold(runs_dir: Path, task_id: str) -> PlannerGold:
    path = pick_gold_trajectory(runs_dir, task_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    task = data.get("task") or {}
    step = next(
        (s for s in data.get("steps", []) if s.get("agent") == "rough_planner"), None
    )
    if step is None:
        raise ValueError(f"{task_id}: no rough_planner step")
    prompt = strip_role_prefix(step_user_text(step))
    if not prompt.strip():
        raise ValueError(f"{task_id}: empty rough_planner gold prompt")
    return PlannerGold(
        task_id=task_id,
        instruction=task.get("instruction") or "",
        prompts=[prompt],
        milestone_intents=[],
        milestone_apis=[],
    )


def build_agent(*, skill_texts: list[str], model_cfg: ModelConfig) -> LlmAgent:
    return LlmAgent(
        name="rough_planner_llm",
        model=model_cfg.name,
        description="Thin-prompt rough planner (MIND-Skill deduction).",
        instruction=static_instruction_provider(
            render_rough_planner_instruction(variant="minimal", skill_texts=skill_texts)
        ),
        output_schema=Plan,
        generate_content_config=meta_generate_config(model_cfg),
    )


async def run(
    gold: PlannerGold, *, skill_texts: list[str], model_cfg: ModelConfig
) -> PlannerDeductionResult:
    agent = build_agent(skill_texts=skill_texts, model_cfg=model_cfg)
    plan = await run_meta_agent_validated(agent, gold.prompts[0], Plan)
    render = render_rough_planner_output(
        plan.thoughts,
        [t.model_dump(mode="json") for t in plan.tasks],
        label="PLANNER_OUTPUT (reconstructed)",
    )
    return PlannerDeductionResult(
        task_id=gold.task_id,
        component=COMPONENT,
        reconstruction=[plan.model_dump(mode="json")],
        render=render,
    )


# Bounded continuation budget for ONE full-system rollout. The controller
# terminates at MAX_CYCLES via force-null-submit (agent.py) -> a real evaluator
# report ALWAYS exists, so §3a's failed_requirements signal is never lost. This
# is the lever that makes the outcome reach SUBMIT: deployment's 15 cycles let a
# rollout run long enough that a wall-timeout cut it BEFORE SUBMIT (no eval, no
# signal). 8 still gives continuation real room to revise (the whole point of
# the full-system loop) while bounding cost. Deviation from deploy's 15 is a
# training knob, disclosed; raise via the constant if revise-depth matters more.
_ROUGH_OUTCOME_MAX_CYCLES = 8
# HARD per-rollout wall. Clean rollouts finish in <13 min (median ~3.5 min), so a
# rollout exceeding this is almost certainly grinding 429 ladders (up to ~19 min
# PER finder call x MAX_CYCLES). We do NOT extend the deadline for 429 waits (the
# earlier grace-extender had no cap, letting a busy-window rollout grind for ~28
# min). Instead 429 waits count against this wall; a rollout that blows it is cut
# and rerun in a clean window rather than wasting an afternoon on one attempt.
_ROUGH_OUTCOME_TIMEOUT_S = 1200.0


async def run_outcome(
    gold: PlannerGold,
    *,
    skill_texts: list[str],
    model_cfg: ModelConfig,
    rpc_url: str,
    work_dir,
    runs_dir=None,  # unused: rough does NOT teacher-force the downstream
    skills_root=None,
):
    """Outcome-grounded rough_planner deduction — FULL SYSTEM, continuation ON.

    Unlike the recon-only loop (planner_common note) and unlike code_planner's
    outcome loop (single teacher-forced executor pass), the rough planner's value
    is inseparable from continuation: for tasks that cannot be fully planned up
    front (e.g. a transfer that only reveals insufficient balance at execution),
    a good rough plan is one whose DIRECTION lets the baseline continuation
    revise into success. Isolating it (continuation off) would score every
    variant FAIL on exactly those tasks and give zero selection signal. So:

      1. the live thin+skill rough planner GENERATES the candidate rough plan;
      2. that plan is injected at cycle 0 of the REAL controller, and the FULL
         deployment runs with continuation/replan ON. finder + continuation stay
         deployment baseline, but the downstream code_planner + executor run thin
         + their OWN per-task skills from the bottom-up library (Path B, when
         skills_root is given) — so the outcome delta isolates the rough skill on
         top of an already-skilled downstream (matches deployment). Without
         skills_root the downstream is the fat no-skill baseline instead;
      3. the real world evaluator yields the outcome + per-requirement detail.

    Single-sample, full-system rollout: variance is larger than the code layers'
    (continuation + multi-cycle) — disclosed; verification uses pass-rate, not a
    single number. Returns (PlannerDeductionResult, DeductionResult); the loop
    reads recon off the first and outcome/§3a-detail off the second.
    """
    planner_result = await run(gold, skill_texts=skill_texts, model_cfg=model_cfg)
    plan = Plan.model_validate(planner_result.reconstruction[0])
    deduction_result = await _run_rough_outcome_controller(
        gold.task_id,
        plan,
        model_cfg=model_cfg,
        rpc_url=rpc_url,
        work_dir=work_dir,
        skills_root=skills_root,
    )
    return planner_result, deduction_result


async def _run_rough_outcome_controller(
    task_id: str,
    plan: Plan,
    *,
    model_cfg: ModelConfig,
    rpc_url: str,
    work_dir,
    skills_root=None,
    experiment_name: str = "mind_skill_rough_outcome",
):
    """Drive the REAL controller for one task with `plan` injected at cycle 0,
    everything else deployment baseline, and read the evaluator outcome.

    Mirrors code_executor.run_deduction's world setup/teardown (per-run RPC
    isolation) and scripts/run_task.py's in-process Runner loop. Returns a
    DeductionResult (records=[]; the outcome lives at the system level)."""
    import asyncio
    import os
    import time
    import uuid
    from pathlib import Path

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from adk_appworld_agent.agent import build_appworld_controller_agent
    from adk_appworld_agent.appworld.auth import AppWorldAuthManager
    from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder
    from adk_appworld_agent.appworld.bootstrap import load_task
    from adk_appworld_agent.appworld.client import AppWorldRpcClient
    from adk_appworld_agent.appworld.holder import AppWorldClientHolder
    from adk_appworld_agent.appworld.world_context import (
        build_world_run_name,
        close_world_safely,
    )
    from adk_appworld_agent.contracts.subagent_output import SubagentStatus
    from adk_appworld_agent.observability.ledger import LEDGER_KEY
    from adk_appworld_agent.orchestration.rate_limit_grace import (
        set_rate_limit_grace_extender,
    )
    from adk_appworld_agent.orchestration.run_config import RunConfig
    from adk_appworld_agent.orchestration.state import (
        BudgetSnapshot,
        Phase,
        RunState,
        RunStatus,
        TaskContext,
    )
    from adk_appworld_agent.orchestration.state_repo import RUN_STATE_KEY
    from adk_appworld_agent.orchestration.subagent_output_store import (
        SubagentOutputStore,
    )
    from adk_appworld_agent.subagents.stubs import DeterministicStubSubagent
    from mind_skill.deduction.code_executor import (
        DeductionResult,
        _extract_counts,
        _failed_requirements,
    )

    t0 = time.monotonic()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # cycle-0 planner that replays the candidate rough plan verbatim. Same
    # {thoughts, tasks} payload shape the real rough planner emits, so the
    # controller converts it to milestones via the identical path.
    injected_planner = DeterministicStubSubagent(
        name="rough_outcome_injected_planner",
        description="Replays the candidate thin+skill rough plan at cycle 0.",
        phase_name=Phase.PLAN,
        stub_status=SubagentStatus.SUCCEEDED,
        stub_payload={
            "thoughts": plan.thoughts,
            "tasks": [t.model_dump(mode="json") for t in plan.tasks],
        },
    )
    # Baseline everywhere else: community finder + code_plan_execute executor,
    # all *_skills empty + *_prompt_variant "full" (RunConfig defaults). PLAN
    # impl left unset (default "llm", != "stub") so continuation is the real
    # fat planner, not a stub.
    # Deployment-matched downstream (Path B): with skills_root, the downstream
    # code_planner + executor run thin + their per-task skills from the bottom-up
    # library (EXECUTE -> code_plan_execute_skill + skills_root + minimal prompt);
    # finder stays community; continuation left at controller default. Without
    # skills_root -> fat no-skill baseline (credit-assignment).
    if skills_root is not None:
        run_config = RunConfig(
            model=model_cfg,
            impls={Phase.FIND: "community", Phase.EXECUTE: "code_plan_execute_skill"},
            skills_root=Path(skills_root),
            skills_library="best",
            skills_prompt_variant="minimal",
        )
    else:
        run_config = RunConfig(
            model=model_cfg,
            impls={Phase.FIND: "community", Phase.EXECUTE: "code_plan_execute"},
        )

    # Experimental "longer wait" override (env MIND_SKILL_LLM_WAIT_S, seconds):
    # the executor's per-event stall timer (retry.llm_call_timeout_s, 150s
    # default) + transport timeout (executor_timeout_s, 240s) both read
    # active_config(). Bump them so a SLOW (not truly hung) gemini stream gets
    # time to emit its next event instead of being stall-cancelled at 150s; the
    # rough wall (MIND_SKILL_ROUGH_WALL_S) is widened to match. All unset =
    # current behavior. Tests "timeout too short" vs "genuinely hung socket".
    _rough_wall = float(
        os.environ.get("MIND_SKILL_ROUGH_WALL_S", _ROUGH_OUTCOME_TIMEOUT_S)
    )
    _wait = float(os.environ.get("MIND_SKILL_LLM_WAIT_S", "0") or 0)
    if _wait > 0:
        from adk_appworld_agent.orchestration.active_config import set_active_config

        run_config = run_config.model_copy(
            update={
                "retry": run_config.retry.model_copy(
                    update={"llm_call_timeout_s": _wait}
                ),
                "executor_timeout_s": _wait + 120.0,
            }
        )
        set_active_config(run_config)

    client = AppWorldRpcClient(addr=rpc_url)
    run_name = f"roughout_{uuid.uuid4().hex[:6]}"
    world_run_name = build_world_run_name(
        experiment_name=experiment_name, run_name=run_name, task_id=task_id
    )
    prev_trace_env = os.environ.get("EXECUTOR_SANDBOX_LOG_PATH")
    prev_max_cycles = os.environ.get("APPWORLD_MAX_CYCLES")
    trace_path = work_dir / f"sandbox_api_calls_{uuid.uuid4().hex[:8]}.jsonl"
    bootstrap = load_task(task_id, world_run_name, rpc_url=rpc_url, client=client)
    AppWorldClientHolder.set_client(client)
    AppWorldAuthHolder.set_auth_manager(AppWorldAuthManager(client))
    os.environ["EXECUTOR_SANDBOX_LOG_PATH"] = str(trace_path)
    # build_appworld_controller_agent reads _env_max_cycles() at build time.
    os.environ["APPWORLD_MAX_CYCLES"] = str(_ROUGH_OUTCOME_MAX_CYCLES)

    instruction = bootstrap.task_instruction or _instruction_fallback(plan)
    block_reason: str | None = None
    try:
        agent = build_appworld_controller_agent(
            run_config=run_config,
            planner_subagent=injected_planner,
            invocation_source=f"rough_outcome:{task_id}",
        )
        initial_state = {
            RUN_STATE_KEY: RunState(
                phase=Phase.BOOTSTRAP,
                status=RunStatus.RUNNING,
                task_context=TaskContext(
                    task_id=task_id,
                    instruction=instruction,
                    task_datetime=bootstrap.task_datetime or "",
                ),
                budget_snapshot=BudgetSnapshot(),
            ).model_dump(mode="json")
        }
        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=experiment_name,
            user_id=experiment_name,
            session_id=task_id,
            state=initial_state,
        )
        runner = Runner(
            agent=agent, session_service=session_service, app_name=experiment_name
        )
        # No grace extension: 429 backoff counts against the HARD wall so a
        # busy-window rollout fails fast (and is rerun) instead of grinding for
        # ~28 min. (The earlier unbounded grace-extender was the real cause of
        # single tasks taking an entire afternoon.)
        set_rate_limit_grace_extender(None)
        try:
            async with asyncio.timeout(_rough_wall):
                async for _event in runner.run_async(
                    user_id=experiment_name,
                    session_id=task_id,
                    new_message=types.Content(parts=[types.Part(text=instruction)]),
                ):
                    pass
        except TimeoutError:
            # >wall ⇒ almost certainly 429-grind (clean rollouts are <13 min) ⇒
            # infra, flag for rerun; do NOT count it as a real outcome.
            raise RoughOutcomeRateLimited(
                f"{task_id}: rough rollout exceeded {_rough_wall:.0f}s "
                "(429 grind) -> rerun in a clean window"
            )

        session = await session_service.get_session(
            app_name=experiment_name, user_id=experiment_name, session_id=task_id
        )
        state_map = session.state or {}
        # Persist the controller trace for diagnosability (project rule): the
        # cycle ledger + final run state make a slow/failed rollout debuggable.
        (work_dir / "controller_ledger.json").write_text(
            json.dumps(state_map.get(LEDGER_KEY, []), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (work_dir / "controller_state.json").write_text(
            json.dumps(state_map.get(RUN_STATE_KEY, {}), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        submit_env = SubagentOutputStore().read(state_map, Phase.SUBMIT)
        submit_payload = submit_env.payload if submit_env is not None else None
        if _rollout_rate_limited(state_map.get(RUN_STATE_KEY), submit_payload):
            # Infra (429 exhaustion), not the rough skill -> never let it count
            # as a real outcome. World cleanup runs in finally; loop reruns later.
            raise RoughOutcomeRateLimited(
                f"{task_id}: rollout aborted on {PROVIDER_RATE_LIMITED}"
            )
        if submit_env is None:
            return DeductionResult(
                task_id=task_id,
                records=[],
                passed=0,
                failed=0,
                total=0,
                cleared=False,
                block_reason=block_reason or "NO_SUBMIT_PHASE",
                submitted_answer=None,
                wall_s=time.monotonic() - t0,
                usage={},
                failed_requirements=[],
            )
        payload = submit_env.payload or {}
        raw_report = payload.get("evaluation_report")
        passed, failed, total = _extract_counts(payload.get("extra") or {}, raw_report)
        return DeductionResult(
            task_id=task_id,
            records=[],
            passed=passed,
            failed=failed,
            total=total,
            cleared=(total > 0 and failed == 0),
            block_reason=block_reason or payload.get("block_reason"),
            submitted_answer=payload.get("submitted_answer"),
            wall_s=time.monotonic() - t0,
            usage={},
            failed_requirements=_failed_requirements(raw_report),
        )
    finally:
        if prev_trace_env is None:
            os.environ.pop("EXECUTOR_SANDBOX_LOG_PATH", None)
        else:
            os.environ["EXECUTOR_SANDBOX_LOG_PATH"] = prev_trace_env
        if prev_max_cycles is None:
            os.environ.pop("APPWORLD_MAX_CYCLES", None)
        else:
            os.environ["APPWORLD_MAX_CYCLES"] = prev_max_cycles
        close_world_safely(client)
        AppWorldAuthHolder.reset()
        AppWorldClientHolder.reset()


def _instruction_fallback(plan: Plan) -> str:
    """Last-resort instruction if the bootstrap one is empty (should not happen
    for corpus tasks); the controller needs a non-empty user message."""
    return plan.thoughts or "Complete the task."


__all__ = ["COMPONENT", "build_agent", "load_gold", "run", "run_outcome"]
