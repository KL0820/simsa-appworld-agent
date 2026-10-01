"""code_planner deduction-lite: per-milestone gold prompts -> live thin+skill
planner -> CodePlanOutput per milestone.

The gold code_plan user prompt (JSON payload + markdown sections) is replayed
verbatim per milestone — exact teacher forcing, zero reconstruction drift on
the input side. Judged by the code_planner recon judge (CODE_PLAN_JUDGE_SYSTEM).
"""

from __future__ import annotations

import json
from pathlib import Path

from google.adk.agents import LlmAgent

from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    render_code_planner_instruction,
    static_instruction_provider,
)
from mind_skill.deduction.gold import (
    payload_from_step_text,
    pick_gold_trajectory,
    step_user_text,
    strip_role_prefix,
)
from mind_skill.deduction.planner_common import PlannerDeductionResult, PlannerGold
from mind_skill.runtime import meta_generate_config, run_meta_agent_validated
from mind_skill.trajectory.render import render_code_planner_trajectory

COMPONENT = "code_planner"


def load_gold(runs_dir: Path, task_id: str) -> PlannerGold:
    path = pick_gold_trajectory(runs_dir, task_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    task = data.get("task") or {}
    by_index: dict[int, dict] = {}
    for step in data.get("steps", []):
        if step.get("agent", "").endswith("code_plan"):
            by_index[int(step.get("milestone_index") or 0)] = step
    if not by_index:
        raise ValueError(f"{task_id}: no code_plan steps")
    prompts, intents, apis = [], [], []
    for index in sorted(by_index):
        text = strip_role_prefix(step_user_text(by_index[index]))
        if not text.strip():
            raise ValueError(f"{task_id} m{index}: empty code_plan gold prompt")
        prompts.append(text)
        payload = payload_from_step_text(text)
        ms = payload.get("milestone") or {}
        intents.append(str(ms.get("intent") or ""))
        apis.append(
            [
                f"{c.get('app_name', '')}.{c.get('api_name', '')}".strip(".")
                for c in (payload.get("candidate_apis") or [])
                if isinstance(c, dict)
            ]
        )
    return PlannerGold(
        task_id=task_id,
        instruction=task.get("instruction") or "",
        prompts=prompts,
        milestone_intents=intents,
        milestone_apis=apis,
    )


def build_agent(*, skill_texts: list[str], model_cfg: ModelConfig) -> LlmAgent:
    return LlmAgent(
        name="code_planner_llm",
        model=model_cfg.name,
        description="Thin-prompt code planner (MIND-Skill deduction).",
        instruction=static_instruction_provider(
            render_code_planner_instruction(variant="minimal", skill_texts=skill_texts)
        ),
        output_schema=CodePlanOutput,
        generate_content_config=meta_generate_config(model_cfg),
    )


async def run(
    gold: PlannerGold, *, skill_texts: list[str], model_cfg: ModelConfig
) -> PlannerDeductionResult:
    agent = build_agent(skill_texts=skill_texts, model_cfg=model_cfg)
    outputs: list[CodePlanOutput] = []
    for prompt in gold.prompts:
        outputs.append(await run_meta_agent_validated(agent, prompt, CodePlanOutput))
    entries = [
        {
            "milestone": gold.milestone_intents[i],
            "apis": gold.milestone_apis[i],
            "code_plan": out.model_dump(mode="json"),
        }
        for i, out in enumerate(outputs)
    ]
    render = render_code_planner_trajectory(
        entries, label="CODE_PLANNER_TRAJECTORY (reconstructed)"
    )
    return PlannerDeductionResult(
        task_id=gold.task_id,
        component=COMPONENT,
        reconstruction=[o.model_dump(mode="json") for o in outputs],
        render=render,
    )


def load_task_skills(
    skills_root, component: str, task_id: str, library: str = "best"
) -> list[str]:
    """Per-task SKILL.md text(s) for a component from a built library
    (skills_root/<component>/<library>/<task_id>/<name>/SKILL.md). [] if none built."""
    from pathlib import Path

    base = Path(skills_root) / component / library / task_id
    return [p.read_text(encoding="utf-8") for p in sorted(base.glob("*/SKILL.md"))]


async def run_outcome(
    gold: PlannerGold,
    *,
    skill_texts: list[str],
    model_cfg: ModelConfig,
    rpc_url: str,
    work_dir,
    runs_dir,
    skills_root=None,
):
    """Outcome-grounded code_planner deduction.

    The planner INPUT stays teacher-forced (gold prompt verbatim); the live
    thin+skill planner GENERATES each milestone's plan; then the executor runs
    those plans as the deployment baseline (fat prompt, NO skill) in the real
    sandbox, and the task goes through the real evaluator. Only the planner
    carries the candidate skill -> the outcome delta is the planner skill's
    (spec §10 credit assignment; single-sample executor noise disclosed).

    Returns (PlannerDeductionResult, DeductionResult).
    """
    import dataclasses

    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        CODE_EXECUTOR_SYSTEM_PROMPT,
    )
    from mind_skill.deduction.code_executor import load_gold_task, run_deduction

    planner_result = await run(gold, skill_texts=skill_texts, model_cfg=model_cfg)

    exec_gold = load_gold_task(runs_dir, gold.task_id)
    generated = [
        CodePlanOutput.model_validate(p) for p in planner_result.reconstruction
    ]
    if len(generated) != len(exec_gold.milestones):
        raise ValueError(
            f"{gold.task_id}: generated {len(generated)} plans for "
            f"{len(exec_gold.milestones)} milestones"
        )
    exec_gold = dataclasses.replace(
        exec_gold,
        milestones=[
            dataclasses.replace(gm, plan=generated[i])
            for i, gm in enumerate(exec_gold.milestones)
        ],
    )
    # Deployment-matched downstream (Path B): the executor runs thin + its OWN
    # per-task skill from the bottom-up library (skills_root/code_executor), not
    # the fat no-skill baseline. Falls back to fat-baseline if no skill is built.
    exec_skills = (
        load_task_skills(skills_root, "code_executor", gold.task_id)
        if skills_root is not None
        else []
    )
    if skills_root is not None and not exec_skills:
        import sys

        # skills_root was given (Path B intended) but the upstream code_executor
        # skill for this task is missing -> downstream silently runs the FAT
        # no-skill baseline. Surface it; do not let a wrong root / broken
        # bottom-up order masquerade as an intended baseline.
        sys.stderr.write(
            f"[skills] code_planner outcome {gold.task_id}: NO downstream "
            f"code_executor skill under {skills_root} -> executor on FAT baseline "
            f"(build code_executor library first / check skills_root)\n"
        )
    deduction_result = await run_deduction(
        exec_gold,
        skill_texts=exec_skills,
        model_cfg=model_cfg,
        rpc_url=rpc_url,
        work_dir=work_dir,
        # None -> run_deduction renders the thin (minimal) executor prompt + skill;
        # fat baseline only when no downstream skill is available.
        executor_instruction=None if exec_skills else CODE_EXECUTOR_SYSTEM_PROMPT,
    )
    return planner_result, deduction_result


__all__ = ["COMPONENT", "build_agent", "load_gold", "run", "run_outcome"]
