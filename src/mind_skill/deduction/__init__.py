"""Deduction harnesses — one module per trained component.

gold.py           — corpus + PASS-trajectory pick + verbatim step-text access
code_executor.py  — teacher-forced sandbox deduction (outcome via real evaluator)
rough_planner.py  — deduction-lite: gold task prompt -> live thin+skill Plan
code_planner.py   — deduction-lite: gold per-milestone prompts -> CodePlanOutput
planner_common.py — shared PlannerGold / PlannerDeductionResult containers
"""

from mind_skill.deduction import code_planner, rough_planner
from mind_skill.deduction.code_executor import (
    DEFAULT_RPC_URL,
    DeductionResult,
    GoldTask,
    load_gold_task,
    run_deduction,
)
from mind_skill.deduction.gold import gold_corpus_task_ids, pick_gold_trajectory
from mind_skill.deduction.planner_common import PlannerDeductionResult, PlannerGold

PLANNER_DEDUCTIONS = {
    rough_planner.COMPONENT: rough_planner,
    code_planner.COMPONENT: code_planner,
}

__all__ = [
    "DEFAULT_RPC_URL",
    "DeductionResult",
    "GoldTask",
    "PLANNER_DEDUCTIONS",
    "PlannerDeductionResult",
    "PlannerGold",
    "gold_corpus_task_ids",
    "load_gold_task",
    "pick_gold_trajectory",
    "run_deduction",
]
