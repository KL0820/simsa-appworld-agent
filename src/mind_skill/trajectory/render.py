"""Trajectory renders shared by induction input and the recon judge.

`render_executor_trajectory` is the spec §9.1 input render (same shape as
build_induction_input.render_executor): per milestone — intent, APIs offered,
code plan, code, result, turns/repairs. It renders BOTH the reference slice
(τ_C, from induction_slices.json) and the reconstruction (τ̂, from a deduction
run) so the recon judge compares like with like.
"""

from __future__ import annotations


def render_executor_trajectory(
    milestone_entries: list[dict], *, label: str = "EXECUTOR_TRAJECTORY"
) -> str:
    """milestone_entries: dicts with milestone / apis / code_plan / code /
    result_summary / turns / repair_count (the executor-slice fields)."""
    lines = [f"{label}:"]
    for i, m in enumerate(milestone_entries):
        lines.append(f"\n--- milestone {i}: {m.get('milestone')}")
        lines.append(f"APIs available: {m.get('apis')}")
        plan = m.get("code_plan") or {}
        if plan.get("plan_steps"):
            lines.append("code plan:")
            for step in plan["plan_steps"]:
                lines.append(f"  - {step}")
        lines.append("code that ran:")
        lines.append("```python")
        lines.append((m.get("code") or "").rstrip())
        lines.append("```")
        lines.append(f"result: {m.get('result_summary')}")
        api_calls = m.get("api_calls")
        if api_calls:
            lines.append(f"api calls: {api_calls}")
        turns = m.get("turns")
        repairs = m.get("repair_count")
        if repairs is None:
            repairs = len(m.get("repair_codes") or [])
        lines.append(f"(turns: {turns}, repairs: {repairs})")
    return "\n".join(lines)


def render_induction_slice(slc: dict) -> tuple[str, str]:
    """induction_slices.json -> (task_instruction, EXECUTOR_TRAJECTORY render)."""
    return slc["instruction"], render_executor_trajectory(slc.get("code_executor", []))


def render_rough_planner_output(
    thoughts: str, milestones: list[dict], *, label: str = "PLANNER_OUTPUT"
) -> str:
    """spec §9.3 render: rationale + ordered milestones (works for both the
    gold slice and a reconstructed Plan)."""
    lines = [f"{label}:", f"rationale: {thoughts}", "milestones:"]
    for i, m in enumerate(milestones):
        task = m.get("task") if isinstance(m, dict) else str(m)
        app = m.get("app") if isinstance(m, dict) else ""
        suffix = f"  [{app}]" if app else ""
        lines.append(f"  - M{i}: {task}{suffix}")
    return "\n".join(lines)


def render_code_planner_trajectory(
    milestone_entries: list[dict], *, label: str = "CODE_PLANNER_TRAJECTORY"
) -> str:
    """spec §9.2 render: per milestone — intent, APIs offered, step plan."""
    lines = [f"{label}:"]
    for i, m in enumerate(milestone_entries):
        lines.append(f"\n--- milestone {i}: {m.get('milestone')}")
        lines.append(f"APIs offered (finder candidates): {m.get('apis')}")
        plan = m.get("code_plan") or {}
        lines.append("step plan:")
        for step in plan.get("plan_steps") or []:
            lines.append(f"  - {step}")
        cs = plan.get("construct_step")
        if cs:
            lines.append(f"  - (construct) {cs}")
        ps = plan.get("print_step")
        if ps:
            lines.append(f"  - (print) {ps}")
    return "\n".join(lines)


def render_component_slice(component: str, slc: dict) -> tuple[str, str]:
    """(task_instruction, reference render) for any trained component."""
    if component == "code_executor":
        return slc["instruction"], render_executor_trajectory(
            slc.get("code_executor", [])
        )
    if component == "code_planner":
        return slc["instruction"], render_code_planner_trajectory(
            slc.get("code_planner", [])
        )
    if component == "rough_planner":
        rp = slc.get("rough_planner") or {}
        return slc["instruction"], render_rough_planner_output(
            rp.get("thoughts") or "", rp.get("milestones") or []
        )
    raise KeyError(f"unknown component: {component!r}")


__all__ = [
    "render_code_planner_trajectory",
    "render_component_slice",
    "render_executor_trajectory",
    "render_induction_slice",
    "render_rough_planner_output",
]
