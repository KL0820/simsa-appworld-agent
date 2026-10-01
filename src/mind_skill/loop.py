"""Closed-loop driver (spec §10): per (component, trajectory) × Q iterations.

For each gold task T:

    P_I = P_I^(0)
    for q in 0..Q-1:
        s_q  = A_I(task_T, slice_T; P_I)          # induction
        τ̂_q  = deduction(T, s_q)                  # teacher-forced, live sandbox
        ℓ    = (outcome, recon, rubric)           # lexicographic
        if ℓ < best: best = s_q                   # paper Algo 1 L12; leak gate removed
        if q < Q-1:
            g   = gradient(P_I, s_q, τ̂_q, feedback)   # never sees τ
            P_I = optimizer(P_I, g) or P_I

EVERY iteration's artifacts are persisted (user requirement: compare q=0/1/2
libraries, not just best-so-far):

    <out_dir>/<task_id>/q{n}/{induction_prompt.txt, skill.md, deduction.json,
                              source_component_trajectory.json,
                              reconstructed_trajectory.json, judgments.json,
                              gradient.txt}
    <out_dir>/<task_id>/result.json        (per-task summary + best pointer)
    <out_dir>/journal.jsonl                (every meta-LLM request/response)

`cli.finalize` then copies this keep-list into the canonical
data/mind_skill/training_runs/<component>/<run_tag>/ (dropping run scratch such as
sandbox_api_calls_*.jsonl) and publishes the per-q libraries to
skills/induced/<component>/<run_tag>/{q0,q1,q2,best}/.

`assemble_libraries` then builds   <skills_root>/<component>/{q0,q1,...,best}/
<task_id>/<skill-name>/SKILL.md   (leaf dir name == frontmatter name, so each
library stays native-ADK loadable).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adk_appworld_agent.orchestration.run_config import ModelConfig
from mind_skill.deduction.code_executor import (
    DEFAULT_RPC_URL,
    GoldTask,
    load_gold_task,
    run_deduction,
)
from mind_skill.deduction.gold import pick_gold_trajectory
from mind_skill.induction.agent import (
    PromptHolder,
    build_induction_agent,
    induce_skill,
)
from mind_skill.induction.schemas import render_skill_md
from mind_skill.judges.losses import (
    LossTriple,
    outcome_loss,
    recon_loss,
    rubric_loss,
)
from mind_skill.judges.recon import build_recon_judge, judge_reconstruction
from mind_skill.judges.rubric import build_rubric_judge, judge_skill
from mind_skill.runtime import meta_model_config
from mind_skill.textgrad.gradient import (
    build_gradient_agent,
    build_gradient_input,
    compute_gradient,
)
from mind_skill.textgrad.optimizer import build_optimizer_agent, optimize_prompt
from mind_skill.trajectory.render import render_executor_trajectory

COMPONENT = "code_executor"  # executor-first (spec §14); planner variants later


# ── slice loading (induction input + recon reference) ─────────────────────────


def load_executor_slice(runs_dir: Path, task_id: str) -> dict:
    """Load a stored slice or derive it from the canonical gold trajectory."""
    traj_path = pick_gold_trajectory(runs_dir, task_id)
    slice_path = traj_path.parent / "induction_slices.json"
    if slice_path.exists():
        return json.loads(slice_path.read_text(encoding="utf-8"))
    from mind_skill.induction.make_induction_slices import slices as build_slices

    data = json.loads(traj_path.read_text(encoding="utf-8"))
    return build_slices(data)


# ── per-task closed loop ──────────────────────────────────────────────────────


@dataclass
class IterationOutcome:
    q: int
    skill_name: str
    skill_md: str
    losses: LossTriple
    deduction_cleared: bool | None  # None = planner loop (no environment outcome)


@dataclass
class TaskLoopResult:
    task_id: str
    iterations: list[IterationOutcome]
    best_q: int | None  # None only if there are zero iterations


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _reconstructed_traj(dres) -> object:
    """Structured reconstructed trajectory tau-hat when the deduction exposes one
    (executor), else the text render (planner deductions)."""
    fn = getattr(dres, "reconstruction_entries", None)
    return fn() if callable(fn) else {"render": dres.reconstruction_render()}


def _iter_meta(component: str, outcome: bool, skills_root) -> dict:
    # Provenance of a persisted iteration: the axes that change what the
    # deduction actually ran (downstream skill source + outcome mode). corpus is
    # keyed out by task_id; component is in the path.
    return {
        "component": component,
        "outcome": bool(outcome),
        "skills_root": str(skills_root) if skills_root else None,
    }


def _resume_meta_ok(q_dir: Path, component: str, outcome: bool, skills_root) -> bool:
    """A persisted q is reusable only if it was produced with the SAME provenance.
    Missing meta.json = legacy run -> treated as compatible (backward-compat);
    corrupt = not reusable (re-run rather than trust)."""
    p = q_dir / "meta.json"
    if not p.exists():
        return True
    try:
        return json.loads(p.read_text(encoding="utf-8")) == _iter_meta(
            component, outcome, skills_root
        )
    except Exception:
        return False


async def run_task_loop(
    task_id: str,
    *,
    runs_dir: Path,
    out_dir: Path,
    appworld_data_root: Path,
    q_iterations: int = 3,
    rpc_url: str = DEFAULT_RPC_URL,
    model_cfg: ModelConfig | None = None,
) -> TaskLoopResult:
    model_cfg = model_cfg or meta_model_config()
    task_dir = out_dir / task_id
    journal = out_dir / "journal.jsonl"

    holder = PromptHolder(COMPONENT)  # fresh P_I^(0) per (C, T)
    induction_agent = build_induction_agent(holder, journal_path=journal)
    recon_agent = build_recon_judge(journal_path=journal)
    rubric_agent = build_rubric_judge(journal_path=journal)
    gradient_agent = build_gradient_agent(journal_path=journal)
    optimizer_agent = build_optimizer_agent(journal_path=journal)

    slc = load_executor_slice(runs_dir, task_id)
    task_instruction = slc["instruction"]
    reference_render = render_executor_trajectory(slc.get("code_executor", []))
    gold: GoldTask = load_gold_task(runs_dir, task_id)

    iterations: list[IterationOutcome] = []
    best: tuple[LossTriple, int] | None = None

    for q in range(q_iterations):
        q_dir = task_dir / f"q{q}"

        # Per-q resume: a fully persisted iteration (incl. the TextGrad step's
        # next prompt, unless final q) is reloaded instead of re-spent. The
        # prompt chain stays exact because prompt.txt snapshots P_I^(q).
        next_prompt = task_dir / f"q{q + 1}" / "induction_prompt.txt"
        if (
            (q_dir / "judgments.json").exists()
            and (q_dir / "skill.md").exists()
            and (q == q_iterations - 1 or next_prompt.exists())
            and _resume_meta_ok(q_dir, COMPONENT, True, None)
        ):
            saved = json.loads((q_dir / "judgments.json").read_text(encoding="utf-8"))
            skill_md = (q_dir / "skill.md").read_text(encoding="utf-8")
            losses = LossTriple(**saved["losses"])
            dres_saved = json.loads(
                (q_dir / "deduction.json").read_text(encoding="utf-8")
            )
            iterations.append(
                IterationOutcome(
                    q=q,
                    skill_name=skill_md.split("name:", 1)[1].split("\n", 1)[0].strip(),
                    skill_md=skill_md,
                    losses=losses,
                    deduction_cleared=bool(dres_saved.get("cleared")),
                )
            )
            if best is None or losses < best[0]:
                best = (losses, q)
            if next_prompt.exists():
                holder.current = next_prompt.read_text(encoding="utf-8")
            continue

        _write(q_dir / "induction_prompt.txt", holder.current)

        skill = await induce_skill(
            induction_agent,
            task_instruction=task_instruction,
            slice_render=reference_render,
        )
        skill_md = render_skill_md(skill)
        _write(q_dir / "skill.md", skill_md)
        _write(q_dir / "meta.json", json.dumps(_iter_meta(COMPONENT, True, None)))

        dres = await run_deduction(
            gold,
            skill_texts=[skill_md],
            model_cfg=model_cfg,
            rpc_url=rpc_url,
            work_dir=q_dir,
        )
        _write(
            q_dir / "deduction.json",
            json.dumps(dres.as_dict(), ensure_ascii=False, indent=2),
        )
        _write(
            q_dir / "source_component_trajectory.json",
            json.dumps(slc.get(COMPONENT, []), ensure_ascii=False, indent=2),
        )
        _write(
            q_dir / "reconstructed_trajectory.json",
            json.dumps(_reconstructed_traj(dres), ensure_ascii=False, indent=2),
        )
        reconstruction_render = dres.reconstruction_render()

        recon_j = await judge_reconstruction(
            recon_agent,
            task_instruction=task_instruction,
            reference_render=reference_render,
            reconstruction_render=reconstruction_render,
        )
        rubric_j = await judge_skill(
            rubric_agent, task_instruction=task_instruction, skill_md=skill_md
        )
        losses = LossTriple(
            outcome=outcome_loss(
                passed=dres.passed, failed=dres.failed, total=dres.total
            ),
            recon=recon_loss(recon_j),
            rubric=rubric_loss(rubric_j),
        )
        _write(
            q_dir / "judgments.json",
            json.dumps(
                {
                    "recon": recon_j.model_dump(mode="json"),
                    "rubric": rubric_j.model_dump(mode="json"),
                    "losses": losses.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        iterations.append(
            IterationOutcome(
                q=q,
                skill_name=skill.name,
                skill_md=skill_md,
                losses=losses,
                deduction_cleared=dres.cleared,
            )
        )
        if (
            best is None or losses < best[0]
        ):  # lexicographic best-so-far (paper Algo 1 L12)
            best = (losses, q)

        if q < q_iterations - 1:
            gradient_text = await compute_gradient(
                gradient_agent,
                build_gradient_input(
                    induction_prompt=holder.current,
                    skill_md=skill_md,
                    reconstruction_render=reconstruction_render,
                    outcome_feedback=dres.outcome_feedback(),
                    recon_judgment=recon_j,
                    rubric_judgment=rubric_j,
                ),
            )
            _write(q_dir / "gradient.txt", gradient_text)
            improved = await optimize_prompt(
                optimizer_agent,
                induction_prompt=holder.current,
                gradient_feedback=gradient_text,
            )
            if improved:
                holder.current = improved
            # else: keep P_I^(q) — a failed optimizer step must not corrupt it.

    result = TaskLoopResult(
        task_id=task_id,
        iterations=iterations,
        best_q=best[1] if best is not None else None,
    )
    _write(
        task_dir / "result.json",
        json.dumps(
            {
                "task_id": task_id,
                "best_q": result.best_q,
                "iterations": [
                    {
                        "q": it.q,
                        "skill_name": it.skill_name,
                        "losses": it.losses.as_dict(),
                        "deduction_cleared": it.deduction_cleared,
                    }
                    for it in iterations
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return result


# ── planner closed loop (recon+rubric; deduction-lite, no sandbox) ───────────


async def run_planner_task_loop(
    task_id: str,
    *,
    component: str,
    runs_dir: Path,
    out_dir: Path,
    appworld_data_root: Path,
    q_iterations: int = 2,
    model_cfg: ModelConfig | None = None,
    outcome: bool = False,
    rpc_url: str = DEFAULT_RPC_URL,
    skills_root: Path | None = None,
) -> TaskLoopResult:
    """run_task_loop's planner sibling: gold input replayed verbatim through
    the live thin-prompt component (deduction/<component>.py).

    outcome=False: recon+rubric only (outcome loss fixed 0.0, spec §7 note).
    outcome=True (code_planner only): generated plans are executed by the
    deployment-baseline executor in the real sandbox -> real outcome loss.
    """
    from mind_skill.deduction import PLANNER_DEDUCTIONS
    from mind_skill.trajectory.render import render_component_slice

    deduction_module = PLANNER_DEDUCTIONS[component]
    if outcome and component not in ("code_planner", "rough_planner"):
        raise ValueError(
            "outcome-grounded planner loop is code_planner / rough_planner-only"
        )

    model_cfg = model_cfg or meta_model_config()
    task_dir = out_dir / task_id
    journal = out_dir / "journal.jsonl"

    holder = PromptHolder(component)
    induction_agent = build_induction_agent(holder, journal_path=journal)
    recon_agent = build_recon_judge(component=component, journal_path=journal)
    rubric_agent = build_rubric_judge(journal_path=journal)
    gradient_agent = build_gradient_agent(journal_path=journal)
    optimizer_agent = build_optimizer_agent(journal_path=journal)

    slc = load_executor_slice(runs_dir, task_id)  # same slices file, all layers
    task_instruction, reference_render = render_component_slice(component, slc)
    gold = deduction_module.load_gold(runs_dir, task_id)

    iterations: list[IterationOutcome] = []
    best: tuple[LossTriple, int] | None = None

    for q in range(q_iterations):
        q_dir = task_dir / f"q{q}"

        next_prompt = task_dir / f"q{q + 1}" / "induction_prompt.txt"
        if (
            (q_dir / "judgments.json").exists()
            and (q_dir / "skill.md").exists()
            and (q == q_iterations - 1 or next_prompt.exists())
            and _resume_meta_ok(q_dir, component, outcome, skills_root)
        ):
            saved = json.loads((q_dir / "judgments.json").read_text(encoding="utf-8"))
            skill_md = (q_dir / "skill.md").read_text(encoding="utf-8")
            losses = LossTriple(**saved["losses"])
            ded_path = q_dir / "deduction.json"
            saved_cleared = None
            if outcome and ded_path.exists():
                saved_cleared = bool(
                    json.loads(ded_path.read_text(encoding="utf-8")).get("cleared")
                )
            iterations.append(
                IterationOutcome(
                    q=q,
                    skill_name=skill_md.split("name:", 1)[1].split("\n", 1)[0].strip(),
                    skill_md=skill_md,
                    losses=losses,
                    deduction_cleared=saved_cleared,
                )
            )
            if best is None or losses < best[0]:
                best = (losses, q)
            if next_prompt.exists():
                holder.current = next_prompt.read_text(encoding="utf-8")
            continue

        _write(q_dir / "induction_prompt.txt", holder.current)

        skill = await induce_skill(
            induction_agent,
            task_instruction=task_instruction,
            slice_render=reference_render,
        )
        skill_md = render_skill_md(skill)
        _write(q_dir / "skill.md", skill_md)
        _write(
            q_dir / "meta.json", json.dumps(_iter_meta(component, outcome, skills_root))
        )

        if outcome:
            dres, exec_dres = await deduction_module.run_outcome(
                gold,
                skill_texts=[skill_md],
                model_cfg=model_cfg,
                rpc_url=rpc_url,
                work_dir=q_dir,
                runs_dir=runs_dir,
                skills_root=skills_root,
            )
            _write(
                q_dir / "deduction.json",
                json.dumps(
                    {"planner": dres.as_dict(), **exec_dres.as_dict()},
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        else:
            exec_dres = None
            dres = await deduction_module.run(
                gold, skill_texts=[skill_md], model_cfg=model_cfg
            )
            _write(
                q_dir / "deduction.json",
                json.dumps(dres.as_dict(), ensure_ascii=False, indent=2),
            )
        _write(
            q_dir / "source_component_trajectory.json",
            json.dumps(slc.get(component, []), ensure_ascii=False, indent=2),
        )
        _write(
            q_dir / "reconstructed_trajectory.json",
            json.dumps(_reconstructed_traj(dres), ensure_ascii=False, indent=2),
        )
        reconstruction_render = dres.reconstruction_render()

        recon_j = await judge_reconstruction(
            recon_agent,
            task_instruction=task_instruction,
            reference_render=reference_render,
            reconstruction_render=reconstruction_render,
        )
        rubric_j = await judge_skill(
            rubric_agent, task_instruction=task_instruction, skill_md=skill_md
        )
        losses = LossTriple(
            outcome=(
                outcome_loss(
                    passed=exec_dres.passed,
                    failed=exec_dres.failed,
                    total=exec_dres.total,
                )
                if exec_dres is not None
                else 0.0  # no environment rollout in the recon-only loop
            ),
            recon=recon_loss(recon_j),
            rubric=rubric_loss(rubric_j),
        )
        _write(
            q_dir / "judgments.json",
            json.dumps(
                {
                    "recon": recon_j.model_dump(mode="json"),
                    "rubric": rubric_j.model_dump(mode="json"),
                    "losses": losses.as_dict(),
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

        iterations.append(
            IterationOutcome(
                q=q,
                skill_name=skill.name,
                skill_md=skill_md,
                losses=losses,
                deduction_cleared=exec_dres.cleared if exec_dres is not None else None,
            )
        )
        if (
            best is None or losses < best[0]
        ):  # lexicographic best-so-far (paper Algo 1 L12)
            best = (losses, q)

        if q < q_iterations - 1:
            gradient_text = await compute_gradient(
                gradient_agent,
                build_gradient_input(
                    induction_prompt=holder.current,
                    skill_md=skill_md,
                    reconstruction_render=reconstruction_render,
                    outcome_feedback=(
                        exec_dres.outcome_feedback()
                        if exec_dres is not None
                        else dres.outcome_feedback()
                    ),
                    recon_judgment=recon_j,
                    rubric_judgment=rubric_j,
                ),
            )
            _write(q_dir / "gradient.txt", gradient_text)
            improved = await optimize_prompt(
                optimizer_agent,
                induction_prompt=holder.current,
                gradient_feedback=gradient_text,
            )
            if improved:
                holder.current = improved

    result = TaskLoopResult(
        task_id=task_id,
        iterations=iterations,
        best_q=best[1] if best is not None else None,
    )
    _write(
        task_dir / "result.json",
        json.dumps(
            {
                "task_id": task_id,
                "best_q": result.best_q,
                "iterations": [
                    {
                        "q": it.q,
                        "skill_name": it.skill_name,
                        "losses": it.losses.as_dict(),
                        "deduction_cleared": it.deduction_cleared,
                    }
                    for it in iterations
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return result


# ── library assembly (per-q + best) ───────────────────────────────────────────


def assemble_libraries(
    out_dir: Path, skills_root: Path, *, q_iterations: int, component: str = COMPONENT
) -> dict:
    """Collect per-task artifacts into per-q and best skill libraries.

    Layout: <skills_root>/<component>/{q0..,best}/<task_id>/<skill-name>/SKILL.md
    (leaf dir name == frontmatter name -> native ADK loadable per library).
    `best` is the lexicographic-min iteration (paper Algo 1). `quarantined`
    stays in the summary schema but is empty unless a task has zero iterations.
    """
    summary: dict = {"libraries": {}, "quarantined": []}
    # Rebuild from scratch: successive assembles (e.g. after re-deriving best_q) must
    # not accumulate stale skill dirs from earlier best selections.
    import shutil

    component_root = skills_root / component
    if component_root.exists():
        shutil.rmtree(component_root)
    results = sorted(Path(out_dir).glob("*/result.json"))

    def _install(library: str, task_id: str, skill_md: str) -> None:
        name = skill_md.split("name:", 1)[1].split("\n", 1)[0].strip()
        dest = skills_root / component / library / task_id / name / "SKILL.md"
        _write(dest, skill_md)
        summary["libraries"].setdefault(library, 0)
        summary["libraries"][library] += 1

    for result_path in results:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_id = result["task_id"]
        task_dir = result_path.parent
        for q in range(q_iterations):
            skill_path = task_dir / f"q{q}" / "skill.md"
            if skill_path.exists():
                _install(f"q{q}", task_id, skill_path.read_text(encoding="utf-8"))
        if result.get("best_q") is None:
            summary["quarantined"].append(task_id)
        else:
            best_path = task_dir / f"q{result['best_q']}" / "skill.md"
            _install("best", task_id, best_path.read_text(encoding="utf-8"))

    _write(
        skills_root / component / "assembly_summary.json",
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    return summary


__all__ = [
    "COMPONENT",
    "IterationOutcome",
    "TaskLoopResult",
    "assemble_libraries",
    "load_executor_slice",
    "run_planner_task_loop",
    "run_task_loop",
]
