"""Planner curation runner (code_planner + rough_planner), fully live.

Unlike the executor runner there is NO cached round-1 to replay: planners have no
prior curation, so every round runs live (gemini judge + full-downstream outcome
deduction via run_outcome, Path B). _1 baselines read from the _1 training log;
_2 baselines computed lazily from the original _2 skill (no _2 outcome log exists).

Downstream executor skills_root = skill_libraries (has every task's executor best).
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from dotenv import load_dotenv

from adk_appworld_agent.repo_paths import repo_env_path

load_dotenv(dotenv_path=repo_env_path())

from mind_skill.curation.curate import (
    MergeVerifier,
    curate_merge_to_convergence,
    make_planner_deduct_one,
)
from mind_skill.paths import GOLD_TRAJECTORIES_DIR, REFINED_DIR, STAGE1_CONSOLIDATED
from mind_skill.runtime import meta_model_config

SK = Path("data/mind_skill/skills")
SKILL_LIB = SK / "skill_libraries"  # downstream executor root + planner skill source
RPC = "tcp://127.0.0.1:4243"


def _baseline_log_dir(comp: str) -> Path | None:
    runs = sorted((Path("logs/mind_skill") / comp).glob("*/"))
    return runs[-1] if runs else None  # the _1 training run dir


def _build_round0(comp: str, round0: Path) -> int:
    """round0 = the _1+_2 planner skills (exclude partial _3), as <task>/<skill>/SKILL.md."""
    if round0.exists():
        shutil.rmtree(round0)
    round0.mkdir(parents=True)
    src = SKILL_LIB / comp / "best"
    n = 0
    for td in sorted(src.iterdir()):
        if td.is_dir() and td.name.rsplit("_", 1)[-1] in ("1", "2"):
            shutil.copytree(td, round0 / td.name)
            n += 1
    return n


def _make_baseline_provider(comp: str):
    """Original _2 (or any) skill md for a task -> lazy baseline compute."""
    src = SKILL_LIB / comp / "best"

    def provider(task_id: str) -> str | None:
        d = src / task_id
        ps = list(d.glob("*/SKILL.md")) if d.is_dir() else []
        return ps[0].read_text(encoding="utf-8") if ps else None

    return provider


async def curate_one(comp: str) -> dict:
    base = REFINED_DIR / comp
    round0 = base / "round0_combined"
    lib = base / STAGE1_CONSOLIDATED
    state = base / "engine_state"
    snap = base / "engine_rounds"
    n0 = _build_round0(comp, round0)
    # RESUME-aware: KEEP merge_eval_cache.json (content-keyed deduction results ->
    # the expensive _2 baselines are reused, not re-run). RESET round progress +
    # the merge-updated baseline_cache (its values are post-merge, not original).
    state.mkdir(parents=True, exist_ok=True)
    for f in (
        "baseline_cache.json",
        "merge_audit.json",
        "independence.json",
        "provenance.json",
        "rounds.json",
        "summary.json",
    ):
        (state / f).unlink(missing_ok=True)
    for p in (lib, snap):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(round0, lib)
    print(
        f"[{comp}] round0 = {n0} skills (_1+_2); downstream executor = skill_libraries",
        flush=True,
    )

    verifier = MergeVerifier(
        deduct_one=make_planner_deduct_one(
            comp,
            runs_dir=GOLD_TRAJECTORIES_DIR,
            model_cfg=meta_model_config(),
            rpc_url=RPC,
            skills_root=SKILL_LIB,
        ),
        work_dir=Path("logs/mind_skill/curation")
        / comp
        / "verify_work",  # stable, not /tmp
        baseline_cache_path=state / "baseline_cache.json",
        merge_eval_cache_path=state / "merge_eval_cache.json",
        audit_path=state / "merge_audit.json",
        baseline_log_dir=_baseline_log_dir(comp),
        baseline_skill_md=_make_baseline_provider(comp),
        verify_timeout_s=600.0,  # cut a hung deduction (gemini stream / sandbox)
        verify_retries=1,  # retry once (stalls are intermittent)
    )
    out = await curate_merge_to_convergence(
        lib,
        threshold=0.80,
        verifier=verifier.verify,
        state_dir=state,
        snapshot_dir=snap,
        max_rounds=2,
    )
    (state / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[{comp}] DONE converged={out['converged']} n {n0}->{out['n_final']} edges={out['edges']}",
        flush=True,
    )
    for rl in out["rounds"]:
        print(
            f"  [{comp}] round{rl['round']}: n {rl['n_start']}->{rl['n_end']} "
            f"merged={rl['merged_count']} rejected={len(rl['rejected'])} "
            f"distinct={len(rl['distinct'])} blocked={len(rl['skipped_blocked'])}",
            flush=True,
        )
    return out


async def main():
    comps = sys.argv[1:] or ["code_planner", "rough_planner"]
    for comp in comps:
        try:
            await curate_one(comp)
        except Exception as exc:
            import traceback

            print(f"[{comp}] FAILED: {exc}", flush=True)
            traceback.print_exc()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
