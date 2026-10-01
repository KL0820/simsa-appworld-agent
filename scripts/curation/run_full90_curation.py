"""Full-90 fresh two-stage curation (_1+_2+_3) — 2026-07-03.

Round0 per component = union of the three induced best libraries (30+30+30):
  induced/<comp>/{<_1 timestamp>, _2smoke_q3, _3full_q3}/best
Stage 1: body-similarity merge to convergence (0.80 pairwise, deduction-gated).
Stage 2: name+desc sharpen to convergence (0.80 pairwise, no deduction).

Baselines are SEEDED from the three training runs' logs (best-q deduction
passed/total) — no baseline recompute. Downstream Path B skills_root =
skills/_90combined (all components' induced best, 90 tasks each).

Previous _1+_2 curation archived at data/mind_skill/_archive/refined_1plus2_20260703
(+ release_thesis_final_1plus2_20260703). This run writes the canonical layout
fresh: refined/<comp>/{round0_combined, stage1_consolidated,
stage2_name_description_refined, engine_state, engine_rounds, stage2_state}.

Run from worktree root (RPC 4243 up):
    PYTHONPATH=src .venv/bin/python scripts/curation/run_full90_curation.py \
        [code_executor code_planner rough_planner]
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

# CURATION90_RESUME=1 -> keep engine state/caches + current library (kill-safe
# continuation; the engine's content-keyed eval cache skips finished deductions).
RESUME = os.environ.get("CURATION90_RESUME") == "1"

sys.path.insert(0, "src")
from dotenv import load_dotenv

from adk_appworld_agent.repo_paths import repo_env_path

load_dotenv(dotenv_path=repo_env_path())

from mind_skill.curation.curate import (
    MergeVerifier,
    _llm_sharpener,
    build_curate_judge,
    build_sharpen_judge,
    curate_merge_to_convergence,
    make_executor_deduct_one,
    make_planner_deduct_one,
    sharpen_to_convergence,
)
from mind_skill.curation.dedup import _llm_resolver
from mind_skill.paths import (
    GOLD_TRAJECTORIES_DIR,
    REFINED_DIR,
    SKILLS_DIR,
    STAGE1_CONSOLIDATED,
    STAGE2_REFINED,
)
from mind_skill.runtime import meta_model_config

INDUCED = SKILLS_DIR / "induced"
COMBINED = SKILLS_DIR / "_90combined"
LOGS = Path("logs/mind_skill")
RPC = "tcp://127.0.0.1:4243"

TAG_1 = {
    "code_executor": "20260627T155739Z",
    "code_planner": "20260627T202503Z",
    "rough_planner": "20260628T024057Z",
}
# per-deduction wall cut: executor = single teacher-forced pass; code_planner adds
# one downstream executor pass; rough_planner runs the FULL controller (heaviest).
VERIFY_TIMEOUT_S = {
    "code_executor": 300.0,
    "code_planner": 900.0,
    "rough_planner": 1800.0,
}


def tags(comp: str) -> list[str]:
    return [TAG_1[comp], "_2smoke_q3", "_3full_q3"]


def build_combined() -> None:
    """skills/_90combined/<comp>/best/<tid>/ — Path B downstream skill root."""
    if COMBINED.exists():
        shutil.rmtree(COMBINED)
    for comp in TAG_1:
        dest = COMBINED / comp / "best"
        dest.mkdir(parents=True)
        for tag in tags(comp):
            for td in sorted((INDUCED / comp / tag / "best").iterdir()):
                if td.is_dir():
                    if (dest / td.name).exists():
                        raise RuntimeError(f"duplicate task {comp}/{td.name}")
                    shutil.copytree(td, dest / td.name)
        n = sum(1 for p in dest.iterdir() if p.is_dir())
        print(f"[combined] {comp}: {n} tasks", flush=True)
        if n != 90:
            raise RuntimeError(f"{comp} combined != 90 ({n})")


def seed_baselines(comp: str, state: Path) -> int:
    """task_id -> [passed, total] from each training run's best-q deduction."""
    cache: dict = {}
    for tag in tags(comp):
        for rp in sorted((LOGS / comp / tag).glob("*/result.json")):
            r = json.loads(rp.read_text(encoding="utf-8"))
            if r.get("best_q") is None:
                continue
            dp = rp.parent / f"q{r['best_q']}" / "deduction.json"
            if dp.exists():
                dd = json.loads(dp.read_text(encoding="utf-8"))
                cache[rp.parent.name] = [
                    int(dd.get("passed", 0)),
                    int(dd.get("total", 0)),
                ]
    state.mkdir(parents=True, exist_ok=True)
    (state / "baseline_cache.json").write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{comp}] seeded {len(cache)} baselines from training logs", flush=True)
    return len(cache)


def baseline_provider(round0: Path):
    """Lazy fallback: original skill md for a task (only used if a log was missing)."""

    def provider(task_id: str) -> str | None:
        d = round0 / task_id
        ps = list(d.glob("*/SKILL.md")) if d.is_dir() else []
        return ps[0].read_text(encoding="utf-8") if ps else None

    return provider


async def stage1(comp: str) -> None:
    base = REFINED_DIR / comp
    round0 = base / "round0_combined"
    lib = base / STAGE1_CONSOLIDATED
    state = base / "engine_state"
    snap = base / "engine_rounds"

    if not (RESUME and round0.exists()):
        if round0.exists():
            shutil.rmtree(round0)
        round0.mkdir(parents=True)
        for tag in tags(comp):
            for td in sorted((INDUCED / comp / tag / "best").iterdir()):
                if td.is_dir():
                    shutil.copytree(td, round0 / td.name)
    n0 = sum(1 for p in round0.iterdir() if p.is_dir())
    if n0 != 90:
        raise RuntimeError(f"{comp} round0 != 90 ({n0})")
    if RESUME and lib.exists() and (state / "baseline_cache.json").exists():
        n_lib = sum(1 for p in lib.iterdir() if p.is_dir())
        print(f"[{comp}] RESUME: keep state + library ({n_lib} skills)", flush=True)
    else:
        for p in (lib, snap, state):
            if p.exists():
                shutil.rmtree(p)
        shutil.copytree(round0, lib)
        seed_baselines(comp, state)

    work = LOGS / "curation90" / comp / "verify_work"
    work.mkdir(parents=True, exist_ok=True)
    if comp == "code_executor":
        deduct = make_executor_deduct_one(
            runs_dir=GOLD_TRAJECTORIES_DIR, model_cfg=meta_model_config(), rpc_url=RPC
        )
    else:
        deduct = make_planner_deduct_one(
            comp,
            runs_dir=GOLD_TRAJECTORIES_DIR,
            model_cfg=meta_model_config(),
            rpc_url=RPC,
            skills_root=COMBINED,
        )
    verifier = MergeVerifier(
        deduct_one=deduct,
        work_dir=work,
        baseline_cache_path=state / "baseline_cache.json",
        merge_eval_cache_path=state / "merge_eval_cache.json",
        audit_path=state / "merge_audit.json",
        baseline_skill_md=baseline_provider(round0),
        verify_timeout_s=VERIFY_TIMEOUT_S[comp],
        verify_retries=1,
    )
    resolver = _llm_resolver(
        build_curate_judge(journal_path=state / "judge_journal.jsonl")
    )
    start_round = 1
    rounds_path = state / "rounds.json"
    if RESUME and rounds_path.exists():
        start_round = len(json.loads(rounds_path.read_text(encoding="utf-8"))) + 1
    print(
        f"[{comp}] STAGE1 start: threshold=0.80 start_round={start_round}", flush=True
    )
    summary = await curate_merge_to_convergence(
        lib,
        threshold=0.80,
        resolver=resolver,
        verifier=verifier.verify,
        state_dir=state,
        snapshot_dir=snap,
        start_round=start_round,
        max_rounds=10,
    )
    n1 = sum(1 for p in lib.iterdir() if p.is_dir())
    print(
        f"[{comp}] STAGE1 done: {n0} -> {n1} skills | {json.dumps(summary, ensure_ascii=False)[:300]}",
        flush=True,
    )


async def stage2(comp: str) -> None:
    base = REFINED_DIR / comp
    src = base / STAGE1_CONSOLIDATED
    dst = base / STAGE2_REFINED
    state = base / "stage2_state"
    snap = base / "stage2_rounds"
    if RESUME and dst.exists() and (state / "sharpened_pairs.json").exists():
        print(f"[{comp}] STAGE2 RESUME: keep {dst.name} + settle-set", flush=True)
    else:
        for p in (dst, state, snap):
            if p.exists():
                shutil.rmtree(p)
        shutil.copytree(src, dst)  # non-destructive: sharpen the COPY
    sharpener = _llm_sharpener(build_sharpen_judge(journal_path=None))
    print(f"[{comp}] STAGE2 start", flush=True)
    summary = await sharpen_to_convergence(
        dst,
        threshold=0.80,
        sharpener=sharpener,
        state_dir=state,
        snapshot_dir=snap,
        max_rounds=10,
    )
    print(
        f"[{comp}] STAGE2 done | {json.dumps(summary, ensure_ascii=False)[:300]}",
        flush=True,
    )


async def main() -> None:
    comps = sys.argv[1:] or ["code_executor", "code_planner", "rough_planner"]
    build_combined()
    for comp in comps:
        await stage1(comp)
        await stage2(comp)
    print("CURATION90-ALL-DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
