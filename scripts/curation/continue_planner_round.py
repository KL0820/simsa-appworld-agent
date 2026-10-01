"""Continue code_planner + rough_planner curation from round 3 to convergence.

Resume mode: does NOT reset engine_lib/state — reuses the round2-output library,
the independence graph, and all caches (baseline_cache holds the current-best per
task, correct as the round3 no-decrease reference). Stops when a round merges 0.
"""

import asyncio
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
SKILL_LIB = SK / "skill_libraries"
RPC = "tcp://127.0.0.1:4243"


def _baseline_log_dir(comp):
    r = sorted((Path("logs/mind_skill") / comp).glob("*/"))
    return r[-1] if r else None


def _provider(comp):
    src = SKILL_LIB / comp / "best"

    def p(t):
        d = src / t
        ps = list(d.glob("*/SKILL.md")) if d.is_dir() else []
        return ps[0].read_text("utf-8") if ps else None

    return p


async def main():
    for comp in ("code_planner", "rough_planner"):
        base = REFINED_DIR / comp
        lib, state, snap = (
            base / STAGE1_CONSOLIDATED,
            base / "engine_state",
            base / "engine_rounds",
        )
        print(f"[{comp}] continue round3+ (resume; keep caches+graph)", flush=True)
        v = MergeVerifier(
            deduct_one=make_planner_deduct_one(
                comp,
                runs_dir=GOLD_TRAJECTORIES_DIR,
                model_cfg=meta_model_config(),
                rpc_url=RPC,
                skills_root=SKILL_LIB,
            ),
            work_dir=Path("logs/mind_skill/curation") / comp / "verify_work",
            baseline_cache_path=state / "baseline_cache.json",
            merge_eval_cache_path=state / "merge_eval_cache.json",
            audit_path=state / "merge_audit.json",
            baseline_log_dir=_baseline_log_dir(comp),
            baseline_skill_md=_provider(comp),
            verify_timeout_s=600.0,
            verify_retries=1,
        )
        out = await curate_merge_to_convergence(
            lib,
            threshold=0.80,
            verifier=v.verify,
            state_dir=state,
            snapshot_dir=snap,
            start_round=3,
            max_rounds=6,
        )
        print(
            f"[{comp}] DONE converged={out['converged']} n_final={out['n_final']}",
            flush=True,
        )
        for rl in out["rounds"]:
            print(
                f"  round{rl['round']}: {rl['n_start']}->{rl['n_end']} merged {rl['merged_count']} "
                f"rej {len(rl['rejected'])} dist {len(rl['distinct'])} infra {len(rl.get('skipped_infra', []))}",
                flush=True,
            )
    print("CONTINUE ALL DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
