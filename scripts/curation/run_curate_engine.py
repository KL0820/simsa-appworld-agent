"""Curation engine runner for code_executor: replay round 1 from cache (0 live),
then run round 2+ live to convergence, with independence-graph + coverage state.

Usage:
  python /tmp/run_curate_engine.py validate   # round-1 reproduction check, NO live
  python /tmp/run_curate_engine.py run         # full to convergence (live round 2+)
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
    ExecutorMergeVerifier,
    _content_sha,
    _md_sha,
    build_curate_judge,
    curate_merge_to_convergence,
)
from mind_skill.curation.dedup import (
    CollisionResolution,
    _llm_resolver,
    load_skill_files,
    render_skill_md,
)
from mind_skill.paths import GOLD_TRAJECTORIES_DIR, REFINED_DIR, STAGE1_CONSOLIDATED
from mind_skill.runtime import meta_model_config

BASE = REFINED_DIR / "code_executor"
ROUND0 = BASE / "round0_combined"  # 60, immutable
ROUND1_EXISTING = BASE / "round1"  # 46, from the old single-round run
DRY = "/tmp/curate_dryrun_exec.json"
R1_RESULT = BASE / "round1_result.json"
ORIG_BASELINES = "/tmp/curate_baselines.json"
BASELINE_LOG_DIR = Path("logs/mind_skill/code_executor/20260627T155739Z")
RPC = "tcp://127.0.0.1:4243"


def seed_state(state_dir: Path) -> None:
    """Seed baseline_cache (33 originals) + merge_eval_cache (round-1's 18 verify
    results, keyed by merged_md sha : task) so the engine replays round 1 with no
    deduction."""
    state_dir.mkdir(parents=True, exist_ok=True)
    # baselines
    shutil.copy(ORIG_BASELINES, state_dir / "baseline_cache.json")
    # eval cache from round-1 audit, matched to dry-run merged_md by source-task set
    dry = json.load(open(DRY))
    by_tasks = {}
    for r in dry:
        if r["verdict"] != "merge":
            continue
        ts = frozenset(s.split(":", 1)[0] for s in (r["skill_1"], r["skill_2"]))
        by_tasks[ts] = r
    audit = json.load(open(R1_RESULT))["audit"]
    eval_cache = {}
    matched = 0
    for a in audit:
        ts = frozenset(pt["task_id"] for pt in a["per_task"])
        dr = by_tasks.get(ts)
        if dr is None:
            print(f"  WARN: no dry-run merge for audit tasks {sorted(ts)}", flush=True)
            continue
        md = render_skill_md(
            dr["merged_name"], dr["merged_description"], dr["merged_body"]
        )
        sha = _md_sha(md)
        for pt in a["per_task"]:
            eval_cache[f"{sha}:{pt['task_id']}"] = {
                "merged_passed": pt["merged_passed"],
                "total": pt["total"],
                "cleared": pt.get("merged_cleared", False),
                "failed_requirements": pt.get("failed_requirements", []),
            }
        matched += 1
    (state_dir / "merge_eval_cache.json").write_text(
        json.dumps(eval_cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"  seeded baselines + eval cache: matched {matched}/{len(audit)} audit merges, "
        f"{len(eval_cache)} task-results",
        flush=True,
    )


def build_resolver(round0_dir: Path, *, allow_live: bool):
    """Cached-then-live resolver: round-1 pairs (original content) hit the dry-run
    verdict cache (keyed by content sha); round-2 pairs (merged content) miss -> live
    gemini judge. In validation, a miss raises (proves round 1 is fully cached)."""
    sha_by_key, index = {}, {}
    for s in load_skill_files(round0_dir):
        sha_by_key[f"{s.task_id}:{s.name}"] = _content_sha(s)
    for r in json.load(open(DRY)):
        ka, kb = r["skill_1"], r["skill_2"]
        if ka in sha_by_key and kb in sha_by_key:
            index[frozenset({sha_by_key[ka], sha_by_key[kb]})] = r
    live = _llm_resolver(build_curate_judge())
    stats = {"hit": 0, "live": 0}

    async def resolve(name, skills, avoid):
        k = frozenset({_content_sha(skills[0]), _content_sha(skills[1])})
        r = index.get(k)
        if r is not None:
            stats["hit"] += 1
            return CollisionResolution(
                verdict=r["verdict"],
                reason=r["reason"],
                merged_name=r["merged_name"],
                merged_description=r["merged_description"],
                merged_body=r["merged_body"],
            )
        stats["live"] += 1
        pair = [f"{s.task_id}:{s.name}" for s in skills]
        if not allow_live:
            raise RuntimeError(f"resolver cache MISS in validation: {pair}")
        print(f"  [live judge] {pair[0]}  +  {pair[1]}", flush=True)
        return await live(name, skills, avoid)

    return resolve, stats


def make_verifier(state_dir: Path, work_dir: Path, *, allow_live: bool):
    v = ExecutorMergeVerifier(
        runs_dir=GOLD_TRAJECTORIES_DIR,
        model_cfg=meta_model_config(),
        rpc_url=RPC,
        work_dir=work_dir,
        baseline_cache_path=state_dir / "baseline_cache.json",
        merge_eval_cache_path=state_dir / "merge_eval_cache.json",
        audit_path=state_dir / "merge_audit.json",
        baseline_log_dir=BASELINE_LOG_DIR,
    )
    if not allow_live:

        async def no_live(*a, **k):
            raise RuntimeError("deduction ran during validation (eval-cache miss)")

        v._run_deduction = no_live
    return v


async def validate():
    print("== VALIDATE: replay round 1 from cache, expect 46, 0 live ==", flush=True)
    W = BASE / "_validate_lib"
    S = BASE / "_validate_state"
    for p in (W, S):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(ROUND0, W)
    seed_state(S)
    resolver, rstats = build_resolver(ROUND0, allow_live=False)
    v = make_verifier(S, Path("/tmp/curate_validate_work"), allow_live=False)
    out = await curate_merge_to_convergence(
        W,
        threshold=0.80,
        resolver=resolver,
        verifier=v.verify,
        state_dir=S,
        max_rounds=1,
    )
    got = sorted(s.name for s in load_skill_files(W))
    want = sorted(s.name for s in load_skill_files(ROUND1_EXISTING))
    print(
        f"  round1 n_end={out['n_final']} (want 46)  live_judges={rstats['live']}",
        flush=True,
    )
    print(f"  names match existing round1: {got == want}", flush=True)
    if got != want:
        only_got = set(got) - set(want)
        only_want = set(want) - set(got)
        print(
            f"  MISMATCH only_in_replay={sorted(only_got)}\n  only_in_existing={sorted(only_want)}",
            flush=True,
        )
    for p in (W, S):
        shutil.rmtree(p)
    print("  PASS" if got == want and rstats["live"] == 0 else "  FAIL", flush=True)


async def run():
    print("== RUN: round 1 (cache) -> convergence (live round 2+) ==", flush=True)
    W = BASE / STAGE1_CONSOLIDATED
    S = BASE / "engine_state"
    SNAP = BASE / "engine_rounds"
    for p in (W, S, SNAP):
        if p.exists():
            shutil.rmtree(p)
    shutil.copytree(ROUND0, W)
    seed_state(S)
    resolver, rstats = build_resolver(ROUND0, allow_live=True)
    v = make_verifier(S, Path("/tmp/curate_engine_work"), allow_live=True)
    # round 1 = cache replay; round 2 = live. Stop after round 2 so the user can
    # inspect + decide whether to continue (per "合併一輪->驗證->再考慮要不要繼續").
    out = await curate_merge_to_convergence(
        W,
        threshold=0.80,
        resolver=resolver,
        verifier=v.verify,
        state_dir=S,
        snapshot_dir=SNAP,
        max_rounds=2,
    )
    (S / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"DONE converged={out['converged']} n_final={out['n_final']} edges={out['edges']} "
        f"live_judges={rstats['live']}",
        flush=True,
    )
    for i, rl in enumerate(out["rounds"], 1):
        print(
            f"  round{i}: n {rl['n_start']}->{rl['n_end']} merged={rl['merged_count']} "
            f"rejected={len(rl['rejected'])} distinct={len(rl['distinct'])} "
            f"blocked={len(rl['skipped_blocked'])} consumed={len(rl['skipped_consumed'])}",
            flush=True,
        )


async def continue_run(start_round: int):
    """Continue the existing engine run (do NOT reset) from `start_round` to
    convergence, reusing persisted graph/coverage/caches."""
    print(f"== CONTINUE from round {start_round} -> convergence ==", flush=True)
    W = BASE / STAGE1_CONSOLIDATED
    S = BASE / "engine_state"
    SNAP = BASE / "engine_rounds"
    assert W.exists() and S.exists(), "no prior engine run to continue"
    resolver, rstats = build_resolver(ROUND0, allow_live=True)
    v = make_verifier(S, Path("/tmp/curate_engine_work"), allow_live=True)
    out = await curate_merge_to_convergence(
        W,
        threshold=0.80,
        resolver=resolver,
        verifier=v.verify,
        state_dir=S,
        snapshot_dir=SNAP,
        start_round=start_round,
        max_rounds=8,
    )
    (S / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"DONE converged={out['converged']} n_final={out['n_final']} edges={out['edges']} "
        f"live_judges={rstats['live']}",
        flush=True,
    )
    for i, rl in enumerate(out["rounds"], 1):
        print(
            f"  round{rl['round']}: n {rl['n_start']}->{rl['n_end']} merged={rl['merged_count']} "
            f"rejected={len(rl['rejected'])} distinct={len(rl['distinct'])} "
            f"blocked={len(rl['skipped_blocked'])} consumed={len(rl['skipped_consumed'])}",
            flush=True,
        )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        asyncio.run(validate())
    elif mode == "continue":
        asyncio.run(continue_run(int(sys.argv[2]) if len(sys.argv) > 2 else 3))
    else:
        asyncio.run(run())
