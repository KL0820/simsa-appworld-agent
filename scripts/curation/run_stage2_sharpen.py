"""Stage-2 sharpen runner (name+desc relabel for selector distinctiveness).

Stage 1 (merge) is done; the converged libraries live in
`data/mind_skill/skills/curated/<comp>/engine_lib` (45/46/48). Stage 2 is the SAME
convergence loop as Stage 1, with the operation swapped from "merge a redundant
pair" to "relabel a confusable pair": each round takes the top-similarity unsettled
name+desc PAIR (cosine >= 0.80, the same cutoff Stage 1 uses), rewrites BOTH members'
name + one-line description to be mutually distinctive (bodies verbatim), and settles
the pair. Pairwise (not connected-component) so a chain of weak look-alikes never
collapses the library into one blob. No deduction / RPC -- cheap LLM relabel only.

NON-DESTRUCTIVE: copies engine_lib -> stage2_final and sharpens the COPY, so the
Stage-1 result stays intact (RULES.md: never destroy experiment data; build a new
versioned output instead of overwriting).

Run from the worktree root, PYTHONPATH=src, Vertex creds in .env:
    .venv/bin/python scripts/curation/run_stage2_sharpen.py [code_executor code_planner rough_planner]
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
    DEFAULT_NAMEDESC_THRESHOLD,
    _llm_sharpener,
    build_sharpen_judge,
    sharpen_to_convergence,
)
from mind_skill.curation.dedup import load_skill_files
from mind_skill.paths import REFINED_DIR, STAGE1_CONSOLIDATED, STAGE2_REFINED


def _snapshot(library_dir: Path) -> dict[str, dict]:
    """home task_id -> {name, description} for the current library state on disk."""
    return {
        sf.task_id: {"name": sf.name, "description": sf.description}
        for sf in load_skill_files(library_dir)
    }


def _preserve_old(base: Path) -> None:
    """Move a prior (flawed connected-component 0.60) Stage-2 run aside, not delete it
    -- it is evidence for why the pairwise 0.80 design replaced it."""
    old = base / STAGE2_REFINED
    if not old.exists():
        return
    dep = base / "_prev_stage2"
    if dep.exists():
        return  # already preserved on an earlier re-run
    dep.mkdir(parents=True)
    shutil.move(str(old), str(dep / STAGE2_REFINED))
    for f in ("stage2_actions.json", "stage2_summary.txt", "stage2_journal.jsonl"):
        p = base / f
        if p.exists():
            shutil.move(str(p), str(dep / f))


async def sharpen_one(comp: str, threshold: float) -> dict:
    base = REFINED_DIR / comp
    src = base / STAGE1_CONSOLIDATED  # Stage-1 converged library (DO NOT mutate)
    dst = base / STAGE2_REFINED  # new versioned Stage-2 output
    journal = base / "stage2_journal.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- Stage-1 not run for {comp}?")
    _preserve_old(base)
    shutil.copytree(src, dst)
    journal.unlink(missing_ok=True)

    before = _snapshot(dst)  # captured BEFORE any rewrite
    sharpener = _llm_sharpener(build_sharpen_judge(journal_path=journal))
    out = await sharpen_to_convergence(
        dst,
        threshold=threshold,
        sharpener=sharpener,
        state_dir=base / "stage2_state",
        snapshot_dir=base / "stage2_rounds",
    )
    after = _snapshot(dst)

    # flatten per-pair renames across all rounds into a before/after table
    rows = []
    for rl in out["rounds"]:
        for sh in rl["sharpened"]:
            for act in sh["renames"]:
                tid = act["from"]["task_id"]
                rows.append(
                    {
                        "round": rl["round"],
                        "task_id": tid,
                        "pair": sh["pair"],
                        "sim": sh["sim"],
                        "reason": sh["reason"],
                        "old_name": before.get(tid, {}).get(
                            "name", act["from"]["name"]
                        ),
                        "old_description": before.get(tid, {}).get("description", ""),
                        "new_name": act["to_name"],
                        "new_description": after.get(tid, {}).get("description", ""),
                    }
                )

    (base / "stage2_actions.json").write_text(
        json.dumps(
            {
                "component": comp,
                "threshold": threshold,
                "converged": out["converged"],
                "n_skills": out["n_final"],
                "settled_pairs": out["settled"],
                "n_rounds": len(out["rounds"]),
                "per_round_sharpened": [r["sharpened_count"] for r in out["rounds"]],
                "renames": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        f"# Stage-2 sharpen -- {comp}",
        f"namedesc cosine threshold = {threshold}  (PAIRWISE, same cutoff as Stage-1 merge; embedding all-mpnet-base-v2)",
        f"library = {dst}",
        f"skills = {out['n_final']}  (count + bodies unchanged; name+desc rewrite only)",
        f"converged = {out['converged']}   settled pairs = {out['settled']}   "
        f"rounds = {len(out['rounds'])}   per-round sharpened = {[r['sharpened_count'] for r in out['rounds']]}",
        "",
    ]
    for r in rows:
        lines += [
            f"[round {r['round']}] pair {r['pair']}  (cosine {r['sim']})",
            f"  task {r['task_id']}",
            f"  name: {r['old_name']}  ->  {r['new_name']}",
            f"  desc-: {r['old_description']}",
            f"  desc+: {r['new_description']}",
            f"  why : {r['reason']}",
            "",
        ]
    (base / "stage2_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"[{comp}] DONE  converged={out['converged']}  settled_pairs={out['settled']}  "
        f"renamed={len(rows)}  n={out['n_final']}  rounds={[r['sharpened_count'] for r in out['rounds']]}  "
        f"-> {dst}",
        flush=True,
    )
    return {
        "comp": comp,
        "n": out["n_final"],
        "settled": out["settled"],
        "renamed": len(rows),
        "converged": out["converged"],
    }


async def main():
    comps = sys.argv[1:] or ["code_executor", "code_planner", "rough_planner"]
    threshold = DEFAULT_NAMEDESC_THRESHOLD
    print(
        f"Stage-2 sharpen: pairwise to convergence, threshold={threshold}", flush=True
    )
    results = []
    for comp in comps:
        try:
            results.append(await sharpen_one(comp, threshold))
        except Exception as exc:
            import traceback

            print(f"[{comp}] FAILED: {exc}", flush=True)
            traceback.print_exc()
    print("\n=== SUMMARY ===", flush=True)
    for r in results:
        print(
            f"  {r['comp']}: {r['n']} skills, {r['settled']} pairs settled, "
            f"{r['renamed']} renamed, converged={r['converged']}",
            flush=True,
        )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
