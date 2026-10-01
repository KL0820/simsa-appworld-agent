"""Assemble curation _2 baselines into training_runs/<comp>/<run>/<task>/ format.

A baseline = the ORIGINAL skill run once through full deduction (run_outcome). That
IS a verified data point reusable for re-training. This captures it in the same
layout as _1 training records: skill.md + deduction.json (+ the full-system trace).
(No prompt.txt/gradient.txt — a baseline is one deduction, not an induction loop.)

Idempotent + re-runnable: reads skill_libraries (original skill) + merge_eval_cache
(the deduction result) + verify_work (the sandbox trace). Re-run at completion to
capture all baselines.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
from mind_skill.curation.curate import _md_sha
from mind_skill.curation.dedup import load_skill_files, render_skill_md

RUN = "curation_baseline"  # the run name under training_runs/<comp>/

for comp in ("code_executor", "code_planner", "rough_planner"):
    sl = Path(f"data/mind_skill/skills/skill_libraries/{comp}/best")
    state = Path(f"data/mind_skill/skills/refined/{comp}/engine_state")
    evalp = state / "merge_eval_cache.json"
    if not (sl.is_dir() and evalp.exists()):
        print(f"[{comp}] skip (no skill_libraries / eval cache)")
        continue
    ev = json.loads(evalp.read_text("utf-8"))
    work = Path(f"logs/mind_skill/curation/{comp}/verify_work")
    out_root = Path(f"data/mind_skill/training_runs/{comp}/{RUN}")
    # original skill md sha -> (task_id, skill SkillFile)
    n = 0
    for sf in load_skill_files(sl):
        md = render_skill_md(sf.name, sf.description, sf.body)
        sha = _md_sha(md)
        ekey = f"{sha}:{sf.task_id}"
        if ekey not in ev:
            continue  # this original skill's baseline wasn't computed in this curation
        rec = out_root / sf.task_id
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "skill.md").write_text(md, encoding="utf-8")
        (rec / "deduction.json").write_text(
            json.dumps(
                {"task_id": sf.task_id, "is_baseline": True, **ev[ekey]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        # full-system trace (controller_state / ledger / sandbox_api_calls), if present
        wd = work / f"merged_{sha}_{sf.task_id}"
        if wd.is_dir():
            for f in wd.iterdir():
                if f.is_file():
                    shutil.copy(f, rec / f.name)
        n += 1
    print(
        f"[{comp}] assembled {n} baseline records -> {out_root}  (trace={'yes' if work.is_dir() else 'results-only'})"
    )
