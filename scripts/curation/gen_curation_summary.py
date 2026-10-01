"""Human-readable curation summary (batch.txt style) from engine_state.

Reads curated/<comp>/engine_state/{rounds,summary}.json and writes
curated/<comp>/curation_summary.txt: merged / rejected / distinct + per-round +
final numbers. (To be folded into the engine during the src/mind_skill reorg so
every curation run auto-emits it.)
"""

import json
import sys
from pathlib import Path

comp = sys.argv[1]
base = Path(f"data/mind_skill/skills/refined/{comp}")
state = base / "engine_state"
rounds = json.loads((state / "rounds.json").read_text("utf-8"))
summ = (
    json.loads((state / "summary.json").read_text("utf-8"))
    if (state / "summary.json").exists()
    else {}
)

merged = sum(len(r["merged"]) for r in rounds)
rejected = sum(len(r["rejected"]) for r in rounds)
distinct = sum(len(r["distinct"]) for r in rounds)
blocked = sum(len(r.get("skipped_blocked", [])) for r in rounds)

L = []
L += ["=" * 64, f"Skill Curation — {comp}", "=" * 64, ""]
n0 = rounds[0]["n_start"] if rounds else "?"
nf = summ.get("n_final", rounds[-1]["n_end"] if rounds else "?")
conv = summ.get("converged")
tail = (
    f"(converged after round {rounds[-1]['round']})"
    if conv
    else f"(stopped at round {rounds[-1]['round']}, not yet converged)"
)
L += [f"  Library:  {n0} -> {nf} skills   {tail}", ""]
L += ["AGGREGATE", "-" * 9, ""]
L += [f"  Merged:   {merged}"]
L += [f"  Rejected: {rejected}   (merge would drop requirement count -> kept separate)"]
L += [f"  Distinct: {distinct}   (judge: genuinely different skills)"]
L += [f"  Blocked:  {blocked}   (already-settled pairs skipped — no rework)", ""]
L += ["PER ROUND", "-" * 9, ""]
for r in rounds:
    L += [
        f"  round{r['round']}: {r['n_start']:>3} -> {r['n_end']:<3}  "
        f"merged {len(r['merged'])}  rejected {len(r['rejected'])}  "
        f"distinct {len(r['distinct'])}  blocked {len(r.get('skipped_blocked', []))}"
    ]
L += [""]
L += ["MERGED", "-" * 6, ""]
for r in rounds:
    for m in r["merged"]:
        st = m.get("source_tasks", [])
        L += [
            f"  [r{r['round']}] {m['pair'][0]}  +  {m['pair'][1]}",
            f"          -> {m.get('merged_name', '?')}   (sim {m['sim']}, verified on {len(st)} tasks, no regression)",
        ]
L += [""]
L += ["REJECTED  (kept separate — merge dropped requirements)", "-" * 52, ""]
for r in rounds:
    for m in r["rejected"]:
        pt = (m.get("verify") or {}).get("per_task", [])
        drops = [
            f"{p['task_id']} {p['baseline_passed']}->{p['merged_passed']}/{p['total']}"
            for p in pt
            if p.get("regressed")
        ]
        L += [
            f"  [r{r['round']}] {m['pair'][0]}  +  {m['pair'][1]}  (sim {m['sim']})",
            f"          regressed: {', '.join(drops) if drops else '(see audit)'}",
        ]
L += [""]
L += ["DISTINCT  (judge: genuinely different)", "-" * 37, ""]
for r in rounds:
    for m in r["distinct"]:
        L += [
            f"  [r{r['round']}] {m['pair'][0]}  +  {m['pair'][1]}  (sim {m['sim']})",
            f"          {m['reason'][:110]}",
        ]

out = base / "curation_summary.txt"
out.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {out}\n")
print("\n".join(L[: L.index("MERGED") + 6]))
