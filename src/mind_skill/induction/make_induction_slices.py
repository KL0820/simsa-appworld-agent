"""Project each gold trajectory into per-subagent INDUCTION SLICES (the inputs fed to
skill induction, per docs/induction_inputs.md). Pure data processing — no LLM calls.

Granularity = per (subagent, trajectory): one slice-bundle per gold task, with the
subagent's steps across milestones. Writes <run_dir>/induction_slices.json next to
each trajectory.json; prints a review sample + a coverage/size summary.

  python src/mind_skill/make_induction_slices.py            # all gold (blind PASS + openbook PASS)
"""

from __future__ import annotations

import json

from mind_skill.paths import GOLD_TRAJECTORIES_DIR

RUNS = GOLD_TRAJECTORIES_DIR


def _tok(o) -> int:
    return len(json.dumps(o, ensure_ascii=False)) // 4


def gold_runs():
    out = []
    for p in sorted(RUNS.glob("*/trajectory.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if (d.get("result") or {}).get("cleared"):
            out.append((d["task"]["task_id"], p, d))
    for p in sorted(RUNS.glob("*/openbook/trajectory.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if (d.get("result") or {}).get("cleared"):
            out.append((d["task"]["task_id"] + " (open-book)", p, d))
    return out


def _ctx_text(s) -> str:
    inp = s.get("input")
    if inp and inp.get("context"):
        return inp["context"][0].get("text", "") or ""
    tn = s.get("turns")
    if tn:
        return ((tn[0].get("request") or {}).get("context", [{}]) or [{}])[0].get(
            "text", ""
        ) or ""
    return ""


def slices(d: dict) -> dict:
    steps = d.get("steps", [])
    ms = (d.get("plan") or {}).get("milestones", [])

    def intent(i):
        return ms[i].get("task") if i < len(ms) else None

    rp = next((s for s in steps if s["agent"] == "rough_planner"), None)
    out = {
        "task_id": d["task"]["task_id"],
        "instruction": d["task"]["instruction"],
        "open_book": (d.get("run_meta") or {}).get("open_book", False),
        # PLAN skill input
        "rough_planner": {
            "instruction": d["task"]["instruction"],
            "thoughts": (rp or {}).get("output", {}).get("rationale"),
            "milestones": ms,
        },
        "finder": [],
        "code_planner": [],
        "code_executor": [],
        "continuation": [],
    }
    for i in range(len(ms)):
        fsteps = [
            s for s in steps if s.get("milestone_index") == i and "finder" in s["agent"]
        ]
        asd = (
            next(
                (
                    s["output"].get("api_selection")
                    for s in fsteps
                    if s["output"].get("api_selection")
                ),
                None,
            )
            or {}
        )
        apis = [a.get("name") for a in asd.get("candidate_apis", [])]
        # FIND skill input
        out["finder"].append(
            {
                "milestone": intent(i),
                "candidates_shown": _ctx_text(fsteps[0])[:1500] if fsteps else "",
                "selected_communities": asd.get("selected_communities"),
                "selected_apis": apis,
                "routing_trace": [r.get("step") for r in asd.get("routing_trace", [])],
            }
        )
        cp = next(
            (
                s
                for s in steps
                if s.get("milestone_index") == i and s["agent"].endswith("code_plan")
            ),
            None,
        )
        ce = next(
            (
                s
                for s in steps
                if s.get("milestone_index") == i and s["agent"].endswith("code_execute")
            ),
            None,
        )
        cp_parsed = (cp or {}).get("output", {}).get("parsed") if cp else None
        if cp:
            # code_planner skill input
            out["code_planner"].append(
                {"milestone": intent(i), "apis": apis, "code_plan": cp_parsed}
            )
        if ce:
            cobj = ce["output"].get("code_execute") or {}
            stdout = (ce.get("result") or {}).get("stdout") or {}
            out["code_executor"].append(
                {  # ★ executor skill input
                    "milestone": intent(i),
                    "apis": apis,
                    "code_plan": cp_parsed,
                    "code": cobj.get("code"),
                    "result_summary": stdout.get("summary"),
                    "turns": len(ce.get("turns") or []),
                    "repair_codes": [
                        ra.get("code")
                        for ra in (cobj.get("repair_attempts") or [])
                        if isinstance(ra, dict) and ra.get("code")
                    ][:2],
                }
            )
    for s in steps:
        if s["agent"] == "continuation":
            p = s["output"].get("parsed") or {}
            out["continuation"].append(
                {"next_action": p.get("next_action"), "rationale": p.get("rationale")}
            )
    return out


def main():
    runs = gold_runs()
    summary = {
        k: {"n": 0, "tok": 0}
        for k in (
            "rough_planner",
            "finder",
            "code_planner",
            "code_executor",
            "continuation",
        )
    }
    sample = None
    for tid, p, d in runs:
        sl = slices(d)
        (p.parent / "induction_slices.json").write_text(
            json.dumps(sl, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for k in summary:
            v = sl[k]
            summary[k]["n"] += 1 if not isinstance(v, list) else len(v)
            summary[k]["tok"] += _tok(v)
        if "3c13f5a" in tid:
            sample = sl
    print(f"處理 {len(runs)} 份 gold；各 induction_slices.json 已寫到各 run 目錄旁。\n")
    print("=== 各 subagent 切片：數量 + 平均 token ===")
    for k, v in summary.items():
        avg = (v["tok"] // len(runs)) if runs else 0
        print(
            f"  {k:16} 共 {v['n']:>3} 切片（{len(runs)} task）  該層每-task ~{avg:>5} tok"
        )
    if sample:
        print("\n" + "=" * 60)
        print("樣板：3c13f5a_2 的 code_executor 切片（旗艦層，給你檢核）")
        print("=" * 60)
        ex = sample["code_executor"]
        print(f"task: {sample['task_id']}  open_book={sample['open_book']}")
        for j, s in enumerate(ex):
            print(f"\n-- milestone {j}: {s['milestone'][:70]}")
            print(
                f"   apis: {s['apis']}  turns: {s['turns']}  repairs: {len(s['repair_codes'])}"
            )
            print(f"   result: {s['result_summary']}")
            print("   code:")
            for ln in (s["code"] or "").splitlines()[:8]:
                print("     ", ln)


if __name__ == "__main__":
    main()
