"""Assemble the canonical SUPERSET trajectory.

Base = the agent's native subagent_io.jsonl (rich: status, outputs, finder
api_selection/routing_trace, executor code_execute incl. repair_attempts,
prior_variables, metrics). Then PATCH the one thing subagent_io lacks — the
executor's verbatim INPUT prompt — from the shim io/ (oracle) or (future)
llm_io.jsonl (gemini). + dedup system_prompts + attach sandbox api_trace + eval.

=> trajectory ⊇ subagent_io.jsonl  (everything it has) + executor input (more).

Usage: python build_trajectory_v2.py <run_root>
  <run_root> has io/ (shim, oracle only) and logs/.../artifacts/{subagent_io.jsonl,
  task_summary.json, evaluation.txt, sandbox_api_calls.jsonl}.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def _jsonl(p: Path):
    return [
        json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()
    ]


def _msgs_to_context(messages):
    out = []
    for m in messages or []:
        text = "".join(
            p.get("text", "") for p in (m.get("parts") or []) if isinstance(p, dict)
        )
        out.append({"role": m.get("role"), "text": text})
    return out


def _parse_shim_req(txt: str):
    sysm = re.search(r"===== SYSTEM =====\n(.*?)\n===== CONTENTS =====", txt, re.S)
    conm = re.search(
        r"===== CONTENTS =====\n(.*?)\n===== RESPONSE NEEDED =====", txt, re.S
    )
    head = re.search(r"kind=(\S+)\s+tools=(\[.*?\])", txt)
    tools = []
    if head:
        try:
            tools = json.loads(head.group(2).replace("'", '"'))
        except Exception:
            tools = []
    return {
        "system": sysm.group(1).strip() if sysm else "",
        "contents": conm.group(1).strip() if conm else "",
        "tools": tools,
    }


def _collect_shim_exec_turns(io_dir: Path):
    """code_plan: one parsed req per code_planner call.
    code_execute: a list of GROUPS; each group = ALL turns of one code_executor
    invocation [{req, resp}, ...] — execute_python turn(s) + submit_final + repairs.
    Captures every turn (turn2+ carry the function_response the model saw), fixing
    the first-turn-only bug.
    """
    cp, ce_groups = [], []
    if not io_dir.is_dir():
        return {"code_plan": cp, "code_execute": ce_groups}
    cur = None
    for p in sorted(io_dir.glob("req_*.txt")):
        seq = int(p.stem.split("_")[1])
        r = _parse_shim_req(p.read_text(encoding="utf-8"))
        resp_p = io_dir / f"resp_{seq:04d}.json"
        resp = None
        if resp_p.exists():
            try:
                resp = json.loads(resp_p.read_text(encoding="utf-8"))
            except Exception:
                resp = None
        if r["system"].startswith("You are the AppWorld code planner"):
            cp.append(r)
            cur = None
        elif "execute_python" in r["tools"]:
            if cur is None:
                cur = []
                ce_groups.append(cur)
            cur.append({"req": r, "resp": resp})
        else:
            cur = None
    return {"code_plan": cp, "code_execute": ce_groups}


def build(run_root: Path):
    art = sorted(run_root.glob("logs/**/artifacts"))[
        -1
    ]  # latest run (re-runs add new ts dirs)
    task_dir = art.parent
    io_recs = _jsonl(art / "subagent_io.jsonl")
    sandbox = (
        _jsonl(art / "sandbox_api_calls.jsonl")
        if (art / "sandbox_api_calls.jsonl").exists()
        else []
    )
    shim = _collect_shim_exec_turns(run_root / "io")

    # eval
    result = None
    ev = art / "evaluation.txt"
    if ev.exists():
        t = ev.read_text(encoding="utf-8", errors="ignore")
        p = re.search(r"Num Passed Tests\s*:\s*(\d+)", t)
        f = re.search(r"Num Failed Tests\s*:\s*(\d+)", t)
        tot = re.search(r"Num Total\s*Tests\s*:\s*(\d+)", t)
        if p and f and tot:
            result = {
                "passed": int(p.group(1)),
                "failed": int(f.group(1)),
                "total": int(tot.group(1)),
                "cleared": int(f.group(1)) == 0,
            }

    system_prompts: dict = {}

    def ref(text, agent):
        if not text:
            return None
        h = _sha(text)
        if h not in system_prompts:
            system_prompts[h] = {"agent": agent, "chars": len(text), "text": text}
        return h

    # run-level meta
    rec0 = io_recs[0]["io_record"]
    tc = (rec0["input"].get("subagent_input") or {}).get("task_context", {})
    cfg0 = rec0["input"].get("model_input", {}).get("generate_content_config", {})
    traj = {
        "schema_version": "1.0",
        "run_meta": {
            "producer": "oracle-claude",
            "model": rec0["input"].get("model_input", {}).get("model"),
            "experiment": task_dir.parent.parent.name,
        },
        "task": {
            "task_id": tc.get("task_id"),
            "instruction": tc.get("instruction"),
            "datetime": tc.get("task_datetime"),
        },
        "config": {
            "sampling": {
                k: cfg0.get(k)
                for k in ("temperature", "top_p", "top_k", "seed", "candidate_count")
            }
        },
        "result": result,
        "plan": None,
        "system_prompts": system_prompts,
        "steps": [],
    }

    def subq(meta):
        keep = (
            "prior_variables",
            "history",
            "milestones",
            "planned_apps",
            "prior_attempts_for_this_milestone",
            "retry_reason",
            "latest_continuation_rationale",
            "last_framework_override",
            "milestone_intent",
            "milestone_index",
            "active_milestone_index",
        )
        return {k: meta[k] for k in keep if k in meta}

    sid = 0
    cp_i = ce_i = sb_i = 0
    for rec in io_recs:
        io = rec["io_record"]
        env = rec.get("envelope", {}) or {}
        agent = io.get("agent", "")
        phase = rec.get("phase")
        inp = io.get("input", {}) or {}
        out = io.get("output", {}) or {}
        meta = ((inp.get("subagent_input") or {}).get("metadata")) or {}
        base = {
            "cycle": meta.get("cycle_n"),
            "milestone_index": meta.get("milestone_index"),
            "phase": phase,
            "attempt": inp.get("attempt"),
            "status": io.get("status"),
            "failure_code": out.get("failure_code"),
            "warnings": out.get("warnings", []),
        }

        def step(
            agent_label,
            input_,
            output,
            result_=None,
            metrics=None,
            *,
            base=base,
            default_metrics=io.get("metrics"),
        ):
            nonlocal sid
            sid += 1
            return {
                "step_id": sid,
                "agent": agent_label,
                **base,
                "input": input_,
                "output": output,
                "result": result_,
                "metrics": metrics or default_metrics,
            }

        if "rough_planner" in agent:
            mi = inp.get("model_input", {})
            traj["plan"] = {
                "thoughts": (out.get("parsed_plan") or {}).get("thoughts"),
                "milestones": (out.get("parsed_plan") or {}).get("tasks", []),
            }
            traj["steps"].append(
                step(
                    "rough_planner",
                    {
                        "system_ref": ref(
                            mi.get("system_instruction"), "rough_planner"
                        ),
                        "context": _msgs_to_context(mi.get("messages")),
                        "tools": [],
                        "tool_config": None,
                        "injected_skills": [],
                        "subagent_input": subq(meta),
                    },
                    {
                        "raw": out.get("raw_llm_text"),
                        "rationale": (out.get("parsed_plan") or {}).get("thoughts"),
                        "parsed": out.get("parsed_plan"),
                    },
                )
            )
        elif "continuation" in agent:
            mi = inp.get("model_input", {})
            traj["steps"].append(
                step(
                    "continuation",
                    {
                        "system_ref": ref(mi.get("system_instruction"), "continuation"),
                        "context": _msgs_to_context(mi.get("messages")),
                        "tools": [],
                        "tool_config": None,
                        "injected_skills": [],
                        "subagent_input": subq(meta),
                    },
                    {
                        "raw": out.get("raw_llm_text"),
                        "rationale": (out.get("parsed_decision") or {}).get(
                            "rationale"
                        ),
                        "parsed": out.get("parsed_decision"),
                    },
                )
            )
        elif "finder" in agent:
            in_calls = inp.get("model_calls") or []
            out_calls = out.get("model_calls") or []
            api_sel = out.get("parsed_api_selection")
            for i, oc in enumerate(out_calls):
                mir = (
                    in_calls[i].get("model_input_raw") if i < len(in_calls) else {}
                ) or {}
                s = step(
                    f"finder:{oc.get('step')}",
                    {
                        "system_ref": ref(
                            mir.get("system_instruction"), f"finder:{oc.get('step')}"
                        ),
                        "context": _msgs_to_context(mir.get("messages")),
                        "tools": [],
                        "tool_config": None,
                        "injected_skills": [],
                        "subagent_input": subq(meta) if i == 0 else None,
                    },
                    {"raw": oc.get("raw_llm_text"), "parsed": oc.get("parsed_json")},
                    metrics={"usage": oc.get("usage"), "retry": oc.get("retry")},
                )
                if i == len(out_calls) - 1 and api_sel:
                    s["output"]["api_selection"] = api_sel
                traj["steps"].append(s)
        elif "executor" in agent:
            payload = env.get("payload", {}) or {}
            pj = out.get("parsed_json", {}) or {}
            code_plan = payload.get("code_plan") or pj.get("code_plan")
            code_exec = payload.get("code_execute") or pj.get("code_execute") or {}
            cp_in = shim["code_plan"][cp_i] if cp_i < len(shim["code_plan"]) else {}
            cp_i += 1
            traj["steps"].append(
                step(
                    f"{agent}:code_plan",
                    {
                        "system_ref": ref(cp_in.get("system"), "code_planner"),
                        "context": [
                            {"role": "user", "text": cp_in.get("contents", "")}
                        ],
                        "tools": [],
                        "tool_config": None,
                        "injected_skills": [],
                        "subagent_input": subq(meta),
                        "_patched_from": "shim_io",
                    },
                    {"parsed": code_plan},
                )
            )
            group = (
                shim["code_execute"][ce_i] if ce_i < len(shim["code_execute"]) else []
            )
            ce_i += 1
            sb = sandbox[sb_i] if sb_i < len(sandbox) else None
            sb_i += 1
            turns = []
            for t in group:  # ← 完整多輪：每通 model 呼叫一筆
                rq = t.get("req") or {}
                rp = t.get("resp") or {}
                if rp.get("type") == "function_call":
                    parts = [
                        {
                            "function_call": {
                                "name": rp.get("name"),
                                "args": rp.get("args", {}),
                            }
                        }
                    ]
                else:
                    parts = [{"text": rp.get("text", "")}]
                turns.append(
                    {
                        "request": {
                            "system_ref": ref(rq.get("system"), "code_executor"),
                            "context": [
                                {"role": "user", "text": rq.get("contents", "")}
                            ],  # turn2+ 含 function_response
                            "tools": rq.get("tools", []),
                            "tool_config": {"mode": "ANY"},
                        },
                        "response": {"parts": parts},
                    }
                )
            ce_step = step(
                f"{agent}:code_execute",
                None,
                {
                    "code_execute": {
                        k: code_exec.get(k)
                        for k in (
                            "code",
                            "tool_calls",
                            "tool_call_count",
                            "parse_error",
                            "llm_raised",
                            "repair_attempts",
                        )
                    },
                    "executor_result": payload.get("executor_result"),
                    "submission_candidate": payload.get("submission_candidate"),
                    "milestone_done": payload.get("milestone_done"),
                },
                result_={
                    "stdout": code_exec.get("stdout_json"),
                    "api_trace": sb.get("api_calls") if sb else None,
                    "committed_variables": (payload.get("executor_result") or {}).get(
                        "variables"
                    ),
                },
            )
            ce_step["turns"] = turns
            ce_step["input"] = None  # input 改由 turns[] 承載
            ce_step["_patched_from"] = "shim_io"
            traj["steps"].append(ce_step)
    return traj


def coverage(traj):
    s = traj["steps"]

    def has(pred):
        return sum(1 for x in s if pred(x))

    return {
        "steps": len(s),
        "with_status": has(lambda x: x.get("status")),
        "with_verbatim_input": has(
            lambda x: (
                bool(x.get("turns"))
                or (
                    x.get("input")
                    and x["input"].get("context")
                    and x["input"]["context"][0].get("text")
                )
            )
        ),
        "executor_turns_captured": has(lambda x: bool(x.get("turns"))),
        "max_turns_in_step": max((len(x.get("turns") or []) for x in s), default=0),
        "executor_input_patched": has(lambda x: bool(x.get("_patched_from"))),
        "with_api_selection": has(lambda x: x["output"].get("api_selection")),
        "with_repair_attempts": has(
            lambda x: (
                x["output"].get("code_execute")
                and x["output"]["code_execute"].get("repair_attempts") is not None
            )
        ),
        "with_committed_vars": has(
            lambda x: x.get("result") and x["result"].get("committed_variables")
        ),
        "with_prior_variables": has(
            lambda x: (
                x.get("input")
                and (x["input"].get("subagent_input") or {}).get("prior_variables")
            )
        ),
        "distinct_system_prompts": len(traj["system_prompts"]),
    }


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    traj = build(root)
    (root / "trajectory.json").write_text(
        json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("wrote trajectory.json")
    print("coverage:", json.dumps(coverage(traj), ensure_ascii=False, indent=2))
