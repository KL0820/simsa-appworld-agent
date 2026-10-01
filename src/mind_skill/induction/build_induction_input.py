"""Assemble the COMPLETE induction input for one (task, layer) — NO LLM call.
Lets you eyeball exactly what the induction model would be fed before running anything.

  python src/mind_skill/build_induction_input.py 3c13f5a_2 code_executor

Reads <run>/induction_slices.json, renders the layer's slice, prepends the layer's
induction prompt, writes induction_inputs/<task>__<layer>.txt and prints it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from mind_skill.paths import GOLD_TRAJECTORIES_DIR

HERE = Path(__file__).resolve()
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
RUNS = GOLD_TRAJECTORIES_DIR
OUT = MIND_SKILL / "induction_inputs"

# ── Induction prompt (code_executor) — faithful to MIND-Skill Fig 7, adapted to take
#    the agent's NATIVE per-milestone executor trajectory instead of a flat solution. ──
EXECUTOR_PROMPT = """You are an expert at distilling a SOLVED task into a reusable PROCEDURAL skill for the
code-writing EXECUTOR of a multi-subagent AppWorld agent.

You are given:
  - TASK_INSTRUCTION: the natural-language task that was solved.
  - EXECUTOR_TRAJECTORY: per milestone, the executor's actual work — the milestone
    intent, the APIs it had available, the Python code it ran, what the code returned,
    and how many turns / repairs it took.

Extract ONE skill capturing the PROCEDURAL / STRUCTURAL how-to that a future executor
would need to solve a DIFFERENT task in the SAME structural family — the non-obvious
steps a naive code-writer would skip or get wrong.

CAPTURE (procedural/structural only):
  - Join-key choice (match on a stable unique identifier, not a display name).
  - Pagination exhaustion (loop until no more pages).
  - Verbatim filter pass-through (feed the instruction's own word into the filter; do
    not invent categories or enumerate allowed values).
  - Counting / aggregation correctness (e.g. include yourself when a cost is split among
    a group "and me"; check every participant role).
  - Action-vs-query (perform the state change when asked to act; do not merely compute).
  - Reading prior-milestone variables and the output / submission contract.

HARD RULES — the skill MUST NOT contain task-specific content:
  (1) No concrete API / method / endpoint names.
  (2) No concrete field / entity / person / app names, no domain nouns.
  (3) No literal thresholds, magic strings, hardcoded category/relation lists, or
      specific answer values.
  LEAKAGE TEST: if a reader could guess WHICH task this came from, it is too specific.

OUTPUT — one valid SKILL.md, exactly this shape, nothing else:
---
name: <kebab-case-slug>
description: <one sentence: when this skill applies; no task specifics>
---
## Overview
<2-3 sentences: the structural problem this family poses>
## When to Apply
<instruction-level signals that this pattern applies, stated generically>
## Procedure
<numbered generic steps; reference roles ("the credential", "the membership set"), not APIs>
## Key Patterns
<the non-obvious structural insights>
## Common Pitfalls
<the mistakes a naive executor makes here, stated generically>
"""

PROMPTS = {"code_executor": EXECUTOR_PROMPT}


def render_executor(slc: dict) -> str:
    L = [f"TASK_INSTRUCTION:\n{slc['instruction']}", "", "EXECUTOR_TRAJECTORY:"]
    for i, m in enumerate(slc.get("code_executor", [])):
        L.append(f"\n--- milestone {i}: {m.get('milestone')}")
        L.append(f"APIs available: {m.get('apis')}")
        cp = m.get("code_plan") or {}
        if cp.get("plan_steps"):
            L.append("code plan:")
            for st in cp["plan_steps"]:
                L.append(f"  - {st}")
        L.append("code that ran:")
        L.append("```python")
        L.append((m.get("code") or "").rstrip())
        L.append("```")
        L.append(f"result: {m.get('result_summary')}")
        L.append(
            f"(turns: {m.get('turns')}, repairs: {len(m.get('repair_codes') or [])})"
        )
    return "\n".join(L)


def build(task: str, layer: str) -> str:
    from mind_skill.loop import load_executor_slice

    slc = load_executor_slice(RUNS, task)
    if layer == "code_executor":
        content = render_executor(slc)
    else:
        raise SystemExit(f"layer {layer} not wired yet (only code_executor)")
    return PROMPTS[layer] + "\n\n" + "=" * 70 + "\n" + content


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "3c13f5a_2"
    layer = sys.argv[2] if len(sys.argv) > 2 else "code_executor"
    full = build(task, layer)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{task}__{layer}.txt"
    f.write_text(full, encoding="utf-8")
    print(f"wrote {f}  ({len(full)} chars ≈ {len(full) // 4} tokens)\n")
    print(full)


if __name__ == "__main__":
    main()
