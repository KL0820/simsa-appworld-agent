"""Claude-CLI oracle backend — each LLM call = a FRESH, CLEAN `claude -p` process.

Clean by construction (empirically verified 2026-06-06):
  --setting-sources ""  → no project/user CLAUDE.md, no memory  (probe: thesis → UNKNOWN)
  --allowedTools ""     → no tools → cannot read ground_truth/solution.py  (probe: /etc/hosts → CANNOT)
  neutral cwd + fresh process per call → zero cross-call/cross-task accumulation.

So the oracle solves each call BLIND from only the request shown — no contamination,
no cheating. Uses the `claude` CLI (subscription), not the API.

NOTE: `claude -p` wraps the Claude Code system prompt around our prompt; we instruct it
to role-play strictly as the model and emit only the response JSON. Robust parse + retry.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

ORACLE_MODEL = os.environ.get("ORACLE_CLAUDE_MODEL", "opus")
_NEUTRAL_CWD = tempfile.mkdtemp(prefix="oracle_clean_")

# Open-book mode (DISCLOSED): if ORACLE_SOLUTION_PATH is set, the task's gold
# solution is appended to each prompt as a CORRECTNESS REFERENCE — used only to
# recover the few tasks blind-solve fails. Outputs must still be derived from each
# step's input (the prompt forbids pasting). Such trajectories are marked
# run_meta.open_book=true downstream. Default (unset) = pure blind/clean oracle.
_REF_PATH = os.environ.get("ORACLE_SOLUTION_PATH")
_REFERENCE = None
if _REF_PATH and Path(_REF_PATH).exists():
    try:
        _REFERENCE = Path(_REF_PATH).read_text(encoding="utf-8")
    except Exception:
        _REFERENCE = None


def claude_call(prompt: str, *, max_retries: int = 10, timeout_s: int = 900) -> str:
    """One clean stateless `claude -p` call → the model's text output (.result).

    Retries with backoff on transient failure / rate limit (subscription 429). The
    batch watchdog wraps this; long sleeps here keep a single call resilient.
    """
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--allowedTools",
        "",
        "--setting-sources",
        "",
        "--model",
        ORACLE_MODEL,
    ]
    last = ""
    for attempt in range(max_retries):
        try:
            res = subprocess.run(
                cmd,
                cwd=_NEUTRAL_CWD,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            last = "timeout"
            time.sleep(min(120, 15 * (attempt + 1)))
            continue
        out = (res.stdout or "").strip()
        try:
            d = json.loads(out)
        except Exception:
            last = (out or res.stderr or "")[:400]
            time.sleep(min(180, 15 * (attempt + 1)))  # likely rate limit / transient
            continue
        if d.get("is_error"):
            last = str(d.get("result"))[:400]
            time.sleep(min(180, 15 * (attempt + 1)))
            continue
        return d.get("result", "") or ""
    raise RuntimeError(f"claude_call failed after {max_retries} retries: {last}")


def parse_oracle_response(text: str) -> dict:
    """Extract the resp dict ({"type": text|function_call|json, ...}) from claude output."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)  # strip markdown fence if any
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i : j + 1])
        raise


_FRAME = (
    "You are the LLM inside an AppWorld multi-subagent agent. Below is the EXACT request "
    "the agent sends to its model: a SYSTEM instruction, the conversation CONTENTS, and "
    "(if present) the available TOOLS. Act as that model would — solve it GENUINELY from "
    "ONLY what is shown; you have no other knowledge of this task and must not invent the "
    "answer.\n\n"
    "Output ONLY one JSON object — no prose, no markdown fences — in EXACTLY one shape:\n"
    '  text     -> {"type":"text","text":"<raw model output; usually a JSON string the parser will load>"}\n'
    '  function -> {"type":"function_call","name":"<tool>","args":{...}}   (when TOOLS force a call)\n'
    '  finder   -> {"type":"json","json":{...}}\n'
)


def build_oracle_prompt(
    kind: str,
    system: str,
    body: str,
    tools: list,
    response_hint: str,
    schema_text: str = "",
) -> str:
    parts = [
        _FRAME,
        f"RESPONSE NEEDED: {response_hint}",
    ]
    if schema_text:
        parts += [
            "",
            "===== OUTPUT SCHEMA (REQUIRED) =====",
            'The JSON you place in the "text" field MUST conform EXACTLY to this JSON Schema:',
            schema_text,
            "Emit ONLY the properties defined above, with their exact names and types; include "
            "every required property; add NO extra keys; never nest a value under a different key "
            "name. A field typed as an enum must be exactly one of its allowed literal strings.",
        ]
    parts += [
        "",
        "===== SYSTEM =====",
        system or "(none)",
        "===== CONTENTS =====",
        body or "(none)",
        "===== TOOLS =====",
        json.dumps(tools) if tools else "(none)",
    ]
    if _REFERENCE:
        parts += [
            "",
            "===== REFERENCE SOLUTION (open-book — correctness reference ONLY) =====",
            _REFERENCE,
            "",
            "Use the reference ONLY to know the correct approach/values so you do not make a "
            "reasoning slip (e.g. an off-by-one share count, a wrong join key, a missed filter). "
            "Your output MUST be derived genuinely from THIS step's SYSTEM + CONTENTS above and stay "
            "consistent with this step's input — do NOT paste final answers; COMPUTE them from the "
            "input the way the reference does, so the trajectory shows the correct procedure.",
        ]
    return "\n".join(parts)
