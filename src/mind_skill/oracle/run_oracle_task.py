"""Oracle runner — drive ONE AppWorld train task through the REAL adk-appworld-agent
controller, with every LLM call answered by a human-in-the-loop (Claude) instead of Gemini.

How it works (the "shim"):
  - A HumanLlm(BaseLlm) is registered into ADK for model name "human". Every ADK
    LlmAgent call (rough_planner / continuation / code_plan / code_execute) lands in
    HumanLlm.generate_content_async, which dumps the EXACT llm_request (system +
    contents + tools, byte-faithful — what Gemini would have received) to a file and
    blocks until a response file is written.
  - The finder uses a direct genai call (not ADK), so call_llm_json is monkeypatched
    to the same file-based handshake.
  - Everything else runs unchanged: real milestones, real candidate-API enrichment,
    real sandbox execution of the code, real evaluate(). Only the brain behind each
    LLM call is swapped Gemini -> Claude. => perfect parity + complete I/O capture.

This is the DEDUCTION side of MIND-Skill (2605.08670) realized in a live environment:
a strong model reconstructs a successful trajectory; the AppWorld evaluator is the
outcome gate. The captured trajectory is the gold input for skill induction.

Isolation: connects to a DEDICATED AppWorld RPC server (default tcp://127.0.0.1:4243),
NOT the ablation server on 4242. Outputs land under src/mind_skill/runs/.

Launch with the project venv python (3.12):
  .venv/bin/python src/mind_skill/run_oracle_task.py
Env overrides: ORACLE_TASK_ID, ORACLE_RPC_URL, ORACLE_IO_DIR.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[
    3
]  # <root>/src/mind_skill/oracle/run_oracle_task.py -> project root
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
RUN_DIR = Path(os.environ.get("ORACLE_RUN_DIR", MIND_SKILL / "runs"))
IO_DIR = Path(os.environ.get("ORACLE_IO_DIR", RUN_DIR / "io"))
LOG_ROOT = RUN_DIR / "logs"

POLL_S = 1.0
_seq = {"n": 0}


def _next_seq() -> int:
    _seq["n"] += 1
    return _seq["n"]


def _sysinstr_to_text(si) -> str:
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    parts = getattr(si, "parts", None)
    if parts:
        return "\n".join(p.text for p in parts if getattr(p, "text", None))
    if isinstance(si, (list, tuple)):
        return "\n".join(_sysinstr_to_text(x) for x in si)
    return str(si)


def _content_to_text(contents) -> str:
    blocks = []
    for c in contents or []:
        role = getattr(c, "role", "?")
        seg = []
        for p in getattr(c, "parts", None) or []:
            if getattr(p, "text", None):
                seg.append(p.text)
            fc = getattr(p, "function_call", None)
            if fc:
                seg.append(
                    f"[FUNCTION_CALL {fc.name}] args={json.dumps(dict(fc.args or {}), ensure_ascii=False, default=str)}"
                )
            fr = getattr(p, "function_response", None)
            if fr:
                seg.append(
                    f"[FUNCTION_RESPONSE {fr.name}] {json.dumps(getattr(fr, 'response', None), ensure_ascii=False, default=str)}"
                )
        blocks.append(f"<<{role}>>\n" + "\n".join(seg))
    return "\n\n".join(blocks)


def _tool_names(cfg) -> list:
    names = []
    for t in getattr(cfg, "tools", None) or []:
        for fd in getattr(t, "function_declarations", None) or []:
            names.append(fd.name)
    return names


def _response_hint(kind: str, tools: list) -> str:
    if tools:
        return (
            f"function_call REQUIRED (ANY-mode forces one of {tools}). "
            'Write resp JSON: {"type":"function_call","name":"<tool>","args":{...}}'
        )
    if kind == "finder":
        return (
            'JSON object. Write resp JSON: {"type":"json","json":{...}}  '
            '(or {"type":"text","text":"<json-string>"})'
        )
    return 'raw model text (usually a JSON string the parser will load). Write resp JSON: {"type":"text","text":"..."}'


def _dump_request(kind: str, system: str, body: str, tools: list, raw_obj) -> int:
    seq = _next_seq()
    stem = IO_DIR / f"req_{seq:04d}"
    meta = {
        "seq": seq,
        "kind": kind,
        "tools": tools,
        "system_chars": len(system),
        "body_chars": len(body),
    }
    stem.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    txt = [
        f"### REQUEST {seq}   kind={kind}   tools={tools}",
        "",
        "===== SYSTEM =====",
        system or "(none)",
        "",
        "===== CONTENTS =====",
        body or "(none)",
        "",
        "===== RESPONSE NEEDED =====",
        _response_hint(kind, tools),
    ]
    stem.with_suffix(".txt").write_text("\n".join(txt), encoding="utf-8")
    try:
        stem.with_suffix(".raw.json").write_text(
            json.dumps(raw_obj, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
    print(
        f"[oracle] >>> req_{seq:04d}  kind={kind} tools={tools} sys={len(system)}ch body={len(body)}ch — WAITING resp_{seq:04d}.json",
        flush=True,
    )
    return seq


async def _await_response(seq: int) -> dict:
    resp = IO_DIR / f"resp_{seq:04d}.json"
    while not resp.exists():
        await asyncio.sleep(POLL_S)
    data = None
    for _ in range(8):
        try:
            data = json.loads(resp.read_text(encoding="utf-8"))
            break
        except Exception:
            await asyncio.sleep(0.3)
    if data is None:
        raise RuntimeError(f"could not parse {resp}")
    print(f"[oracle] <<< resp_{seq:04d}  type={data.get('type')}", flush=True)
    return data


# ---- the shim ----
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.models.registry import LLMRegistry  # noqa: E402
from google.genai import types  # noqa: E402


class HumanLlm(BaseLlm):
    @classmethod
    def supported_models(cls):
        return [r"human.*"]

    async def generate_content_async(self, llm_request, stream: bool = False):
        cfg = getattr(llm_request, "config", None)
        system = _sysinstr_to_text(getattr(cfg, "system_instruction", None))
        body = _content_to_text(getattr(llm_request, "contents", None))
        tools = _tool_names(cfg)
        try:
            raw = json.loads(llm_request.model_dump_json(exclude_none=True))
        except Exception:
            raw = {"note": "model_dump_json failed"}
        seq = _dump_request("adk", system, body, tools, raw)
        data = await _await_response(seq)
        if data.get("type") == "function_call":
            part = types.Part(
                function_call=types.FunctionCall(
                    name=data["name"], args=data.get("args", {}) or {}
                )
            )
        else:
            part = types.Part(text=data.get("text", "") or "")
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
            turn_complete=True,
        )


def _install():
    import adk_appworld_agent.subagents.finder.llm as finllm
    import adk_appworld_agent.subagents.finder.routing as finrt

    async def human_call_llm_json(user_prompt, sys_prompt, *, model_cfg=None):
        seq = _dump_request(
            "finder",
            sys_prompt or "",
            user_prompt or "",
            [],
            {"sys_prompt": sys_prompt, "user_prompt": user_prompt},
        )
        data = await _await_response(seq)
        parsed = data.get("json")
        if parsed is None and data.get("text"):
            try:
                parsed = json.loads(data["text"])
            except Exception:
                parsed = {}
        return {
            "parsed_json": parsed if isinstance(parsed, dict) else {},
            "usage": finllm._usage_from_response(None),
            "retry": finllm._empty_retry_metrics(),
        }

    LLMRegistry.register(HumanLlm)
    finllm.call_llm_json = human_call_llm_json
    finrt.call_llm_json = human_call_llm_json
    print(
        "[oracle] HumanLlm registered (model='human') + finder call_llm_json patched",
        flush=True,
    )


def main() -> int:
    os.chdir(
        AGENT_ROOT
    )  # data paths (communities.json etc.) resolve relative to project root
    IO_DIR.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    for f in IO_DIR.glob("req_*"):
        f.unlink()
    for f in IO_DIR.glob("resp_*"):
        f.unlink()

    _install()
    sys.path.insert(0, str(AGENT_ROOT / "scripts"))
    import run_task

    sys.argv = [
        "run_task.py",
        "--task_id",
        os.environ.get("ORACLE_TASK_ID", "2a163ab_1"),
        "--model_name",
        "human",
        "--experiment_name",
        "oracle_train",
        "--log_root",
        str(LOG_ROOT),
        "--rpc_url",
        os.environ.get("ORACLE_RPC_URL", "tcp://127.0.0.1:4243"),
        "--plan",
        "rough",
        "--find",
        "community",
        "--execute",
        "code_plan_execute",
        "--skills",
        "off",  # gold trajectories MUST NOT read the skill library (induction leakage)
        "--keep-debug",
        "--timeout",
        "999999",
    ]
    print(
        f"[oracle] launching run_task: task={sys.argv[2]} rpc={sys.argv[12]} io={IO_DIR}",
        flush=True,
    )
    return run_task.main()


if __name__ == "__main__":
    raise SystemExit(main())
