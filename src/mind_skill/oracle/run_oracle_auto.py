"""Autonomous oracle runner — drive ONE AppWorld task through the real agent, with every
LLM call answered by a FRESH clean `claude -p` (see oracle_backend.py), no human in the loop.

Same as run_oracle_task.py but the response source is claude_call (autonomous + clean)
instead of a blocking file handshake. Still writes req_*.txt / resp_*.json so
build_trajectory.py consumes it unchanged.

Launch (project venv 3.12), one task, needs a dedicated AppWorld RPC server (default 4243):
  ORACLE_TASK_ID=82e2fac_2 ORACLE_RUN_DIR=$PWD/src/mind_skill/runs/82e2fac_2 \
    .venv/bin/python src/mind_skill/run_oracle_auto.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel

# this module lives at <root>/src/mind_skill/oracle/run_oracle_auto.py
HERE = Path(__file__).resolve()
AGENT_ROOT = HERE.parents[3]  # <root> (worktree/repo root; has scripts/)
MIND_SKILL = HERE.parents[1]  # <root>/src/mind_skill
RUN_DIR = Path(os.environ.get("ORACLE_RUN_DIR", MIND_SKILL / "runs"))
IO_DIR = Path(os.environ.get("ORACLE_IO_DIR", RUN_DIR / "io"))
LOG_ROOT = RUN_DIR / "logs"

sys.path.insert(0, str(HERE.parent))  # oracle/ dir — for the sibling oracle_backend
from oracle_backend import (  # noqa: E402
    build_oracle_prompt,
    claude_call,
    parse_oracle_response,
)

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


def _response_schema_to_text(cfg) -> str:
    """Serialize the ADK output_schema (config.response_schema) to JSON-Schema text.

    The real pipeline forces each subagent's output shape via Gemini structured
    output (config.response_schema). The claude oracle gets no such enforcement, so
    unless we hand it the schema explicitly it free-forms an off-contract shape
    (the parsed JSON then fails Plan / CodePlanOutput / ContinuationDecision).
    """
    schema = getattr(cfg, "response_schema", None)
    if schema is None:
        return ""
    try:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            obj = schema.model_json_schema()
        elif isinstance(schema, BaseModel):
            obj = schema.model_dump(exclude_none=True, mode="json")
        elif isinstance(schema, dict):
            obj = schema
        else:
            return str(schema)
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(schema)


def _response_hint(kind: str, tools: list) -> str:
    if tools:
        return (
            f"function_call REQUIRED (ANY-mode forces one of {tools}). "
            '{"type":"function_call","name":"<tool>","args":{...}}'
        )
    if kind == "finder":
        return 'JSON object: {"type":"json","json":{...}}'
    return 'raw model text (usually a JSON string the parser will load): {"type":"text","text":"..."}'


def _dump_request(
    kind: str, system: str, body: str, tools: list, schema_text: str = ""
) -> int:
    seq = _next_seq()
    stem = IO_DIR / f"req_{seq:04d}"
    stem.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "seq": seq,
                "kind": kind,
                "tools": tools,
                "system_chars": len(system),
                "body_chars": len(body),
                "schema_chars": len(schema_text),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
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
    if schema_text:
        lines += ["", "===== OUTPUT SCHEMA =====", schema_text]
    stem.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    return seq


def _write_resp(seq: int, data: dict) -> None:
    (IO_DIR / f"resp_{seq:04d}.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )


async def _ask_claude(
    kind: str, system: str, body: str, tools: list, schema_text: str = ""
) -> dict:
    seq = _dump_request(kind, system, body, tools, schema_text)
    prompt = build_oracle_prompt(
        kind, system, body, tools, _response_hint(kind, tools), schema_text
    )
    print(
        f"[oracle-auto] >>> req_{seq:04d} kind={kind} tools={tools} sys={len(system)}ch body={len(body)}ch — calling claude -p",
        flush=True,
    )
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, claude_call, prompt)
    try:
        data = parse_oracle_response(text)
    except Exception:
        data = {"type": "text", "text": text}
    _write_resp(seq, data)
    print(f"[oracle-auto] <<< resp_{seq:04d} type={data.get('type')}", flush=True)
    return data


# ---- the backend shim ----
from google.adk.models.base_llm import BaseLlm  # noqa: E402
from google.adk.models.llm_response import LlmResponse  # noqa: E402
from google.adk.models.registry import LLMRegistry  # noqa: E402
from google.genai import types  # noqa: E402


class ClaudeCliLlm(BaseLlm):
    @classmethod
    def supported_models(cls):
        return [r"human.*"]

    async def generate_content_async(self, llm_request, stream: bool = False):
        cfg = getattr(llm_request, "config", None)
        system = _sysinstr_to_text(getattr(cfg, "system_instruction", None))
        body = _content_to_text(getattr(llm_request, "contents", None))
        tools = _tool_names(cfg)
        schema_text = _response_schema_to_text(cfg)
        data = await _ask_claude("adk", system, body, tools, schema_text)
        if data.get("type") == "function_call":
            part = types.Part(
                function_call=types.FunctionCall(
                    name=data.get("name"), args=data.get("args", {}) or {}
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

    async def claude_call_llm_json(user_prompt, sys_prompt, *, model_cfg=None):
        data = await _ask_claude("finder", sys_prompt or "", user_prompt or "", [])
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

    LLMRegistry.register(ClaudeCliLlm)
    finllm.call_llm_json = claude_call_llm_json
    finrt.call_llm_json = claude_call_llm_json
    print(
        "[oracle-auto] ClaudeCliLlm registered (model='human') + finder patched",
        flush=True,
    )


def main() -> int:
    os.chdir(AGENT_ROOT)
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
        os.environ.get("ORACLE_TASK_ID", "82e2fac_2"),
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
        f"[oracle-auto] launching run_task: task={sys.argv[2]} rpc={sys.argv[12]} model={os.environ.get('ORACLE_CLAUDE_MODEL', 'opus')}",
        flush=True,
    )
    return run_task.main()


if __name__ == "__main__":
    raise SystemExit(main())
