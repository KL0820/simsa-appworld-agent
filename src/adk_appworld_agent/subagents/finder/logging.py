from __future__ import annotations

from typing import Any

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.observability.subagent_logs import (
    SubagentLogSpec,
    metrics_with_seconds,
)
from adk_appworld_agent.orchestration.state import Phase

COMMUNITY_FINDER_LOG_SPEC = SubagentLogSpec(
    agent_name="finder_subagent_community",
    phase=Phase.FIND,
    expected_output="ApiSelection",
)


def community_finder_io_record(
    *,
    task_id: str,
    status: str,
    subagent_input: SubagentInput,
    subagent_input_text: str,
    envelope: SubagentEnvelope,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = envelope.payload if isinstance(envelope.payload, dict) else {}
    # Envelope-level io is canonical; payload["io"] is a legacy fallback.
    io_payload = (
        envelope.io
        if isinstance(envelope.io, dict) and envelope.io
        else (payload.get("io") if isinstance(payload, dict) else {})
    )
    if not isinstance(io_payload, dict):
        io_payload = {}
    io_input = io_payload.get("input") if isinstance(io_payload, dict) else {}
    io_output = io_payload.get("output") if isinstance(io_payload, dict) else {}

    input_record: dict[str, Any] = {
        "phase": subagent_input.phase.value,
        "attempt": subagent_input.attempt,
        "subagent_input_text": subagent_input_text,
        "subagent_input": io_input.get("subagent_input"),
        "task_instruction": io_input.get("task_instruction"),
        "finder_instruction": io_input.get("finder_instruction"),
        "task_datetime": io_input.get("task_datetime"),
        "planned_apps": io_input.get("planned_apps", []),
        "model_input_raw": io_input.get("model_input_raw"),
        "model_calls": io_input.get("model_calls", []),
    }
    output_record: dict[str, Any] = {
        "phase": envelope.phase.value,
        "subagent_name": envelope.subagent_name,
        "attempt": envelope.attempt,
        "failure_code": envelope.failure_code,
        "warnings": envelope.warnings,
        "raw_llm_text": io_output.get("raw_llm_text"),
        "parsed_api_selection": io_output.get("parsed_api_selection"),
        "model_calls": io_output.get("model_calls", []),
    }
    if "error" in io_output:
        output_record["error"] = io_output.get("error")

    return {
        "io_format": "subagent_io.v1",
        "task_id": task_id,
        "agent": COMMUNITY_FINDER_LOG_SPEC.agent_name,
        "expected_output": COMMUNITY_FINDER_LOG_SPEC.expected_output,
        "status": status,
        "input": input_record,
        "output": output_record,
        "metrics": metrics_with_seconds(metrics),
    }


__all__ = [
    "COMMUNITY_FINDER_LOG_SPEC",
    "community_finder_io_record",
]
