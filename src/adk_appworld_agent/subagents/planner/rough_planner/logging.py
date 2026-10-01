from __future__ import annotations

from typing import Any

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.observability.subagent_logs import (
    SubagentLogSpec,
    metrics_with_seconds,
)
from adk_appworld_agent.orchestration.state import Phase

ROUGH_PLANNER_LOG_SPEC = SubagentLogSpec(
    agent_name="rough_planner_subagent",
    phase=Phase.PLAN,
    expected_output="Plan",
)


def _io_mode() -> str:
    return "subagent_io.v1"


def rough_plan_io_record(
    *,
    task_id: str,
    status: str,
    subagent_input: SubagentInput,
    subagent_input_text: str,
    raw_event_text: str,
    envelope: SubagentEnvelope,
    outer_event_count: int,
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
    model_input = io_input.get("model_input") if isinstance(io_input, dict) else None
    raw_llm_text = (
        io_output.get("raw_llm_text") if isinstance(io_output, dict) else None
    )
    io_mode = _io_mode()

    input_record: dict[str, Any] = {
        "phase": subagent_input.phase.value,
        "attempt": subagent_input.attempt,
        "subagent_input_text": subagent_input_text,
        "rendered_prompt": io_input.get("rendered_prompt"),
        "model_input_raw": model_input,
    }
    output_record: dict[str, Any] = {
        "phase": envelope.phase.value,
        "subagent_name": envelope.subagent_name,
        "attempt": envelope.attempt,
        "failure_code": envelope.failure_code,
        "warnings": envelope.warnings,
        "raw_llm_text": raw_llm_text,
        "raw_event_text": raw_event_text,
    }

    record: dict[str, Any] = {
        "io_format": io_mode,
        "agent": ROUGH_PLANNER_LOG_SPEC.agent_name,
        "expected_output": ROUGH_PLANNER_LOG_SPEC.expected_output,
        "status": status,
        "input": input_record,
        "output": output_record,
        "metrics": metrics_with_seconds(metrics),
    }
    return record


def rough_plan_error_io_record(
    *,
    task_id: str,
    status: str,
    task_datetime: str,
    error: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    io_mode = _io_mode()
    return {
        "io_format": io_mode,
        "agent": ROUGH_PLANNER_LOG_SPEC.agent_name,
        "expected_output": ROUGH_PLANNER_LOG_SPEC.expected_output,
        "status": status,
        "input": {"task_datetime": task_datetime},
        "output": {"error": error},
        "metrics": metrics_with_seconds(metrics),
    }


__all__ = [
    "ROUGH_PLANNER_LOG_SPEC",
    "rough_plan_error_io_record",
    "rough_plan_io_record",
]
