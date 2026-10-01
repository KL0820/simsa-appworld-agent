from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from adk_appworld_agent.orchestration.state import Phase


def _system_text_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()


def _fence_for(body: str) -> str:
    """Pick a fence delimiter that does not collide with backticks inside body.

    Markdown closes a ```N-backtick``` fence on the first matching N-backtick
    line. Prompt bodies (system / user) often embed ``` python and ``` text
    nested fences from the prompt template; if we wrap with the default
    ``` (3 backticks), the inner ``` closes the outer fence and any markdown
    headings after that leak to the document level. Use the longest
    backtick run inside body + 1 (minimum 3) so the outer fence cannot be
    closed prematurely.
    """
    longest = 0
    current = 0
    for ch in body:
        if ch == "`":
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return "`" * max(3, longest + 1)


def _render_system_block(
    *,
    system_text: str,
    seen: dict[str, str],
    call_label: str,
) -> list[str]:
    """Render a `#### system` block with hash-collapse for repeats.

    `seen` maps a system-text hash to the call label of its first occurrence
    (e.g. ``"call 1"``, ``"PLAN/rough_planner_subagent attempt 1 call 1"``).
    First occurrence is shown in full; later calls with the identical text
    fold into a one-line banner pointing back at the first call. Different
    text (e.g. a Repair Required block was added) shows in full again with
    its own hash.
    """
    normalized = _normalize_multiline(system_text)
    if not normalized:
        return ["  System: (empty)", ""]

    digest = _system_text_hash(normalized)
    char_count = len(normalized)
    first_call = seen.get(digest)
    if first_call is None:
        seen[digest] = call_label
        lines = [
            f"  System (first occurrence · {char_count} chars · hash={digest}):",
            "",
        ]
        for ln in normalized.split("\n"):
            lines.append(f"    {ln}" if ln else "")
        lines.append("")
        return lines

    # Collapse identical repeats into a one-line banner.
    return [
        f"  System: unchanged from {first_call} · {char_count} chars · hash={digest}",
        "",
    ]


SUBAGENT_IO_FILENAME = "subagent_io.txt"

METRIC_KEYS: tuple[str, ...] = (
    "wall_ms",
    "subagent_calls",
    "envelope_received",
    "llm_calls",
    "llm_call_attempts",
    "usage_event_count",
    "timeout_count",
    "prompt_tokens",
    "completion_tokens",
    "thoughts_tokens",
    "total_tokens",
)


@dataclass(frozen=True)
class TaskLogContext:
    task_id: str
    instruction: str
    task_datetime: str
    index: int
    total: int
    is_blacklisted: bool = False


@dataclass(frozen=True)
class SubagentLogSpec:
    agent_name: str
    phase: Phase
    expected_output: str


def normalize_metrics(
    metrics: dict[str, Any] | None, *, fallback_wall_ms: int = 0
) -> dict[str, int]:
    raw = metrics or {}
    normalized = {key: int(raw.get(key) or 0) for key in METRIC_KEYS}
    if normalized["wall_ms"] == 0 and fallback_wall_ms:
        normalized["wall_ms"] = int(fallback_wall_ms)
    if normalized["llm_call_attempts"] == 0 and normalized["llm_calls"]:
        normalized["llm_call_attempts"] = normalized["llm_calls"]
    if normalized["total_tokens"] == 0:
        normalized["total_tokens"] = (
            normalized["prompt_tokens"]
            + normalized["completion_tokens"]
            + normalized["thoughts_tokens"]
        )
    return normalized


def zero_metrics(wall_ms: int = 0) -> dict[str, int]:
    return normalize_metrics({}, fallback_wall_ms=wall_ms)


def metrics_with_seconds(metrics: dict[str, Any] | None) -> dict[str, int | float]:
    raw = metrics or {}
    normalized = normalize_metrics(raw)
    wall_s = raw.get("wall_s")
    return {
        "wall_s": wall_s
        if isinstance(wall_s, int | float)
        else round(normalized["wall_ms"] / 1000.0, 3),
        "llm_calls": normalized["llm_calls"],
        "llm_call_attempts": normalized["llm_call_attempts"],
        "usage_event_count": normalized["usage_event_count"],
        "timeout_count": normalized["timeout_count"],
        "prompt_tokens": normalized["prompt_tokens"],
        "completion_tokens": normalized["completion_tokens"],
        "thoughts_tokens": normalized["thoughts_tokens"],
        "total_tokens": normalized["total_tokens"],
    }


def format_metrics(metrics: dict[str, Any] | None) -> str:
    normalized = normalize_metrics(metrics)
    return (
        f"wall_ms={normalized['wall_ms']} "
        f"subagent_calls={normalized['subagent_calls']} "
        f"envelope_received={normalized['envelope_received']} "
        f"llm_calls(observed/attempted)="
        f"{normalized['llm_calls']}/{normalized['llm_call_attempts']} "
        f"timeouts={normalized['timeout_count']} "
        f"tokens(prompt={normalized['prompt_tokens']}, "
        f"completion={normalized['completion_tokens']}, "
        f"thoughts={normalized['thoughts_tokens']}, "
        f"total={normalized['total_tokens']})"
    )


def _try_parse_json_object(text: str | None) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _to_pretty_json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _normalize_multiline(text: str | None) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n")


def _model_input_from_record(input_block: dict[str, Any]) -> dict[str, Any]:
    model_input = input_block.get("model_input_raw") or input_block.get("model_input")
    return model_input if isinstance(model_input, dict) else {}


def _user_text_from_record(
    input_block: dict[str, Any], model_input: dict[str, Any]
) -> str:
    messages = model_input.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0]
        if isinstance(first, dict):
            parts = first.get("parts")
            if isinstance(parts, list) and parts:
                first_part = parts[0]
                if isinstance(first_part, dict) and isinstance(
                    first_part.get("text"), str
                ):
                    return first_part.get("text") or ""
    rendered_prompt = input_block.get("rendered_prompt")
    return rendered_prompt if isinstance(rendered_prompt, str) else ""


def _model_calls_from_record(
    input_block: dict[str, Any],
    output_block: dict[str, Any],
) -> list[dict[str, Any]]:
    input_calls = input_block.get("model_calls")
    output_calls = output_block.get("model_calls")
    if not isinstance(input_calls, list):
        input_calls = []
    if not isinstance(output_calls, list):
        output_calls = []

    calls: list[dict[str, Any]] = []
    for index in range(max(len(input_calls), len(output_calls))):
        input_call = (
            input_calls[index]
            if index < len(input_calls) and isinstance(input_calls[index], dict)
            else {}
        )
        output_call = (
            output_calls[index]
            if index < len(output_calls) and isinstance(output_calls[index], dict)
            else {}
        )
        step = input_call.get("step") or output_call.get("step") or f"call_{index + 1}"
        model_input = (
            input_call.get("model_input_raw")
            or input_call.get("model_input")
            or output_call.get("model_input_raw")
            or {}
        )
        calls.append(
            {
                "step": str(step),
                "model_input_raw": model_input if isinstance(model_input, dict) else {},
                "raw_llm_text": output_call.get("raw_llm_text")
                or input_call.get("raw_llm_text")
                or "",
                "parsed_json": output_call.get("parsed_json")
                or input_call.get("parsed_json")
                or None,
                "code": output_call.get("code") or input_call.get("code") or "",
                "raw_stdout": output_call.get("raw_stdout")
                or input_call.get("raw_stdout")
                or "",
                "parse_error": output_call.get("parse_error")
                or input_call.get("parse_error")
                or "",
                "tool_calls": output_call.get("tool_calls")
                or input_call.get("tool_calls")
                or [],
                "event_diagnostics": output_call.get("event_diagnostics")
                or input_call.get("event_diagnostics")
                or [],
                "repair_attempts": output_call.get("repair_attempts")
                or input_call.get("repair_attempts")
                or [],
            }
        )
    return calls


def _subagent_input_metadata(input_record: dict[str, Any]) -> dict[str, Any]:
    subagent_input = input_record.get("subagent_input")
    if isinstance(subagent_input, dict) and isinstance(
        subagent_input.get("metadata"), dict
    ):
        return subagent_input["metadata"]
    subagent_input_text = input_record.get("subagent_input_text")
    if isinstance(subagent_input_text, str) and subagent_input_text.strip():
        parsed = _try_parse_json_object(subagent_input_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("metadata"), dict):
            return parsed["metadata"]
    return {}


def _copy_milestone_context(input_record: dict[str, Any]) -> None:
    metadata = _subagent_input_metadata(input_record)
    for key in (
        "planned_apps",
        "milestone_id",
        "milestone_intent",
        "milestone_index",
        "milestone_total",
        "prior_variables",
    ):
        if key in metadata and key not in input_record:
            input_record[key] = metadata[key]


def _render_repair_attempts_block(repair_attempts: list[dict]) -> list[str]:
    """Render the executor's repair_attempts list as a per-attempt timeline.

    Each attempt gets `code` / `stdout` / `parse error` blocks plus a
    `triggered by` one-liner pointing at the previous attempt's failure
    (when present). This makes input → output causality across retries
    readable without scrolling through the embedded `## Repair required`
    section in the next attempt's system prompt.
    """
    if not repair_attempts:
        return []
    n = len(repair_attempts)
    lines = [
        f"  Attempts ({n} code execution attempt{'s' if n != 1 else ''} on this milestone):",
        "",
    ]
    prev_one_liner: str | None = None
    for idx, attempt in enumerate(repair_attempts, start=1):
        if not isinstance(attempt, dict):
            continue
        code = attempt.get("code") if isinstance(attempt.get("code"), str) else ""
        raw_stdout = (
            attempt.get("raw_stdout")
            if isinstance(attempt.get("raw_stdout"), str)
            else ""
        )
        parse_error = (
            attempt.get("parse_error")
            if isinstance(attempt.get("parse_error"), str)
            else ""
        )
        llm_raised = (
            attempt.get("llm_raised")
            if isinstance(attempt.get("llm_raised"), str)
            else ""
        )
        usage = attempt.get("usage") if isinstance(attempt.get("usage"), dict) else {}
        attempt_one_liner = _attempt_failure_one_liner(parse_error, llm_raised)

        header = f"  Attempt {idx}"
        if prev_one_liner:
            header += f" -- triggered by attempt {idx - 1}: {prev_one_liner}"
        elif idx == 1:
            header += " -- initial"
        lines.append(header)
        lines.append("  " + "." * min(len(header) - 2, 60))
        lines.append("")
        if usage:
            usage_summary = (
                f"    tokens: prompt={usage.get('prompt_tokens') or 0} "
                f"completion={usage.get('completion_tokens') or 0} "
                f"thoughts={usage.get('thoughts_tokens') or 0}"
            )
            lines.extend([usage_summary, ""])
        if code:
            lines.append("    Code:")
            for ln in _normalize_multiline(code).split("\n"):
                lines.append(f"      {ln}" if ln else "")
            lines.append("")
        if raw_stdout:
            lines.append("    Stdout:")
            for ln in _normalize_multiline(raw_stdout).split("\n"):
                lines.append(f"      {ln}" if ln else "")
            lines.append("")
        if parse_error:
            lines.append("    Parse error:")
            for ln in _normalize_multiline(parse_error).split("\n"):
                lines.append(f"      {ln}" if ln else "")
            lines.append("")
        if llm_raised:
            lines.append("    LLM raised:")
            for ln in _normalize_multiline(llm_raised).split("\n"):
                lines.append(f"      {ln}" if ln else "")
            lines.append("")

        prev_one_liner = attempt_one_liner
    return lines


def _attempt_failure_one_liner(parse_error: str, llm_raised: str) -> str | None:
    for value in (parse_error, llm_raised):
        if not value or not str(value).strip():
            continue
        compact = " ".join(str(value).split())
        if len(compact) > 160:
            return compact[:157].rstrip() + "..."
        return compact
    return None


def _compact_parsed_output(value: object) -> object:
    """Strip duplicate per-attempt detail from the report-level parsed output.

    The executor's parsed_json carries `code_execute.{code, raw_stdout,
    parse_error, tool_calls, event_diagnostics, repair_attempts}`. All of
    that is rendered above as a per-attempt timeline (see
    `_render_repair_attempts_block`), so re-dumping it as JSON here just
    doubles the byte count. Keep `mode`, `stdout_json`, `tool_call_count`,
    `llm_raised` (cheap structured fields a reader actually scans) and
    replace the verbose ones with a `_<n> attempts shown above_` banner.
    """
    if not isinstance(value, dict):
        return value
    code_execute = value.get("code_execute")
    if not isinstance(code_execute, dict):
        return value
    repair_attempts = code_execute.get("repair_attempts")
    if not isinstance(repair_attempts, list) or not repair_attempts:
        return value

    compact_code_execute: dict = {}
    for key in ("mode", "stdout_json", "tool_call_count", "llm_raised"):
        if key in code_execute:
            compact_code_execute[key] = code_execute[key]
    n = len(repair_attempts)
    compact_code_execute["_attempts_rendered_above"] = (
        f"{n} attempt{'s' if n != 1 else ''}; see `#### attempts` section"
    )

    compact = dict(value)
    compact["code_execute"] = compact_code_execute
    return compact


def _render_model_calls_markdown(
    model_calls: list[dict[str, Any]],
    *,
    seen_system: dict[str, str] | None = None,
    call_label_prefix: str = "",
) -> list[str]:
    if not model_calls:
        return []

    lines = ["", "## model calls", ""]
    if seen_system is None:
        seen_system = {}
    for index, call in enumerate(model_calls, start=1):
        model_input = (
            call.get("model_input_raw")
            if isinstance(call.get("model_input_raw"), dict)
            else {}
        )
        system_text = (
            model_input.get("system_instruction")
            if isinstance(model_input.get("system_instruction"), str)
            else ""
        )
        user_text = _user_text_from_record({}, model_input)
        raw_llm_text = (
            call.get("raw_llm_text")
            if isinstance(call.get("raw_llm_text"), str)
            else ""
        )
        execute_python_code = (
            call.get("code") if isinstance(call.get("code"), str) else ""
        )
        execute_python_stdout = (
            call.get("raw_stdout") if isinstance(call.get("raw_stdout"), str) else ""
        )
        parse_error = (
            call.get("parse_error") if isinstance(call.get("parse_error"), str) else ""
        )
        tool_calls = (
            call.get("tool_calls") if isinstance(call.get("tool_calls"), list) else []
        )
        event_diagnostics = (
            call.get("event_diagnostics")
            if isinstance(call.get("event_diagnostics"), list)
            else []
        )
        repair_attempts = (
            call.get("repair_attempts")
            if isinstance(call.get("repair_attempts"), list)
            else []
        )
        parsed_json = (
            call.get("parsed_json")
            if isinstance(call.get("parsed_json"), dict)
            else None
        )
        parsed_or_raw = (
            parsed_json
            or _try_parse_json_object(raw_llm_text)
            or {"raw_llm_text": _normalize_multiline(raw_llm_text)}
        )

        step_name = call.get("step") or f"call_{index}"
        call_label = (
            f"{call_label_prefix}call {index} ({step_name})"
            if call_label_prefix
            else f"call {index}"
        )
        attempt_count_suffix = (
            f" ({len(repair_attempts)} attempts)" if repair_attempts else ""
        )
        lines.extend(
            [
                f"[{index}] {step_name}{attempt_count_suffix}",
                "-" * min(len(f"[{index}] {step_name}{attempt_count_suffix}"), 60),
                "",
            ]
        )
        lines.extend(
            _render_system_block(
                system_text=system_text, seen=seen_system, call_label=call_label
            )
        )
        normalized_user = _normalize_multiline(user_text)
        lines.append("  User:")
        for ln in normalized_user.split("\n"):
            lines.append(f"    {ln}" if ln else "")
        lines.append("")
        if repair_attempts:
            # The repair_attempts timeline is the canonical record of what
            # the executor did across retries. Suppress the duplicate
            # top-level `code` / `stdout` / `parse_error` / `tool_calls` /
            # `event_diagnostics` (they're identical to the last attempt)
            # to avoid printing the same content twice.
            lines.extend(_render_repair_attempts_block(repair_attempts))
        else:
            if tool_calls:
                tc_text = _to_pretty_json(tool_calls)
                lines.append("  Tool calls:")
                for ln in tc_text.split("\n"):
                    lines.append(f"    {ln}")
                lines.append("")
            if execute_python_code:
                lines.append("  execute_python code:")
                for ln in _normalize_multiline(execute_python_code).split("\n"):
                    lines.append(f"    {ln}" if ln else "")
                lines.append("")
            if execute_python_stdout:
                lines.append("  execute_python stdout:")
                for ln in _normalize_multiline(execute_python_stdout).split("\n"):
                    lines.append(f"    {ln}" if ln else "")
                lines.append("")
            if parse_error:
                lines.append("  Parse error:")
                for ln in _normalize_multiline(parse_error).split("\n"):
                    lines.append(f"    {ln}" if ln else "")
                lines.append("")
            if event_diagnostics:
                ed_text = _to_pretty_json(event_diagnostics)
                lines.append("  Event diagnostics:")
                for ln in ed_text.split("\n"):
                    lines.append(f"    {ln}")
                lines.append("")
        out_text = _to_pretty_json(parsed_or_raw)
        lines.append("  Output:")
        for ln in out_text.split("\n"):
            lines.append(f"    {ln}")
        lines.append("")
    return lines


def _task_context_from_subagent_input(input_block: dict[str, Any]) -> dict[str, Any]:
    subagent_input = input_block.get("subagent_input")
    if isinstance(subagent_input, dict) and isinstance(
        subagent_input.get("task_context"), dict
    ):
        return subagent_input["task_context"]
    subagent_input_text = input_block.get("subagent_input_text")
    if isinstance(subagent_input_text, str) and subagent_input_text.strip():
        parsed = _try_parse_json_object(subagent_input_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("task_context"), dict):
            return parsed["task_context"]
    return {}


def _subagent_input_summary(input_block: dict[str, Any]) -> dict[str, Any]:
    task_context = _task_context_from_subagent_input(input_block)
    summary: dict[str, Any] = {}
    for key in ("task_id", "instruction", "task_datetime"):
        value = task_context.get(key)
        if value not in (None, ""):
            summary[key] = value
    for key in (
        "task_instruction",
        "finder_instruction",
        "task_datetime",
        "planned_apps",
        "milestone_id",
        "milestone_intent",
        "milestone_index",
        "milestone_total",
        "prior_variables",
    ):
        value = input_block.get(key)
        if value not in (None, "", [], {}):
            summary[key] = value
    return summary


def _render_subagent_input_summary(input_block: dict[str, Any]) -> list[str]:
    summary = _subagent_input_summary(input_block)
    if not summary:
        return []
    summary_text = _to_pretty_json(summary)
    return [
        "",
        "SUBAGENT INPUT",
        "--------------",
        "",
        *(f"  {ln}" for ln in summary_text.split("\n")),
    ]


def io_record_from_subagent_output(
    phase_name: str, item: dict[str, Any]
) -> dict[str, Any] | None:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    # Envelope-level io is the canonical location; payload["io"] is a legacy
    # fallback for cached data written before the io-sidecar refactor.
    io_payload = item.get("io")
    if not isinstance(io_payload, dict) or not io_payload:
        io_payload = payload.get("io")
    if not isinstance(io_payload, dict):
        return None
    cache_source = (
        payload.get("cache_source")
        if isinstance(payload.get("cache_source"), str)
        else None
    )
    metrics = item.get("metrics") or payload.get("metrics") or {}
    replayed = bool(cache_source) or bool(
        isinstance(metrics, dict) and metrics.get("replayed")
    )
    if "io_format" in io_payload and isinstance(io_payload.get("input"), dict):
        _copy_milestone_context(io_payload["input"])
        output_block = (
            io_payload.get("output")
            if isinstance(io_payload.get("output"), dict)
            else None
        )
        if output_block is not None:
            if cache_source and not output_block.get("cache_source"):
                output_block["cache_source"] = cache_source
            if replayed and "replayed" not in output_block:
                output_block["replayed"] = True
        return io_payload

    input_record = dict(io_payload.get("input") or {})
    output_record = dict(io_payload.get("output") or {})
    _copy_milestone_context(input_record)
    input_record.setdefault("phase", phase_name)
    input_record.setdefault("attempt", item.get("attempt"))
    output_record.setdefault("phase", phase_name)
    output_record.setdefault("subagent_name", item.get("subagent_name"))
    output_record.setdefault("attempt", item.get("attempt"))
    output_record.setdefault("failure_code", item.get("failure_code"))
    output_record.setdefault("warnings", item.get("warnings") or [])
    if cache_source:
        output_record.setdefault("cache_source", cache_source)
    if replayed:
        output_record.setdefault("replayed", True)
    return {
        "io_format": "subagent_io.v1",
        "agent": item.get("subagent_name"),
        "expected_output": io_payload.get("expected_output") or "",
        "status": item.get("status"),
        "input": input_record,
        "output": output_record,
        "metrics": metrics,
    }


def render_subagent_io_markdown(
    io_record: dict[str, Any],
    *,
    seen_system: dict[str, str] | None = None,
) -> list[str]:
    """Render one subagent call using the shared human-facing report format.

    `seen_system` (optional) lets the caller share a system-text dedup map
    across multiple records (e.g. the per-task subagent IO sink). When
    omitted, dedup is local to this single record.
    """

    input_block = (
        io_record.get("input") if isinstance(io_record.get("input"), dict) else {}
    )
    output_block = (
        io_record.get("output") if isinstance(io_record.get("output"), dict) else {}
    )
    metrics = (
        io_record.get("metrics") if isinstance(io_record.get("metrics"), dict) else {}
    )
    model_input = _model_input_from_record(input_block)
    model_calls = _model_calls_from_record(input_block, output_block)

    system_text = (
        model_input.get("system_instruction")
        if isinstance(model_input.get("system_instruction"), str)
        else ""
    )
    user_text = _user_text_from_record(input_block, model_input)
    raw_llm_text = (
        output_block.get("raw_llm_text")
        if isinstance(output_block.get("raw_llm_text"), str)
        else ""
    )
    parsed_output = _try_parse_json_object(raw_llm_text)
    parsed_or_raw = (
        parsed_output
        if parsed_output is not None
        else {"raw_llm_text": _normalize_multiline(raw_llm_text)}
    )
    parsed_or_raw = _compact_parsed_output(parsed_or_raw)
    cost_summary = metrics_with_seconds(metrics)
    phase = output_block.get("phase") or input_block.get("phase") or ""
    attempt = output_block.get("attempt") or input_block.get("attempt") or ""
    agent = output_block.get("subagent_name") or io_record.get("agent") or ""
    context = {
        "phase": phase,
        "agent": agent,
        "attempt": attempt,
        "status": io_record.get("status"),
        "failure_code": output_block.get("failure_code"),
    }
    if output_block.get("replayed"):
        context["source"] = "cache"
        if output_block.get("cache_source"):
            context["cache_source"] = output_block["cache_source"]
    for key in (
        "planned_apps",
        "milestone_id",
        "milestone_intent",
        "milestone_index",
        "milestone_total",
    ):
        if key in input_block:
            context[key] = input_block[key]

    title_bar = "=" * 64
    context_text = _to_pretty_json(context)
    lines = [
        title_bar,
        "SUBAGENT IO REPORT",
        title_bar,
        "",
        "PHASE / AGENT / ATTEMPT",
        "-----------------------",
        "",
        *(f"  {ln}" for ln in context_text.split("\n")),
    ]
    if model_calls:
        lines.extend(_render_subagent_input_summary(input_block))
        attempt_label = (
            f"{phase}/{agent} attempt {attempt} " if phase or agent or attempt else ""
        )
        lines.extend(
            _render_model_calls_markdown(
                model_calls,
                seen_system=seen_system,
                call_label_prefix=attempt_label,
            )
        )
    else:
        seen = seen_system if seen_system is not None else {}
        attempt_label = (
            f"{phase}/{agent} attempt {attempt} call 1"
            if phase or agent or attempt
            else "call 1"
        )
        system_lines = _render_system_block(
            system_text=system_text, seen=seen, call_label=attempt_label
        )
        lines.extend(["", "SYSTEM", "------", ""])
        lines.extend(system_lines)
        normalized_user_top = _normalize_multiline(user_text)
        lines.extend(["", "USER", "----", ""])
        for ln in normalized_user_top.split("\n"):
            lines.append(f"  {ln}" if ln else "")
    parsed_text = _to_pretty_json(parsed_or_raw)
    cost_text = _to_pretty_json(cost_summary)
    lines.extend(
        [
            "",
            "PARSED OUTPUT",
            "-------------",
            "",
            *(f"  {ln}" for ln in parsed_text.split("\n")),
            "",
            "COST",
            "----",
            "",
            *(f"  {ln}" for ln in cost_text.split("\n")),
        ]
    )
    return lines


__all__ = [
    "METRIC_KEYS",
    "SUBAGENT_IO_FILENAME",
    "SubagentLogSpec",
    "TaskLogContext",
    "format_metrics",
    "io_record_from_subagent_output",
    "metrics_with_seconds",
    "normalize_metrics",
    "render_subagent_io_markdown",
    "zero_metrics",
]
