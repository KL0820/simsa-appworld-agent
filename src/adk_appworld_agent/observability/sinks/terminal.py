from __future__ import annotations

from adk_appworld_agent.observability.events import (
    OrchestrationEvent,
    SubagentCompleted,
)
from adk_appworld_agent.observability.metrics import compact_tokens, normalize_metrics


def _short_text(text: object, *, limit: int = 90) -> str:
    if not isinstance(text, str):
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _api_key(item: object) -> str | None:
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        return None
    app = item.get("app")
    name = item.get("name") or item.get("api_name")
    if isinstance(app, str) and isinstance(name, str):
        return f"{app}.{name}"
    key = item.get("key") or item.get("api_key")
    return key if isinstance(key, str) else None


def _join_limited(items: list[str], *, limit: int = 8) -> str:
    visible = items[:limit]
    suffix = f", +{len(items) - limit} more" if len(items) > limit else ""
    return ", ".join(visible) + suffix


def _numbered_plan_lines(plan: object, *, limit: int = 3) -> list[str]:
    if not isinstance(plan, list) or not plan:
        return []

    lines: list[str] = []
    for index, step in enumerate(plan[:limit], start=1):
        text = _short_text(step, limit=120)
        if text:
            lines.append(f"     {index}. {text}")
    if len(plan) > limit:
        lines.append(f"     ... +{len(plan) - limit} more")
    return lines


def _code_plan_lines(payload: dict) -> list[str]:
    code_plan = payload.get("code_plan")
    if not isinstance(code_plan, dict):
        return []

    raw_steps = code_plan.get("plan_steps")
    if not isinstance(raw_steps, list):
        # Backwards compat for older runs.
        raw_steps = code_plan.get("plan")
    plan: list[str] = list(raw_steps) if isinstance(raw_steps, list) else []
    construct = str(code_plan.get("construct_step") or "").strip()
    print_step = str(code_plan.get("print_step") or "").strip()
    for trailing in (construct, print_step):
        if trailing:
            plan.append(trailing)

    step_lines = _numbered_plan_lines(plan)
    output_variable = code_plan.get("output_variable")
    output_name = ""
    if isinstance(output_variable, dict):
        output_name = str(output_variable.get("name") or "")
    step_count = len(plan)

    summary = f"  -> code_plan steps={step_count}"
    if output_name:
        summary += f" output={output_name}"
    return [summary, *step_lines]


def _subagent_input_metadata(item: dict) -> dict:
    # Envelope-level io is the canonical source; payload["io"] is a legacy
    # fallback for cached / pre-refactor data.
    io_payload = item.get("io") if isinstance(item.get("io"), dict) else {}
    if not io_payload:
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        io_payload = payload.get("io") if isinstance(payload.get("io"), dict) else {}
    input_payload = (
        io_payload.get("input") if isinstance(io_payload.get("input"), dict) else {}
    )
    subagent_input = input_payload.get("subagent_input")
    if isinstance(subagent_input, dict) and isinstance(
        subagent_input.get("metadata"), dict
    ):
        return subagent_input["metadata"]
    return {}


def _milestone_detail_line(item: dict) -> str | None:
    metadata = _subagent_input_metadata(item)
    idx = metadata.get("milestone_index")
    total = metadata.get("milestone_total")
    parts: list[str] = []
    if isinstance(idx, int) and isinstance(total, int) and total:
        parts.append(f"milestone={idx + 1}/{total}")
    planned_apps = metadata.get("planned_apps")
    if isinstance(planned_apps, list) and planned_apps:
        parts.append("app=" + ",".join(str(app) for app in planned_apps))
    intent = _short_text(metadata.get("milestone_intent"), limit=90)
    if intent:
        parts.append(f'intent="{intent}"')
    return "  -> " + " ".join(parts) if parts else None


def terminal_lines_for_subagent(phase_name: str, item: dict) -> list[str]:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    raw_metrics = (
        item.get("metrics")
        if isinstance(item.get("metrics"), dict)
        else payload.get("metrics")
    )
    metrics = normalize_metrics(raw_metrics if isinstance(raw_metrics, dict) else None)
    status = item.get("status")
    mark = "OK" if status == "SUCCEEDED" else "FAIL"
    replayed = bool(isinstance(raw_metrics, dict) and raw_metrics.get("replayed"))
    source = " cache" if payload.get("cache_source") or replayed else ""
    wall_s = metrics["wall_ms"] / 1000.0
    header = (
        f"[{phase_name:<6} {str(item.get('subagent_name') or ''):<24} {mark}{source}]"
    )
    metrics_line = (
        f"  attempt={item.get('attempt')} llm={metrics['llm_calls']} "
        f"wall={wall_s:.1f}s tokens={compact_tokens(metrics['total_tokens'])}"
    )

    lines = [header.rstrip(), metrics_line.rstrip()]
    milestone_detail = _milestone_detail_line(item)
    if milestone_detail:
        lines.append(milestone_detail)
    if phase_name == "PLAN":
        tasks = payload.get("tasks")
        if isinstance(tasks, list) and tasks:
            apps = [
                str(task.get("app") or "unknown")
                for task in tasks
                if isinstance(task, dict)
            ]
            lines.append(f"  -> {len(tasks)} tasks: {_join_limited(apps, limit=10)}")
    elif phase_name == "FIND":
        matched = (
            payload.get("matched_apps")
            if isinstance(payload.get("matched_apps"), list)
            else []
        )
        candidate_apis = (
            payload.get("candidate_apis")
            if isinstance(payload.get("candidate_apis"), list)
            else []
        )
        keys = [key for key in (_api_key(item) for item in candidate_apis) if key]
        lines.append(
            f"  -> matched={','.join(str(app) for app in matched) or 'none'} "
            f"apis={len(keys)}"
        )
        if keys:
            lines.append(f"     {_join_limited(keys, limit=10)}")
    elif phase_name == "EXECUTE":
        lines.append(
            f"  -> finalize={payload.get('finalize_called')} "
            f"done={payload.get('milestone_done')} tools={payload.get('tool_call_count') or 0}"
        )
        lines.extend(_code_plan_lines(payload))
    elif phase_name == "SUBMIT":
        extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
        lines.append(
            f"  -> eval={extra.get('passed')}/{extra.get('total')} "
            f"block={payload.get('block_reason')}"
        )

    failure = item.get("failure_code")
    if failure:
        lines.append(f"  -> failure={failure}")
    return lines


class TerminalEventPrinter:
    def handle(self, event: OrchestrationEvent) -> None:
        if not isinstance(event, SubagentCompleted):
            return
        for line in terminal_lines_for_subagent(event.phase_name, event.raw_output):
            print(line, flush=True)
