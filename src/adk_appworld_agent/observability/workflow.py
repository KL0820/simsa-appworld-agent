from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from adk_appworld_agent.observability.metrics import aggregate_subagent_metrics

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    cleaned = _ANSI_RE.sub("", text)
    return "\n".join(line.rstrip() for line in cleaned.splitlines())


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def write_workflow(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def candidate_api_keys(candidate_apis: list[dict] | None) -> list[str]:
    keys: list[str] = []
    for item in candidate_apis or []:
        if not isinstance(item, dict):
            continue
        app = item.get("app")
        name = item.get("name")
        if isinstance(app, str) and isinstance(name, str):
            keys.append(f"{app}.{name}")
    return sorted(keys)


def _api_key(app: object, api_name: object) -> str | None:
    if not isinstance(app, str) or not isinstance(api_name, str):
        return None
    return f"{app}.{api_name}"


def _shape_score(shape: dict | None) -> tuple[int, int]:
    if not isinstance(shape, dict):
        return (0, 0)
    count_hint = shape.get("count_hint")
    count_score = count_hint if isinstance(count_hint, int) else 0
    keys = shape.get("keys")
    key_score = len(keys) if isinstance(keys, list) else 0
    return (count_score, key_score)


def _select_better_shape(current: dict | None, candidate: dict | None) -> dict | None:
    if not isinstance(candidate, dict):
        return current
    if not isinstance(current, dict):
        return candidate
    if _shape_score(candidate) >= _shape_score(current):
        return candidate
    return current


def _collect_execute_api_summary(
    *,
    sandbox_trace: list[dict],
    candidate_keys: list[str],
) -> dict:
    api_docs_lookups: list[str] = []
    called_counter: Counter[str] = Counter()
    failed_api_calls: list[str] = []
    unexpected_api_calls: list[str] = []
    api_result_shapes: dict[str, dict] = {}
    api_call_trace: list[dict] = []
    candidate_set = set(candidate_keys)

    for block in sandbox_trace:
        if not isinstance(block, dict):
            continue
        for api_call in block.get("api_calls", []):
            if not isinstance(api_call, dict):
                continue
            app = api_call.get("app")
            api_name = api_call.get("api_name")
            api_key = _api_key(app, api_name)
            if api_key is None:
                continue
            status = str(api_call.get("status") or "")
            kind = str(api_call.get("kind") or "")

            if kind.startswith("api_docs."):
                if kind == "api_docs.show_api_descriptions":
                    api_docs_lookups.append(f"{app}.*")
                else:
                    api_docs_lookups.append(api_key)
                continue

            called_counter[api_key] += 1
            if status and status != "ok" and api_key not in failed_api_calls:
                failed_api_calls.append(api_key)
            if (
                candidate_set
                and api_key not in candidate_set
                and api_key not in unexpected_api_calls
            ):
                unexpected_api_calls.append(api_key)
            if status == "ok":
                api_result_shapes[api_key] = _select_better_shape(
                    api_result_shapes.get(api_key),
                    api_call.get("result_shape"),
                ) or api_result_shapes.get(api_key)

            api_call_trace.append(
                {
                    "api_key": api_key,
                    "kwargs": api_call.get("kwargs") or {},
                    "status": status or "ok",
                    "result_shape": api_call.get("result_shape"),
                    "error": api_call.get("error"),
                }
            )

    return {
        "api_docs_lookups": sorted(set(api_docs_lookups)),
        "called_apis": sorted(called_counter.keys()),
        "api_call_counts": dict(sorted(called_counter.items())),
        "failed_api_calls": failed_api_calls,
        "unexpected_api_calls": unexpected_api_calls,
        "api_result_shapes": dict(sorted(api_result_shapes.items())),
        "api_call_trace": api_call_trace,
    }


def _summary_text(value: object, *, limit: int = 160) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + f"...<+{len(text) - limit}c>"
    return text


def _executor_variables_count(payload: dict) -> int:
    exec_result = payload.get("executor_result") or {}
    variables = exec_result.get("variables") if isinstance(exec_result, dict) else None
    return len(variables) if isinstance(variables, list) else 0


def _submission_shape(candidate: object) -> dict:
    if not isinstance(candidate, dict):
        return {}
    return {
        "task_type_hint": candidate.get("task_type_hint"),
        "answer_type": candidate.get("answer_type"),
        "source": candidate.get("source"),
    }


def _submission_summary(submit_payload: dict) -> dict:
    extra = submit_payload.get("extra") or {}
    raw_report = submit_payload.get("evaluation_report")
    if isinstance(raw_report, dict):
        for key in ("result", "report", "message"):
            value = raw_report.get(key)
            if value is not None:
                report_text = str(value)
                break
        else:
            report_text = ""
    else:
        report_text = str(raw_report or "")
    report_text = _strip_ansi(report_text).strip()
    return {
        "submitted_answer": submit_payload.get("submitted_answer"),
        "passed": extra.get("passed"),
        "failed": extra.get("failed"),
        "total": extra.get("total"),
        "report_excerpt": report_text or None,
        "sub_test_results": _parse_sub_test_results(report_text) if report_text else [],
    }


def _parse_sub_test_results(report_text: str) -> list[dict]:
    """Parse the live evaluator report into a list of sub-test outcomes.

    Each entry is ``{"status": "pass"|"fail", "requirement": "<docstring>",
    "error": "<assertion text or None>"}``. Used by the markdown renderer
    so analysis can see which specific sub-tests failed without scrolling
    through the full report.
    """
    results: list[dict] = []
    if not report_text:
        return results
    pass_pattern = re.compile(r">>\s*Passed Requirement\s*\n(.+?)(?=>>|\Z)", re.DOTALL)
    fail_pattern = re.compile(r">>\s*Failed Requirement\s*\n(.+?)(?=>>|\Z)", re.DOTALL)
    for block in pass_pattern.findall(report_text):
        requirement = _first_real_paragraph(block)
        if requirement:
            results.append(
                {"status": "pass", "requirement": requirement, "error": None}
            )
    for block in fail_pattern.findall(report_text):
        requirement = _first_real_paragraph(block)
        error = _extract_assertion_error(block)
        if requirement:
            results.append(
                {"status": "fail", "requirement": requirement, "error": error}
            )
    return results


def _first_real_paragraph(block: str) -> str:
    """First non-empty, non-code-fence line group in a Pass/Fail block.

    Stops at section separators (`---`, `=====`, or Unicode box-drawing
    `─` lines used by the AppWorld evaluator's "Passes/Fails" headers)
    so the last Pass entry doesn't accidentally absorb the next section.
    """
    parts: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            if parts:
                break
            continue
        if stripped.startswith("```"):
            if parts:
                break
            continue
        if stripped.startswith("---") or stripped.startswith("====="):
            if parts:
                break
            continue
        # Unicode box-drawing separator used by AppWorld evaluator banners
        # (e.g. "──── Fails ────"). The line is mostly box chars + a
        # section name. Treat any line whose first 4 chars contain `─`
        # as a separator.
        if "─" in stripped[:4]:
            if parts:
                break
            continue
        parts.append(stripped)
    return " ".join(parts).strip()


def _extract_assertion_error(block: str) -> str | None:
    """Pull the AssertionError / failure detail out of a fail block.

    Captures the multi-line body following ``AssertionError:`` (e.g.
    ``set() == {a, b, c}\\nIn right but not left: [...]``) until a blank
    line or a section separator, joining with ` | ` so the whole error
    fits on a single sub-test result line.
    """
    lines = block.splitlines()
    # Find "AssertionError" line first; if absent, fall back to the
    # first post-code-fence non-trivial line.
    assertion_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("AssertionError"):
            assertion_idx = i
            break

    if assertion_idx is not None:
        parts: list[str] = [lines[assertion_idx].strip()]
        for line in lines[assertion_idx + 1 :]:
            stripped = line.strip()
            if not stripped:
                break
            if stripped.startswith("```"):
                break
            if stripped.startswith("---") or stripped.startswith("====="):
                break
            if "─" in stripped[:4]:
                break
            parts.append(stripped)
        # Drop trailing empty entries; join with ' | ' so the whole error
        # reads on a single line of the sub-test entry.
        compact = " | ".join(p for p in parts if p)
        # Trim long error blobs to avoid swamping the sub-test list.
        if len(compact) > 240:
            compact = compact[:237].rstrip() + "..."
        return compact

    # No explicit AssertionError; fall back to first non-trivial line
    # after a code fence (legacy formats).
    after_fence = False
    for line in lines:
        stripped = line.strip()
        if stripped == "```":
            after_fence = True
            continue
        if after_fence and stripped and not stripped.startswith("---"):
            return stripped
    return None


def _compact_api_call(api_call: dict) -> dict:
    return {
        key: value for key, value in api_call.items() if key not in {"result_preview"}
    }


def _compact_final_state(final_state: dict) -> dict:
    budget = final_state.get("budget_snapshot")
    return {
        "phase": final_state.get("phase"),
        "status": final_state.get("status"),
        "budget_snapshot": budget if isinstance(budget, dict) else {},
        "replan_count": final_state.get("replan_count"),
        "retry_count": final_state.get("retry_count"),
        "completion_block_reason": final_state.get("completion_block_reason"),
        "milestone_count": len(final_state.get("milestones") or []),
        "active_milestone_index": final_state.get("active_milestone_index"),
        "named_variable_count": len(final_state.get("named_variables") or {}),
        # Attempt-budget economic stop (§4.2): typed terminal class + the
        # no-progress streak that triggered it, so /analyze-task-fail can tell a
        # clean budget-spent stop (best-so-far) from a real failure and spot a
        # too-early stop. (Field was `give_up_reason` before 2026-06-22.)
        "economic_stop_reason": final_state.get("economic_stop_reason"),
        "no_progress_streak": final_state.get("no_progress_streak"),
    }


def _wall_s_from_ms(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) / 1000.0, 3)


def _metrics_wall_s(metrics: dict | None) -> float | None:
    if not isinstance(metrics, dict):
        return None
    return _wall_s_from_ms(metrics.get("wall_ms"))


def _compact_one_line(text: str, *, limit: int = 200) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _execute_failure_one_liner(payload: dict) -> str | None:
    """Extract a short root-cause line from an executor's last failed attempt.

    Looks at code_execute.repair_attempts[-1].parse_error / .llm_raised first,
    then falls back to code_execute.parse_error / .llm_raised on the run itself.
    Returns None when the executor finished cleanly (no parse_error / llm_raised).
    """
    code_execute = payload.get("code_execute") if isinstance(payload, dict) else {}
    if not isinstance(code_execute, dict):
        return None

    repair_attempts = code_execute.get("repair_attempts")
    if isinstance(repair_attempts, list):
        for attempt in reversed(repair_attempts):
            if not isinstance(attempt, dict):
                continue
            for key in ("parse_error", "llm_raised"):
                value = attempt.get(key)
                if isinstance(value, str) and value.strip():
                    return _compact_one_line(value)

    for key in ("parse_error", "llm_raised"):
        value = code_execute.get(key)
        if isinstance(value, str) and value.strip():
            return _compact_one_line(value)

    return None


def _aggregate_failure_one_liner(
    execute_runs: list[dict],
    submit_payload: dict,
) -> str | None:
    """Pick a one-line failure summary for the task overview.

    Prefer the latest failed executor's `failure_one_liner`. Fall back to the
    submit phase's evaluation report excerpt (compacted) when the executor
    finalized cleanly but the evaluator still rejected the answer.
    """
    for run in reversed(execute_runs or []):
        if not isinstance(run, dict):
            continue
        if run.get("status") == "FAILED":
            line = run.get("failure_one_liner")
            if isinstance(line, str) and line.strip():
                return line
    extra = submit_payload.get("extra") if isinstance(submit_payload, dict) else {}
    if isinstance(extra, dict):
        passed = extra.get("passed")
        total = extra.get("total")
        if isinstance(passed, int) and isinstance(total, int) and passed < total:
            return _compact_one_line(
                f"evaluator rejected answer ({passed}/{total} passed)"
            )
    return None


def _eval_summary(result: dict) -> dict:
    passed = result.get("passed")
    failed = result.get("failed")
    total = result.get("total")
    passed_all = None
    if isinstance(passed, int) and isinstance(failed, int) and isinstance(total, int):
        passed_all = (failed == 0) and (passed == total)
    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "passed_all": passed_all,
    }


def _active_milestone_label(final_state: dict, milestone_total: int) -> str | None:
    active_index = final_state.get("active_milestone_index")
    if not isinstance(active_index, int) or milestone_total <= 0:
        return None
    return f"{active_index + 1}/{milestone_total}"


def _planner_milestones(plan_payload: dict, final_state: dict) -> list[dict]:
    raw_milestones = plan_payload.get("milestones")
    if not isinstance(raw_milestones, list) or not raw_milestones:
        raw_tasks = plan_payload.get("tasks")
        if isinstance(raw_tasks, list) and raw_tasks:
            raw_milestones = [
                {
                    "id": None,
                    "intent": task.get("task"),
                    "app": task.get("app"),
                }
                for task in raw_tasks
                if isinstance(task, dict)
            ]
        else:
            raw_milestones = final_state.get("milestones") or []

    milestones: list[dict] = []
    for index, milestone in enumerate(raw_milestones):
        if not isinstance(milestone, dict):
            continue
        milestones.append(
            {
                "index": index,
                "id": milestone.get("id"),
                "intent": milestone.get("intent"),
                "app": milestone.get("app"),
            }
        )
    return milestones


def _find_metadata(find_item: dict) -> dict:
    # Envelope-level io is canonical; payload["io"] is a legacy fallback.
    io_payload = find_item.get("io") if isinstance(find_item.get("io"), dict) else {}
    if not io_payload:
        payload = (
            find_item.get("payload")
            if isinstance(find_item.get("payload"), dict)
            else {}
        )
        io_payload = payload.get("io") if isinstance(payload.get("io"), dict) else {}
    io_input = (
        io_payload.get("input") if isinstance(io_payload.get("input"), dict) else {}
    )
    subagent_input = (
        io_input.get("subagent_input")
        if isinstance(io_input.get("subagent_input"), dict)
        else {}
    )
    metadata = (
        subagent_input.get("metadata")
        if isinstance(subagent_input.get("metadata"), dict)
        else {}
    )
    return metadata


def _find_run_summary(find_item: dict) -> dict:
    payload = (
        find_item.get("payload") if isinstance(find_item.get("payload"), dict) else {}
    )
    metrics = (
        find_item.get("metrics") if isinstance(find_item.get("metrics"), dict) else {}
    )
    metadata = _find_metadata(find_item)
    cache_source = (
        payload.get("cache_source")
        if isinstance(payload.get("cache_source"), str)
        else None
    )
    replayed = bool(cache_source) or bool(metrics.get("replayed"))
    return {
        "attempt": find_item.get("attempt"),
        "milestone_index": metadata.get("milestone_index"),
        "milestone_id": metadata.get("milestone_id"),
        "milestone_intent": metadata.get("milestone_intent"),
        "planned_apps": metadata.get("planned_apps"),
        "selected_apis": candidate_api_keys(payload.get("candidate_apis")),
        "llm_calls": metrics.get("llm_calls"),
        "llm_call_attempts": metrics.get("llm_call_attempts"),
        "retry_count": metrics.get("retry_count"),
        "backoff_s": _wall_s_from_ms(metrics.get("retry_backoff_ms")),
        "rate_limit_count": metrics.get("rate_limit_count"),
        "provider_error_count": metrics.get("provider_error_count"),
        "provider_issue_likely": bool(metrics.get("provider_error_count")),
        "wall_s": _metrics_wall_s(metrics),
        "replayed": replayed,
        "cache_source": cache_source,
    }


def _slice_sandbox_trace_by_execute(
    execute_items: list[dict],
    sandbox_trace: list[dict],
) -> list[list[dict]]:
    blocks = [block for block in sandbox_trace if isinstance(block, dict)]
    grouped: list[list[dict]] = []
    cursor = 0
    for exec_item in execute_items:
        payload = (
            exec_item.get("payload")
            if isinstance(exec_item.get("payload"), dict)
            else {}
        )
        raw_count = payload.get("tool_call_count")
        block_count = raw_count if isinstance(raw_count, int) and raw_count > 0 else 0
        grouped.append(blocks[cursor : cursor + block_count])
        cursor += block_count
    if cursor < len(blocks) and grouped:
        grouped[-1].extend(blocks[cursor:])
    return grouped


def _execute_run_summary(
    exec_item: dict,
    *,
    sandbox_blocks: list[dict],
    candidate_keys: list[str],
) -> dict:
    payload = (
        exec_item.get("payload") if isinstance(exec_item.get("payload"), dict) else {}
    )
    # Envelope-level metrics is canonical; payload["metrics"] is a legacy fallback.
    metrics = (
        exec_item.get("metrics") if isinstance(exec_item.get("metrics"), dict) else {}
    )
    if not metrics:
        metrics = (
            payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
        )
    api_summary = _collect_execute_api_summary(
        sandbox_trace=sandbox_blocks,
        candidate_keys=candidate_keys,
    )
    code_plan = _code_plan_summary(payload.get("code_plan"))
    output_variable_preview = _output_variable_preview(payload)
    return {
        "attempt": exec_item.get("attempt"),
        "milestone_index": payload.get("milestone_index"),
        "status": exec_item.get("status"),
        "failure_code": exec_item.get("failure_code"),
        "tool_calls": payload.get("tool_call_count") or 0,
        "called_apis": api_summary["called_apis"],
        "failed_api_calls": api_summary["failed_api_calls"],
        "unexpected_api_calls": api_summary["unexpected_api_calls"],
        "api_call_trace": api_summary["api_call_trace"],
        "wall_s": _metrics_wall_s(metrics) or 0.0,
        "llm_calls": metrics.get("llm_calls") or 0,
        "code_plan": code_plan,
        "output_variable_preview": output_variable_preview,
        "failure_one_liner": _execute_failure_one_liner(payload),
    }


def _output_variable_preview(payload: dict) -> str | None:
    """Pull the executor's stdout_json `value` as a compact preview.

    The value is the milestone's actual output that becomes the next
    milestone's prior_variable. We render it (truncated) so analysis can
    see what flowed forward without jumping to the next milestone's
    input section.
    """
    code_execute = payload.get("code_execute") if isinstance(payload, dict) else None
    if not isinstance(code_execute, dict):
        return None
    stdout_json = code_execute.get("stdout_json")
    if not isinstance(stdout_json, dict):
        return None
    value = stdout_json.get("value")
    if value is None:
        return None
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return text


def _code_plan_summary(raw_code_plan: object) -> dict | None:
    if not isinstance(raw_code_plan, dict):
        return None
    # CodePlanOutput dumped via model_dump exposes plan_steps / construct_step
    # / print_step. Rebuild the rendered numbered list here so the summary
    # mirrors what the executor actually saw.
    raw_steps = raw_code_plan.get("plan_steps")
    if not isinstance(raw_steps, list):
        # Backwards compat for older runs serialised before the schema split.
        raw_steps = raw_code_plan.get("plan")
    if not isinstance(raw_steps, list):
        return None
    steps = [str(step).strip() for step in raw_steps if str(step).strip()]
    construct = str(raw_code_plan.get("construct_step") or "").strip()
    print_step = str(raw_code_plan.get("print_step") or "").strip()
    for trailing in (construct, print_step):
        if trailing:
            steps.append(trailing)
    if not steps:
        return None
    output_variable = raw_code_plan.get("output_variable")
    output_name = ""
    output_description = ""
    if isinstance(output_variable, dict):
        output_name = str(output_variable.get("name") or "").strip()
        output_description = str(output_variable.get("description") or "").strip()
    return {
        "steps": steps,
        "output_variable": {
            "name": output_name,
            "description": output_description,
        },
    }


def _latest_run_by_milestone(
    runs_by_milestone: dict[int, list[dict]], milestone_index: int
) -> dict | None:
    runs = runs_by_milestone.get(milestone_index) or []
    return runs[-1] if runs else None


def _milestone_digest(
    milestone: dict,
    *,
    find_run: dict | None,
    execute_run: dict | None,
) -> dict:
    digest = {
        "index": milestone.get("index"),
        "app": milestone.get("app"),
        "goal": milestone.get("intent"),
        "find": {
            "status": "SUCCEEDED" if find_run is not None else "MISSING",
            "selected_apis": (find_run or {}).get("selected_apis", []),
            "wall_s": (find_run or {}).get("wall_s"),
            "llm_calls": (find_run or {}).get("llm_calls"),
            "llm_call_attempts": (find_run or {}).get("llm_call_attempts"),
            "retry_count": (find_run or {}).get("retry_count"),
            "backoff_s": (find_run or {}).get("backoff_s"),
            "rate_limit_count": (find_run or {}).get("rate_limit_count"),
            "provider_issue_likely": (find_run or {}).get(
                "provider_issue_likely", False
            ),
            "replayed": (find_run or {}).get("replayed", False),
            "cache_source": (find_run or {}).get("cache_source"),
        },
        "execute": {
            "status": (execute_run or {}).get("status") or "MISSING",
            "called_apis": (execute_run or {}).get("called_apis", []),
            "failed_api_calls": (execute_run or {}).get("failed_api_calls", []),
            "unexpected_api_calls": (execute_run or {}).get("unexpected_api_calls", []),
            "api_call_trace": (execute_run or {}).get("api_call_trace", []),
            "tool_calls": (execute_run or {}).get("tool_calls", 0),
            "wall_s": (execute_run or {}).get("wall_s"),
            "llm_calls": (execute_run or {}).get("llm_calls"),
            "code_plan": (execute_run or {}).get("code_plan"),
            "output_variable_preview": (execute_run or {}).get(
                "output_variable_preview"
            ),
        },
    }
    failure_code = (execute_run or {}).get("failure_code")
    if failure_code is not None:
        digest["execute"]["failure_code"] = failure_code
    failure_one_liner = (execute_run or {}).get("failure_one_liner")
    if failure_one_liner:
        digest["execute"]["failure_one_liner"] = failure_one_liner
    return digest


def _subagent_summary_rows(subagent_outputs: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for phase_name, items in subagent_outputs.items():
        valid_items = [item for item in items or [] if isinstance(item, dict)]
        if not valid_items:
            continue
        latest = valid_items[-1]
        totals = aggregate_subagent_metrics({phase_name: valid_items})
        cache_hits = 0
        for item in valid_items:
            payload = (
                item.get("payload") if isinstance(item.get("payload"), dict) else {}
            )
            metrics = (
                item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            )
            if payload.get("cache_source") or metrics.get("replayed"):
                cache_hits += 1
        rows.append(
            {
                "phase": phase_name,
                "agent": latest.get("subagent_name"),
                "attempts": len(valid_items),
                "status": latest.get("status"),
                "failure_code": latest.get("failure_code"),
                "wall_s": _wall_s_from_ms(totals.get("wall_ms")),
                "llm_calls": totals.get("llm_calls", 0),
                "llm_call_attempts": totals.get("llm_call_attempts", 0),
                "prompt_tokens": totals.get("prompt_tokens", 0),
                "completion_tokens": totals.get("completion_tokens", 0),
                "thoughts_tokens": totals.get("thoughts_tokens", 0),
                "total_tokens": totals.get("total_tokens", 0),
                "cache_hits": cache_hits,
            }
        )
    return rows


def _add(
    records: list[dict],
    *,
    phase: str,
    subagent: str,
    kind: str,
    summary: str,
    data: dict | None = None,
    t_s: float | None = None,
) -> None:
    record = {
        "phase": phase,
        "subagent": subagent,
        "kind": kind,
        "summary": summary,
    }
    if t_s is not None:
        record["t_s"] = round(t_s, 3)
    if data:
        record["data"] = data
    records.append(record)


def build_workflow_records(
    *,
    ledger: list[dict],
    subagent_outputs: dict[str, list[dict]],
    executor_trace: list[dict],
    sandbox_trace: list[dict],
) -> list[dict]:
    records: list[dict] = []

    for entry in ledger:
        phase_decision = entry.get("phase_decision")
        if not phase_decision:
            continue
        phase = str(
            entry.get("state_after", {}).get("phase")
            or entry.get("state_before", {}).get("phase")
            or ""
        )
        _add(
            records,
            phase=phase or "CONTROLLER",
            subagent="controller",
            kind="transition",
            summary=str(phase_decision),
            data={
                "failure_code": entry.get("failure_code"),
                "budget_snapshot": (entry.get("state_after") or {}).get(
                    "budget_snapshot"
                ),
            },
        )

    plan_items = subagent_outputs.get("PLAN", [])
    if plan_items:
        latest = plan_items[-1]
        payload = latest.get("payload") or {}
        subagent_name = str(latest.get("subagent_name") or "planner")
        milestones = _planner_milestones(payload, {})
        metrics = latest.get("metrics") or payload.get("metrics") or {}
        for ms in milestones:
            _add(
                records,
                phase="PLAN",
                subagent=subagent_name,
                kind="milestone",
                summary=f"[{ms.get('id', '?')}] {str(ms.get('intent') or '')[:80]}",
                data={
                    "id": ms.get("id"),
                    "intent": ms.get("intent"),
                    "app": ms.get("app"),
                },
            )
        plan_warnings = (
            latest.get("warnings") if isinstance(latest.get("warnings"), list) else []
        )
        _add(
            records,
            phase="PLAN",
            subagent=subagent_name,
            kind="result",
            summary=f"{len(milestones)} milestone(s) planned"
            + (
                f" — {', '.join(str(m.get('id') or '?') for m in milestones)}"
                if milestones
                else ""
            ),
            data={
                "milestone_count": len(milestones),
                "source": "cache" if payload.get("cache_source") else "llm",
                "cache_source": payload.get("cache_source"),
                "metrics": metrics,
                "failure_code": latest.get("failure_code"),
                "llm_raised": payload.get("llm_raised"),
                "warnings": plan_warnings,
            },
        )

    find_items = subagent_outputs.get("FIND", [])
    for latest in find_items:
        payload = latest.get("payload") or {}
        metrics = latest.get("metrics") or {}
        metadata = _find_metadata(latest)
        find_context = {
            "attempt": latest.get("attempt"),
            "milestone_index": metadata.get("milestone_index"),
            "milestone_id": metadata.get("milestone_id"),
            "milestone_intent": metadata.get("milestone_intent"),
        }
        for trace in payload.get("routing_trace", []):
            if not isinstance(trace, dict):
                continue
            trace_data = dict(trace)
            trace_data.update(
                {key: value for key, value in find_context.items() if value is not None}
            )
            step = str(trace.get("step", "find_step"))
            if step == "planned_app_filter":
                _add(
                    records,
                    phase="FIND",
                    subagent="finder",
                    kind="planned_app_filter",
                    summary="planned apps: "
                    + ", ".join(trace.get("matched_apps") or []),
                    data=trace_data,
                )
            elif step == "community_select":
                _add(
                    records,
                    phase="FIND",
                    subagent="finder",
                    kind="community_select",
                    summary="selected communities: "
                    + ", ".join(trace.get("selected_communities") or []),
                    data=trace_data,
                )
            elif step == "seed_candidates":
                _add(
                    records,
                    phase="FIND",
                    subagent="finder",
                    kind="seed_candidates",
                    summary=(
                        f"{trace.get('candidate_count')} candidate apis "
                        f"from {trace.get('community_count')} communities"
                    ),
                    data=trace_data,
                )
            elif step == "seed_filter":
                selected = trace.get("selected_api_keys") or []
                _add(
                    records,
                    phase="FIND",
                    subagent="finder",
                    kind="seed_filter",
                    summary=f"kept {len(selected)} apis",
                    data=trace_data,
                )
            elif step == "dependency_round":
                exact_selected = trace.get("exact_selected") or []
                fallback_selected = trace.get("fallback_selected") or []
                _add(
                    records,
                    phase="FIND",
                    subagent="finder",
                    kind="dependency_round",
                    summary=(
                        f"round {trace.get('round')}: exact={len(exact_selected)} "
                        f"fallback={len(fallback_selected)}"
                    ),
                    data=trace_data,
                )
        cache_source = (
            payload.get("cache_source")
            if isinstance(payload.get("cache_source"), str)
            else None
        )
        replayed = bool(cache_source) or bool(metrics.get("replayed"))
        warnings = (
            latest.get("warnings") if isinstance(latest.get("warnings"), list) else []
        )
        find_summary = (
            f"finder returned {payload.get('candidate_count', 0)} candidate apis"
        )
        if replayed:
            find_summary += " (cache replay)"
        _add(
            records,
            phase="FIND",
            subagent=str(latest.get("subagent_name") or "finder"),
            kind="result",
            summary=find_summary,
            data={
                "matched_apps": payload.get("matched_apps"),
                "selected_communities": payload.get("selected_communities"),
                "final_candidate_apis": candidate_api_keys(
                    payload.get("candidate_apis")
                ),
                "metrics": metrics,
                "source": "cache" if replayed else "llm",
                "cache_source": cache_source,
                "warnings": warnings,
                **{
                    key: value
                    for key, value in find_context.items()
                    if value is not None
                },
            },
        )

    sandbox_iter = iter(sandbox_trace)
    sandbox_next = next(sandbox_iter, None)
    execute_python_count = 0
    for record in executor_trace:
        t_ms = record.get("t_ms")
        t_s = (float(t_ms) / 1000.0) if isinstance(t_ms, (int, float)) else None
        for part in record.get("parts", []):
            kind = part.get("kind")
            if kind == "call" and part.get("name") == "execute_python":
                execute_python_count += 1
                code = (part.get("args") or {}).get("code") or ""
                _add(
                    records,
                    phase="EXECUTE",
                    subagent="executor",
                    kind="tool_call",
                    summary=f"execute_python #{execute_python_count}",
                    t_s=t_s,
                    data={
                        "code_char_count": part.get("code_char_count", len(code)),
                        "usage": record.get("usage"),
                    },
                )
                if isinstance(sandbox_next, dict):
                    for api_call in sandbox_next.get("api_calls", []):
                        if not isinstance(api_call, dict):
                            continue
                        api_kind = str(api_call.get("kind") or "sandbox_call")
                        app = api_call.get("app")
                        api_name = api_call.get("api_name")
                        status = api_call.get("status")
                        summary = (
                            f"{app}.{api_name} -> {status}"
                            if app and api_name
                            else api_kind
                        )
                        _add(
                            records,
                            phase="EXECUTE",
                            subagent="sandbox",
                            kind=api_kind,
                            summary=summary,
                            t_s=t_s,
                            data=_compact_api_call(api_call),
                        )
                    sandbox_next = next(sandbox_iter, None)
            elif kind == "resp" and part.get("name") == "execute_python":
                response = part.get("response")
                response_len = part.get("response_char_count")
                if response_len is None:
                    response_len = len(str(response or ""))
                _add(
                    records,
                    phase="EXECUTE",
                    subagent="executor",
                    kind="tool_response",
                    summary=f"execute_python returned {response_len} chars",
                    t_s=t_s,
                    data={"usage": record.get("usage")},
                )
            elif kind == "text":
                text = part.get("text")
                text_len = part.get("text_char_count")
                if text_len is None:
                    text_len = len(str(text or ""))
                _add(
                    records,
                    phase="EXECUTE",
                    subagent="executor",
                    kind="model_text",
                    summary=f"model text {text_len} chars",
                    t_s=t_s,
                )

    execute_items = subagent_outputs.get("EXECUTE", [])
    for exec_item in execute_items:
        payload = exec_item.get("payload") or {}
        ms_id = payload.get("milestone_id")
        ms_idx = payload.get("milestone_index", 0)
        ms_total = payload.get("milestone_total", 1)
        milestone_label = f"[{ms_id}] ({ms_idx + 1}/{ms_total}) " if ms_id else ""
        status_label = exec_item.get("status", "?")
        failure = exec_item.get("failure_code")
        variables_count = _executor_variables_count(payload)
        _add(
            records,
            phase="EXECUTE",
            subagent=str(exec_item.get("subagent_name") or "executor"),
            kind="result",
            summary=(
                f"{milestone_label}{status_label}"
                + (f" [{failure}]" if failure else "")
            ),
            data={
                "milestone_id": ms_id,
                "milestone_index": ms_idx,
                "milestone_total": ms_total,
                "tool_call_count": payload.get("tool_call_count"),
                "final_response_chars": len(str(payload.get("final_response") or "")),
                "submission_candidate": _submission_shape(
                    payload.get("submission_candidate")
                ),
                "finalize_called": payload.get("finalize_called"),
                "finalize_nudge_used": payload.get("finalize_nudge_used"),
                "no_progress_nudge_used": payload.get("no_progress_nudge_used"),
                "docs_only_nudge_used": payload.get("docs_only_nudge_used"),
                "auto_finalize_used": payload.get("auto_finalize_used"),
                "milestone_done": payload.get("milestone_done"),
                # Grounded self-assess verdict {ok, summary, problem, backstop}
                # so the analyst/viewer sees WHY a milestone was flipped NOT-DONE
                # (or suppressed by the subset-underfetch precision backstop).
                "self_assess": (payload.get("code_execute") or {}).get("self_assess"),
                "variables_count": variables_count,
                "metrics": payload.get("metrics") or exec_item.get("metrics") or {},
                "failure_code": failure,
                "llm_raised": payload.get("llm_raised"),
            },
        )

    submit_items = subagent_outputs.get("SUBMIT", [])
    if submit_items:
        latest = submit_items[-1]
        payload = latest.get("payload") or {}
        extra = payload.get("extra") or {}
        _add(
            records,
            phase="SUBMIT",
            subagent=str(latest.get("subagent_name") or "completion_gate"),
            kind="result",
            summary=(
                f"{payload.get('status')} "
                f"({extra.get('passed')}/{extra.get('total')} passed)"
            ),
            data={
                "block_reason": payload.get("block_reason"),
                "evaluation": extra,
                "failure_code": latest.get("failure_code"),
            },
        )

    for idx, record in enumerate(records, start=1):
        record["seq"] = idx
    return records


def derive_failure_point(*, result: dict) -> str | None:
    if (
        result.get("status") == "TIMED_OUT"
        or result.get("block_reason") == "TASK_TIMEOUT"
    ):
        return f"{result.get('phase') or 'FAILED'}::TASK_TIMEOUT"
    if result.get("submit_block"):
        return f"SUBMIT::{result['submit_block']}"
    if result.get("exec_failure"):
        return f"EXECUTE::{result['exec_failure']}"
    if result.get("block_reason"):
        return f"{result.get('phase') or 'FAILED'}::{result['block_reason']}"
    return None


def build_task_summary(
    *,
    task_id: str,
    result: dict,
    final_state: dict,
    subagent_outputs: dict[str, list[dict]],
    sandbox_trace: list[dict],
    workflow_path: str | None,
    evaluation_report_path: str | None,
    command: str | None = None,
) -> dict:
    execute_items = [
        item
        for item in (subagent_outputs.get("EXECUTE") or [])
        if isinstance(item, dict)
    ]
    submit_payload = (subagent_outputs.get("SUBMIT") or [{}])[-1].get("payload") or {}
    find_items = [
        item for item in (subagent_outputs.get("FIND") or []) if isinstance(item, dict)
    ]
    plan_payload = (subagent_outputs.get("PLAN") or [{}])[-1].get("payload") or {}
    plan_milestones = _planner_milestones(plan_payload, final_state)
    find_runs = [_find_run_summary(item) for item in find_items]
    find_runs_by_milestone: dict[int, list[dict]] = {}
    for run in find_runs:
        milestone_index = run.get("milestone_index")
        if isinstance(milestone_index, int):
            find_runs_by_milestone.setdefault(milestone_index, []).append(run)
    sandbox_groups = _slice_sandbox_trace_by_execute(execute_items, sandbox_trace)
    execute_runs: list[dict] = []
    execute_runs_by_milestone: dict[int, list[dict]] = {}
    for index, exec_item in enumerate(execute_items):
        payload = (
            exec_item.get("payload")
            if isinstance(exec_item.get("payload"), dict)
            else {}
        )
        milestone_index = payload.get("milestone_index")
        if not isinstance(milestone_index, int):
            continue
        latest_find = _latest_run_by_milestone(find_runs_by_milestone, milestone_index)
        candidate_keys = latest_find.get("selected_apis", []) if latest_find else []
        execute_run = _execute_run_summary(
            exec_item,
            sandbox_blocks=sandbox_groups[index] if index < len(sandbox_groups) else [],
            candidate_keys=candidate_keys,
        )
        execute_runs.append(execute_run)
        execute_runs_by_milestone.setdefault(milestone_index, []).append(execute_run)

    milestone_digests = [
        _milestone_digest(
            milestone,
            find_run=_latest_run_by_milestone(
                find_runs_by_milestone, int(milestone.get("index") or 0)
            ),
            execute_run=_latest_run_by_milestone(
                execute_runs_by_milestone, int(milestone.get("index") or 0)
            ),
        )
        for milestone in plan_milestones
    ]
    if result.get("status") == "TIMED_OUT":
        active_index = final_state.get("active_milestone_index")
        if isinstance(active_index, int) and 0 <= active_index < len(milestone_digests):
            phase = str(result.get("phase") or "")
            active = milestone_digests[active_index]
            if phase == "FIND" and active["find"]["status"] == "MISSING":
                active["find"]["status"] = "TIMED_OUT"
            if phase == "EXECUTE" and active["execute"]["status"] == "MISSING":
                active["execute"]["status"] = "TIMED_OUT"
    find_retry_count = sum(int(run.get("retry_count") or 0) for run in find_runs)
    find_backoff_ms = sum(
        int(round((run.get("backoff_s") or 0) * 1000)) for run in find_runs
    )
    rate_limit_count = sum(int(run.get("rate_limit_count") or 0) for run in find_runs)
    provider_issue_likely = any(
        bool(run.get("provider_issue_likely")) for run in find_runs
    )
    find_cache_hits = sum(1 for run in find_runs if run.get("replayed"))
    aggregate_metrics = aggregate_subagent_metrics(subagent_outputs)

    return {
        "task_id": task_id,
        "instruction": ((final_state.get("task_context") or {}).get("instruction")),
        "datetime": ((final_state.get("task_context") or {}).get("task_datetime")),
        "status": result.get("status"),
        "phase": result.get("phase"),
        "wall_s": _wall_s_from_ms(result.get("task_wall_ms")),
        "eval": _eval_summary(result),
        "aggregate_metrics": aggregate_metrics,
        "overview": {
            "plan_source": "cache" if plan_payload.get("cache_source") else "llm",
            "cache_source": plan_payload.get("cache_source"),
            "milestones_total": len(plan_milestones),
            "find_completed": len(find_runs),
            "find_cache_hits": find_cache_hits,
            "execute_completed": len(execute_runs),
            "active_milestone": _active_milestone_label(
                final_state, len(plan_milestones)
            ),
            "find_retry_count": find_retry_count,
            "find_backoff_s": _wall_s_from_ms(find_backoff_ms) or 0.0,
            "rate_limit_count": rate_limit_count,
            "provider_issue_likely": provider_issue_likely,
            "submit_status": submit_payload.get("status"),
            "submit_forced_fallback": bool(submit_payload.get("forced_fallback")),
            "submit_original_failure_code": submit_payload.get("original_failure_code"),
            "block_reason": result.get("block_reason"),
            "failure_point": derive_failure_point(result=result),
            "failure_one_liner": _aggregate_failure_one_liner(
                execute_runs, submit_payload
            ),
            "policy_source": result.get("policy_source"),
        },
        "subagents": _subagent_summary_rows(subagent_outputs),
        "milestones": milestone_digests,
        "submission": _submission_summary(submit_payload),
        "run": {
            "command": command,
            "workflow_path": workflow_path,
            "evaluation_report_path": evaluation_report_path,
        },
    }


__all__ = [
    "build_task_summary",
    "build_workflow_records",
    "candidate_api_keys",
    "read_jsonl",
    "write_workflow",
]
