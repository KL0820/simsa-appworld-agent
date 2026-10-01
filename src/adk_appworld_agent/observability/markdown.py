from __future__ import annotations

import json
from typing import Any

from adk_appworld_agent.observability.metrics import compact_number
from adk_appworld_agent.observability.subagent_logs import render_subagent_io_markdown


def json_block(value: object) -> list[str]:
    return ["```json", json.dumps(value, indent=2, ensure_ascii=False), "```"]


def format_wall_s(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.1f}s"
    return "NA"


def format_eval(summary: dict[str, Any]) -> str:
    eval_summary = summary.get("eval") if isinstance(summary.get("eval"), dict) else {}
    passed = eval_summary.get("passed")
    total = eval_summary.get("total")
    if passed is None or total is None:
        return "NA"
    return f"{passed}/{total} passed"


def _format_ratio(passed: object, total: object) -> str:
    if passed is None or total is None:
        return "NA"
    return f"{passed}/{total}"


def _status_mark(status: object) -> str:
    return "OK" if str(status) in {"SUCCEEDED", "COMPLETED", "COMPLETE"} else "FAIL"


def _safe_table_text(value: object, *, limit: int | None = None) -> str:
    text = str(value or "").replace("|", "\\|").strip()
    if limit is not None and len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _list_lines(title: str, value: object) -> list[str]:
    """Plain-text section: 'Title:' then indented bullet list."""
    lines = [f"  {title}:"]
    if not isinstance(value, list) or not value:
        return [*lines, "    None"]
    items = [str(item) for item in value if item not in (None, "")]
    if not items:
        return [*lines, "    None"]
    return [*lines, *(f"    {item}" for item in items)]


def _section_header(title: str, level: int = 1) -> list[str]:
    """Plain-text section header. level=0 is top (===), level=1 is sub (---)."""
    if level <= 0:
        bar = "=" * 64
        return [bar, title, bar]
    return [title, "-" * len(title)]


def _padded_table(rows: list[list[str]], indent: str = "  ") -> list[str]:
    """Render rows as space-padded columns. First row treated as header.

    Rows must all have the same column count. Header row is followed by a
    separator made of dashes.
    """
    if not rows:
        return []
    n_cols = len(rows[0])
    widths = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            if i < n_cols:
                widths[i] = max(widths[i], len(str(cell)))
    out = []
    for r_idx, row in enumerate(rows):
        cells = [str(row[i]).ljust(widths[i]) for i in range(n_cols)]
        out.append(indent + "  ".join(cells).rstrip())
        if r_idx == 0:
            sep = ["-" * widths[i] for i in range(n_cols)]
            out.append(indent + "  ".join(sep))
    return out


def _api_call_trace_lines(value: object) -> list[str]:
    """Plain-text per-call API trace — one line per call."""
    lines = ["  API call trace:"]
    if not isinstance(value, list) or not value:
        return [*lines, "    None"]

    rendered: list[str] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        api_key = entry.get("api_key") or "?"
        kwargs = entry.get("kwargs") or {}
        kw_text = ", ".join(f"{k}={_format_kwarg_value(v)}" for k, v in kwargs.items())
        status = str(entry.get("status") or "ok")
        shape = entry.get("result_shape") or {}
        shape_text = _format_result_shape(shape)
        error = entry.get("error")
        line = f"    {api_key}({kw_text}) -> {shape_text}"
        if status and status != "ok":
            line += f" [{status}]"
        if error:
            err_text = str(error).splitlines()[0][:120]
            line += f" -- {err_text}"
        rendered.append(line)

    # Collapse consecutive identical lines (same api+kwargs+shape) into "(xN)".
    collapsed: list[str] = []
    for line in rendered:
        if collapsed and collapsed[-1].rstrip(" (xN0123456789)") == line:
            base = collapsed[-1]
            base_no_count = base
            for marker in (
                " (x2)",
                " (x3)",
                " (x4)",
                " (x5)",
                " (x6)",
                " (x7)",
                " (x8)",
                " (x9)",
            ):
                if base.endswith(marker):
                    base_no_count = base[: -len(marker)]
                    count = int(marker[3:-1]) + 1
                    collapsed[-1] = f"{base_no_count} (x{count})"
                    break
            else:
                collapsed[-1] = f"{base} (x2)"
        else:
            collapsed.append(line)

    if not collapsed:
        return [*lines, "    None"]
    return [*lines, *collapsed]


def _format_kwarg_value(value: object) -> str:
    """Format a kwarg value compactly for the API call trace.

    Note: upstream `_sanitize_kwargs` runs `repr()` on every value, so
    incoming strings are already Python-repr formatted (e.g. `"'partner'"`,
    `"42"`, `"[1, 2]"`). Special placeholders like `"<redacted>"` are not
    repr'd. We pass repr'd content through as-is to avoid double-quoting,
    and quote raw plain strings only when they aren't already enclosed.
    """
    if value is None:
        return "None"
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, str):
        # Already-repr'd or already-bracketed forms: emit as-is.
        already_quoted = (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        )
        already_structured = value.startswith("[") or value.startswith("{")
        if already_quoted or already_structured:
            if len(value) > 40:
                return value[:37].rstrip() + "..."
            return value
        # Plain string (e.g. "<redacted>") — wrap in double quotes.
        if len(value) > 40:
            return f'"{value[:37]}..."'
        return f'"{value}"'
    if isinstance(value, list):
        if len(value) > 3:
            return f"list[{len(value)}]"
        return "[" + ", ".join(_format_kwarg_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{...}" if value else "{}"
    return str(value)


def _format_result_shape(shape: object) -> str:
    if not isinstance(shape, dict):
        return "?"
    typ = shape.get("type") or "?"
    if typ == "list":
        n = shape.get("count_hint")
        return f"list[{n}]" if n is not None else "list"
    if typ == "object":
        keys = shape.get("keys")
        if isinstance(keys, list) and keys:
            preview = ",".join(str(k) for k in keys[:4])
            more = "..." if len(keys) > 4 else ""
            return f"{{{preview}{more}}}"
        return "{...}"
    if typ == "none":
        return "None"
    return str(typ)


def _code_plan_lines(value: object) -> list[str]:
    lines = ["  Code Plan:"]
    if not isinstance(value, dict):
        return [*lines, "    None"]
    output_variable = (
        value.get("output_variable")
        if isinstance(value.get("output_variable"), dict)
        else {}
    )
    output_name = str(output_variable.get("name") or "").strip()
    output_description = str(output_variable.get("description") or "").strip()
    if output_name or output_description:
        variable_text = output_name or "unnamed"
        if output_description:
            variable_text = f"{variable_text} - {output_description}"
        lines.append(f"    Output variable: {variable_text}")
    steps = value.get("steps") if isinstance(value.get("steps"), list) else []
    clean_steps = [str(step).strip() for step in steps if str(step).strip()]
    if not clean_steps:
        return [*lines, "    None"]
    lines.append("")
    lines.extend(f"    {idx}. {step}" for idx, step in enumerate(clean_steps, start=1))
    return lines


def _subagent_rows(summary: dict[str, Any]) -> list[str]:
    """Plain-text subagent table — padded columns, no pipe characters."""
    subagents = (
        summary.get("subagents") if isinstance(summary.get("subagents"), list) else []
    )
    table: list[list[str]] = [
        ["Phase", "Agent", "Attempts", "LLM calls", "Tokens", "Wall", "Result"]
    ]
    for item in subagents:
        if not isinstance(item, dict):
            continue
        table.append(
            [
                str(item.get("phase") or ""),
                str(item.get("agent") or ""),
                str(item.get("attempts") or 0),
                str(compact_number(item.get("llm_calls") or 0)),
                str(compact_number(item.get("total_tokens") or 0)),
                format_wall_s(item.get("wall_s")),
                _status_mark(item.get("status")),
            ]
        )
    return _padded_table(table)


def _milestone_rows(summary: dict[str, Any]) -> list[str]:
    """Plain-text milestone table."""
    milestones = (
        summary.get("milestones") if isinstance(summary.get("milestones"), list) else []
    )
    table: list[list[str]] = [["#", "App", "Intent", "Status", "Find", "Exec"]]
    for item in milestones:
        if not isinstance(item, dict):
            continue
        find = item.get("find") if isinstance(item.get("find"), dict) else {}
        execute = item.get("execute") if isinstance(item.get("execute"), dict) else {}
        index = item.get("index")
        display_index = str(int(index) + 1) if isinstance(index, int) else ""
        table.append(
            [
                display_index,
                _safe_table_text(item.get("app")),
                _safe_table_text(item.get("goal"), limit=90),
                str(execute.get("status") or find.get("status") or ""),
                f"{format_wall_s(find.get('wall_s'))} / {compact_number(find.get('llm_calls') or 0)} calls",
                f"{format_wall_s(execute.get('wall_s'))} / {compact_number(execute.get('llm_calls') or 0)} calls",
            ]
        )
    return _padded_table(table)


def _milestone_detail_lines(summary: dict[str, Any]) -> list[str]:
    """Plain-text per-milestone detail blocks."""
    lines: list[str] = []
    milestones = (
        summary.get("milestones") if isinstance(summary.get("milestones"), list) else []
    )
    for item in milestones:
        if not isinstance(item, dict):
            continue
        find = item.get("find") if isinstance(item.get("find"), dict) else {}
        execute = item.get("execute") if isinstance(item.get("execute"), dict) else {}
        index = item.get("index")
        display_index = int(index) + 1 if isinstance(index, int) else "?"
        app = item.get("app") or "unknown"
        intent = str(item.get("goal") or "").strip()
        header = f"[Milestone {display_index}] {app} - {intent}"
        lines.extend(
            [
                header,
                "-" * min(len(header), 80),
                "",
                *_list_lines("Find APIs", find.get("selected_apis")),
                "",
                *_list_lines("Called APIs", execute.get("called_apis")),
                "",
                *_list_lines("Failed APIs", execute.get("failed_api_calls")),
                "",
                *_list_lines("Unexpected APIs", execute.get("unexpected_api_calls")),
                "",
                *_api_call_trace_lines(execute.get("api_call_trace")),
                "",
                *_code_plan_lines(execute.get("code_plan")),
                "",
                *_committed_variable_lines(
                    execute.get("code_plan"), execute.get("output_variable_preview")
                ),
                "",
            ]
        )
    return lines


def _committed_variable_lines(code_plan: object, value_preview: object) -> list[str]:
    """Plain-text 'Committed variable' section."""
    lines = ["  Committed variable:"]
    name = ""
    description = ""
    if isinstance(code_plan, dict):
        ov = code_plan.get("output_variable")
        if isinstance(ov, dict):
            name = str(ov.get("name") or "").strip()
            description = str(ov.get("description") or "").strip()
    if not name and not value_preview:
        return [*lines, "    None"]
    if name:
        head = f"    {name}"
        if description:
            head += f" - {description}"
        lines.append(head)
    if value_preview:
        preview_text = str(value_preview)
        if len(preview_text) > 480:
            preview_text = preview_text[:480].rstrip() + " ..."
        lines.append("    value preview:")
        for ln in preview_text.split("\n"):
            lines.append(f"      {ln}")
    return lines


def _submission_lines(summary: dict[str, Any]) -> list[str]:
    submission = (
        summary.get("submission")
        if isinstance(summary.get("submission"), dict)
        else None
    )
    if not submission:
        return ["  None"]
    answer = submission.get("submitted_answer")
    passed = submission.get("passed")
    failed = submission.get("failed")
    total = submission.get("total")
    report = submission.get("report_excerpt")
    sub_results = (
        submission.get("sub_test_results")
        if isinstance(submission.get("sub_test_results"), list)
        else []
    )
    if answer is None and passed is None and report is None:
        return ["  None"]
    lines: list[str] = []
    if answer is not None:
        ans_text = str(answer)
        if len(ans_text) > 240:
            ans_text = ans_text[:240].rstrip() + " ..."
        lines.append(f"  Submitted answer: {ans_text}")
    if passed is not None or total is not None:
        eval_line = (
            f"  Evaluator:        {passed if passed is not None else '?'}/"
            f"{total if total is not None else '?'} passed"
            + (f" ({failed} failed)" if failed else "")
        )
        lines.append(eval_line)
    if sub_results:
        lines.append("")
        lines.append("  Sub-test results:")
        for entry in sub_results:
            mark = "[PASS]" if entry.get("status") == "pass" else "[FAIL]"
            req = str(entry.get("requirement") or "").strip()
            error = entry.get("error")
            line = f"    {mark} {req}"
            if error:
                line += f" -- {error}"
            lines.append(line)
    if report:
        lines.append("")
        lines.append("  Full evaluator report:")
        for ln in report.split("\n"):
            lines.append(f"    {ln}")
    return lines


def _failure_lines(summary: dict[str, Any]) -> list[str]:
    overview = (
        summary.get("overview") if isinstance(summary.get("overview"), dict) else {}
    )
    failures: list[str] = []

    root = overview.get("submit_original_failure_code") or overview.get("block_reason")
    failure_point = overview.get("failure_point")
    forced_fallback = overview.get("submit_forced_fallback")
    one_liner = overview.get("failure_one_liner")

    if root:
        failures.append(f"  Root: {root}")
        if one_liner:
            failures.append(f"        {one_liner}")
    if failure_point and failure_point != f"FAILED::{root}":
        cascade = f"  Cascade: {failure_point}"
        if forced_fallback:
            cascade += " (FORCED null fallback)"
        failures.append(cascade)
    elif forced_fallback and root:
        failures.append("  Cascade: FORCED null fallback to SUBMIT")

    if overview.get("provider_issue_likely"):
        retries = overview.get("find_retry_count") or 0
        failures.append(f"  FIND provider retries observed: {retries}")
    policy_source = overview.get("policy_source")
    if policy_source:
        failures.append(f"  Policy: {policy_source}")
    if not failures:
        failures.append("  None")
    return failures


def _markers_line(summary: dict[str, Any]) -> str | None:
    overview = (
        summary.get("overview") if isinstance(summary.get("overview"), dict) else {}
    )
    markers: list[str] = []

    if overview.get("submit_forced_fallback"):
        markers.append("forced_fallback")

    cache_source = overview.get("cache_source")
    plan_source = overview.get("plan_source")
    find_cache_hits = overview.get("find_cache_hits") or 0
    cache_bits: list[str] = []
    if plan_source == "cache" or cache_source:
        cache_bits.append("plan")
    if isinstance(find_cache_hits, int) and find_cache_hits > 0:
        cache_bits.append(f"find×{find_cache_hits}")
    if cache_bits:
        markers.append("cache=" + "+".join(cache_bits))

    if overview.get("provider_issue_likely"):
        markers.append("provider_retries")

    milestones = (
        summary.get("milestones") if isinstance(summary.get("milestones"), list) else []
    )
    failed_milestone_count = sum(
        1
        for milestone in milestones
        if isinstance(milestone, dict)
        and isinstance(milestone.get("execute"), dict)
        and milestone["execute"].get("status") == "FAILED"
    )
    if failed_milestone_count:
        markers.append(f"milestone_fail×{failed_milestone_count}")

    if not markers:
        return None
    return "Markers: " + " | ".join(markers)


def render_task_summary_markdown(summary: dict[str, Any]) -> str:
    """Render the task summary as plain text (no markdown syntax).

    Despite the legacy `_markdown` function name, output is intended for
    direct viewing in plain-text editors. Sections use UPPERCASE headers
    underlined with `===` (top) or `---` (sub); fields are aligned with
    spaces; tables use padded columns instead of pipes; no `**bold**`,
    `## headings`, or fenced code blocks.
    """
    overview = (
        summary.get("overview") if isinstance(summary.get("overview"), dict) else {}
    )
    aggregate = (
        summary.get("aggregate_metrics")
        if isinstance(summary.get("aggregate_metrics"), dict)
        else {}
    )
    block = overview.get("block_reason") or summary.get("phase") or "NA"
    forced_fallback = overview.get("submit_forced_fallback")
    instruction = str(summary.get("instruction") or "").strip()
    one_liner = overview.get("failure_one_liner")
    markers_line = _markers_line(summary)

    title_bar = "=" * 64
    task_id = str(summary.get("task_id") or "")
    title = f"Task {task_id}"

    header_lines = [
        f"  Status        {summary.get('status')}",
        f"  Eval          {format_eval(summary)}",
        f"  Block         {block}",
        f"  Wall          {format_wall_s(summary.get('wall_s'))}",
        f"  LLM calls     {compact_number(aggregate.get('llm_calls') or 0)}",
        f"  Tokens        {compact_number(aggregate.get('total_tokens') or 0)}",
    ]
    if forced_fallback:
        header_lines.append("  Submission    FORCED null fallback")
    if markers_line:
        header_lines.extend(["", f"  {markers_line}"])
    if one_liner:
        header_lines.extend(["", f"  Failure: {one_liner}"])

    lines = [
        title_bar,
        title,
        title_bar,
        "",
        *header_lines,
        "",
        "",
        "TASK INSTRUCTION",
        "----------------",
        "",
        instruction or "NA",
        "",
        "",
        "SUBAGENTS",
        "---------",
        "",
        *_subagent_rows(summary),
        "",
        "",
        "MILESTONES",
        "----------",
        "",
        *_milestone_rows(summary),
        "",
        "",
        "MILESTONE DETAILS",
        "-----------------",
        "",
        *_milestone_detail_lines(summary),
        "",
        "SUBMISSION",
        "----------",
        "",
        *_submission_lines(summary),
        "",
        "",
        "FAILURES",
        "--------",
        "",
        *_failure_lines(summary),
    ]
    command = (
        (summary.get("run") or {}).get("command")
        if isinstance(summary.get("run"), dict)
        else None
    )
    if command:
        lines.extend(["", "", "COMMAND", "-------", "", f"  {command}"])
    return "\n".join(lines).rstrip() + "\n"


def _percentile_50(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return (float(sorted_values[mid - 1]) + float(sorted_values[mid])) / 2.0


def _format_int(value: float) -> str:
    return compact_number(int(round(value)))


def _format_float(value: float) -> str:
    return f"{value:.1f}"


def _scenario_goal_completion(completed_tasks: list[dict]) -> tuple[int, int]:
    """Compute SGC only when every observed scenario has all three variants.

    A one-variant ablation or partial run cannot establish scenario completion.
    Return (0, 0) for unavailable SGC instead of reporting a misleading 100%.
    """
    by_scenario: dict[str, dict[str, bool]] = {}
    for task in completed_tasks:
        task_id = task.get("task_id") or ""
        stem, separator, variant = task_id.rpartition("_")
        if not separator or variant not in {"1", "2", "3"}:
            return 0, 0
        by_scenario.setdefault(stem, {})[variant] = bool(
            (task.get("result") or {}).get("passed_all")
        )
    if any(set(variants) != {"1", "2", "3"} for variants in by_scenario.values()):
        return 0, 0
    total = len(by_scenario)
    all_pass = sum(all(variants.values()) for variants in by_scenario.values())
    return all_pass, total


def _aggregate_table_lines(completed_tasks: list[dict], total_tasks: int) -> list[str]:
    completed_results = [t.get("result") or {} for t in completed_tasks]
    done = len(completed_results)
    passed = sum(1 for r in completed_results if r.get("passed_all"))
    pass_pct = f"{(passed / total_tasks * 100):.1f}%" if total_tasks else "0.0%"
    sgc_passed, sgc_total = _scenario_goal_completion(completed_tasks)
    sgc_pct = f"{(sgc_passed / sgc_total * 100):.1f}%" if sgc_total else "0.0%"
    test_passed = sum(int(r.get("passed") or 0) for r in completed_results)
    test_total = sum(int(r.get("total") or 0) for r in completed_results)
    test_pass_pct = f"{(test_passed / test_total * 100):.1f}%" if test_total else "0.0%"

    walls = [float(r.get("wall_s") or 0.0) for r in completed_results]
    llms = [float(r.get("llm_calls") or 0) for r in completed_results]
    toks = [float(r.get("total_tokens") or 0) for r in completed_results]

    table: list[list[str]] = [["Metric", "Total", "Avg", "P50"]]
    table.append(["Tasks done", f"{done} / {total_tasks}", "-", "-"])
    table.append(
        ["TGC (tasks passed)", f"{passed} / {total_tasks} ({pass_pct})", "-", "-"]
    )
    table.append(
        [
            "SGC (scenarios all-variant pass)",
            f"{sgc_passed} / {sgc_total} ({sgc_pct})"
            if sgc_total and done == total_tasks
            else "N/A (requires all three variants per scenario)",
            "-",
            "-",
        ]
    )
    table.append(
        [
            "Test pass",
            f"{_format_ratio(test_passed, test_total)} ({test_pass_pct})",
            "-",
            "-",
        ]
    )
    if completed_results:
        table.append(
            [
                "Wall (s)",
                _format_float(sum(walls)),
                _format_float(sum(walls) / done),
                _format_float(_percentile_50(walls)),
            ]
        )
        table.append(
            [
                "LLM calls",
                _format_int(sum(llms)),
                _format_float(sum(llms) / done),
                _format_int(_percentile_50(llms)),
            ]
        )
        table.append(
            [
                "Total tokens",
                _format_int(sum(toks)),
                _format_int(sum(toks) / done),
                _format_int(_percentile_50(toks)),
            ]
        )
    return _padded_table(table)


def _per_task_section(task: dict) -> list[str]:
    task_id = task.get("task_id", "")
    instruction = (task.get("instruction") or "").strip()
    state = task.get("state")  # "completed" / "pending"
    instruction_lines = (
        ["", "  Task instruction:", f"    {instruction}"] if instruction else []
    )
    if state == "pending":
        header = f"[{task_id}] PENDING"
        return [header, *instruction_lines] if instruction_lines else [header]
    result = task.get("result") or {}
    passed = result.get("passed")
    total = result.get("total")
    eval_str = (
        f"{passed}/{total}" if (passed is not None and total is not None) else "NA"
    )
    mark = "PASS" if result.get("passed_all") else "FAIL"
    wall_s = result.get("wall_s") or 0
    llm = int(result.get("llm_calls") or 0)
    tokens = int(result.get("total_tokens") or 0)
    header = (
        f"[{task_id}] {mark} {eval_str}  | {_format_float(wall_s)}s | "
        f"{llm} llm | {compact_number(tokens)} tok"
    )
    lines = [header]
    if instruction_lines:
        lines.extend(instruction_lines)
    block_reason = result.get("block_reason")
    failure_code = result.get("exec_failure")
    extras: list[str] = []
    if block_reason:
        extras.append(f"block: {block_reason}")
    if failure_code:
        extras.append(f"failure: {failure_code}")
    # infra 備註: flag tasks whose result may be perturbed by the gemini
    # streaming hang (finder 429s / provider issue). block: already carries
    # EXECUTOR_LLM_STALLED / TASK_TIMEOUT.
    rate_limit = int(result.get("rate_limit_count") or 0)
    if rate_limit or result.get("provider_issue_likely"):
        extras.append(f"infra⚠ finder429={rate_limit}")
    if extras:
        lines.append("  " + " | ".join(extras))
    return lines


def render_batch_results_markdown(state: dict[str, Any]) -> str:
    """Render the per-batch results markdown.

    `state` shape:
      {
        "run_name": str,
        "updated_at": str (ISO 8601),
        "config": {plan, find, execute, policy},
        "tasks": [
          {"task_id": str, "instruction": str, "state": "pending"} or
          {"task_id": str, "instruction": str, "state": "completed",
           "result": {passed: int, total: int, passed_all: bool, wall_s: float,
                      llm_calls: int, total_tokens: int,
                      block_reason: str|None, exec_failure: str|None}}
        ]
      }
    """
    run_name = state.get("run_name", "")
    updated_at = state.get("updated_at", "")
    cfg = state.get("config") if isinstance(state.get("config"), dict) else {}
    tasks = state.get("tasks") if isinstance(state.get("tasks"), list) else []
    completed_tasks = [t for t in tasks if t.get("state") == "completed"]

    title_bar = "=" * 64
    lines = [
        title_bar,
        f"Batch Results: {run_name}",
        title_bar,
        "",
    ]
    if updated_at:
        lines.append(f"  Updated: {updated_at}")
    for key in ("plan", "find", "execute", "verify", "policy"):
        value = cfg.get(key)
        if value:
            lines.append(f"  {key.capitalize()}:    {value}")
    cache_dir = cfg.get("cache_dir") or ""
    cache_read = cfg.get("cache_read") or ""
    if cache_dir:
        lines.append(f"  Cache:    {cache_dir} (read={cache_read or 'none'})")
    else:
        lines.append("  Cache:    none")
    lines.extend(
        [
            "",
            "",
            "AGGREGATE",
            "---------",
            "",
            *_aggregate_table_lines(completed_tasks, total_tasks=len(tasks)),
            "",
            "",
            "PER TASK",
            "--------",
            "",
        ]
    )
    for task in tasks:
        lines.extend(_per_task_section(task))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "format_eval",
    "format_wall_s",
    "json_block",
    "render_batch_results_markdown",
    "render_subagent_io_markdown",
    "render_task_summary_markdown",
]
