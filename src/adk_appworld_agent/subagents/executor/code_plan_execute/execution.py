"""Stage-1/Stage-2 LLM execution machinery for the code_plan_execute executor.

Run/stream/repair loops, stall detection, event/usage extraction, deterministic
skeletons, and the run-result dataclasses. Extracted from agent.py in the
2026-06-22 behavior-preserving split."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import threading
import uuid
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# Monotonic counter over Stage-2 stall cancellations within one process. Lets the
# diagnostic log line up "[stall-cancel] seq=K" against the next finder
# "[429-probe]"/"rate-limited" line to test whether a cancelled executor stream
# precedes a finder 429 (the shared-state hypothesis).
_STALL_CANCEL_COUNTER = itertools.count(1)
_AGEN_CLOSE_GRACE_S = 5.0

from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_appworld_agent.contracts.code_plan import CodePlanOutput, VariableSpec
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.content_utils import content_to_text
from adk_appworld_agent.orchestration.rate_limit_grace import grant_rate_limit_grace
from adk_appworld_agent.subagents.executor.appworld_tools import (
    EXECUTOR_CANDIDATE_APIS_ENV,
    EXECUTOR_GUARDRAILS_ENV,
    EXECUTOR_PRIOR_VARIABLES_ENV,
    EXECUTOR_TASK_DATETIME_ENV,
    clean_execute_python_code,
    repair_generated_source_artifacts,
)
from adk_appworld_agent.subagents.executor.code_plan_execute._primitives import (
    _MAX_CODE_EXECUTE_ATTEMPTS,
    _empty_usage,
    _llm_stall_timeout_seconds,
    _model_input_raw,
    _output_variable_name,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompt_builders import (
    _enrich_candidate_apis,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.prompts import (
    CODE_EXECUTOR_SYSTEM_PROMPT,
)
from adk_appworld_agent.subagents.executor.code_plan_execute.results import (
    _parse_execute_stdout,
)
from adk_appworld_agent.subagents.executor.guardrails import (
    GeneratedCodeValidationError,
    unwrap_printed_program_source,
)
from adk_appworld_agent.subagents.utils.retry import (
    RateLimitExhausted,
    is_rate_limit,
    is_retryable,
    retry_agent_stream,
)


@dataclass
class _CodePlanRun:
    plan: CodePlanOutput | None
    prompt: str
    model_input_raw: dict
    raw_text: str
    parsed_json: dict | None
    parse_error: str | None
    llm_raised: str | None
    usage: dict[str, int]
    rate_limit_exhausted: bool = False


@dataclass
class _CodeExecuteRun:
    stdout_json: dict | None  # parsed {"value":..., "answer":..., "summary":...}
    finalize_args: dict | None  # FINALIZE_MARKER fallback (in-sandbox finalize() call)
    raw_stdout: str
    code: str | None
    model_input_raw: dict
    raw_text: str
    parse_error: str | None
    llm_raised: str | None
    tool_call_count: int
    tool_calls: list[dict]
    event_diagnostics: list[dict]
    usage: dict[str, int]
    stalled: bool = False
    repair_attempts: list[dict] = field(default_factory=list)
    submit_final_args: dict | None = None  # Z: SUBMIT_FINAL tool call args
    stream_error: str | None = None  # raw stream exception text (pre-classification)
    rate_limit_exhausted: bool = False  # unified 429 ladder fully consumed


async def _run_llm_code_planner(
    *,
    inner_agent: LlmAgent,
    subagent_input: SubagentInput,
    prompt: str,
    model_input_raw: dict,
) -> _CodePlanRun:
    session_service = InMemorySessionService()
    app_name = f"code_planner_{subagent_input.task_context.task_id}"
    user_id = "code_planner"
    session_id = f"code_plan_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(
        agent=inner_agent,
        session_service=session_service,
        app_name=app_name,
    )
    raw_text = ""
    usage = _empty_usage()

    async def _attempt(attempt: int) -> None:
        nonlocal raw_text, session_id, usage
        if attempt > 0:
            session_id = f"code_plan_{uuid.uuid4().hex[:8]}"
            await session_service.create_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
            raw_text = ""
            usage = _empty_usage()
        async with aclosing(
            runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            )
        ) as agen:
            async for event in agen:
                for part in getattr(event.content, "parts", None) or []:
                    fc = getattr(part, "function_call", None)
                    if fc is not None and fc.name == "load_skill":
                        usage["skill_loads"] = usage.get("skill_loads", 0) + 1
                event_usage = _extract_usage(event)
                if event_usage is not None:
                    usage["llm_calls"] += 1
                    usage["prompt_tokens"] += event_usage["prompt"]
                    usage["completion_tokens"] += event_usage["completion"]
                    usage["thoughts_tokens"] += event_usage["thoughts"]
                text = content_to_text(event.content)
                if text:
                    raw_text = text

    rate_limit_exhausted = False
    try:
        llm_raised = await retry_agent_stream(_attempt, label="code_plan")
    except RateLimitExhausted as exc:
        llm_raised = str(exc)
        rate_limit_exhausted = True
    if raw_text and usage["llm_calls"] == 0:
        usage["llm_calls"] = 1
    plan, parsed_json, parse_error = _parse_code_plan(raw_text)
    if llm_raised is not None:
        sys.stderr.write(f"[code_plan] failed after retries: {llm_raised}\n")
        sys.stderr.flush()
    elif parse_error is not None:
        sys.stderr.write(f"[code_plan] invalid output: {parse_error}\n")
        sys.stderr.flush()
    return _CodePlanRun(
        plan=plan,
        prompt=prompt,
        model_input_raw=model_input_raw,
        raw_text=raw_text,
        parsed_json=parsed_json,
        parse_error=parse_error,
        llm_raised=llm_raised,
        usage=usage,
        rate_limit_exhausted=rate_limit_exhausted,
    )


async def _run_llm_code_executor(
    *,
    inner_agent: LlmAgent,
    subagent_input: SubagentInput,
    prompt: str,
    model_input_raw: dict,
) -> _CodeExecuteRun:
    meta = subagent_input.metadata or {}
    candidate_apis = meta.get("candidate_apis")
    prior_variables = meta.get("prior_variables")

    # Inject env vars so the sandbox can load them inside execute_python
    if isinstance(candidate_apis, list) and candidate_apis:
        os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = json.dumps(
            _enrich_candidate_apis(candidate_apis), ensure_ascii=False
        )
    else:
        os.environ.pop(EXECUTOR_CANDIDATE_APIS_ENV, None)
    if isinstance(prior_variables, dict) and prior_variables:
        os.environ[EXECUTOR_PRIOR_VARIABLES_ENV] = json.dumps(
            prior_variables, ensure_ascii=False
        )
    else:
        os.environ.pop(EXECUTOR_PRIOR_VARIABLES_ENV, None)
    if subagent_input.task_context.task_datetime:
        os.environ[EXECUTOR_TASK_DATETIME_ENV] = (
            subagent_input.task_context.task_datetime
        )
    else:
        os.environ.pop(EXECUTOR_TASK_DATETIME_ENV, None)
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"

    session_service = InMemorySessionService()
    app_name = f"code_executor_{subagent_input.task_context.task_id}"
    user_id = "code_executor"
    session_id = f"code_exec_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    runner = Runner(
        agent=inner_agent,
        session_service=session_service,
        app_name=app_name,
    )

    try:
        return await _run_code_executor_repair_loop(
            runner=runner,
            session_service=session_service,
            app_name=app_name,
            user_id=user_id,
            subagent_input=subagent_input,
            base_prompt=prompt,
            base_model_input_raw=model_input_raw,
        )
    finally:
        os.environ.pop(EXECUTOR_CANDIDATE_APIS_ENV, None)
        os.environ.pop(EXECUTOR_GUARDRAILS_ENV, None)
        os.environ.pop(EXECUTOR_PRIOR_VARIABLES_ENV, None)
        os.environ.pop(EXECUTOR_TASK_DATETIME_ENV, None)


async def _run_attempt_isolated(
    coro_factory: Callable[[], Awaitable["_CodeExecuteRun"]],
) -> "_CodeExecuteRun":
    """Run one executor attempt on a FRESH event loop in a short-lived thread.

    genai binds its aiohttp connector to the running event loop; a
    gemini-2.5-flash socket stall (python-genai #1893) plus the unawaited
    AsyncClient.aclose() scheduled on GC (#1709) leave a half-dead connection
    that, on the SHARED per-task loop, makes every subsequent attempt observe
    0 events (the measured 5x events_seen=0 cascade). Giving each attempt its
    own throwaway loop means a poisoned connection dies with that loop and can
    never reach the next attempt/phase. The attempt's own per-event stall timer
    still bounds each individual call; this only breaks the cross-attempt
    cascade. The coroutine is BUILT inside the worker thread (coro_factory) so
    it is created and run on the same fresh loop.
    """
    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller loop
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    # Await the worker without blocking the main loop (join runs on a pool
    # thread). The worker returns once the attempt completes — including a clean
    # stalled=True return after its own per-event timer fires — so there is no
    # need to forcibly abandon it; its fresh loop is discarded regardless.
    await asyncio.get_running_loop().run_in_executor(None, thread.join)
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["result"]  # type: ignore[return-value]


async def _run_code_executor_repair_loop(
    *,
    runner: Runner,
    session_service: InMemorySessionService,
    app_name: str,
    user_id: str,
    subagent_input: SubagentInput,
    base_prompt: str,
    base_model_input_raw: dict,
) -> _CodeExecuteRun:
    attempts: list[dict] = []
    last_run: _CodeExecuteRun | None = None
    repair_prompt: str | None = None
    attempt_index = 0
    transient_count = 0
    rate_limit_count = 0
    provider_delays = active_config().retry.provider_backoff_delays

    while attempt_index < _MAX_CODE_EXECUTE_ATTEMPTS:
        prompt = repair_prompt or base_prompt
        model_input_raw = _model_input_raw(
            model=str(base_model_input_raw.get("model") or "unknown"),
            system_prompt=str(
                base_model_input_raw.get("system_instruction")
                or CODE_EXECUTOR_SYSTEM_PROMPT
            ),
            prompt=prompt,
            tool_config=base_model_input_raw.get("tool_config"),
        )

        def _make_attempt(
            prompt=prompt,
            model_input_raw=model_input_raw,
            attempt_index=attempt_index,
        ) -> Awaitable[_CodeExecuteRun]:
            return _run_code_executor_attempt(
                runner=runner,
                session_service=session_service,
                app_name=app_name,
                user_id=user_id,
                subagent_input=subagent_input,
                prompt=prompt,
                model_input_raw=model_input_raw,
                attempt_index=attempt_index,
            )

        if active_config().executor_isolate_llm_loop:
            # Fresh event loop per attempt — a poisoned genai connection dies
            # with the throwaway loop instead of cascading (see RunConfig).
            run = await _run_attempt_isolated(_make_attempt)
        else:
            run = await _make_attempt()
        attempts.append(_summarize_code_execute_attempt(attempt_index, run))
        run.repair_attempts = list(attempts)
        run.usage = _sum_code_execute_attempt_usage(attempts)
        last_run = run

        if run.stdout_json is not None or run.finalize_args is not None:
            return run

        # Unified provider ladder: a rate-limited stream retries
        # the SAME prompt — never a repair prompt, the 429 text must not
        # leak into any LLM input. On exhaustion the run is marked and the
        # subagent emits PROVIDER_RATE_LIMITED (clean abort + rerun later).
        rate_limit_msg = None
        if not run.stalled:
            if run.stream_error and is_rate_limit(run.stream_error):
                rate_limit_msg = run.stream_error
            elif run.llm_raised and is_rate_limit(run.llm_raised):
                rate_limit_msg = run.llm_raised
        if rate_limit_msg is not None:
            if rate_limit_count >= len(provider_delays):
                sys.stderr.write(
                    f"[code_exec] rate-limit ladder exhausted: {rate_limit_msg[:120]}\n"
                )
                sys.stderr.flush()
                run.rate_limit_exhausted = True
                return run
            delay = provider_delays[rate_limit_count]
            sys.stderr.write(
                f"[code_exec] attempt {attempt_index + 1} rate-limited, "
                f"waiting {delay:.0f}s "
                f"({rate_limit_count + 1}/{len(provider_delays)})\n"
            )
            sys.stderr.flush()
            grant_rate_limit_grace(delay)
            await asyncio.sleep(delay)
            rate_limit_count += 1
            continue

        # Transient short-circuit: stall / 5xx should retry the same
        # attempt with backoff rather than burn a repair-prompt slot. Same
        # prompt + transient infra failure → no repair-instruction is going
        # to fix it; the LLM just needs another chance once the upstream
        # provider recovers. Budget is exhausted after len(active_config().retry.executor_stall_backoff)
        # transient retries; then we fall through to the normal failure path.
        # (429 never reaches here — handled by the unified ladder above.)
        is_transient = run.stalled or bool(
            run.llm_raised and is_retryable(run.llm_raised)
        )
        if is_transient and transient_count < len(
            active_config().retry.executor_stall_backoff
        ):
            delay = active_config().retry.executor_stall_backoff[transient_count]
            label = "stalled" if run.stalled else (run.llm_raised or "")[:80]
            sys.stderr.write(
                f"[code_exec] attempt {attempt_index + 1} transient ({label}), "
                f"retry {transient_count + 1}/{len(active_config().retry.executor_stall_backoff)} "
                f"in {delay:.0f}s\n"
            )
            sys.stderr.flush()
            await asyncio.sleep(delay)
            transient_count += 1
            continue

        if not _should_repair_code_execute_run(run):
            _log_code_execute_failure(run)
            return run

        if attempt_index >= _MAX_CODE_EXECUTE_ATTEMPTS - 1:
            break

        repair_prompt = _build_code_execute_repair_prompt(
            base_prompt=base_prompt,
            failed_run=run,
            next_attempt=attempt_index + 2,
            max_attempts=_MAX_CODE_EXECUTE_ATTEMPTS,
        )
        attempt_index += 1

    if last_run is None:
        return _empty_code_execute_run(
            model_input_raw=base_model_input_raw,
            parse_error="code executor did not run any attempts",
            llm_raised="code executor did not run any attempts",
            repair_attempts=attempts,
        )
    _log_code_execute_failure(last_run)
    return last_run


async def _run_code_executor_attempt(
    *,
    runner: Runner,
    session_service: InMemorySessionService,
    app_name: str,
    user_id: str,
    subagent_input: SubagentInput,
    prompt: str,
    model_input_raw: dict,
    attempt_index: int,
) -> _CodeExecuteRun:
    session_id = f"code_exec_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    raw_text = ""
    raw_stdout = ""
    code: str | None = None
    tool_call_count = 0
    tool_calls: list[dict] = []
    submit_final_args: dict | None = None
    event_diagnostics: list[dict] = []
    usage = _empty_usage()
    stream_error: str | None = None
    # Z: defend against degenerate empty-code execute_python loops
    # (f323bae_2 2026-05-31: 173 consecutive `execute_python(code="")` calls
    # before the 40-min wall timeout finally killed the task). Empty calls
    # do not advance the milestone and just burn LLM round-trips, but the
    # per-event stall timer resets on each call so the loop never breaks
    # from the inside. We cap consecutive empty calls AND total
    # execute_python calls.
    consecutive_empty_executes = 0
    real_execute_calls = 0
    _MAX_CONSECUTIVE_EMPTY_EXECUTES = 2
    _MAX_TOTAL_EXECUTE_CALLS = 6

    stall_timeout_s = _llm_stall_timeout_seconds()
    stalled = False
    try:
        async with aclosing(
            runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            )
        ) as agen:
            while True:
                try:
                    event = await asyncio.wait_for(
                        agen.__anext__(), timeout=stall_timeout_s
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    stalled = True
                    stream_error = (
                        f"Stage 2 executor attempt {attempt_index + 1} stalled "
                        f"(no LLM event for {stall_timeout_s:g}s)"
                    )
                    _md = subagent_input.metadata or {}
                    print(
                        f"[stall-cancel] seq={next(_STALL_CANCEL_COUNTER)} "
                        f"task={subagent_input.task_context.task_id} "
                        f"milestone={_md.get('milestone_id')} "
                        f"attempt={attempt_index + 1} timeout_s={stall_timeout_s:g} "
                        f"events_seen={len(event_diagnostics)} "
                        f"last_code_len={len(code) if code else 0} "
                        f"(cancelling Stage-2 stream)",
                        file=sys.stderr,
                        flush=True,
                    )
                    break
                event_diagnostics.append(_extract_event_diagnostics(event))
                event_tool_calls = _extract_tool_calls(event)
                loop_guard_triggered = False
                if event_tool_calls:
                    tool_calls.extend(event_tool_calls)
                    for tool_call in event_tool_calls:
                        name = tool_call.get("name")
                        if name == "execute_python":
                            tool_call_count += 1
                            args = tool_call.get("args")
                            code_arg = (
                                args.get("code") if isinstance(args, dict) else None
                            )
                            is_empty = not (
                                isinstance(code_arg, str) and code_arg.strip()
                            )
                            if is_empty:
                                consecutive_empty_executes += 1
                                if (
                                    consecutive_empty_executes
                                    >= _MAX_CONSECUTIVE_EMPTY_EXECUTES
                                ):
                                    stream_error = (
                                        f"Stage 2 executor attempt "
                                        f"{attempt_index + 1} stuck in empty-code "
                                        f"loop ({consecutive_empty_executes} "
                                        f"consecutive execute_python calls with "
                                        f"empty `code`)"
                                    )
                                    loop_guard_triggered = True
                                    break
                            else:
                                consecutive_empty_executes = 0
                                real_execute_calls += 1
                                code = _clean_code_arg(code_arg)
                                _log_code_execute_call(code)
                                if real_execute_calls > _MAX_TOTAL_EXECUTE_CALLS:
                                    stream_error = (
                                        f"Stage 2 executor attempt "
                                        f"{attempt_index + 1} exceeded total "
                                        f"execute_python budget "
                                        f"({real_execute_calls} > "
                                        f"{_MAX_TOTAL_EXECUTE_CALLS})"
                                    )
                                    loop_guard_triggered = True
                                    break
                        elif name == "submit_final":
                            args = tool_call.get("args")
                            if isinstance(args, dict):
                                # Capture only the first submit_final call —
                                # Z contract says EXACTLY ONCE. Later calls
                                # are ignored to preserve the first commit.
                                if submit_final_args is None:
                                    submit_final_args = args
                        elif name == "load_skill":
                            # Native-skill condition: pure counter (round1
                            # spec Part C) — no break, no capture. A run with
                            # skill_loads == 0 means "skill never triggered",
                            # not "skill ineffective".
                            usage["skill_loads"] = usage.get("skill_loads", 0) + 1
                if loop_guard_triggered:
                    break
                event_usage = _extract_usage(event)
                if event_usage is not None:
                    usage["llm_calls"] += 1
                    usage["prompt_tokens"] += event_usage["prompt"]
                    usage["completion_tokens"] += event_usage["completion"]
                    usage["thoughts_tokens"] += event_usage["thoughts"]
                text = content_to_text(event.content)
                if text:
                    raw_text = text
                result_str = _extract_execute_python_result(event)
                if result_str is not None:
                    raw_stdout = result_str
                    # Z: don't break on execute_python result — keep listening
                    # so the LlmAgent has a chance to call submit_final next.
                    # Loop ends naturally on StopAsyncIteration (LlmAgent
                    # finishes its turn after submit_final) or on stall.
                if submit_final_args is not None:
                    # Z: submit_final captured — done. Break to avoid waiting
                    # on extra events the LlmAgent might emit.
                    break
    except Exception as exc:
        stream_error = str(exc)

    if raw_text and usage["llm_calls"] == 0:
        usage["llm_calls"] = 1

    stdout_json, finalize_args, parse_error = _parse_execute_stdout(raw_stdout)
    if stream_error is not None:
        parse_error = stream_error
    elif tool_call_count == 0 and submit_final_args is None:
        # Z: at least one of execute_python or submit_final must fire,
        # otherwise the executor produced no usable output channel.
        parse_error = _no_execute_python_call_error(raw_text, event_diagnostics)
    elif submit_final_args is not None:
        # Z: structured submit_final args present → don't penalise the
        # missing print(json.dumps(...)) shape in stdout.
        parse_error = None

    llm_raised = (
        stream_error
        if code is None and tool_call_count == 0 and submit_final_args is None
        else None
    )
    return _CodeExecuteRun(
        stdout_json=stdout_json,
        finalize_args=finalize_args,
        submit_final_args=submit_final_args,
        raw_stdout=raw_stdout,
        code=code,
        model_input_raw=model_input_raw,
        raw_text=raw_text,
        parse_error=parse_error,
        llm_raised=llm_raised,
        tool_call_count=tool_call_count,
        tool_calls=tool_calls,
        event_diagnostics=event_diagnostics,
        usage=usage,
        stalled=stalled,
        stream_error=stream_error,
    )


def _should_repair_code_execute_run(run: _CodeExecuteRun) -> bool:
    """Repair only failures that came after the model produced code."""
    if run.stdout_json is not None or run.finalize_args is not None:
        return False
    return run.code is not None and bool(run.parse_error or run.llm_raised)


def _build_code_execute_repair_prompt(
    *,
    base_prompt: str,
    failed_run: _CodeExecuteRun,
    next_attempt: int,
    max_attempts: int,
) -> str:
    diagnostics = _code_execute_repair_diagnostics(failed_run)
    parts = [
        base_prompt,
        "",
        "## Repair required",
        f"The previous code execution attempt failed. This is repair attempt {next_attempt} of {max_attempts}.",
        "Rewrite the complete Python program. Do not return a diff or patch.",
        "Call execute_python exactly once with the complete corrected program.",
        "Keep the same output contract: the final stdout line must be json.dumps with a dict containing `value`.",
        "",
        "### Previous code",
        "```python",
        failed_run.code or "",
        "```",
        "",
        "### Failure diagnostics",
        diagnostics,
    ]
    return "\n".join(parts)


def _code_execute_repair_diagnostics(run: _CodeExecuteRun) -> str:
    lines: list[str] = []
    if run.parse_error:
        lines.append(f"- parse_or_execution_error: {run.parse_error}")
    if run.llm_raised:
        lines.append(f"- stream_or_provider_error: {run.llm_raised}")
    lines.append(f"- execute_python_tool_calls: {run.tool_call_count}")
    if run.raw_stdout:
        lines += [
            "- raw_stdout:",
            "```text",
            _truncate_for_prompt(run.raw_stdout, 4000),
            "```",
        ]
    if run.event_diagnostics:
        lines += [
            "- event_diagnostics:",
            "```json",
            json.dumps(run.event_diagnostics[-3:], ensure_ascii=False, indent=2),
            "```",
        ]
    return "\n".join(lines) if lines else "- no diagnostics captured"


def _summarize_code_execute_attempt(attempt_index: int, run: _CodeExecuteRun) -> dict:
    return {
        "attempt": attempt_index + 1,
        "code": run.code,
        "raw_stdout": run.raw_stdout,
        "stdout_json": run.stdout_json,
        "parse_error": run.parse_error,
        "llm_raised": run.llm_raised,
        "tool_call_count": run.tool_call_count,
        "tool_calls": run.tool_calls,
        "event_diagnostics": run.event_diagnostics,
        "usage": run.usage,
    }


def _sum_code_execute_attempt_usage(attempts: list[dict]) -> dict[str, int]:
    total = _empty_usage()
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in total:
            total[key] += int(usage.get(key, 0) or 0)
    return total


def _empty_code_execute_run(
    *,
    model_input_raw: dict,
    parse_error: str,
    llm_raised: str | None,
    repair_attempts: list[dict],
) -> _CodeExecuteRun:
    return _CodeExecuteRun(
        stdout_json=None,
        finalize_args=None,
        submit_final_args=None,
        raw_stdout="",
        code=None,
        model_input_raw=model_input_raw,
        raw_text="",
        parse_error=parse_error,
        llm_raised=llm_raised,
        tool_call_count=0,
        tool_calls=[],
        event_diagnostics=[],
        usage=_empty_usage(),
        repair_attempts=repair_attempts,
    )


def _log_code_execute_call(code: str) -> None:
    sys.stderr.write(
        f"[code_exec] execute_python called "
        f"(code_len={len(code)}):\n"
        f"{'─' * 60}\n"
        f"{code}\n"
        f"{'─' * 60}\n"
    )
    sys.stderr.flush()


def _log_code_execute_failure(run: _CodeExecuteRun) -> None:
    if run.llm_raised is not None:
        sys.stderr.write(f"[code_exec] failed: {run.llm_raised}\n")
        sys.stderr.flush()
    elif run.parse_error is not None:
        sys.stderr.write(f"[code_exec] stdout parse error: {run.parse_error}\n")
        sys.stderr.flush()


def _truncate_for_prompt(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...<truncated {len(value) - limit} chars>"


# ── Code cleaning ─────────────────────────────────────────────────────────────


def _clean_code_arg(raw: str) -> str:
    code = clean_execute_python_code(raw)
    try:
        unwrapped = unwrap_printed_program_source(code)
    except GeneratedCodeValidationError:
        return code
    # unwrap pulled the inner program out of a wrapping string literal via
    # ast.value.value, which Python evaluates as a real string and
    # collapses any nested `\n` escape sequences into newline characters.
    # Re-escape so subsequent compile/validation sees valid Python.
    return repair_generated_source_artifacts(unwrapped)


# ── Event inspection helpers ──────────────────────────────────────────────────


def _extract_tool_calls(event: Event) -> list[dict]:
    if not event.content or not event.content.parts:
        return []
    tool_calls: list[dict] = []
    for part in event.content.parts:
        fc = getattr(part, "function_call", None)
        if not fc:
            continue
        name = getattr(fc, "name", None)
        args = getattr(fc, "args", None) or {}
        tool_calls.append(
            {
                "name": str(name or ""),
                "args": _json_safe_args(args if isinstance(args, dict) else {}),
            }
        )
    return tool_calls


def _json_safe_args(args: dict) -> dict:
    try:
        json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        return {str(key): str(value) for key, value in args.items()}
    return args


def _extract_execute_python_result(event: Event) -> str | None:
    if not event.content or not event.content.parts:
        return None
    for part in event.content.parts:
        fr = getattr(part, "function_response", None)
        if not fr or getattr(fr, "name", None) != "execute_python":
            continue
        response = getattr(fr, "response", None)
        result = response.get("result") if isinstance(response, dict) else response
        return result if isinstance(result, str) else str(result or "")
    return None


def _extract_event_diagnostics(event: Event) -> dict:
    diag: dict = {}
    for attr in ("id", "author", "invocation_id", "branch"):
        value = getattr(event, attr, None)
        if value not in (None, ""):
            diag[attr] = value

    content = getattr(event, "content", None)
    if content is not None:
        role = getattr(content, "role", None)
        if role:
            diag["role"] = role
        parts = getattr(content, "parts", None) or []
        part_diags = []
        for part in parts:
            part_diag: dict = {}
            text = getattr(part, "text", None)
            if isinstance(text, str) and text:
                part_diag["text_len"] = len(text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = getattr(fc, "args", None) or {}
                call_diag = {"name": getattr(fc, "name", None)}
                if isinstance(args, dict):
                    call_diag["arg_keys"] = sorted(args.keys())
                    code_arg = args.get("code")
                    if isinstance(code_arg, str):
                        call_diag["code_len"] = len(code_arg)
                part_diag["function_call"] = call_diag
            fr = getattr(part, "function_response", None)
            if fr is not None:
                response = getattr(fr, "response", None)
                response_diag = {
                    "name": getattr(fr, "name", None),
                    "response_type": type(response).__name__,
                }
                if isinstance(response, dict):
                    result = response.get("result")
                    response_diag["response_keys"] = sorted(response.keys())
                    if isinstance(result, str):
                        response_diag["result_len"] = len(result)
                part_diag["function_response"] = response_diag
            if part_diag:
                part_diags.append(part_diag)
        if part_diags:
            diag["parts"] = part_diags

    usage = _extract_usage(event)
    if usage is not None:
        diag["usage"] = usage
    return diag


def _no_execute_python_call_error(raw_text: str, event_diagnostics: list[dict]) -> str:
    if raw_text.strip():
        return "execute_python tool call was not observed; model returned text instead"
    if event_diagnostics:
        return "execute_python tool call was not observed; see event_diagnostics"
    return "execute_python tool call was not observed; no model events were captured"


def _extract_usage(event: Event) -> dict | None:
    um = getattr(event, "usage_metadata", None)
    if um is None:
        return None
    prompt = getattr(um, "prompt_token_count", None) or 0
    completion = getattr(um, "candidates_token_count", None) or 0
    thoughts = getattr(um, "thoughts_token_count", None) or 0
    if prompt == 0 and completion == 0 and thoughts == 0:
        return None
    return {"prompt": prompt, "completion": completion, "thoughts": thoughts}


# ── Deterministic skeleton helpers (test / non-LLM mode) ─────────────────────


def _build_deterministic_plan(metadata: dict) -> CodePlanOutput:
    milestone_intent = str(
        metadata.get("milestone_intent") or "Complete the milestone."
    )
    output_name = _output_variable_name(metadata)
    return CodePlanOutput(
        plan_steps=[
            f"Understand the milestone intent: {milestone_intent}",
            "Use the candidate APIs and prior variables when the real Stage 2 executor is enabled.",
        ],
        construct_step="Prepare result_dict with a value for the planned output variable.",
        print_step="Print json.dumps(result_dict) as the final stdout line.",
        output_variable=VariableSpec(
            name=output_name,
            description=(
                "Deterministic skeleton output stand-in for milestone: "
                f"{milestone_intent}"
            ),
        ),
    )


def _build_deterministic_execute_run(
    metadata: dict,
    plan: CodePlanOutput,
    model_input_raw: dict,
) -> _CodeExecuteRun:
    milestone_id = str(metadata.get("milestone_id") or "")
    result_value = {
        "mode": "code_execute_deterministic_skeleton",
        "milestone_id": milestone_id,
        "planned_steps": len(plan.numbered_steps()),
    }
    stdout_json = {
        "value": result_value,
        "answer": "null",
        "summary": "Code-execute skeleton completed without calling tools.",
        "description": "Deterministic skeleton placeholder for non-LLM mode.",
    }
    return _CodeExecuteRun(
        stdout_json=stdout_json,
        finalize_args=None,
        submit_final_args=None,
        raw_stdout=json.dumps(stdout_json, ensure_ascii=False),
        code=None,
        model_input_raw=model_input_raw,
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=0,
        tool_calls=[],
        event_diagnostics=[],
        usage=_empty_usage(),
    )


# ── Code plan parser ──────────────────────────────────────────────────────────


def _parse_code_plan(
    text: str,
) -> tuple[CodePlanOutput | None, dict | None, str | None]:
    try:
        data = json.loads(text)
        plan = CodePlanOutput.model_validate(data)
    except Exception as exc:
        return None, None, str(exc)
    return plan, plan.model_dump(mode="json"), None


# ── IO payload builders ───────────────────────────────────────────────────────
