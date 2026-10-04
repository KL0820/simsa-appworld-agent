"""Run one AppWorld task through the controller and dump the trajectory.

Usage:
    python scripts/run_task.py --task_id 6104387_3 --experiment_name local_dev

Prereqs:
    - AppWorld zerorpc server running on tcp://127.0.0.1:4242 (see adk-api-agent).
    - Install runtime extras: `pip install -e '.[runtime]'`.

Current behavior:
    - `--find community --execute llm` runs the community Finder + LLM Executor path.
    - `--plan rough --find community --execute stub` runs per-milestone Finder
      while mocking Executor.
    - `--find stub --execute stub` keeps the old plumbing-only path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path

from adk_appworld_agent.repo_paths import load_repository_env

load_repository_env()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_appworld_agent.agent import build_appworld_controller_agent
from adk_appworld_agent.appworld.auth import AppWorldAuthManager
from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder
from adk_appworld_agent.appworld.bootstrap import load_task
from adk_appworld_agent.appworld.client import AppWorldRpcClient
from adk_appworld_agent.appworld.holder import AppWorldClientHolder
from adk_appworld_agent.appworld.world_context import (
    build_world_run_name,
    close_world_safely,
)
from adk_appworld_agent.observability.events import (
    EventBus,
    SubagentCompleted,
    TaskCompleted,
    subagent_completed_events,
)
from adk_appworld_agent.observability.ledger import LEDGER_KEY
from adk_appworld_agent.observability.live_timeline import LiveTimeline
from adk_appworld_agent.observability.log_files import (
    COMPACT_LOG_DETAIL,
    LOG_DETAIL_ENV,
    RAW_LOG_DETAIL,
    clean_ansi,
    extract_report_text,
    format_seconds,
    log_timestamp,
    prune_experiment_runs,
    write_evaluation_report,
    write_jsonl,
)
from adk_appworld_agent.observability.paths import artifacts_dir
from adk_appworld_agent.observability.sinks.debug_artifacts import DebugArtifactsSink
from adk_appworld_agent.observability.sinks.subagent_io import SubagentIoMarkdownSink
from adk_appworld_agent.observability.sinks.subagent_io_jsonl import SubagentIoJsonlSink
from adk_appworld_agent.observability.sinks.task_summary import TaskSummaryMarkdownSink
from adk_appworld_agent.observability.sinks.terminal import (
    TerminalEventPrinter,
    terminal_lines_for_subagent,
)
from adk_appworld_agent.observability.subagent_logs import (
    io_record_from_subagent_output,
)
from adk_appworld_agent.observability.workflow import (
    build_task_summary,
    read_jsonl,
)
from adk_appworld_agent.orchestration.active_config import set_active_config
from adk_appworld_agent.orchestration.cache.policy import CacheLayer, CachePolicy
from adk_appworld_agent.orchestration.cache.store import JsonlSubagentCacheStore
from adk_appworld_agent.orchestration.rate_limit_grace import (
    set_rate_limit_grace_extender,
)
from adk_appworld_agent.orchestration.run_config import (
    LogConfig,
    ModelConfig,
    RetryConfig,
    RunConfig,
    dump_resolved_config,
    load_run_config,
)
from adk_appworld_agent.orchestration.state import (
    BudgetSnapshot,
    Phase,
    RunState,
    RunStatus,
    TaskContext,
)
from adk_appworld_agent.orchestration.state_repo import RUN_STATE_KEY
from adk_appworld_agent.orchestration.subagent_output_store import SUBAGENT_OUTPUTS_KEY

_PASS_RE = re.compile(r"Num Passed Tests\s*:\s*(\d+)")
_FAIL_RE = re.compile(r"Num Failed Tests\s*:\s*(\d+)")
_TOTAL_RE = re.compile(r"Num Total\s*Tests\s*:\s*(\d+)")


def _parse_cache_set(raw: str | None) -> set[str] | None:
    """Parse the env-string form of a cache subagent allow-list.

    Empty / unset → empty set (deny all). ``all`` → None (allow any subagent).
    Comma-separated names → that set.
    """

    if raw is None:
        return set()
    value = raw.strip()
    if not value:
        return set()
    if value.lower() == "all":
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _build_cache_layer_from_env() -> CacheLayer | None:
    """Read-only subagent cache from env vars.

    `SUBAGENT_CACHE_DIR` selects the cache root.
    `SUBAGENT_CACHE_READ` selects which subagents may hit the cache
        (`all`, comma-separated names, or empty to disable reads).
    `SUBAGENT_CACHE_VERSION` overrides the spec-declared version (for
    invalidation).

    Writes are always disabled at runtime; cache files are produced by
    `scripts/build_cache.py` from log artifacts, not by the runtime.
    """

    cache_dir = os.getenv("SUBAGENT_CACHE_DIR")
    if not cache_dir:
        return None
    store = JsonlSubagentCacheStore(Path(cache_dir))
    policy = CachePolicy(
        read=_parse_cache_set(os.getenv("SUBAGENT_CACHE_READ")),
        write=set(),
        version_override=os.getenv("SUBAGENT_CACHE_VERSION") or None,
    )
    return CacheLayer(store=store, policy=policy)


LIVE_PHASE_ORDER = (
    Phase.PLAN.value,
    Phase.VERIFY.value,
    Phase.FIND.value,
    Phase.EXECUTE.value,
    Phase.SUBMIT.value,
    Phase.COMPLETE.value,
    Phase.FAILED.value,
    Phase.BOOTSTRAP.value,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AppWorld task through the multi-agent runtime."
    )
    parser.add_argument("--task_id", type=str, default=os.getenv("TASK_ID"))
    parser.add_argument(
        "--experiment_name", type=str, default=os.getenv("EXPERIMENT_NAME", "local_dev")
    )
    parser.add_argument(
        "--rpc_url", type=str, default=os.getenv("RPC_URL", "tcp://127.0.0.1:4242")
    )
    parser.add_argument("--log_root", type=str, default="logs")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an experiment config file (JSON RunConfig). When set it is "
        "the single source of truth; model/impl/skills CLI args are ignored.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=os.getenv("RUN_NAME"),
        help="Existing batch run directory name under <log_root>/<experiment_name>.",
    )
    parser.add_argument(
        "--find",
        dest="find_impl",
        default=os.getenv("FIND_IMPL", "community"),
        help="Subagent impl name for Phase.FIND (registry key).",
    )
    parser.add_argument(
        "--plan",
        dest="plan_impl",
        default=os.getenv("PLAN_IMPL", "stub"),
        help="Subagent impl name for Phase.PLAN (registry key).",
    )
    parser.add_argument(
        "--verify",
        dest="verify_impl",
        default=os.getenv("VERIFY_IMPL", "stub"),
        help="Subagent impl name for Phase.VERIFY (registry key). Default 'stub' is a no-op pass-through.",
    )
    parser.add_argument(
        "--execute",
        dest="execute_impl",
        default=os.getenv("EXECUTE_IMPL", "code_plan_execute"),
        help="Subagent impl name for Phase.EXECUTE (registry key).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("TASK_TIMEOUT_S", "2400")),
        help="Hard per-task timeout in seconds. The run aborts after this limit.",
    )
    parser.add_argument(
        "--model_name", type=str, default=os.getenv("MODEL_NAME", "gemini-2.5-flash")
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("MODEL_TEMPERATURE", "0"))
    )
    parser.add_argument(
        "--top_p", type=float, default=float(os.getenv("MODEL_TOP_P", "1"))
    )
    parser.add_argument("--top_k", type=int, default=int(os.getenv("MODEL_TOP_K", "1")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("MODEL_SEED", "123")))
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=int(os.getenv("MODEL_MAX_OUTPUT_TOKENS", "0")),
        help="0 keeps the provider default.",
    )
    parser.add_argument(
        "--show_raw_eval_report",
        action="store_true",
        help="Print the cleaned full evaluator report in addition to the summary.",
    )
    parser.add_argument(
        "--keep-debug",
        dest="keep_debug",
        action="store_true",
        help="Keep workflow, trajectory, executor, sandbox, and ledger debug JSONL files.",
    )
    parser.add_argument(
        "--overwrite-task-log",
        dest="overwrite_task_log",
        action="store_true",
        help="Allow overwriting an existing completed task log directory.",
    )
    # ── MIND-Skill: one switch, same pattern as --plan/--find/--execute ──
    # --skills rewrites the PLAN/EXECUTE impls to the registry skill variants:
    #   best|q0|q1|q2  -> rough_skill + code_plan_execute_skill (per-task PUSH;
    #                     held-out tasks retrieve top-K by description)
    #   native_<lib>   -> *_skill_native (ADK SkillToolset PULL via load_skill)
    #   thin           -> skill impls with NO skills (frozen thin prompts only)
    parser.add_argument(
        "--skills",
        # DEFAULT = thin+skill (best library), the thesis method (2026-06-22).
        # Pass --skills off for the no-skill baseline ablation.
        default=os.getenv("SKILLS_IMPL", "best"),
        choices=[
            "off",
            "best",
            "q0",
            "q1",
            "q2",
            "thin",
            "native_best",
            "native_q0",
            "native_q1",
            "native_q2",
        ],
        help="skill condition (rewrites plan/execute impls to the skill registry variants)",
    )
    parser.add_argument(
        "--skills_root",
        type=Path,
        default=Path(
            os.getenv("SKILLS_ROOT", "data/mind_skill/skills/release/thesis_final")
        ),
        help="root of <component>/<library>/<task_id>/<name>/SKILL.md",
    )
    parser.add_argument(
        "--skills_k",
        type=int,
        default=int(os.getenv("SKILLS_K", "3")),
        help="held-out retrieval top-K (spec §12 default 3)",
    )
    parser.add_argument(
        "--skills_prompt",
        choices=["minimal", "full"],
        default=os.getenv("SKILLS_PROMPT", "minimal"),
        help="prompt variant for skill injection: 'minimal' (frozen Fig-8 thin, "
        "default) or 'full' (skills on top of the constraint-preserving prompt).",
    )
    return parser.parse_args()


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    max_executor_retries = _max_executor_retries_from_env()
    plan_impl, execute_impl = args.plan_impl, args.execute_impl
    skills_library = "best"
    if args.skills != "off":
        native = args.skills.startswith("native_")
        plan_impl = "rough_skill_native" if native else "rough_skill"
        execute_impl = (
            "code_plan_execute_skill_native" if native else "code_plan_execute_skill"
        )
        if args.skills == "thin":
            skills_library = "none"
        else:
            skills_library = args.skills.removeprefix("native_")
    return RunConfig(
        model=ModelConfig(
            name=args.model_name,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            candidate_count=1,
            max_output_tokens=max(0, args.max_output_tokens),
        ),
        retry=RetryConfig(max_executor_retries=max_executor_retries),
        log=LogConfig(log_root=Path(args.log_root), keep_debug=bool(args.keep_debug)),
        impls={
            Phase.FIND: args.find_impl,
            Phase.PLAN: plan_impl,
            Phase.VERIFY: args.verify_impl,
            Phase.EXECUTE: execute_impl,
        },
        timeout_s=args.timeout,
        skills_root=args.skills_root,
        skills_library=skills_library,
        skills_k=args.skills_k,
        skills_prompt_variant=args.skills_prompt,
    )


def _apply_env_ablation_overrides(run_config: RunConfig) -> RunConfig:
    """Env-gated ablation toggles applied AFTER config resolution, so an A/B
    differs ONLY in the toggled flag regardless of whether the run built its
    config from CLI args or from a --config JSON file (the batch path uses the
    latter, which otherwise bypasses arg-level defaults). Default OFF ==
    baseline behaviour. Currently: EXECUTOR_SEES_PRIOR_MS_RETURNS=1 surfaces
    earlier milestones' observed api returns to the downstream code_planner
    (vis-gap-apitrace intermediate-visibility bridge)."""
    raw = os.getenv("EXECUTOR_SEES_PRIOR_MS_RETURNS")
    if raw is not None:
        run_config = run_config.model_copy(
            update={"executor_sees_prior_milestone_returns": raw == "1"}
        )
    return run_config


def _max_executor_retries_from_env() -> int:
    raw = os.getenv("CONTROLLER_MAX_EXECUTOR_RETRIES", "1")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1


def _seed_run_state(
    task_id: str,
    instruction: str,
    task_datetime: str,
) -> dict:
    task_context = TaskContext(
        task_id=task_id,
        instruction=instruction,
        task_datetime=task_datetime,
    )
    state = RunState(
        phase=Phase.BOOTSTRAP,
        status=RunStatus.RUNNING,
        task_context=task_context,
        budget_snapshot=BudgetSnapshot(),
    )
    return {RUN_STATE_KEY: state.model_dump(mode="json")}


def _clean_ansi(text: str) -> str:
    return clean_ansi(text)


def _extract_eval_counts(report_text: str) -> tuple[int, int, int] | None:
    clean = _clean_ansi(report_text)
    passed_match = _PASS_RE.search(clean)
    failed_match = _FAIL_RE.search(clean)
    total_match = _TOTAL_RE.search(clean)
    if not (passed_match and failed_match and total_match):
        return None
    return (
        int(passed_match.group(1)),
        int(failed_match.group(1)),
        int(total_match.group(1)),
    )


def _extract_report_text(report: object) -> str:
    return extract_report_text(report)


def _event_line(record: dict) -> str:
    calls = record.get("function_calls") or []
    responses = record.get("function_responses") or []
    text = record.get("text")
    if calls or responses or text:
        return (
            f"[event] author={record['author']} "
            f"calls={[fc['name'] for fc in calls]} "
            f"responses={[fr['name'] for fr in responses]} "
            f"text={(text or '')[:80]!r}"
        )
    delta_keys = record.get("state_delta_keys") or []
    return f"[event] author={record['author']} state_delta={delta_keys}"


def _short_text(text: object, *, limit: int = 100) -> str:
    if not isinstance(text, str):
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _phase_sort_key(phase_name: str) -> tuple[int, str]:
    try:
        return (LIVE_PHASE_ORDER.index(phase_name), phase_name)
    except ValueError:
        return (len(LIVE_PHASE_ORDER), phase_name)


def _subagent_output_key(phase_name: str, item: dict) -> tuple[str, str, str]:
    return (
        phase_name,
        str(item.get("subagent_name") or ""),
        str(item.get("attempt") or ""),
    )


def _subagent_input_metadata(item: dict) -> dict:
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


def _milestone_summary(item: dict) -> str:
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
    return " ".join(parts)


def _stage_result_line(phase_name: str, item: dict) -> str:
    return _stage_result_lines(phase_name, item)[0]


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


def _plan_task_lines(payload: dict) -> list[str]:
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return []
    lines: list[str] = []
    for index, task in enumerate(tasks[:8], 1):
        if not isinstance(task, dict):
            continue
        app = task.get("app") or "unknown"
        text = _short_text(task.get("task"), limit=96)
        lines.append(f"  plan {index}. {app}: {text}")
    if len(tasks) > 8:
        lines.append(f"  plan ... +{len(tasks) - 8} more")
    return lines


def _candidate_api_lines(payload: dict) -> list[str]:
    candidate_apis = payload.get("candidate_apis")
    if not isinstance(candidate_apis, list):
        return []
    keys = [key for key in (_api_key(item) for item in candidate_apis) if key]
    if not keys:
        return []
    return [f"  apis: {_join_limited(keys, limit=10)}"]


def _stage_result_lines(phase_name: str, item: dict) -> list[str]:
    return terminal_lines_for_subagent(phase_name, item)


def _state_progress_line(raw_state: dict) -> str:
    phase = raw_state.get("phase")
    status = raw_state.get("status")
    block = raw_state.get("completion_block_reason")
    milestones = raw_state.get("milestones")
    active_idx = raw_state.get("active_milestone_index")
    milestone_text = ""
    if isinstance(milestones, list) and milestones and isinstance(active_idx, int):
        milestone_text = f" active_milestone={active_idx + 1}/{len(milestones)}"
        if 0 <= active_idx < len(milestones):
            active = milestones[active_idx]
            if isinstance(active, dict):
                app = _short_text(active.get("app"), limit=24)
                intent = _short_text(active.get("intent"), limit=110)
                if app:
                    milestone_text += f" app={app}"
                if intent:
                    milestone_text += f' intent="{intent}"'
    block_text = f" block={block}" if block else ""
    return f"[state] phase={phase} status={status}{milestone_text}{block_text}"


def _event_progress_lines(
    event,
    *,
    seen_outputs: set[tuple[str, str, str]],
    last_state_line: str | None,
    terminal_bus: EventBus | None = None,
    timeline: LiveTimeline | None = None,
) -> tuple[list[str], str | None]:
    if not event.actions:
        return [], last_state_line
    delta = event.actions.state_delta or {}
    lines: list[str] = []

    raw_outputs = delta.get(SUBAGENT_OUTPUTS_KEY)
    if isinstance(raw_outputs, dict):
        for phase_name in sorted(raw_outputs, key=_phase_sort_key):
            items = raw_outputs.get(phase_name) or []
            for item in items:
                if not isinstance(item, dict):
                    continue
                key = _subagent_output_key(phase_name, item)
                if key in seen_outputs:
                    continue
                seen_outputs.add(key)
                if timeline is not None:
                    timeline.record_output(phase_name, item)
                record = io_record_from_subagent_output(phase_name, item)
                if terminal_bus is not None and record is not None:
                    terminal_bus.emit(
                        SubagentCompleted(
                            phase_name=phase_name,
                            io_record=record,
                            raw_output=item,
                        )
                    )
                else:
                    lines.extend(_stage_result_lines(phase_name, item))

    raw_state = delta.get(RUN_STATE_KEY)
    if isinstance(raw_state, dict):
        if timeline is not None:
            timeline.record_state(raw_state)
        state_line = _state_progress_line(raw_state)
        if state_line != last_state_line:
            lines.append(state_line)
            last_state_line = state_line
    return lines, last_state_line


def _print_find_payload(payload: dict, metrics: dict) -> None:
    matched_apps = payload.get("matched_apps") or []
    selected_communities = payload.get("selected_communities") or []
    print(
        "  matched_apps="
        + (", ".join(matched_apps) if matched_apps else "none")
        + f"  fallback_used={payload.get('fallback_used')}  "
        + f"candidate_count={payload.get('candidate_count')}"
    )
    if selected_communities:
        print("  selected_communities=" + ", ".join(selected_communities))
    if metrics:
        print(
            f"  finder_llm_calls={metrics.get('llm_calls')}  "
            f"dependency_rounds={metrics.get('dependency_rounds')}  "
            f"wall={format_seconds(metrics.get('wall_ms'))}"
        )


def _print_execute_payload(payload: dict) -> None:
    metrics = payload.get("metrics") or {}
    print(
        f"  tool_calls={payload.get('tool_call_count')}  "
        f"wall={format_seconds(metrics.get('wall_ms'))}  "
        f"llm_calls={metrics.get('llm_calls')}  "
        f"tokens=p{metrics.get('prompt_tokens')}/c{metrics.get('completion_tokens')}/t{metrics.get('thoughts_tokens')}"
    )
    candidate = payload.get("submission_candidate")
    if isinstance(candidate, dict):
        print(
            f"  submission_candidate type={candidate.get('task_type_hint')}/"
            f"{candidate.get('answer_type')} source={candidate.get('source')}"
        )
    if payload.get("llm_raised"):
        print(f"  llm_raised={payload['llm_raised']}")


def _print_submit_payload(payload: dict, *, show_raw_eval_report: bool) -> None:
    extra = payload.get("extra") or {}
    print(
        f"  completion={payload.get('status')}  "
        f"eval={extra.get('passed')}/{extra.get('total')}  "
        f"failed={extra.get('failed')}  "
        f"report_parsed={extra.get('report_parsed')}"
    )
    if payload.get("block_reason"):
        print(f"  block_reason={payload['block_reason']}")
    if show_raw_eval_report:
        report_text = _extract_report_text(payload.get("evaluation_report"))
        if report_text:
            print("  evaluator_report:")
            print(_clean_ansi(report_text))


def _print_payload(
    phase_name: str,
    payload: dict,
    *,
    metrics: dict,
    show_raw_eval_report: bool,
) -> None:
    if phase_name == Phase.FIND.value:
        _print_find_payload(payload, metrics)
        return
    if phase_name == Phase.EXECUTE.value:
        _print_execute_payload(payload)
        return
    if phase_name == Phase.SUBMIT.value:
        _print_submit_payload(payload, show_raw_eval_report=show_raw_eval_report)
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2000])


def _write_subagent_io_markdown(
    log_dir: Path,
    artifact_dir: Path,
    subagent_outputs: dict[str, list[dict]],
) -> Path | None:
    markdown_sink = SubagentIoMarkdownSink(log_dir)
    jsonl_sink = SubagentIoJsonlSink(artifact_dir)
    bus = EventBus([markdown_sink, jsonl_sink])
    for event in subagent_completed_events(
        subagent_outputs,
        phase_order=list(LIVE_PHASE_ORDER),
        io_record_factory=io_record_from_subagent_output,
    ):
        bus.emit(event)
    jsonl_sink.flush()
    return markdown_sink.flush()


def _completed_task_output_paths(log_dir: Path) -> list[Path]:
    artifact_dir = log_dir / "artifacts"
    return [
        log_dir / "task_summary.txt",
        log_dir / "subagent_io.txt",
        artifact_dir / "task_summary.json",
        artifact_dir / "subagent_io.jsonl",
        artifact_dir / "workflow.jsonl",
        artifact_dir / "trajectory.jsonl",
        artifact_dir / "executor.jsonl",
        artifact_dir / "sandbox_api_calls.jsonl",
    ]


def _existing_completed_task_outputs(log_dir: Path) -> list[Path]:
    return [path for path in _completed_task_output_paths(log_dir) if path.exists()]


def _ensure_task_log_can_be_written(log_dir: Path, *, overwrite: bool) -> None:
    existing = _existing_completed_task_outputs(log_dir)
    if not existing or overwrite:
        return
    preview = ", ".join(str(path) for path in existing[:4])
    more = f", ... +{len(existing) - 4} more" if len(existing) > 4 else ""
    raise FileExistsError(
        "Existing completed task outputs found for this run_name/task_id. "
        "Use a new --run_name or pass --overwrite-task-log if replacement is intentional: "
        f"{preview}{more}"
    )


async def _run(args: argparse.Namespace) -> int:
    if not args.task_id:
        print("ERROR: --task_id is required (or set TASK_ID env).")
        return 2

    keep_debug = bool(getattr(args, "keep_debug", False))
    os.environ[LOG_DETAIL_ENV] = RAW_LOG_DETAIL if keep_debug else COMPACT_LOG_DETAIL
    if getattr(args, "config", None):
        run_config = load_run_config(args.config)
    else:
        run_config = _run_config_from_args(args)
    run_config = _apply_env_ablation_overrides(run_config)
    # Install the run's resolved config so every deep helper (retry ladders,
    # dependency graph, http timeout, ...) reads THIS run's settings.
    set_active_config(run_config)

    experiment_dir = Path(args.log_root) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    run_dir = experiment_dir / (args.run_name or log_timestamp())
    log_dir = run_dir / args.task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        _ensure_task_log_can_be_written(
            log_dir,
            overwrite=bool(getattr(args, "overwrite_task_log", False)),
        )
    except FileExistsError as exc:
        print(f"ERROR: {exc}")
        return 2
    artifact_dir = artifacts_dir(log_dir)
    # Record the exact resolved settings of this run (no un-logged env).
    dump_resolved_config(run_config, artifact_dir)
    timestamp = run_dir.name
    command_text = shlex.join([os.sys.executable, *os.sys.argv])
    command_path = artifact_dir / "command.txt"
    command_path.write_text(command_text + "\n", encoding="utf-8")
    workflow_path = artifact_dir / "workflow.jsonl"
    trajectory_path = artifact_dir / "trajectory.jsonl"
    executor_path = artifact_dir / "executor.jsonl"
    sandbox_trace_path = artifact_dir / "sandbox_api_calls.jsonl"
    timeline_path = os.getenv("SIMSA_PUBLIC_TIMELINE_PATH")
    timeline = (
        LiveTimeline(Path(timeline_path), sandbox_trace_path)
        if timeline_path
        else None
    )
    tmp_executor = Path(
        tempfile.NamedTemporaryFile(
            prefix="executor_", suffix=".jsonl", delete=False
        ).name
    )

    print("=" * 60)
    print(f"Task ID         : {args.task_id}")
    print(f"Experiment      : {args.experiment_name}")
    print(f"RPC URL         : {args.rpc_url}")
    print(f"Command file    : {command_path}")
    if keep_debug:
        print(f"Workflow file   : {workflow_path}")
    print("=" * 60)

    client = AppWorldRpcClient(addr=args.rpc_url)
    world_run_name = build_world_run_name(
        experiment_name=args.experiment_name,
        run_name=run_dir.name,
        task_id=args.task_id,
    )
    bootstrap = load_task(
        args.task_id, world_run_name, rpc_url=args.rpc_url, client=client
    )
    if timeline is not None:
        timeline.record_task(args.task_id, bootstrap.task_instruction)
    print(f"\nTask Instruction:\n{bootstrap.task_instruction}\n")

    AppWorldClientHolder.set_client(client)
    AppWorldAuthHolder.set_auth_manager(AppWorldAuthManager(client))
    os.environ["EXECUTOR_LOG_PATH"] = str(executor_path if keep_debug else tmp_executor)
    # Sandbox API trace is the canonical analyst evidence for P4 cluster
    # diagnosis (search-result over-trust). Always persist to artifacts/, not
    # gated on --keep-debug. result_items field includes bounded snapshot of
    # each API call's full return — three-layer size cap in appworld_tools.
    os.environ["EXECUTOR_SANDBOX_LOG_PATH"] = str(sandbox_trace_path)
    task_t0 = time.monotonic()

    try:
        cache_layer = _build_cache_layer_from_env()
        if cache_layer is not None:
            cache_dir = os.getenv("SUBAGENT_CACHE_DIR", "")
            cache_read = os.getenv("SUBAGENT_CACHE_READ", "")
            print(
                f"Subagent cache: {cache_dir} (read={cache_read or 'none'})",
                flush=True,
            )
        invocation_source = f"run_task:{args.task_id}" if args.task_id else "run_task"
        print(
            "orchestration_policy=cycle-based (LlmPolicy retired in step 4 cull)",
            flush=True,
        )
        agent = build_appworld_controller_agent(
            cache_layer=cache_layer,
            invocation_source=invocation_source,
            run_config=run_config,
        )
        initial_state = _seed_run_state(
            args.task_id,
            bootstrap.task_instruction,
            bootstrap.task_datetime,
        )
        if timeline is not None:
            timeline.record_state(initial_state[RUN_STATE_KEY])

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name=args.experiment_name,
            user_id=args.experiment_name,
            session_id=args.task_id,
            state=initial_state,
        )

        runner = Runner(
            agent=agent,
            session_service=session_service,
            app_name=args.experiment_name,
        )

        import sys

        timeout_hit = False
        seen_outputs: set[tuple[str, str, str]] = set()
        last_state_line: str | None = None
        terminal_bus = EventBus([TerminalEventPrinter()])
        sink = trajectory_path.open("w", encoding="utf-8") if keep_debug else None
        # Cumulative rate-limit (429/provider) backoff time granted as grace —
        # excluded from the task wall budget so infra throttling cannot become a
        # spurious TASK_TIMEOUT. Recorded in the summary for observability.
        rate_limit_grace_holder = [0.0]
        try:
            try:
                async with asyncio.timeout(args.timeout) as _timeout_cm:

                    def _extend_deadline(secs: float) -> None:
                        # Push the wall deadline forward by the rate-limit wait,
                        # so 429 backoff is not charged against agent compute.
                        rate_limit_grace_holder[0] += secs
                        when = _timeout_cm.when()
                        if when is not None:
                            _timeout_cm.reschedule(when + secs)

                    set_rate_limit_grace_extender(_extend_deadline)
                    async for event in runner.run_async(
                        user_id=args.experiment_name,
                        session_id=args.task_id,
                        new_message=types.Content(
                            parts=[types.Part(text=bootstrap.task_instruction)]
                        ),
                    ):
                        progress_lines, last_state_line = _event_progress_lines(
                            event,
                            seen_outputs=seen_outputs,
                            last_state_line=last_state_line,
                            terminal_bus=terminal_bus,
                            timeline=timeline,
                        )
                        for line in progress_lines:
                            print(line, flush=True)

                        if not keep_debug:
                            continue
                        text_part = (
                            event.content.parts[0].text
                            if event.content
                            and event.content.parts
                            and event.content.parts[0].text
                            else None
                        )
                        function_calls = []
                        function_responses = []
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if getattr(part, "function_call", None):
                                    function_calls.append(
                                        {
                                            "name": part.function_call.name,
                                            "args": dict(part.function_call.args or {}),
                                        }
                                    )
                                if getattr(part, "function_response", None):
                                    resp = part.function_response.response
                                    function_responses.append(
                                        {
                                            "name": part.function_response.name,
                                            "response_preview": str(resp)[:200],
                                        }
                                    )
                        record = {
                            "invocation_id": event.invocation_id,
                            "author": event.author,
                            "state_delta_keys": sorted(
                                (event.actions.state_delta or {}).keys()
                            )
                            if event.actions
                            else [],
                            "has_content": event.content is not None,
                            "text": text_part,
                            "function_calls": function_calls,
                            "function_responses": function_responses,
                        }
                        if sink is not None:
                            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                            sink.flush()
                        print(_event_line(record), flush=True)
                        sys.stdout.flush()
            except TimeoutError:
                timeout_hit = True
                grace = rate_limit_grace_holder[0]
                grace_note = (
                    f" (excludes {grace:.0f}s rate-limit grace)" if grace > 0 else ""
                )
                print(
                    f"\nTIMEOUT: task exceeded {args.timeout:.0f}s of compute{grace_note}",
                    flush=True,
                )
        finally:
            set_rate_limit_grace_extender(None)
            if sink is not None:
                sink.close()

        session = await session_service.get_session(
            app_name=args.experiment_name,
            user_id=args.experiment_name,
            session_id=args.task_id,
        )
        ledger = (session.state or {}).get(LEDGER_KEY, [])
        final_state = (session.state or {}).get(RUN_STATE_KEY, {})

        print("\n--- Ledger ---")
        for entry in ledger:
            state_after = (
                entry.get("state_after")
                if isinstance(entry.get("state_after"), dict)
                else {}
            )
            print(
                f"{entry['phase_decision']:<45}"
                f"failure_code={entry.get('failure_code')}  "
                f"budget={state_after.get('budget_snapshot')}"
            )

        if keep_debug:
            print("\n--- Final RunState ---")
            print(json.dumps(final_state, indent=2, ensure_ascii=False))
        else:
            print(
                "\nFinal RunState: "
                f"phase={final_state.get('phase')} "
                f"status={final_state.get('status')} "
                f"block={final_state.get('completion_block_reason')}"
            )

        subagent_outputs = (session.state or {}).get(SUBAGENT_OUTPUTS_KEY, {})
        subagent_io_path = _write_subagent_io_markdown(
            log_dir, artifact_dir, subagent_outputs
        )
        artifact_paths: dict[str, str] = {}
        if keep_debug:
            ledger_path = write_jsonl(artifact_dir / "ledger.jsonl", ledger)
            artifact_paths["ledger_path"] = str(ledger_path)
            artifact_paths["trajectory_path"] = str(trajectory_path)
            artifact_paths["executor_path"] = str(executor_path)
            if sandbox_trace_path.exists():
                artifact_paths["sandbox_api_calls_path"] = str(sandbox_trace_path)
        evaluation_report_path = None
        submit_items_for_report = subagent_outputs.get(Phase.SUBMIT.value, [])
        if (keep_debug or args.show_raw_eval_report) and submit_items_for_report:
            submit_payload = submit_items_for_report[-1].get("payload") or {}
            report_path = write_evaluation_report(
                artifact_dir,
                timestamp,
                submit_payload.get("evaluation_report"),
            )
            evaluation_report_path = (
                str(report_path) if report_path is not None else None
            )

        if keep_debug:
            print("\n--- Subagent outputs (per phase) ---")
            ordered_phase_names = [
                phase_name
                for phase_name in LIVE_PHASE_ORDER
                if phase_name in subagent_outputs
            ] + [
                phase_name
                for phase_name in sorted(subagent_outputs)
                if phase_name not in LIVE_PHASE_ORDER
            ]
            for phase_name in ordered_phase_names:
                items = subagent_outputs.get(phase_name) or []
                for item in items:
                    milestone = _milestone_summary(item)
                    milestone_text = f" {milestone}" if milestone else ""
                    print(
                        f"[{phase_name}] subagent={item.get('subagent_name')} "
                        f"status={item.get('status')} failure={item.get('failure_code')}"
                        f"{milestone_text}"
                    )
                    payload = item.get("payload") or {}
                    if payload:
                        _print_payload(
                            phase_name,
                            payload,
                            metrics=item.get("metrics") or {},
                            show_raw_eval_report=args.show_raw_eval_report,
                        )

        executor_records = read_jsonl(executor_path if keep_debug else tmp_executor)
        sandbox_records = read_jsonl(sandbox_trace_path)
        if keep_debug:
            debug_sink = DebugArtifactsSink(
                log_dir,
                workflow_path=workflow_path,
                trajectory_path=trajectory_path,
                executor_path=executor_path,
                sandbox_trace_path=sandbox_trace_path,
            )
            debug_sink.write_workflow(
                ledger=ledger,
                subagent_outputs=subagent_outputs,
                executor_trace=executor_records,
                sandbox_trace=sandbox_records,
            )
        counts = None
        submit_items = subagent_outputs.get(Phase.SUBMIT.value, [])
        if submit_items:
            submit_payload = submit_items[-1].get("payload") or {}
            counts = _extract_eval_counts(
                _extract_report_text(submit_payload.get("evaluation_report"))
            )
        latest_policy_source = next(
            (
                entry.get("policy_source")
                for entry in reversed(ledger)
                if isinstance(entry, dict) and entry.get("policy_source")
            ),
            None,
        )
        # Unified 429 label — TIMED_OUT-style status so the post-batch
        # retry flow can scan for it. Wall timeout takes precedence (the
        # harder stop); both classes are rerun-only, never analyzed as
        # agent failures.
        rate_limit_hit = (
            final_state.get("completion_block_reason") == "PROVIDER_RATE_LIMITED"
        )
        result = {
            "task_id": args.task_id,
            "phase": final_state.get("phase"),
            "status": (
                "TIMED_OUT"
                if timeout_hit
                else "RATE_LIMITED"
                if rate_limit_hit
                else final_state.get("status")
            ),
            "block_reason": "TASK_TIMEOUT"
            if timeout_hit
            else final_state.get("completion_block_reason"),
            "exec_failure": (
                (subagent_outputs.get(Phase.EXECUTE.value) or [{}])[-1].get(
                    "failure_code"
                )
            ),
            "submit_block": (
                (subagent_outputs.get(Phase.SUBMIT.value) or [{}])[-1].get("payload")
                or {}
            ).get("block_reason"),
            "passed": counts[0] if counts else None,
            "failed": counts[1] if counts else None,
            "total": counts[2] if counts else None,
            "task_wall_ms": int((time.monotonic() - task_t0) * 1000),
            "rate_limit_grace_s": round(rate_limit_grace_holder[0], 1),
            "policy_source": latest_policy_source,
        }
        task_summary = build_task_summary(
            task_id=args.task_id,
            result=result,
            final_state=final_state,
            subagent_outputs=subagent_outputs,
            sandbox_trace=sandbox_records,
            workflow_path=str(workflow_path) if keep_debug else None,
            evaluation_report_path=evaluation_report_path,
            command=command_text,
        )
        if timeline is not None:
            timeline.record_result(task_summary)
        task_summary_sink = TaskSummaryMarkdownSink(log_dir)
        EventBus([task_summary_sink]).emit(
            TaskCompleted(
                task_id=args.task_id,
                summary=task_summary,
                subagent_outputs=subagent_outputs,
            )
        )
        task_summary_path = task_summary_sink.flush()

        # Machine-readable sidecar for run_batch.py to parse without re-deriving
        # data from markdown.
        task_summary_json_path = artifact_dir / "task_summary.json"
        task_summary_json_path.write_text(
            json.dumps(task_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"\nWrote {len(ledger)} ledger entries.")
        if keep_debug:
            print(f"Workflow -> {workflow_path}")
        if keep_debug and executor_path.exists():
            print(f"Executor trace -> {executor_path}")
        if keep_debug and artifact_paths.get("sandbox_api_calls_path"):
            print(f"Sandbox API trace -> {artifact_paths['sandbox_api_calls_path']}")
        if keep_debug and artifact_paths.get("ledger_path"):
            print(f"Ledger file -> {artifact_paths['ledger_path']}")
        if subagent_io_path is not None:
            print(f"Subagent IO -> {subagent_io_path}")
        if evaluation_report_path is not None:
            print(f"Evaluator report -> {evaluation_report_path}")
        if task_summary_path is not None:
            print(f"Task summary -> {task_summary_path}")
        return 124 if timeout_hit else 0
    finally:
        os.environ.pop("EXECUTOR_LOG_PATH", None)
        os.environ.pop("EXECUTOR_SANDBOX_LOG_PATH", None)
        os.environ.pop(LOG_DETAIL_ENV, None)
        try:
            if tmp_executor.exists() and not keep_debug:
                tmp_executor.unlink()
        except OSError:
            pass
        close_world_safely(client)
        AppWorldAuthHolder.reset()
        AppWorldClientHolder.reset()
        prune_experiment_runs(experiment_dir, keep_last=5)


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
