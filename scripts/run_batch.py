"""Batch-run AppWorld tasks through the controller."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

from adk_appworld_agent.repo_paths import load_repository_env

load_repository_env()

warnings.filterwarnings("ignore", message="authlib.jose module is deprecated.*")
warnings.filterwarnings(
    "ignore",
    message=r"\[EXPERIMENTAL\] feature FeatureName.PLUGGABLE_AUTH is enabled.*",
)
warnings.filterwarnings("ignore", category=Warning, module=r"authlib\._joserfc_helpers")
warnings.filterwarnings(
    "ignore", category=Warning, module=r"google\.adk\.features\._feature_decorator"
)

_showwarning = warnings.showwarning


def _show_relevant_warning(message, category, filename, lineno, file=None, line=None):
    text = str(message)
    if text.startswith("authlib.jose module is deprecated"):
        return
    if "[EXPERIMENTAL] feature FeatureName.PLUGGABLE_AUTH is enabled" in text:
        return
    _showwarning(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _show_relevant_warning

from adk_appworld_agent.appworld_paths import find_appworld_root
from adk_appworld_agent.observability.events import (
    BatchFinished,
    BatchStarted,
    BatchTaskCompleted,
    EventBus,
)
from adk_appworld_agent.observability.paths import artifacts_dir
from adk_appworld_agent.observability.sinks.batch_results import (
    BatchResultsMarkdownSink,
)
from adk_appworld_agent.observability.subagent_logs import TaskLogContext
from adk_appworld_agent.orchestration.run_config import load_run_config
from adk_appworld_agent.task_sets import (
    DATASET_CHOICES,
    VARIANT_CHOICES,
    task_set_path,
)

_APPWORLD_ROOT = find_appworld_root()
DEFAULT_TASKS_DIR = _APPWORLD_ROOT / "data" / "tasks"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run the full controller pipeline."
    )
    parser.add_argument("--dataset", choices=DATASET_CHOICES)
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="full")
    parser.add_argument(
        "--tasks_dir",
        type=str,
        default=os.getenv("APPWORLD_TASKS_DIR", str(DEFAULT_TASKS_DIR)),
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default=None,
        help="Comma-separated task ids for a targeted-only run.",
    )
    parser.add_argument(
        "--extra_task_ids",
        type=str,
        default=None,
        help="Comma-separated task ids appended to the selected dataset, with duplicates removed.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Reuse an existing run dir under <log_root>/<experiment_name> (resume).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap tasks from --dataset (0 = all). Explicit task ids are unaffected.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=os.getenv("EXPERIMENT_NAME", "rough_plan_refactor"),
    )
    parser.add_argument(
        "--find",
        dest="find_impl",
        default=os.getenv("FIND_IMPL", "community"),
        help="Phase.FIND subagent impl key.",
    )
    parser.add_argument(
        "--plan",
        dest="plan_impl",
        default=os.getenv("PLAN_IMPL", "rough"),
        help="Phase.PLAN subagent impl key.",
    )
    parser.add_argument(
        "--verify",
        dest="verify_impl",
        default=os.getenv("VERIFY_IMPL", "stub"),
        help="Phase.VERIFY subagent impl key. Default 'stub' is a no-op pass-through.",
    )
    parser.add_argument(
        "--execute",
        dest="execute_impl",
        default=os.getenv("EXECUTE_IMPL", "code_plan_execute"),
        help="Phase.EXECUTE subagent impl key.",
    )
    parser.add_argument(
        "--rpc_url",
        type=str,
        default=os.getenv("RPC_URL", "tcp://127.0.0.1:4242"),
        help="AppWorld RPC URL.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Log root passed to run_task.py.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("TASK_TIMEOUT_S", "2400")),
        help="Per-task timeout in seconds.",
    )
    parser.add_argument(
        "--model_name", type=str, default=os.getenv("MODEL_NAME", "gemini-2.5-flash")
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an experiment config file (JSON RunConfig); passed through "
        "to run_task.py as the single source of truth for this batch.",
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
        help="Pass through to run_task.py: skill library per task "
        "(default 'best' = thin+skill, the thesis method; 'off' = no-skill baseline).",
    )
    parser.add_argument(
        "--skills_root",
        type=str,
        default=os.getenv("SKILLS_ROOT", "data/mind_skill/skills/release/thesis_final"),
        help="Pass through to run_task.py.",
    )
    parser.add_argument(
        "--skills_prompt",
        default=os.getenv("SKILLS_PROMPT", "minimal"),
        choices=["minimal", "full"],
        help="Pass through to run_task.py: skill-injection prompt variant "
        "('minimal' frozen thin, default; 'full' skills on the constraint prompt).",
    )
    parser.add_argument(
        "--show_raw_eval_report",
        action="store_true",
        help="Pass through to run_task.py.",
    )
    parser.add_argument(
        "--keep-debug",
        dest="keep_debug",
        action="store_true",
        help="Pass through to run_task.py.",
    )
    return parser.parse_args()


def _parse_split_line(line: str) -> tuple[str, bool] | None:
    """Parse a single line of a split file.

    Format: `<task_id>` for a normal task, or `<task_id> SKIP [reason]` to mark
    the task as blacklisted (auto-fail without running). Returns
    (task_id, is_blacklisted), or None for empty / comment-only lines.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(None, 1)
    task_id = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    # Match 'SKIP' as a whole token (so a task whose name happens to contain
    # those letters would not falsely trigger; we tokenize the suffix).
    is_blacklisted = "SKIP" in rest.split()
    return task_id, is_blacklisted


def _load_task_ids(args: argparse.Namespace) -> list[tuple[str, bool]]:
    """Return list of (task_id, is_blacklisted) tuples.

    Base selection comes from explicit task ids, a fixed dataset+variant, or a
    legacy split file. Extra task ids are appended after offset/limit and
    deduplicated without changing the base order.
    """
    if getattr(args, "task_ids", None):
        items = [
            (task_id.strip(), False)
            for task_id in args.task_ids.split(",")
            if task_id.strip()
        ]
    else:
        items = _load_base_selection(args)

    extras = [
        task_id.strip()
        for task_id in (getattr(args, "extra_task_ids", None) or "").split(",")
        if task_id.strip()
    ]
    if not items and not extras:
        raise SystemExit("ERROR: --dataset, --task_ids, or --extra_task_ids required")
    return _append_unique_task_ids(items, extras)


def _load_base_selection(args: argparse.Namespace) -> list[tuple[str, bool]]:
    dataset = getattr(args, "dataset", None)
    if dataset == "extra_only":
        return []
    if dataset == "quick_smoke":
        if getattr(args, "variant", "full") != "full":
            raise SystemExit("ERROR: quick_smoke only supports variant=full")
        task_set_file = task_set_path("quick_smoke")
    else:
        if dataset is None:
            return []
        task_set_file = task_set_path(dataset)
    if not task_set_file.exists():
        raise SystemExit(f"ERROR: task set file not found: {task_set_file}")

    items: list[tuple[str, bool]] = []
    for line in task_set_file.read_text().splitlines():
        parsed = _parse_split_line(line)
        if parsed is not None:
            items.append(parsed)
    variant = getattr(args, "variant", "full")
    if dataset and variant != "full":
        items = [item for item in items if item[0].endswith(f"_{variant}")]
    limit = args.limit if args.limit > 0 else len(items)
    return items[args.offset : args.offset + limit]


def _append_unique_task_ids(
    items: list[tuple[str, bool]],
    extra_task_ids: list[str],
) -> list[tuple[str, bool]]:
    result = list(items)
    seen = {task_id for task_id, _ in items}
    for task_id in extra_task_ids:
        if task_id not in seen:
            result.append((task_id, False))
            seen.add(task_id)
    return result


def _load_task_specs(task_id: str, args: argparse.Namespace) -> tuple[str, str]:
    specs_path = Path(args.tasks_dir) / task_id / "specs.json"
    try:
        payload = json.loads(specs_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Task specs not found: {specs_path}") from exc

    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Task specs missing instruction: {specs_path}")

    task_datetime = payload.get("datetime")
    return instruction, task_datetime if isinstance(task_datetime, str) else ""


def _load_task_contexts(args: argparse.Namespace) -> list[TaskLogContext]:
    items = _load_task_ids(args)
    total = len(items)
    tasks: list[TaskLogContext] = []
    for index, (task_id, is_blacklisted) in enumerate(items, 1):
        instruction, task_datetime = _load_task_specs(task_id, args)
        tasks.append(
            TaskLogContext(
                task_id=task_id,
                instruction=instruction,
                task_datetime=task_datetime,
                index=index,
                total=total,
                is_blacklisted=is_blacklisted,
            )
        )
    return tasks


def _task_selection_record(
    args: argparse.Namespace,
    tasks: list[TaskLogContext],
) -> dict[str, object]:
    """Describe the reproducible task selection separately from model config."""
    return {
        "dataset": getattr(args, "dataset", None),
        "variant": getattr(args, "variant", "full"),
        "explicit_task_ids": getattr(args, "task_ids", None),
        "extra_task_ids": [
            task_id.strip()
            for task_id in (getattr(args, "extra_task_ids", None) or "").split(",")
            if task_id.strip()
        ],
        "offset": getattr(args, "offset", 0),
        "limit": getattr(args, "limit", 0),
        "resolved_task_ids": [task.task_id for task in tasks],
    }


_BLACKLIST_AGGREGATE_KEYS = (
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
    "retry_count",
    "retry_backoff_ms",
    "rate_limit_count",
    "provider_error_count",
)


def _write_blacklisted_summary(path: Path, task: TaskLogContext) -> dict:
    """Write a synthetic task_summary.json for a blacklisted task and return
    the same dict (callers feed it directly into BatchTaskCompleted to avoid
    re-reading via _load_fresh_task_summary, whose mtime check would reject
    a file written outside the subprocess timing window)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "datetime": task.task_datetime,
        "status": "FAILED",
        "phase": "BLACKLISTED",
        "wall_s": 0.0,
        "eval": None,
        "aggregate_metrics": {key: 0 for key in _BLACKLIST_AGGREGATE_KEYS},
        "overview": {
            "block_reason": "BLACKLISTED",
            "exec_failure": None,
        },
        "failures": [
            {
                "code": "BLACKLISTED",
                "message": (
                    "Task blacklisted in split file via 'SKIP' suffix; not executed."
                ),
            }
        ],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _default_run_dir(args: argparse.Namespace) -> Path:
    run_name = getattr(args, "run_name", None)
    if run_name:
        # resume path: reuse an existing batch run dir (used with --task_ids
        # to fill in the not-yet-completed tasks of an interrupted batch)
        return Path(getattr(args, "log_root", "logs")) / args.experiment_name / run_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return Path(getattr(args, "log_root", "logs")) / args.experiment_name / timestamp


def _task_log_dir(run_dir: Path, task_id: str) -> Path:
    return run_dir / task_id


def _model_config_record(args: argparse.Namespace) -> dict:
    if getattr(args, "config", None):
        model = load_run_config(args.config).model
        return {
            "name": model.name,
            "temperature": model.temperature,
            "top_p": model.top_p,
            "top_k": model.top_k,
            "seed": model.seed,
            "max_output_tokens": model.max_output_tokens or "provider_default",
        }
    max_tokens = (
        args.max_output_tokens if args.max_output_tokens > 0 else "provider_default"
    )
    return {
        "name": args.model_name,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "max_output_tokens": max_tokens,
    }


def _model_config_summary(args: argparse.Namespace) -> str:
    config = _model_config_record(args)
    return (
        f"name={config['name']}, temperature={config['temperature']}, top_p={config['top_p']}, "
        f"top_k={config['top_k']}, seed={config['seed']}, max_output_tokens={config['max_output_tokens']}"
    )


def _batch_config_record(args: argparse.Namespace) -> dict:
    """Describe the effective implementations, not ignored CLI defaults."""
    record = {
        "plan": args.plan_impl,
        "find": args.find_impl,
        "verify": args.verify_impl,
        "execute": args.execute_impl,
        "policy": "cycle-based",
        "cache_dir": os.getenv("SUBAGENT_CACHE_DIR", ""),
        "cache_read": os.getenv("SUBAGENT_CACHE_READ", ""),
    }
    if getattr(args, "config", None):
        config = load_run_config(args.config)
        # Missing implementations are resolved by the runtime registry, not
        # by CLI arguments (the JSON configuration is authoritative).
        for name in ("plan", "find", "verify", "execute"):
            record[name] = config.impls.get(name.upper(), "registry-default")
        record["cache_dir"] = str(config.cache.dir or "")
        record["cache_read"] = ",".join(config.cache.read)
    elif getattr(args, "skills", "best") != "off":
        native = getattr(args, "skills", "best").startswith("native_")
        record["plan"] = "rough_skill_native" if native else "rough_skill"
        record["execute"] = (
            "code_plan_execute_skill_native" if native else "code_plan_execute_skill"
        )
    return record


def _controller_command(
    task_id: str, args: argparse.Namespace, *, run_name: str
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_task.py",
        "--task_id",
        task_id,
        "--experiment_name",
        args.experiment_name,
        "--rpc_url",
        args.rpc_url,
        "--log_root",
        args.log_root,
        "--run_name",
        run_name,
        "--find",
        args.find_impl,
        "--plan",
        args.plan_impl,
        "--verify",
        args.verify_impl,
        "--execute",
        args.execute_impl,
        "--timeout",
        str(args.timeout),
        "--model_name",
        args.model_name,
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--seed",
        str(args.seed),
    ]
    if args.max_output_tokens > 0:
        cmd.extend(["--max_output_tokens", str(args.max_output_tokens)])
    skills = getattr(args, "skills", "best")
    if skills != "off":
        cmd.extend(
            [
                "--skills",
                skills,
                "--skills_root",
                str(
                    getattr(
                        args,
                        "skills_root",
                        "data/mind_skill/skills/release/thesis_final",
                    )
                ),
                "--skills_prompt",
                getattr(args, "skills_prompt", "minimal"),
            ]
        )
    if args.show_raw_eval_report:
        cmd.append("--show_raw_eval_report")
    if _keep_debug(args):
        cmd.append("--keep-debug")
    if getattr(args, "config", None):
        cmd.extend(["--config", args.config])
    return cmd


def _keep_debug(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "keep_debug", False))


def _copy_stream(pipe, log_file, terminal) -> None:
    try:
        for line in iter(pipe.readline, ""):
            log_file.write(line)
            log_file.flush()
            terminal.write(line)
            terminal.flush()
    finally:
        pipe.close()


def _run_logged_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> int:
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout_thread = threading.Thread(
            target=_copy_stream,
            args=(proc.stdout, stdout_file, sys.stdout),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_copy_stream,
            args=(proc.stderr, stderr_file, sys.stderr),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        returncode = proc.wait()
        stdout_thread.join()
        stderr_thread.join()
        return returncode


def _write_command_file(task_dir: Path, cmd: list[str]) -> Path:
    command_path = artifacts_dir(task_dir) / "command.txt"
    command_path.write_text(shlex.join(cmd) + "\n", encoding="utf-8")
    return command_path


def _load_fresh_task_summary(path: Path, *, started_at_epoch: float) -> dict | None:
    if not path.exists():
        return None
    try:
        if path.stat().st_mtime < started_at_epoch:
            return None
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return summary if isinstance(summary, dict) else None


async def _run_all_controller(args: argparse.Namespace) -> int:
    task_contexts = _load_task_contexts(args)
    run_dir = _default_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    selection_record = _task_selection_record(args, task_contexts)
    (run_dir / "task_selection.json").write_text(
        json.dumps(selection_record, indent=2) + "\n",
        encoding="utf-8",
    )
    run_name = run_dir.name
    workspace_root = Path(__file__).resolve().parents[1]

    # Write the canonical task list to logs/<exp>/task_ids.txt so /resume-batch
    # has a deterministic source-of-truth for "what was the original plan".
    # Only write on first invocation (resumes pass a subset and must not
    # overwrite). One task_id per line.
    exp_root = run_dir.parent
    task_ids_file = exp_root / "task_ids.txt"
    if not task_ids_file.exists():
        task_ids_file.write_text(
            "\n".join(tc.task_id for tc in task_contexts) + "\n",
            encoding="utf-8",
        )

    print("Controller batch mode", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Model config: {_model_config_summary(args)}", flush=True)
    batch_config = _batch_config_record(args)
    print(
        "Subagent impls: "
        + ", ".join(
            f"{name}={batch_config[name]}"
            for name in ("find", "plan", "verify", "execute")
        ),
        flush=True,
    )
    print(f"RPC URL: {args.rpc_url}", flush=True)
    print(
        "Task selection: "
        f"dataset={selection_record['dataset'] or '-'}, "
        f"variant={selection_record['variant']}, "
        f"extra={len(selection_record['extra_task_ids'])}, "
        f"resolved={len(selection_record['resolved_task_ids'])}",
        flush=True,
    )
    cache_dir_env = os.getenv("SUBAGENT_CACHE_DIR")
    cache_read_env = os.getenv("SUBAGENT_CACHE_READ", "")
    if cache_dir_env:
        print(
            f"Subagent cache: {cache_dir_env} (read={cache_read_env or 'none'})",
            flush=True,
        )
    print(
        "Per-task wrapper logs: <run directory>/<task_id>/artifacts/{stdout,stderr}.log",
        flush=True,
    )

    batch_results_sink = BatchResultsMarkdownSink(run_dir)
    bus = EventBus([batch_results_sink])
    bus.emit(
        BatchStarted(
            run_name=run_name,
            config=batch_config,
            tasks=[
                {"task_id": task.task_id, "instruction": task.instruction}
                for task in task_contexts
            ],
        )
    )
    print(f"Batch results: {batch_results_sink.path}", flush=True)

    blacklisted_ids = [task.task_id for task in task_contexts if task.is_blacklisted]
    if blacklisted_ids:
        print(
            f"Blacklisted ({len(blacklisted_ids)} task(s), auto-fail, no run): "
            f"{','.join(blacklisted_ids)}",
            flush=True,
        )

    failures = 0
    for task in task_contexts:
        task_dir = _task_log_dir(run_dir, task.task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir = artifacts_dir(task_dir)

        if task.is_blacklisted:
            print(
                f"\nTask ID: {task.task_id} ({task.index}/{task.total}) "
                f"-- BLACKLISTED, skipping",
                flush=True,
            )
            summary = _write_blacklisted_summary(
                artifact_dir / "task_summary.json", task
            )
            bus.emit(
                BatchTaskCompleted(
                    task_id=task.task_id,
                    instruction=task.instruction,
                    index=task.index,
                    total=task.total,
                    summary=summary,
                    returncode=0,
                )
            )
            failures += (
                1  # blacklisted entries count as failures for exit-code purposes
            )
            continue

        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"
        print(f"\nTask ID: {task.task_id} ({task.index}/{task.total})", flush=True)
        cmd = _controller_command(task.task_id, args, run_name=run_name)
        _write_command_file(task_dir, cmd)
        started_at_epoch = time.time()
        started = time.monotonic()
        returncode = _run_logged_command(
            cmd,
            cwd=workspace_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        wall_ms = int((time.monotonic() - started) * 1000)

        status = "SUCCEEDED" if returncode == 0 else "FAILED"
        print(
            f"<<< EXIT controller status={status} returncode={returncode} wall={wall_ms / 1000:.2f}s",
            flush=True,
        )
        if returncode != 0:
            failures += 1

        summary_path = artifact_dir / "task_summary.json"
        summary = _load_fresh_task_summary(
            summary_path,
            started_at_epoch=started_at_epoch,
        )
        bus.emit(
            BatchTaskCompleted(
                task_id=task.task_id,
                instruction=task.instruction,
                index=task.index,
                total=task.total,
                summary=summary,
                returncode=returncode,
            )
        )

    bus.emit(BatchFinished(run_name=run_name))
    return 1 if failures else 0


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run_all_controller(args))


if __name__ == "__main__":
    raise SystemExit(main())
