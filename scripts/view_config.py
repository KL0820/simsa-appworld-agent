"""Open an interactive browser launcher for AppWorld batch runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from adk_appworld_agent.config_viewer import (
    apply_cli_overrides,
    build_launcher_view,
    discover_config_presets,
    render_config_html,
)
from adk_appworld_agent.orchestration.run_config import RunConfig
from adk_appworld_agent.subagents.registry import available_subagent_impls
from adk_appworld_agent.task_sets import DATASET_CHOICES, VARIANT_CHOICES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs"
_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--experiment_name")
    parser.add_argument("--dataset", choices=DATASET_CHOICES)
    parser.add_argument("--variant", choices=VARIANT_CHOICES, default="full")
    parser.add_argument("--extra_task_ids")
    parser.add_argument("--rpc_url", default="tcp://127.0.0.1:4242")
    parser.add_argument("--log_root", default="logs")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)

    # Explicit launcher CLI overrides. None means "fall through to preset/default".
    parser.add_argument("--model_name")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--candidate_count", type=int)
    parser.add_argument("--max_output_tokens", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--find", dest="find_impl")
    parser.add_argument("--plan", dest="plan_impl")
    parser.add_argument("--verify", dest="verify_impl")
    parser.add_argument("--execute", dest="execute_impl")
    parser.add_argument(
        "--skills",
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
    )
    parser.add_argument("--skills_root")
    parser.add_argument("--skills_prompt", choices=["minimal", "full"])
    parser.add_argument(
        "--keep-debug", dest="keep_debug", action="store_true", default=None
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the launcher HTML and exit. Start/validate require the live server.",
    )
    return parser.parse_args()


def _base_config(config_path: Path | None) -> RunConfig:
    return RunConfig.from_file(config_path) if config_path else RunConfig()


def _cli_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "model_name",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "candidate_count",
        "max_output_tokens",
        "timeout",
        "find_impl",
        "plan_impl",
        "verify_impl",
        "execute_impl",
        "skills",
        "skills_root",
        "skills_prompt",
        "keep_debug",
    )
    return {name: getattr(args, name) for name in names}


def _experiment_name(config_path: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    return config_path.stem if config_path else "appworld_run"


def validate_launch_payload(payload: object) -> tuple[RunConfig, dict[str, Any]]:
    """Validate browser input before any file or process side effect."""
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    try:
        config = RunConfig.model_validate(payload.get("config"))
    except ValidationError as exc:
        raise ValueError(f"RunConfig validation failed: {exc}") from exc
    for phase, impl in config.impls.items():
        if impl not in available_subagent_impls(phase):
            raise ValueError(
                f"Unknown {phase.value} implementation: {impl}. "
                f"Available: {', '.join(available_subagent_impls(phase))}"
            )

    raw_batch = payload.get("batch")
    if not isinstance(raw_batch, dict):
        raise ValueError("batch must be a JSON object.")
    batch = {
        "experiment_name": str(raw_batch.get("experiment_name") or "").strip(),
        "dataset": str(raw_batch.get("dataset") or "").strip(),
        "variant": str(raw_batch.get("variant") or "full").strip(),
        "extra_task_ids": str(raw_batch.get("extra_task_ids") or "").strip(),
        "rpc_url": str(raw_batch.get("rpc_url") or "").strip(),
        "log_root": str(raw_batch.get("log_root") or "").strip(),
        "offset": int(raw_batch.get("offset") or 0),
        "limit": int(raw_batch.get("limit") or 0),
    }
    if not batch["experiment_name"] or not _EXPERIMENT_RE.fullmatch(
        batch["experiment_name"]
    ):
        raise ValueError(
            "experiment_name may contain only letters, numbers, dot, underscore, and dash."
        )
    if batch["dataset"] not in DATASET_CHOICES:
        raise ValueError("Choose a supported dataset.")
    if batch["variant"] not in VARIANT_CHOICES:
        raise ValueError("Choose a supported variant.")
    if batch["dataset"] in {"quick_smoke", "extra_only"} and batch["variant"] != "full":
        raise ValueError(f"{batch['dataset']} only supports the full variant.")
    if batch["dataset"] == "extra_only" and not batch["extra_task_ids"]:
        raise ValueError("Enter at least one Extra Task ID.")
    if not batch["rpc_url"]:
        raise ValueError("rpc_url is required.")
    if not batch["log_root"]:
        raise ValueError("log_root is required.")
    if batch["offset"] < 0 or batch["limit"] < 0:
        raise ValueError("offset and limit must be non-negative.")
    config = config.model_copy(
        update={
            "log": config.log.model_copy(update={"log_root": Path(batch["log_root"])})
        }
    )
    return config, batch


def prepare_launch(
    payload: object,
    *,
    project_root: Path = PROJECT_ROOT,
    write_config: bool,
) -> dict[str, Any]:
    """Resolve browser input into one generated config and one batch command."""
    config, batch = validate_launch_payload(payload)
    config_path = (
        _write_generated_config(config, batch["experiment_name"], project_root)
        if write_config
        else project_root / "logs" / "_launcher" / "configs" / "<generated>.json"
    )
    command = _batch_command(config_path, batch, project_root)
    return {
        "config": config,
        "batch": batch,
        "config_path": config_path,
        "command": command,
        "command_text": shlex.join(command),
    }


def _write_generated_config(
    config: RunConfig,
    experiment_name: str,
    project_root: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (
        project_root
        / "logs"
        / "_launcher"
        / "configs"
        / f"{timestamp}_{experiment_name}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.to_json() + "\n", encoding="utf-8")
    return path


def _batch_command(
    config_path: Path,
    batch: dict[str, Any],
    project_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(project_root / "scripts" / "run_batch.py"),
        "--config",
        str(config_path),
        "--experiment_name",
        batch["experiment_name"],
        "--rpc_url",
        batch["rpc_url"],
        "--log_root",
        batch["log_root"],
        "--offset",
        str(batch["offset"]),
        "--limit",
        str(batch["limit"]),
    ]
    command.extend(["--dataset", batch["dataset"], "--variant", batch["variant"]])
    if batch["extra_task_ids"]:
        command.extend(["--extra_task_ids", batch["extra_task_ids"]])
    return command


class LauncherState:
    """Own the single child process started by one launcher server."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root
        self.process: subprocess.Popen[str] | None = None
        self.cancelled_pid: int | None = None
        self.lock = threading.Lock()

    def validate(self, payload: object) -> dict[str, Any]:
        launch = prepare_launch(
            payload, project_root=self.project_root, write_config=False
        )
        return {"command": launch["command_text"]}

    def launch(self, payload: object) -> dict[str, Any]:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError(
                    f"A batch is already running (PID {self.process.pid})."
                )
            launch = prepare_launch(
                payload, project_root=self.project_root, write_config=True
            )
            self.process = subprocess.Popen(
                launch["command"],
                cwd=self.project_root,
                text=True,
                start_new_session=True,
            )
            self.cancelled_pid = None
            return {
                "pid": self.process.pid,
                "config_path": str(launch["config_path"]),
                "command": launch["command_text"],
            }

    def cancel(self) -> dict[str, Any]:
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                raise RuntimeError("No batch is currently running.")
            returncode = _stop_process_group(process)
            self.cancelled_pid = process.pid
            return {
                "state": "cancelled",
                "pid": process.pid,
                "returncode": returncode,
            }

    def status(self) -> dict[str, Any]:
        with self.lock:
            if self.process is None:
                return {"state": "idle"}
            returncode = self.process.poll()
            state = "running" if returncode is None else "finished"
            if returncode is not None and self.process.pid == self.cancelled_pid:
                state = "cancelled"
            return {
                "state": state,
                "pid": self.process.pid,
                "returncode": returncode,
            }


def _stop_process_group(
    process: subprocess.Popen[str],
    *,
    grace_s: float = 5.0,
) -> int:
    """Stop the launched batch and all descendants in its process group."""
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait()


def _serve(
    page: bytes,
    host: str,
    port: int,
    *,
    open_browser: bool,
    state: LauncherState,
) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send_bytes(200, "text/html; charset=utf-8", page)
                return
            if self.path == "/api/status":
                self._send_json(200, state.status())
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/api/cancel":
                try:
                    result = state.cancel()
                except RuntimeError as exc:
                    self._send_json(400, {"error": str(exc)})
                    return
                self._send_json(200, result)
                return
            if self.path not in ("/api/validate", "/api/launch"):
                self.send_error(404)
                return
            try:
                payload = self._read_json()
                result = (
                    state.validate(payload)
                    if self.path == "/api/validate"
                    else state.launch(payload)
                )
            except (ValueError, RuntimeError) as exc:
                self._send_json(400, {"error": str(exc)})
                return
            self._send_json(200, result)

        def _read_json(self) -> object:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("Invalid request body size.")
            return json.loads(self.rfile.read(length))

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._send_bytes(status, "application/json; charset=utf-8", body)

        def _send_bytes(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{server.server_port}/"
    print(f"AppWorld run launcher: {url}")
    print("CLI values override preset/default values in the initial form.")
    print("Press Ctrl-C to stop the launcher; a started batch continues independently.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    args = _parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "ERROR: the launcher can execute local commands and only binds to loopback hosts."
        )
        return 2
    config_path = args.config.resolve() if args.config else None
    raw_cli_overrides = _cli_config_overrides(args)
    config, cli_sources = apply_cli_overrides(
        _base_config(config_path),
        raw_cli_overrides,
    )
    presets = discover_config_presets(CONFIGS_DIR, selected_path=config_path)
    for preset in presets:
        preset_config, _ = apply_cli_overrides(
            RunConfig.model_validate(preset["config"]),
            raw_cli_overrides,
        )
        preset["config"] = preset_config.model_dump(mode="json")
    view = build_launcher_view(
        config,
        config_path=config_path,
        presets=presets,
        experiment_name=_experiment_name(config_path, args.experiment_name),
        dataset=args.dataset,
        variant=args.variant,
        extra_task_ids=args.extra_task_ids,
        rpc_url=args.rpc_url,
        log_root=args.log_root,
        offset=args.offset,
        limit=args.limit,
        cli_sources=cli_sources,
    )
    page = render_config_html(view)
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(page, encoding="utf-8")
        print(f"Wrote {output}")
        return 0
    _serve(
        page.encode("utf-8"),
        args.host,
        args.port,
        open_browser=not args.no_open,
        state=LauncherState(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
