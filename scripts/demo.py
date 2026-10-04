"""Run one AppWorld task and build a local HTML walkthrough.

This is a convenience launcher, not a replacement for AppWorld installation.
The walkthrough omits raw model and sandbox I/O, but task text should still be
reviewed before the HTML is shared.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from adk_appworld_agent.repo_paths import load_repository_env

from scripts.build_task_viewer import build_viewer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ID = "13547f5_2"


def _required_runtime() -> tuple[Path, Path]:
    load_repository_env()
    root_raw = os.getenv("APPWORLD_ROOT", "")
    root = Path(root_raw).expanduser() if root_raw else Path()
    if not root_raw or not (root / "src" / "appworld").is_dir() or not (root / "data").is_dir():
        raise RuntimeError("Set APPWORLD_ROOT in .env to an installed AppWorld checkout with src/appworld and data.")
    appworld_python = Path(os.getenv("APPWORLD_PYTHON") or root / ".venv" / "bin" / "python")
    if not appworld_python.is_file():
        raise RuntimeError("AppWorld Python was not found. Set APPWORLD_PYTHON in .env to its environment's python executable.")
    if not os.getenv("GOOGLE_API_KEY"):
        vertex_ready = (
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
            and os.getenv("GOOGLE_CLOUD_PROJECT")
            and os.getenv("GOOGLE_CLOUD_LOCATION")
        )
        if not vertex_ready:
            raise RuntimeError("Set GOOGLE_API_KEY in .env, or configure Vertex AI credentials as shown in .env.example.")
    return root, appworld_python


def _free_local_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, port: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("AppWorld RPC server exited during startup; inspect appworld_server.log.")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError("AppWorld RPC server did not become ready within 30 seconds; inspect appworld_server.log.")


def _run_demo(task_id: str) -> int:
    appworld_root, appworld_python = _required_runtime()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"portfolio_demo_{run_name}"
    run_dir = ROOT / "logs" / experiment_name / "run"
    run_dir.mkdir(parents=True)
    port = _free_local_port()
    server_env = os.environ.copy()
    server_env["APPWORLD_ROOT"] = str(appworld_root)
    server_env["RPC_PORT"] = str(port)
    server_log_path = run_dir / "appworld_server.log"
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [str(appworld_python), str(ROOT / "scripts" / "appworld_rpc_server.py")],
            cwd=appworld_root,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, port)
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_task.py"),
                "--task_id", task_id,
                "--config", str(ROOT / "configs" / "portfolio_smoke.json"),
                "--experiment_name", experiment_name,
                "--run_name", "run",
                "--rpc_url", f"tcp://127.0.0.1:{port}",
                "--log_root", str(ROOT / "logs"),
            ]
            result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=False)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    summary_path = run_dir / task_id / "artifacts" / "task_summary.json"
    if summary_path.is_file():
        view_path = build_viewer(summary_path, run_dir / task_id / "task_view.html")
        print(f"\nTask walkthrough: {view_path}")
        print(f"Task summary: {summary_path}")
        print("Review task text before sharing either file.")
    else:
        print(f"\nNo task summary was produced. Inspect logs under {run_dir}.", file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    args = parser.parse_args()
    try:
        return _run_demo(args.task_id)
    except RuntimeError as exc:
        print(f"Demo setup error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
