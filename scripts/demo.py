"""Preview SIMSA with Python alone, or run one real AppWorld task locally."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

from scripts.demo_dashboard import make_server, read_events


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_ID = "13547f5_2"


def _required_runtime() -> tuple[Path, Path]:
    from adk_appworld_agent.repo_paths import load_repository_env

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


def _append_event(path: Path, event_type: str, **details: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": event_type, "at": time.time(), **details}, ensure_ascii=False) + "\n")


def _preview(path: Path) -> int:
    """Show illustrative data only, never a stored AppWorld trajectory."""
    events = [
        ("task", {"task_id": "EXAMPLE · NOT AN APPWORLD TASK", "instruction": "Illustrative task: find two products in a catalog and draft a short comparison."}),
        ("phase_started", {"phase": "PLAN"}),
        ("plan", {"milestones": [{"app": "catalog", "task": "Find matching products"}, {"app": "notes", "task": "Save a concise comparison"}]}),
        ("phase_started", {"phase": "FIND", "milestone_index": 0}),
        ("retrieval", {"apps": ["catalog"], "apis": ["catalog.search_products", "catalog.get_product"], "candidate_count": 2}),
        ("phase_started", {"phase": "EXECUTE", "milestone_index": 0}),
        ("execution", {"summary": "Found two products matching the requested criteria.", "called_apis": ["catalog.search_products", "catalog.get_product"], "tool_call_count": 2, "milestone_done": True}),
        ("control", {"action": "continue", "reason": "The catalog lookup is complete; the comparison still needs to be saved."}),
        ("phase_started", {"phase": "FIND", "milestone_index": 1}),
        ("retrieval", {"apps": ["notes"], "apis": ["notes.create_note"], "candidate_count": 1}),
        ("phase_started", {"phase": "EXECUTE", "milestone_index": 1}),
        ("execution", {"summary": "Saved the comparison note.", "called_apis": ["notes.create_note"], "tool_call_count": 1, "milestone_done": True}),
        ("control", {"action": "submit", "reason": "Both planned steps are complete."}),
        ("phase_started", {"phase": "SUBMIT"}),
        ("submission", {"status": "Submitted"}),
        ("preview_complete", {}),
    ]
    for event_type, details in events:
        _append_event(path, event_type, **details)
        time.sleep(0.45)
    return 0


def _run_live(task_id: str, timeline_path: Path) -> int:
    """Start AppWorld, run the model, then retain a static local record."""
    from scripts.build_task_viewer import build_viewer

    appworld_root, appworld_python = _required_runtime()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    experiment_name = f"portfolio_demo_{run_name}"
    run_dir = ROOT / "logs" / experiment_name / "run"
    run_dir.mkdir(parents=True)
    port = _free_local_port()
    server_env = os.environ.copy()
    server_env.update({"APPWORLD_ROOT": str(appworld_root), "RPC_PORT": str(port)})
    server_log_path = run_dir / "appworld_server.log"
    _append_event(timeline_path, "phase_started", phase="BOOTSTRAP", status="Starting AppWorld")
    with server_log_path.open("w", encoding="utf-8") as server_log:
        server = subprocess.Popen(
            [str(appworld_python), str(ROOT / "scripts" / "appworld_rpc_server.py")],
            cwd=appworld_root, env=server_env, stdout=server_log, stderr=subprocess.STDOUT,
        )
        try:
            _wait_for_server(server, port)
            command = [
                sys.executable, str(ROOT / "scripts" / "run_task.py"),
                "--task_id", task_id,
                "--config", str(ROOT / "configs" / "portfolio_smoke.json"),
                "--experiment_name", experiment_name,
                "--run_name", "run",
                "--rpc_url", f"tcp://127.0.0.1:{port}",
                "--log_root", str(ROOT / "logs"),
            ]
            run_env = os.environ.copy()
            run_env["SIMSA_PUBLIC_TIMELINE_PATH"] = str(timeline_path)
            with (run_dir / "agent_console.log").open("w", encoding="utf-8") as agent_log:
                result = subprocess.run(command, cwd=ROOT, env=run_env, stdout=agent_log, stderr=subprocess.STDOUT, check=False)
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
        print(f"\nStatic walkthrough: {view_path}", flush=True)
        print(f"Task summary: {summary_path}", flush=True)
    else:
        print(f"\nNo task summary. Inspect {run_dir / 'agent_console.log'}", file=sys.stderr, flush=True)
    return result.returncode


def _serve(mode: str, task_id: str, *, open_browser: bool, exit_after_run: bool) -> int:
    timeline_dir = Path(tempfile.mkdtemp(prefix="simsa-dashboard-")) if mode == "preview" else ROOT / "logs" / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    timeline_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = timeline_dir / "events.jsonl"
    server = make_server(timeline_path, mode=mode)
    url = f"http://127.0.0.1:{server.server_port}/"
    result = {"code": 0}

    def work() -> None:
        try:
            result["code"] = _preview(timeline_path) if mode == "preview" else _run_live(task_id, timeline_path)
            if result["code"] and not any(event.get("type") == "evaluation" for event in read_events(timeline_path)):
                _append_event(timeline_path, "error", message="The agent stopped early. Inspect the local run logs.")
        except (RuntimeError, OSError) as exc:
            result["code"] = 2
            _append_event(timeline_path, "error", message=str(exc))
            print(f"Demo error: {exc}", file=sys.stderr, flush=True)
        finally:
            if exit_after_run:
                server.shutdown()

    print(f"{mode.upper()} dashboard: {url}", flush=True)
    if mode == "live":
        print(f"Local timeline: {timeline_path}", flush=True)
    if open_browser:
        webbrowser.open(url)
    worker = threading.Thread(target=work, daemon=False)
    worker.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping dashboard. Run logs remain local.", flush=True)
    finally:
        server.server_close()
        worker.join()
    return result["code"]


def _serve_replay(path: Path, *, open_browser: bool) -> int:
    if not path.is_file() or not read_events(path):
        print(f"No timeline events found at {path}", file=sys.stderr)
        return 2
    server = make_server(path, mode="replay")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"RECORDED LIVE RUN: {url}", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preview", action="store_true", help="Illustrative UI with Python standard library only")
    group.add_argument("--live", action="store_true", help="Real AppWorld task and model calls")
    group.add_argument("--replay", type=Path, metavar="EVENTS_JSONL", help="View a locally recorded live run")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID, help="AppWorld task for --live")
    parser.add_argument("--no-open", action="store_true", help="Do not open the system browser")
    parser.add_argument("--exit-after-run", action="store_true", help="Exit when the task finishes (for automation)")
    args = parser.parse_args()
    if args.replay is not None:
        return _serve_replay(args.replay, open_browser=not args.no_open)
    return _serve("preview" if args.preview else "live", args.task_id, open_browser=not args.no_open, exit_after_run=args.exit_after_run)


if __name__ == "__main__":
    raise SystemExit(main())
