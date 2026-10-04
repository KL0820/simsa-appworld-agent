"""Local, dependency-free HTTP dashboard for a SIMSA event timeline."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from scripts.timeline_stages import build_stages


ASSETS = Path(__file__).resolve().parent / "assets"
ASSET_TYPES = {
    "/": ("dashboard.html", "text/html; charset=utf-8"),
    "/dashboard.css": ("dashboard.css", "text/css; charset=utf-8"),
    "/dashboard.js": ("dashboard.js", "text/javascript; charset=utf-8"),
}


def read_events(path: Path) -> list[dict]:
    """Return complete JSONL records; ignore a line still being written."""
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def make_server(timeline_path: Path, *, mode: str, port: int = 0) -> ThreadingHTTPServer:
    """Bind a dashboard on loopback; all browser content stays on this machine."""
    if mode not in {"preview", "live", "replay"}:
        raise ValueError("mode must be preview, live, or replay")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/api/events":
                events = read_events(timeline_path)
                body = json.dumps(
                    {"mode": mode, "events": events, "stages": build_stages(events)},
                    ensure_ascii=False,
                ).encode("utf-8")
                self._reply(body, "application/json; charset=utf-8")
                return
            asset = ASSET_TYPES.get(route)
            if asset is None:
                self.send_error(404)
                return
            file_name, mime_type = asset
            self._reply((ASSETS / file_name).read_bytes(), mime_type)

        def _reply(self, body: bytes, mime_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; style-src 'self'; script-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)
