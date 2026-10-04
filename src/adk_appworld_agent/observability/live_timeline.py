"""Write a small, allowlisted event stream for the local demo dashboard.

The research logs contain prompts, API arguments, and sandbox results. This
stream intentionally contains only task stages, milestone text, API names, and
the benchmark evaluator counts. It is still local task data, not a public log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _short(value: object, limit: int = 400) -> str:
    if not isinstance(value, str):
        return ""
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def _milestones(tasks: object) -> list[dict[str, str]]:
    if not isinstance(tasks, list):
        return []
    return [
        {"app": _short(task.get("app"), 80), "task": _short(task.get("task"))}
        for task in tasks
        if isinstance(task, dict)
    ]


def _api_names(candidates: object) -> list[str]:
    if not isinstance(candidates, list):
        return []
    names: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        app = _short(candidate.get("app"), 80)
        name = _short(candidate.get("name"), 120)
        if name:
            names.append(f"{app}.{name}" if app and not name.startswith(f"{app}.") else name)
    return list(dict.fromkeys(names))


class LiveTimeline:
    """Append safe dashboard events as each agent stage finishes."""

    def __init__(self, path: Path, sandbox_trace_path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        self.path = path
        self.sandbox_trace_path = sandbox_trace_path
        self._last_state: tuple[object, ...] | None = None
        self._sandbox_lines_seen = 0

    def emit(self, event_type: str, **details: object) -> None:
        event = {"type": event_type, "at": time.time(), **details}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def record_task(self, task_id: str, instruction: str) -> None:
        self.emit("task", task_id=_short(task_id, 80), instruction=_short(instruction, 1200))

    def record_state(self, state: dict) -> None:
        phase = state.get("phase")
        active_index = state.get("active_milestone_index")
        status = state.get("status")
        signature = (phase, active_index, status)
        if not isinstance(phase, str) or signature == self._last_state:
            return
        self._last_state = signature
        self.emit(
            "phase_started",
            phase=phase,
            milestone_index=active_index if isinstance(active_index, int) else None,
            status=_short(status, 80),
        )

    def record_output(self, phase: str, output: dict) -> None:
        payload = output.get("payload") or {}
        if not isinstance(payload, dict):
            return
        if "tasks" in payload:
            self.emit("plan", milestones=_milestones(payload.get("tasks")))
        elif "candidate_apis" in payload:
            matched_apps = payload.get("matched_apps")
            self.emit(
                "retrieval",
                apps=[_short(app, 80) for app in matched_apps] if isinstance(matched_apps, list) else [],
                apis=_api_names(payload.get("candidate_apis")),
                candidate_count=payload.get("candidate_count"),
            )
        elif "code_execute" in payload:
            result = payload.get("executor_result") or {}
            self.emit(
                "execution",
                summary=_short(result.get("summary")) if isinstance(result, dict) else "",
                called_apis=self._new_called_apis(),
                tool_call_count=payload.get("tool_call_count"),
                milestone_done=bool(payload.get("milestone_done")),
            )
        elif "next_action" in payload:
            self.emit(
                "control",
                action=_short(payload.get("next_action"), 120),
                reason=_short(payload.get("rationale")),
            )
        elif phase == "SUBMIT":
            self.emit("submission", status=_short(payload.get("status"), 120))

    def record_result(self, summary: dict) -> None:
        evaluation = summary.get("eval") or {}
        self.emit(
            "evaluation",
            status=_short(summary.get("status"), 120),
            passed=evaluation.get("passed"),
            total=evaluation.get("total"),
            passed_all=evaluation.get("passed_all"),
        )

    def _new_called_apis(self) -> list[str]:
        if not self.sandbox_trace_path.is_file():
            return []
        lines = self.sandbox_trace_path.read_text(encoding="utf-8").splitlines()
        new_lines = lines[self._sandbox_lines_seen :]
        self._sandbox_lines_seen = len(lines)
        names: list[str] = []
        for line in new_lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            for call in record.get("api_calls") or []:
                if not isinstance(call, dict):
                    continue
                app = _short(call.get("app"), 80)
                name = _short(call.get("api_name"), 120)
                if name:
                    names.append(f"{app}.{name}" if app and not name.startswith(f"{app}.") else name)
        return list(dict.fromkeys(names))
