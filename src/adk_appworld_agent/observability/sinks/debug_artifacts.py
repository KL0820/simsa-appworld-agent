from __future__ import annotations

from pathlib import Path

from adk_appworld_agent.observability.log_files import write_jsonl
from adk_appworld_agent.observability.workflow import build_workflow_records


class DebugArtifactsSink:
    def __init__(
        self,
        log_dir: Path,
        *,
        workflow_path: Path,
        trajectory_path: Path,
        executor_path: Path,
        sandbox_trace_path: Path,
    ) -> None:
        self.log_dir = log_dir
        self.workflow_path = workflow_path
        self.trajectory_path = trajectory_path
        self.executor_path = executor_path
        self.sandbox_trace_path = sandbox_trace_path

    def write_workflow(
        self,
        *,
        ledger: list[dict],
        subagent_outputs: dict[str, list[dict]],
        executor_trace: list[dict],
        sandbox_trace: list[dict],
    ) -> Path:
        records = build_workflow_records(
            ledger=ledger,
            subagent_outputs=subagent_outputs,
            executor_trace=executor_trace,
            sandbox_trace=sandbox_trace,
        )
        return write_jsonl(self.workflow_path, records)
