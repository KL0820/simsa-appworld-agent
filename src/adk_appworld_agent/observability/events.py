from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OrchestrationEvent:
    """Marker base for events consumed by observability sinks."""


@dataclass(frozen=True)
class SubagentCompleted(OrchestrationEvent):
    phase_name: str
    io_record: dict
    raw_output: dict


@dataclass(frozen=True)
class TaskCompleted(OrchestrationEvent):
    task_id: str
    summary: dict
    subagent_outputs: dict[str, list[dict]]


@dataclass(frozen=True)
class BatchStarted(OrchestrationEvent):
    """Emitted once at batch start. Lets sinks initialize the pending task list."""

    run_name: str
    config: dict  # plan/find/execute impls, model, policy, etc.
    tasks: list[dict]  # each: {task_id, instruction}


@dataclass(frozen=True)
class BatchTaskCompleted(OrchestrationEvent):
    """Emitted after each task subprocess returns."""

    task_id: str
    instruction: str
    index: int  # 1-based position within the batch
    total: int
    summary: (
        dict | None
    )  # task_summary content, or None if subprocess failed before producing one
    returncode: int


@dataclass(frozen=True)
class BatchFinished(OrchestrationEvent):
    run_name: str


class EventSink(Protocol):
    def handle(self, event: OrchestrationEvent) -> None: ...


class EventBus:
    def __init__(self, sinks: list[EventSink] | None = None) -> None:
        self._sinks = list(sinks or [])

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def emit(self, event: OrchestrationEvent) -> None:
        for sink in self._sinks:
            sink.handle(event)


def subagent_completed_events(
    subagent_outputs: dict[str, list[dict]],
    *,
    phase_order: list[str],
    io_record_factory,
) -> list[SubagentCompleted]:
    ordered_phase_names = [
        phase_name for phase_name in phase_order if phase_name in subagent_outputs
    ] + [phase_name for phase_name in subagent_outputs if phase_name not in phase_order]
    events: list[SubagentCompleted] = []
    for phase_name in ordered_phase_names:
        for item in subagent_outputs.get(phase_name) or []:
            if not isinstance(item, dict):
                continue
            record = io_record_factory(phase_name, item)
            if record is not None:
                events.append(
                    SubagentCompleted(
                        phase_name=phase_name,
                        io_record=record,
                        raw_output=item,
                    )
                )
    return events
