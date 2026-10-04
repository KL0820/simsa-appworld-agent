"""Turn low-level agent events into one readable card per workflow stage."""

from __future__ import annotations


PHASE_STAGE = {
    "BOOTSTRAP": "bootstrap",
    "FIND": "retrieval",
    "EXECUTE": "execution",
    "SUBMIT": "submission",
    "COMPLETE": "evaluation",
}
RESULT_TYPES = {"plan", "retrieval", "execution", "control", "submission", "evaluation"}


def _pending_stage(event: dict, stage_type: str) -> dict:
    return {
        "type": stage_type,
        "stage_status": "running",
        "at": event.get("at"),
        "milestone_index": event.get("milestone_index"),
    }


def _latest_running(stages: list[dict], stage_type: str) -> dict | None:
    for stage in reversed(stages):
        if stage["type"] == stage_type and stage["stage_status"] == "running":
            return stage
    return None


def build_stages(events: list[dict]) -> list[dict]:
    """Pair each phase's loading event with its result, without fake steps."""
    stages: list[dict] = []
    plan_created = False
    bootstrap_seen = False
    for event in events:
        event_type = event.get("type")
        if event_type == "task":
            bootstrap = _latest_running(stages, "bootstrap")
            if bootstrap is not None:
                bootstrap["stage_status"] = "complete"
                bootstrap["at"] = event.get("at")
            continue
        if event_type == "phase_started":
            phase = event.get("phase")
            if phase == "BOOTSTRAP":
                if bootstrap_seen:
                    continue
                bootstrap_seen = True
            stage_type = "control" if phase == "PLAN" and plan_created else (
                "plan" if phase == "PLAN" else PHASE_STAGE.get(phase)
            )
            if stage_type and _latest_running(stages, stage_type) is None:
                stages.append(_pending_stage(event, stage_type))
            continue
        if event_type in RESULT_TYPES:
            stage = _latest_running(stages, event_type)
            if stage is None:
                stage = _pending_stage(event, event_type)
                stages.append(stage)
            stage.update(event)
            stage["stage_status"] = "complete"
            if event_type == "plan":
                plan_created = True
            continue
        if event_type == "error":
            stages.append({**event, "stage_status": "error"})
    return stages
