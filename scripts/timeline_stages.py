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
ITEM_STAGE_TYPES = {"retrieval", "execution", "control"}


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


def _apply_plan_revision(plan_items: list[dict], index: int, stage: dict) -> list[dict]:
    """Continuation revisions replace only the writable plan tail segment."""
    revised = stage.get("revised_milestones")
    action = str(stage.get("action", "")).upper()
    if not isinstance(revised, list) or not revised or action not in {"RETRY", "ADVANCE"}:
        return plan_items
    if not all(isinstance(item, dict) and item.get("task") for item in revised):
        return plan_items
    start = index + (1 if action == "ADVANCE" else 0)
    return plan_items[:start] + revised + plan_items[start + len(revised):]


def _attach_plan_context(stages: list[dict]) -> None:
    """Identify the plan item and outer loop cycle for each working stage."""
    plan_items: list[dict] = []
    cycles: dict[int, int] = {}
    next_cycle: set[int] = set()
    active_index: int | None = None
    for stage in stages:
        if stage["type"] == "plan" and isinstance(stage.get("milestones"), list):
            plan_items = stage["milestones"]
        if stage["type"] not in ITEM_STAGE_TYPES:
            continue
        index = stage.get("milestone_index")
        if not isinstance(index, int) and stage["type"] == "control":
            index = active_index
        if not isinstance(index, int) or not 0 <= index < len(plan_items):
            continue
        active_index = index
        if stage["type"] in {"retrieval", "execution"}:
            if index in next_cycle:
                cycles[index] = cycles.get(index, 1) + 1
                next_cycle.remove(index)
            else:
                cycles.setdefault(index, 1)
        item = plan_items[index]
        if not isinstance(item, dict):
            continue
        stage["plan_item"] = {
            "number": index + 1,
            "total": len(plan_items),
            "app": item.get("app", ""),
            "task": item.get("task", ""),
            "cycle": cycles.get(index, 1),
        }
        if stage["type"] == "control" and stage["stage_status"] == "complete":
            plan_items = _apply_plan_revision(plan_items, index, stage)
        if stage["type"] == "control" and stage["stage_status"] == "complete":
            next_cycle.add(index)


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
    _attach_plan_context(stages)
    return stages


def build_display_blocks(stages: list[dict]) -> list[dict]:
    """Nest each work item's repeated cycles below one visible plan item."""
    blocks: list[dict] = []
    for stage in stages:
        item = stage.get("plan_item")
        if not isinstance(item, dict):
            blocks.append({"kind": "stage", "stage": stage})
            continue

        previous = blocks[-1] if blocks else None
        if previous is None or previous["kind"] != "plan_item" or previous["item"]["number"] != item["number"]:
            previous = {"kind": "plan_item", "item": item, "cycles": []}
            blocks.append(previous)

        cycles = previous["cycles"]
        if not cycles or cycles[-1]["number"] != item["cycle"]:
            cycles.append({"number": item["cycle"], "item": item, "stages": []})
        cycles[-1]["stages"].append(stage)
    return blocks
