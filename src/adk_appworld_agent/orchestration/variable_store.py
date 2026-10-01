from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from adk_appworld_agent.contracts.executor_result import MemoryVariable
from adk_appworld_agent.contracts.limits import (
    PRIOR_VARIABLES_MAX_LENGTH,
    VARIABLE_KEYS_PREVIEW_MAX_COUNT,
    VARIABLE_PREVIEW_MAX_LENGTH,
)
from adk_appworld_agent.subagents.utils.data_summarizer import summarize_data_for_llm

if TYPE_CHECKING:
    from adk_appworld_agent.orchestration.state import RunState


# Re-exported for backward compat with any callers that import the old names.
DEFAULT_VARIABLE_SUMMARY_LIMIT = PRIOR_VARIABLES_MAX_LENGTH
DEFAULT_VARIABLE_PREVIEW_LIMIT = VARIABLE_PREVIEW_MAX_LENGTH


class VariableStore:
    """Typed view over RunState's loose variable dictionaries."""

    def __init__(self, state: RunState) -> None:
        self._state = state

    def commit(self, var: MemoryVariable) -> MemoryVariable:
        value = json.loads(var.value_json)
        enriched = var.model_copy(
            update={
                "type_name": var.type_name or type(value).__name__,
                "count_items": _count_items(value),
                "created_at": var.created_at or _utc_now_iso(),
            }
        )
        self._state.named_variables[enriched.name] = enriched.model_dump(mode="json")
        if enriched.name not in self._state.variable_creation_order:
            self._state.variable_creation_order.append(enriched.name)
        return enriched

    def get(self, name: str) -> MemoryVariable | None:
        raw = self._state.named_variables.get(name)
        if not isinstance(raw, dict):
            return None
        try:
            return MemoryVariable.model_validate(raw)
        except Exception:
            return None

    def names(self) -> list[str]:
        ordered = [
            name
            for name in self._state.variable_creation_order
            if name in self._state.named_variables
        ]
        for name in self._state.named_variables:
            if name not in ordered:
                ordered.append(name)
        return ordered

    def summary(
        self,
        names: list[str] | None = None,
        *,
        last_n: int | None = None,
        max_length: int = DEFAULT_VARIABLE_SUMMARY_LIMIT,
    ) -> str:
        selected_names = self._selected_names(names=names, last_n=last_n)
        blocks = [
            block
            for name in selected_names
            if (block := self._summary_block(name)) is not None
        ]
        if not blocks:
            return ""
        return _truncate_text(
            "\n\n".join(blocks),
            limit=max_length,
            notice="prior variable summary truncated",
        )

    def preview(
        self, name: str, *, max_length: int = DEFAULT_VARIABLE_PREVIEW_LIMIT
    ) -> str:
        var = self.get(name)
        if var is None:
            return ""
        return _value_preview(var.value_json, max_length=max_length)

    def _selected_names(
        self, *, names: list[str] | None, last_n: int | None
    ) -> list[str]:
        selected = list(names) if names is not None else self.names()
        if last_n is not None and last_n >= 0:
            selected = selected[-last_n:]
        return selected

    def _summary_block(self, name: str) -> str | None:
        var = self.get(name)
        if var is None:
            return None
        lines = [
            f"### {var.name}",
            f'- access: prior_variable_values["{var.name}"]',
            f"- type: {var.type_name or 'unknown'}",
            f"- items: {var.count_items}",
        ]
        if var.description:
            lines.append(f"- description: {var.description}")
        if var.source_milestone_id:
            lines.append(f"- source_milestone_id: {var.source_milestone_id}")
        # Surface top-level field names for dict / list-of-dict values so the
        # next milestone's executor can write `prior_variable_values["x"]["id"]`
        # without guessing the inner shape. Observed gap on 986aa4e_2 m3.
        fields = _top_level_fields(var.value_json)
        if fields:
            lines.append(f"- accessible_fields: {fields}")
        # Value preview via the generic data summarizer (entry 4 helper).
        # For list[dict] / dict shapes, this surfaces schema + distributions
        # + sample rows that the next milestone can plan against. Falls
        # back to raw JSON for primitives or non-serializable values.
        try:
            value = json.loads(var.value_json) if var.value_json else None
        except (TypeError, ValueError):
            value = None
        lines.append("- value summary:")
        for sline in summarize_data_for_llm(value).splitlines():
            lines.append(f"  {sline}")
        return "\n".join(lines)


def _count_items(value: Any) -> int:
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    return 1


def _top_level_fields(value_json: str) -> list[str]:
    """Return the top-level dict keys for a JSON-encoded value.

    For dicts: keys of the dict itself. For list-of-dicts: keys of the first
    item (representative shape). Returns [] for primitives or empty containers.
    Used by `_summary_block` so the next milestone's executor can see the
    accessible field names without parsing the preview JSON manually.
    """
    try:
        value = json.loads(value_json)
    except (TypeError, ValueError):
        return []
    if isinstance(value, dict):
        return list(value.keys())[:VARIABLE_KEYS_PREVIEW_MAX_COUNT]
    if isinstance(value, list) and value and isinstance(value[0], dict):
        return list(value[0].keys())[:VARIABLE_KEYS_PREVIEW_MAX_COUNT]
    return []


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value_preview(value_json: str, *, max_length: int) -> str:
    try:
        value = json.loads(value_json)
        rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        rendered = value_json
    return _truncate_text(
        rendered, limit=max_length, notice="variable preview truncated"
    )


def _truncate_text(text: str, *, limit: int, notice: str) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    keep = max(0, limit - len(notice) - 10)
    return f"{text[:keep]}\n...<{notice}: {len(text) - keep} chars>..."


__all__ = [
    "DEFAULT_VARIABLE_PREVIEW_LIMIT",
    "DEFAULT_VARIABLE_SUMMARY_LIMIT",
    "VariableStore",
]
