"""Smart-generic data summarizer for LLM-friendly tool output / variable rendering.

Used by:
  - api_trace rendering in continuation_planner history (replaces the
    legacy first-N + 16-key-priority approach)
  - prior_variables rendering in variable_store (per variable value preview)

Approach: classify each field by data shape (no AppWorld-specific
keywords), then render schema + distributions + ranges + samples.

Constants live in `contracts/limits.py`. No magic numbers here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from adk_appworld_agent.contracts.limits import (
    CATEGORICAL_NUMERIC_MAX_UNIQUE_FRAC,
    CATEGORICAL_NUMERIC_MIN_UNIQUE,
    DISTRIBUTION_TOP_K,
    FREE_FORM_AVG_LEN_THRESHOLD,
    HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH,
    ID_FIELD_PATTERN,
    ID_UNIQUE_RATIO_THRESHOLD,
    ISO_DATETIME_PATTERN,
    SAMPLE_COUNT,
    SAMPLE_FIELD_MAX_CHARS,
)

_ID_FIELD_RE = re.compile(ID_FIELD_PATTERN, re.I)
_ISO_DATETIME_RE = re.compile(ISO_DATETIME_PATTERN)


# ── Shape / type detection ─────────────────────────────────────────────


def _is_iso_datetime(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(_ISO_DATETIME_RE.match(value))


def _looks_like_id(field: str, values: list) -> bool:
    if not _ID_FIELD_RE.search(field):
        return False
    non_null = [v for v in values if v is not None]
    if not non_null:
        return False
    return len(set(non_null)) / len(non_null) >= ID_UNIQUE_RATIO_THRESHOLD


def _type_label(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "datetime" if _is_iso_datetime(value) else "str"
    if isinstance(value, dict):
        return f"dict({len(value)} keys)"
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _infer_schema(items: list[dict]) -> dict[str, str]:
    """Union all keys across items, infer type per field from first non-null value."""
    schema: dict[str, str] = {}
    ordered_keys: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for k in item.keys():
            if k not in schema:
                ordered_keys.append(k)
                schema[k] = "null"
    for k in ordered_keys:
        for item in items:
            if isinstance(item, dict) and k in item and item[k] is not None:
                schema[k] = _type_label(item[k])
                break
    return schema


def _collect_field(items: list[dict], field: str) -> list:
    return [item.get(field) if isinstance(item, dict) else None for item in items]


def _classify_field(field: str, values: list, n_items: int) -> str:
    """Return one of:
    constant, id, datetime_range, numeric_range, categorical, free_form,
    nested_dict, nested_list, or 'mixed' (fallback).
    """
    non_null = [v for v in values if v is not None]
    if not non_null:
        return "constant"  # all null

    # Uniqueness via repr / json.dumps for hashable comparison.
    unique: set = set()
    for v in non_null:
        try:
            unique.add(
                v
                if isinstance(v, (int, float, str, bool))
                else json.dumps(v, ensure_ascii=False, sort_keys=True)
            )
        except (TypeError, ValueError):
            unique.add(str(v))
    n_unique = len(unique)

    if n_unique == 1 and len(non_null) == n_items:
        return "constant"
    if _looks_like_id(field, values):
        return "id"

    sample = non_null[0]
    if isinstance(sample, bool):
        return "categorical"
    if isinstance(sample, (int, float)):
        numeric_cap = max(
            CATEGORICAL_NUMERIC_MIN_UNIQUE,
            int(n_items * CATEGORICAL_NUMERIC_MAX_UNIQUE_FRAC),
        )
        if n_unique > numeric_cap:
            return "numeric_range"
        return "categorical"
    if isinstance(sample, str):
        if all(_is_iso_datetime(v) for v in non_null):
            return "datetime_range"
        # Free-form text vs categorical:
        # Only call it free-form when strings are genuinely long. Short
        # categorical strings (description / name / title) stay categorical
        # even when 100% unique — for small n the specific values are still
        # informative for the LLM.
        avg_len = sum(len(v) for v in non_null) / len(non_null)
        if avg_len > FREE_FORM_AVG_LEN_THRESHOLD:
            return "free_form"
        return "categorical"
    if isinstance(sample, dict):
        return "nested_dict"
    if isinstance(sample, list):
        return "nested_list"
    return "mixed"


# ── Sample rendering ───────────────────────────────────────────────────


def _render_sample_item(
    item: dict, *, max_field_chars: int = SAMPLE_FIELD_MAX_CHARS
) -> str:
    """Render one sample item — every field listed, per-field value truncation.

    No whole-string truncation: later fields are never hidden by an early
    long field's value. Long values get per-field "..." marker but the
    field itself stays visible.
    """
    parts: list[str] = []
    for k, v in item.items():
        if isinstance(v, str):
            disp = repr(v if len(v) <= max_field_chars else v[:max_field_chars] + "…")
        elif isinstance(v, (dict, list)):
            s = json.dumps(v, ensure_ascii=False)
            disp = s if len(s) <= max_field_chars else s[:max_field_chars] + "…"
        elif v is None:
            disp = "null"
        else:
            disp = str(v)
        parts.append(f"{k}={disp}")
    return "{" + ", ".join(parts) + "}"


# ── Main summarizer ────────────────────────────────────────────────────


def summarize_list_of_dicts(
    items: list[dict],
    *,
    max_uniques: int = DISTRIBUTION_TOP_K,
    max_samples: int = SAMPLE_COUNT,
) -> list[str]:
    """Render summary lines for a list of dicts (most common AppWorld shape)."""
    n = len(items)
    if n == 0:
        return ["result: list[0] (empty)"]

    schema = _infer_schema(items)
    lines = [
        f"result: list[{n}]",
        f"schema: {{{', '.join(f'{k}: {v}' for k, v in schema.items())}}}",
    ]

    constants_lines: list[str] = []
    constant_fields: set[str] = set()
    distributions_lines: list[str] = []
    ranges_lines: list[str] = []
    ids_lines: list[str] = []
    free_form_lines: list[str] = []
    raw_only_fields: list[str] = []

    diversity_field: str | None = None
    diversity_top: list[tuple] = []

    for field in schema:
        values = _collect_field(items, field)
        cls = _classify_field(field, values, n)

        if cls == "constant":
            v = next((x for x in values if x is not None), None)
            v_repr = repr(v)
            if len(v_repr) > HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH:
                v_repr = v_repr[:HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH] + "…"
            constants_lines.append(f"  {field} = {v_repr} (all {n})")
            constant_fields.add(field)
        elif cls == "id":
            non_null = [v for v in values if v is not None]
            ids_lines.append(f"  {field}: {len(set(non_null))} unique values")
        elif cls == "datetime_range":
            non_null = sorted([v for v in values if v is not None])
            ranges_lines.append(f"  {field}: [{non_null[0]} .. {non_null[-1]}]")
        elif cls == "numeric_range":
            non_null = sorted([v for v in values if v is not None])
            median = non_null[len(non_null) // 2]
            ranges_lines.append(
                f"  {field}: [{non_null[0]}, {non_null[-1]}], median {median}"
            )
        elif cls == "categorical":
            non_null = [v for v in values if v is not None]
            counter = Counter(non_null)
            top = counter.most_common(max_uniques)
            uniq = len(counter)
            more = uniq - len(top)
            tail = f", ... {more} more" if more > 0 else ""
            top_str = ", ".join(f"{v!r}: {c}" for v, c in top)
            distributions_lines.append(f"  {field} ({uniq} unique): {top_str}{tail}")
            # Diversity sample axis: prefer field with most unique values (within bound).
            if 2 <= uniq <= max_uniques * 3 and (
                diversity_field is None or uniq > len(diversity_top)
            ):
                diversity_field = field
                diversity_top = top
        elif cls == "free_form":
            non_null = [v for v in values if v is not None]
            uniq = len(set(non_null))
            avg = sum(len(v) for v in non_null) // len(non_null)
            free_form_lines.append(
                f"  {field}: {uniq} unique values, avg len {avg} chars"
            )
        elif cls == "nested_dict":
            raw_only_fields.append(f"  {field}: dict")
        elif cls == "nested_list":
            all_empty = all(
                (
                    isinstance(it, dict)
                    and isinstance(it.get(field), list)
                    and len(it[field]) == 0
                )
                for it in items
            )
            if all_empty:
                constants_lines.append(f"  {field} = [] (all {n})")
                constant_fields.add(field)
            else:
                # `it.get(field, [])` returns None when the key exists with
                # value None (the default `[]` only applies when the key is
                # missing entirely). Coerce None → empty list before len()
                # to avoid `TypeError: object of type 'NoneType' has no len()`.
                # Surfaced by per-API aggregator change (2026-05-17) which
                # started rendering dict-pool items per-group; prior path
                # dropped that pool entirely so this branch was unreachable
                # for those cases.
                lens = [
                    len(it.get(field) or []) for it in items if isinstance(it, dict)
                ]
                raw_only_fields.append(
                    f"  {field}: list, lengths [{min(lens)}..{max(lens)}]"
                )
        else:
            # Default fallback for any unrecognized class (mixed type / future).
            non_null = [v for v in values if v is not None]
            uniq = len(set(repr(v) for v in non_null))
            raw_only_fields.append(f"  {field}: {cls}, {uniq} unique values")

    if constants_lines:
        lines.append("constants (all-same):")
        lines.extend(constants_lines)
    if distributions_lines:
        lines.append("distributions:")
        lines.extend(distributions_lines)
    if ranges_lines:
        lines.append("ranges:")
        lines.extend(ranges_lines)
    if free_form_lines:
        lines.append("free-form fields:")
        lines.extend(free_form_lines)
    if ids_lines:
        lines.append("ids (skipped distribution):")
        lines.extend(ids_lines)
    if raw_only_fields:
        lines.append(
            "raw-only fields (nested / mixed / unhandled — full values in samples):"
        )
        lines.extend(raw_only_fields)

    # Sample selection
    samples: list[dict] = []
    if diversity_field and diversity_top:
        seen_values: set = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            v = item.get(diversity_field)
            if (
                v in (top_value for top_value, _ in diversity_top)
                and v not in seen_values
            ):
                samples.append(item)
                seen_values.add(v)
                if len(samples) >= max_samples:
                    break
    if not samples:
        samples = items[:max_samples]
    if samples:
        lines.append(f"samples (diverse by {diversity_field or 'first-N'}):")
        for s in samples:
            lines.append(f"  {_render_sample_item(s)}")

    return lines


def summarize_dict(value: dict) -> list[str]:
    """Render a single dict (e.g., single-object API return)."""
    schema = {k: _type_label(v) for k, v in value.items()}
    lines = [
        f"result: dict ({len(value)} fields)",
        f"schema: {{{', '.join(f'{k}: {v}' for k, v in schema.items())}}}",
        "value:",
    ]
    for k, v in value.items():
        rendered = json.dumps(v, ensure_ascii=False)
        if len(rendered) > HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH:
            rendered = rendered[:HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH] + "…"
        lines.append(f"  {k}: {rendered}")
    return lines


def summarize_data_for_llm(data: Any) -> str:
    """Top-level entry: shape detect + dispatch to typed summarizer."""
    if isinstance(data, list):
        if not data:
            return "result: list[0] (empty)"
        if all(isinstance(x, dict) for x in data):
            return "\n".join(summarize_list_of_dicts(data))
        # list of primitives — show with sample
        sample = json.dumps(data[:3], ensure_ascii=False)
        if len(sample) > HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH:
            sample = sample[:HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH] + "…"
        return f"result: list[{len(data)}] (non-dict items, samples: {sample})"
    if isinstance(data, dict):
        return "\n".join(summarize_dict(data))
    return f"result: scalar / other — value={data!r}"


__all__ = [
    "summarize_data_for_llm",
    "summarize_list_of_dicts",
    "summarize_dict",
]
