from __future__ import annotations

import json
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path

from adk_appworld_agent.appworld_paths import find_repo_root
from adk_appworld_agent.orchestration.active_config import active_config
from adk_appworld_agent.orchestration.run_config import (
    ModelConfig,
    model_config_from_env,
)
from adk_appworld_agent.subagents.finder.api_specs import get_api_spec
from adk_appworld_agent.subagents.finder.community_finder.prompts import (
    API_DEP_FILTER_SYS_PROMPT,
    API_DEP_FILTER_USER_PROMPT,
    APP_SELECT_SYS_PROMPT,
    APP_SELECT_USER_PROMPT,
    BATCHED_DEP_FILTER_SYS_PROMPT,
    BATCHED_DEP_FILTER_USER_PROMPT,
    COMMUNITY_SELECT_SYS_PROMPT,
    COMMUNITY_SELECT_USER_PROMPT,
    FILTER_SYS_PROMPT,
    FILTER_USER_PROMPT,
)
from adk_appworld_agent.subagents.finder.llm import call_llm_json

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
# API-graph data lives at the repo top-level data/ (out of src/), not next to
# the finder module — see appworld_paths.find_repo_root.
_DATA_DIR = find_repo_root() / "data" / "api_graph"
_COMMUNITIES_PATH = _DATA_DIR / "leiden_communities_with_summaries.json"
_API_DEPENDENCY_INDEX_PATH = _DATA_DIR / "api_dependency_index.json"
_APP_CONTEXT_PATH = _DATA_DIR / "app_context.md"
_BEHAVIOR_GUIDELINES_PATH = _DATA_DIR / "behavior_guidelines.md"

# Empty-filter coverage backstop: when communities were selected (graph routing
# worked) but the per-op seed_filter LLM rejected EVERY candidate, the finder
# would otherwise surface candidate_apis=[] and the executor, with no allow-list,
# guesses non-existent API names until MAX_CYCLES (0d01c76_2: file_system_c0
# selected, filter returned [], executor guessed read_text_file/read_file/
# get_file_content, all rejected). Instead, fall back to the selected
# communities' own ops (graph-grounded — NO op-name leak into community
# selection, so the format-C / no-op-list contribution is preserved). Capped so
# a multi-community selection cannot flood the weak model. candidate_apis is an
# allow-list, not a mandate, so a pure-compute milestone that needs no API is
# free to ignore the surfaced ops — but the targeted measurement must confirm
# compute-only milestones do not start making spurious calls.
_EMPTY_FILTER_BACKSTOP_CAP = 40


@lru_cache(maxsize=1)
def _load_communities() -> dict[str, dict]:
    payload = json.loads(_COMMUNITIES_PATH.read_text(encoding="utf-8"))
    communities = payload.get("communities", {})
    return communities if isinstance(communities, dict) else {}


def _api_dependency_index_path() -> Path:
    """Resolve the dependency-graph file from the active config — a filename
    under data/ (e.g. api_dependency_index_llm_only.json for the pure-LLM
    graph) or an absolute path. Replaces the per-worktree API_DEP_INDEX env."""
    name = active_config().retrieval.dependency_index_file
    candidate = Path(name)
    return candidate if candidate.is_absolute() else _DATA_DIR / name


@lru_cache(maxsize=4)
def _load_api_dependency_index_cached(path_str: str) -> dict[str, list[dict]]:
    path = Path(path_str)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("consumer_to_producers", {})
    return raw if isinstance(raw, dict) else {}


def _load_api_dependency_index() -> dict[str, list[dict]]:
    # Cache keyed by the resolved path so switching graphs via config reloads.
    return _load_api_dependency_index_cached(str(_api_dependency_index_path()))


def _load_forward_dependency_index() -> dict[str, list[dict]]:
    """Invert consumer_to_producers -> producer_op : [{consumer_op, id_field, source}].

    Used only by optional forward dependency expansion. Derived at load time from
    the SAME api_dependency_index.json — no extra data file.
    """
    forward: dict[str, list[dict]] = {}
    for consumer_op, edges in _load_api_dependency_index().items():
        for edge in edges:
            producer_op = str(edge.get("producer_op", ""))
            if not producer_op:
                continue
            forward.setdefault(producer_op, []).append(
                {
                    "consumer_op": consumer_op,
                    "id_field": edge.get("id_field"),
                    "source": edge.get("source"),
                }
            )
    return forward


def _load_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _split_op_id(op_id: str) -> tuple[str, str] | None:
    if "__" not in op_id:
        return None
    return tuple(op_id.split("__", 1))


def _api_key(app_name: str, api_name: str) -> str:
    return f"{app_name}.{api_name}"


def _api_key_from_spec(api: dict) -> str:
    return _api_key(str(api.get("app_name", "")), str(api.get("api_name", "")))


def _op_id_from_spec(api: dict) -> str:
    return f"{api.get('app_name', '')}__{api.get('api_name', '')}"


def _display_op_id(op_id: str) -> str:
    parts = _split_op_id(op_id)
    if parts is None:
        return op_id
    return _api_key(*parts)


def _required_params(api: dict) -> list[dict]:
    return [
        param
        for param in api.get("parameters", [])
        if isinstance(param, dict) and param.get("required")
    ]


def _all_param_names(api: dict) -> set[str]:
    return {
        str(param.get("name"))
        for param in api.get("parameters", [])
        if isinstance(param, dict) and isinstance(param.get("name"), str)
    }


def _format_required_params(required: list[dict]) -> str:
    if not required:
        return "(none)"
    parts: list[str] = []
    for param in required:
        desc = str(param.get("description", "")).split(".")[0].strip()
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        parts.append(f"{name} ({desc})" if desc else name)
    return ", ".join(parts) if parts else "(none)"


def _get_response_fields(api: dict) -> list[str]:
    success = api.get("response_schemas", {}).get("success", {})
    if isinstance(success, dict):
        return [str(key) for key in success.keys()]
    if isinstance(success, list) and success and isinstance(success[0], dict):
        return [str(key) for key in success[0].keys()]
    return []


def _build_available_fields(selected_apis: list[dict]) -> set[str]:
    fields: set[str] = set()
    for api in selected_apis:
        fields.update(_get_response_fields(api))
    return fields


def _build_available_field_counts(selected_apis: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for api in selected_apis:
        counts.update(_get_response_fields(api))
    return counts


def _available_fields_from_other_selected_apis(
    consumer_api: dict,
    available_field_counts: Counter[str],
) -> set[str]:
    consumer_fields = Counter(_get_response_fields(consumer_api))
    available_fields = set()
    for field_name, count in available_field_counts.items():
        if count - consumer_fields.get(field_name, 0) > 0:
            available_fields.add(field_name)
    return available_fields


def _generate_config_snapshot(model_cfg: ModelConfig) -> dict:
    return {
        "response_mime_type": "application/json",
        "temperature": model_cfg.temperature,
        "top_p": model_cfg.top_p,
        "top_k": model_cfg.top_k,
        "candidate_count": model_cfg.candidate_count,
        "seed": model_cfg.seed,
        "max_output_tokens": model_cfg.max_output_tokens or None,
    }


def _model_input_record(
    user_prompt: str, sys_prompt: str, model_cfg: ModelConfig
) -> dict:
    return {
        "model": model_cfg.name,
        "system_instruction": sys_prompt,
        "messages": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generate_content_config": _generate_config_snapshot(model_cfg),
    }


def _empty_usage_metrics() -> dict[str, int]:
    return {
        "usage_event_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thoughts_tokens": 0,
        "total_tokens": 0,
    }


def _empty_retry_metrics() -> dict[str, int | str | None]:
    return {
        "retry_count": 0,
        "retry_backoff_ms": 0,
        "rate_limit_count": 0,
        "provider_error_count": 0,
        "last_error": None,
    }


def _normalize_usage_metrics(usage: object) -> dict[str, int]:
    if not isinstance(usage, dict):
        return _empty_usage_metrics()
    return {
        "usage_event_count": int(usage.get("usage_event_count") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "thoughts_tokens": int(usage.get("thoughts_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _normalize_retry_metrics(retry: object) -> dict[str, int | str | None]:
    if not isinstance(retry, dict):
        return _empty_retry_metrics()
    return {
        "retry_count": int(retry.get("retry_count") or 0),
        "retry_backoff_ms": int(retry.get("retry_backoff_ms") or 0),
        "rate_limit_count": int(retry.get("rate_limit_count") or 0),
        "provider_error_count": int(retry.get("provider_error_count") or 0),
        "last_error": str(retry.get("last_error") or "") or None,
    }


def _split_llm_result(
    result: object,
) -> tuple[dict, dict[str, int], dict[str, int | str | None]]:
    if isinstance(result, dict) and ("parsed_json" in result or "usage" in result):
        parsed = result.get("parsed_json")
        return (
            parsed if isinstance(parsed, dict) else {},
            _normalize_usage_metrics(result.get("usage")),
            _normalize_retry_metrics(result.get("retry")),
        )
    if isinstance(result, dict):
        return result, _empty_usage_metrics(), _empty_retry_metrics()
    return {}, _empty_usage_metrics(), _empty_retry_metrics()


def _accumulate_usage_metrics(total: dict[str, int], usage: dict[str, int]) -> None:
    for key in (
        "usage_event_count",
        "prompt_tokens",
        "completion_tokens",
        "thoughts_tokens",
        "total_tokens",
    ):
        total[key] += int(usage.get(key) or 0)


def _accumulate_retry_metrics(
    total: dict[str, int | str | None],
    retry: dict[str, int | str | None],
) -> None:
    for key in (
        "retry_count",
        "retry_backoff_ms",
        "rate_limit_count",
        "provider_error_count",
    ):
        total[key] = int(total.get(key) or 0) + int(retry.get(key) or 0)
    if retry.get("last_error"):
        total["last_error"] = retry.get("last_error")


async def _call_llm_json_for_step(
    *,
    step: str,
    user_prompt: str,
    sys_prompt: str,
    model_cfg: ModelConfig,
    io_trace: list[dict] | None,
) -> tuple[dict, dict[str, int], dict[str, int | str | None]]:
    result = await call_llm_json(user_prompt, sys_prompt, model_cfg=model_cfg)
    parsed_json, usage, retry = _split_llm_result(result)
    if io_trace is not None:
        io_trace.append(
            {
                "step": step,
                "model_input_raw": _model_input_record(
                    user_prompt, sys_prompt, model_cfg
                ),
                "raw_llm_text": json.dumps(parsed_json, ensure_ascii=False),
                "parsed_json": parsed_json,
                "usage": usage,
                "retry": retry,
            }
        )
    return parsed_json, usage, retry


def _build_app_catalog(communities: dict[str, dict]) -> str:
    """One line per app: 'app: capability-area; ...' — built from the official
    community-graph titles. Catalog for the stage-1 app-select (fires only when
    the planner gave no usable app). No op-name leak (titles only, format-C)."""
    by_app: dict[str, list[str]] = {}
    for community in communities.values():
        if not isinstance(community, dict):
            continue
        app = str(community.get("app", "")).strip()
        title = str(community.get("title", "")).strip()
        if app and title:
            by_app.setdefault(app, []).append(title)
    lines = []
    for app in sorted(by_app):
        lines.append(f"- {app}: {'; '.join(by_app[app][:6])}")
    return "\n".join(lines)


def _build_community_text(
    communities: dict[str, dict], app_filter: set[str] | None = None
) -> str:
    """Format per-community text for the community-finder prompt.

    Per `api_graph_refactor_spec.md` §Step 7 (2026-05-30 lock): use
    format C — title + summary + op count only, NO op-name list. The
    spec frames community summary as a retrieval abstraction layer; if
    we leak op names back into the prompt, the LLM bypasses the
    abstraction by pattern-matching on names, which weakens the
    contribution claim and reintroduces the noise the community
    structure was meant to suppress.
    """
    by_app: dict[str, list[dict]] = {}
    for community in communities.values():
        if not isinstance(community, dict):
            continue
        app = str(community.get("app", ""))
        if app_filter and app not in app_filter:
            continue
        by_app.setdefault(app, []).append(community)

    lines: list[str] = []
    for app in sorted(by_app):
        lines.append(f"=== {app} ===")
        for community in sorted(
            by_app[app], key=lambda item: int(item.get("index", 0))
        ):
            community_id = str(community.get("community_id", ""))
            title = str(community.get("title", "")).strip()
            summary = str(community.get("summary", "")).strip()
            # Count only — the op-name list itself is intentionally dropped
            # (spec §Step 7 format C). Keep the count so the planner can
            # gauge community breadth without leaking names.
            op_ids = [
                op
                for op in community.get("operations", [])
                if isinstance(op, str) and "__" in op
            ]
            op_count = len(op_ids)
            # Read/write capability split, derived from each op's HTTP METHOD
            # (GET = read/retrieve vs write). This is the most abstract
            # capability signal — it leaks NO op names (preserving format-C),
            # but it disambiguates similarly-titled communities the summaries
            # alone confuse: e.g. file_system_c0 "...and content updates"
            # (0 read / 11 write) vs file_system_c2 which actually holds the
            # read-content op (4 read / 1 write). Without it, a "read the file
            # content" milestone mis-selects the write-only community and the
            # read op is never surfaced (0d01c76_2 → MAX_CYCLES).
            read_n = 0
            write_n = 0
            for op_id in op_ids:
                parts = _split_op_id(op_id)
                if parts is None:
                    continue
                spec = get_api_spec(parts[0], parts[1])
                method = str((spec or {}).get("method", "")).upper()
                if method == "GET":
                    read_n += 1
                else:
                    write_n += 1
            lines.append(
                f"[{community_id}] {title} "
                f"({op_count} APIs: {read_n} read/retrieve, {write_n} create/modify/delete)"
            )
            if summary:
                lines.append(f"  {summary}")
        lines.append("")
    return "\n".join(lines)


def _apps_from_planner(
    planned_apps: list[str] | None,
    all_apps: list[str],
) -> tuple[list[str], list[str], bool]:
    requested = planned_apps or []
    valid_apps: list[str] = []
    seen: set[str] = set()
    for app in requested:
        if isinstance(app, str) and app in all_apps and app not in seen:
            valid_apps.append(app)
            seen.add(app)
    invalid_apps = sorted(
        {str(app) for app in requested if isinstance(app, str) and app not in all_apps}
    )
    if valid_apps:
        return valid_apps, invalid_apps, False
    return list(all_apps), invalid_apps, True


def _build_api_filter_text(candidates: list[dict]) -> str:
    # Full-spec rendering (2026-06-08): present each candidate API in full — all
    # parameters (name, type, required/optional, description) + response fields —
    # rather than the prior compact "[key] METHOD - desc + Required params" line.
    # Presenting the API completely (not a truncated view) is the natural
    # representation for a selection prompt; the compact form was an artifact.
    lines: list[str] = []
    for api in candidates:
        key = _api_key_from_spec(api)
        method = str(api.get("method", "")).upper()
        desc = str(api.get("description", "")).strip()
        lines.append(f"[{key}] {method} - {desc}")
        params = api.get("parameters", [])
        param_lines: list[str] = []
        for param in params if isinstance(params, list) else []:
            if not isinstance(param, dict):
                continue
            name = str(param.get("name", "")).strip()
            if not name:
                continue
            ptype = str(param.get("type", "")).strip()
            req = "required" if param.get("required") else "optional"
            pdesc = str(param.get("description", "")).strip()
            meta = ", ".join(x for x in (ptype, req) if x)
            head = f"    - {name}" + (f" ({meta})" if meta else "")
            param_lines.append(head + (f": {pdesc}" if pdesc else ""))
        if param_lines:
            lines.append("  Parameters:")
            lines.extend(param_lines)
        response_fields = _get_response_fields(api)
        if response_fields:
            lines.append(f"  Returns: {', '.join(response_fields)}")
    return "\n".join(lines)


def _build_unsatisfied_params_text(selected_apis: list[dict]) -> str:
    available_fields = _build_available_fields(selected_apis)
    unsatisfied: dict[str, list[str]] = {}
    for api in selected_apis:
        for param in _required_params(api):
            param_name = str(param.get("name", ""))
            if not param_name or param_name in available_fields:
                continue
            unsatisfied.setdefault(param_name, []).append(_api_key_from_spec(api))
    if not unsatisfied:
        return "(none)"
    return "\n".join(
        f"  {param_name} (needed by: {', '.join(sorted(apis))})"
        for param_name, apis in sorted(unsatisfied.items())
    )


def _build_dependency_candidate_text(candidates: list[dict]) -> str:
    lines: list[str] = []
    for api in candidates:
        key = _api_key_from_spec(api)
        method = str(api.get("method", "")).upper()
        desc = str(api.get("description", "")).strip()
        response_fields = ", ".join(_get_response_fields(api)) or "(none)"
        lines.append(f"[{key}] {method} - {desc}")
        lines.append(f"  Response fields: {response_fields}")
    return "\n".join(lines)


def _format_param_list(params: list[str]) -> str:
    if not params:
        return "- (none)"
    return "\n".join(f"- {param}" for param in params)


def _format_evidence_pair(edge: dict) -> str:
    relation = f"{_display_op_id(edge.get('producer_op', ''))} -> {_display_op_id(edge.get('consumer_op', ''))}"
    id_field = str(edge.get("id_field", "")).strip()
    source = str(edge.get("source", "")).strip()
    if id_field:
        relation += f" via {id_field}"
    if source:
        relation += f" ({source})"
    return relation


def _build_dependency_evidence_text(evidence_pairs: list[dict]) -> str:
    if not evidence_pairs:
        return "- (none)"
    return "\n".join(f"- {_format_evidence_pair(pair)}" for pair in evidence_pairs)


@lru_cache(maxsize=1)
def _build_api_to_community_map() -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    for community_id, community in _load_communities().items():
        for op_id in community.get("operations", []):
            parts = _split_op_id(op_id)
            if parts is not None:
                mapping[parts] = community_id
    return mapping


def _collect_api_specs_from_ops(
    op_ids: list[str],
    *,
    exclude: set[tuple[str, str]] | None = None,
) -> tuple[list[dict], list[str]]:
    seen: set[tuple[str, str]] = set(exclude or set())
    api_specs: list[dict] = []
    missing_ops: list[str] = []
    for op_id in op_ids:
        parts = _split_op_id(op_id)
        if parts is None or parts in seen:
            continue
        seen.add(parts)
        spec = get_api_spec(*parts)
        if spec is None:
            missing_ops.append(op_id)
            continue
        api_specs.append(spec)
    return api_specs, missing_ops


def _collect_api_specs_from_communities(
    community_ids: list[str],
    communities: dict[str, dict],
    *,
    exclude: set[tuple[str, str]] | None = None,
) -> list[dict]:
    op_ids: list[str] = []
    for community_id in community_ids:
        community = communities.get(community_id)
        if community is not None:
            op_ids.extend(community.get("operations", []))
    api_specs, _ = _collect_api_specs_from_ops(op_ids, exclude=exclude)
    return api_specs


def _extract_selected_candidates(
    candidate_map: dict[str, dict],
    result: dict,
    *,
    existing_keys: set[str] | None = None,
) -> list[dict]:
    selected_keys = result.get("selected", [])
    if not isinstance(selected_keys, list):
        return []
    existing = existing_keys or set()
    unknown = [
        key for key in selected_keys if key not in candidate_map and key not in existing
    ]
    if unknown:
        logger.warning("community finder returned unknown candidate keys: %s", unknown)
    return [candidate_map[key] for key in selected_keys if key in candidate_map]


async def _select_generic_candidates(
    *,
    task_instruction: str,
    candidates: list[dict],
    selected_apis: list[dict],
    behavior_guidelines: str,
    app_context: str,
    model_cfg: ModelConfig,
    io_trace: list[dict] | None = None,
    trace_step: str = "api_filter",
) -> tuple[list[dict], dict[str, int], dict[str, int | str | None]]:
    candidate_map = {_api_key_from_spec(api): api for api in candidates}
    existing_keys = {_api_key_from_spec(api) for api in selected_apis}
    user_prompt = FILTER_USER_PROMPT.format(
        task_instruction=task_instruction,
        unsatisfied_params=_build_unsatisfied_params_text(selected_apis),
        api_list=_build_api_filter_text(candidates),
    )
    sys_prompt = FILTER_SYS_PROMPT.format(
        behavior_guidelines=behavior_guidelines,
        app_context=app_context,
    )
    result, usage, retry = await _call_llm_json_for_step(
        step=trace_step,
        user_prompt=user_prompt,
        sys_prompt=sys_prompt,
        model_cfg=model_cfg,
        io_trace=io_trace,
    )
    return (
        _extract_selected_candidates(
            candidate_map, result, existing_keys=existing_keys
        ),
        usage,
        retry,
    )


async def _select_api_dependency_candidates(
    *,
    task_instruction: str,
    request: dict,
    model_cfg: ModelConfig,
    io_trace: list[dict] | None = None,
    trace_step: str = "api_dependency_filter",
) -> tuple[list[dict], dict[str, int], dict[str, int | str | None]]:
    candidates = request["candidate_apis"]
    candidate_map = {_api_key_from_spec(api): api for api in candidates}
    user_prompt = API_DEP_FILTER_USER_PROMPT.format(
        task_instruction=task_instruction,
        consumer_api_display=_display_op_id(request["consumer_op"]),
        relevant_params=_format_param_list(request["relevant_unresolved_params"]),
        dependency_evidence=_build_dependency_evidence_text(request["evidence_pairs"]),
        candidate_text=_build_dependency_candidate_text(candidates),
    )
    result, usage, retry = await _call_llm_json_for_step(
        step=trace_step,
        user_prompt=user_prompt,
        sys_prompt=API_DEP_FILTER_SYS_PROMPT,
        model_cfg=model_cfg,
        io_trace=io_trace,
    )
    return _extract_selected_candidates(candidate_map, result), usage, retry


def _build_batched_dependency_text(requests: list[dict]) -> str:
    """Per-consumer block: consumer full spec + ITS candidate prerequisites'
    full specs grouped under it. Keeps the consumer->producer relationship
    (not one pool of producers + one pool of consumers); no per-parameter
    dependency evidence — the model reasons from the full specs.
    """
    blocks: list[str] = []
    for request in requests:
        parts = _split_op_id(request["consumer_op"])
        consumer_spec = get_api_spec(*parts) if parts is not None else None
        consumer_text = (
            _build_api_filter_text([consumer_spec])
            if consumer_spec is not None
            else f"[{_display_op_id(request['consumer_op'])}]"
        )
        candidate_text = _build_api_filter_text(request["candidate_apis"])
        blocks.append(
            "=== Consumer API (needs prerequisites) ===\n"
            f"{consumer_text}\n"
            "Candidate prerequisite APIs for the above consumer:\n"
            f"{candidate_text}"
        )
    return "\n\n".join(blocks)


async def _select_api_dependency_candidates_batched(
    *,
    task_instruction: str,
    requests: list[dict],
    model_cfg: ModelConfig,
    io_trace: list[dict] | None = None,
    trace_step: str = "batched_api_dependency_filter",
) -> tuple[list[dict], dict[str, int], dict[str, int | str | None]]:
    """ONE LLM call for ALL consumers in a dependency layer (replaces the
    per-consumer loop). Candidates stay grouped under their consumer in the
    prompt; returns the union of selected prerequisite producers.
    """
    candidate_map: dict[str, dict] = {}
    for request in requests:
        for api in request["candidate_apis"]:
            candidate_map[_api_key_from_spec(api)] = api
    user_prompt = BATCHED_DEP_FILTER_USER_PROMPT.format(
        task_instruction=task_instruction,
        grouped_text=_build_batched_dependency_text(requests),
    )
    result, usage, retry = await _call_llm_json_for_step(
        step=trace_step,
        user_prompt=user_prompt,
        sys_prompt=BATCHED_DEP_FILTER_SYS_PROMPT,
        model_cfg=model_cfg,
        io_trace=io_trace,
    )
    return _extract_selected_candidates(candidate_map, result), usage, retry


def _merge_selected_apis(
    all_filtered: dict[str, dict], selected: list[dict]
) -> list[dict]:
    newly_added: list[dict] = []
    for api in selected:
        key = _api_key_from_spec(api)
        if not key or key in all_filtered:
            continue
        all_filtered[key] = api
        newly_added.append(api)
    return newly_added


def _forward_fallback_community_ids(
    selected_producers: list[dict],
    forward_index: dict[str, list[dict]],
    api_to_community: dict[tuple[str, str], str],
    exclude_cids: set[str],
) -> list[str]:
    """Forward dependency expansion (deterministic, NO LLM).

    For each selected op treated as a PRODUCER, surface the communities of its
    EXACT-edge consumers (APIs that act on the resource it produces). Brings a
    "create X -> act on X" next-step API into routing scope even when Leiden split
    it into a different community. The existing fallback LLM filter then prunes for
    relevance, so recall rises without forward noise reaching the executor.
    """
    cids: set[str] = set()
    for api in selected_producers:
        producer_op = _op_id_from_spec(api)
        for edge in forward_index.get(producer_op, []):
            if edge.get("source") != "exact":  # precision: high-confidence edges only
                continue
            parts = _split_op_id(str(edge.get("consumer_op", "")))
            if parts is None:
                continue
            cid = api_to_community.get(parts)
            if cid and cid not in exclude_cids:
                cids.add(cid)
    return sorted(cids)


def _fallback_community_ids_from_consumer(
    consumer_api: dict,
    communities: dict[str, dict],
    api_to_community: dict[tuple[str, str], str],
    exclude_cids: set[str],
) -> list[str]:
    key = (str(consumer_api.get("app_name", "")), str(consumer_api.get("api_name", "")))
    consumer_community_id = api_to_community.get(key)
    if not consumer_community_id:
        return []
    community = communities.get(consumer_community_id, {})
    community_ids = [
        str(dep.get("community_id", "")).strip()
        for dep in community.get("dependencies", [])
        if isinstance(dep, dict) and str(dep.get("community_id", "")).strip()
    ]
    return sorted({cid for cid in community_ids if cid not in exclude_cids})


def _fallback_community_ids_from_edges(
    dependency_edges: list[dict],
    exclude_cids: set[str],
) -> list[str]:
    community_ids = [
        str(edge.get("producer_community_id", "")).strip()
        for edge in dependency_edges
        if str(edge.get("producer_community_id", "")).strip()
    ]
    return sorted({cid for cid in community_ids if cid not in exclude_cids})


def _make_api_dependency_request(
    *,
    consumer_api: dict,
    dependency_index: dict[str, list[dict]],
    available_field_counts: Counter[str],
    selected_api_keys: set[tuple[str, str]],
    communities: dict[str, dict],
    api_to_community: dict[tuple[str, str], str],
    exclude_fallback_cids: set[str],
) -> tuple[dict | None, list[str]]:
    consumer_op = _op_id_from_spec(consumer_api)
    consumer_param_names = _all_param_names(consumer_api)
    available_fields = _available_fields_from_other_selected_apis(
        consumer_api, available_field_counts
    )
    dependency_edges = dependency_index.get(consumer_op, [])
    matched_edges = [
        edge
        for edge in dependency_edges
        if edge.get("id_field") in consumer_param_names
    ]
    if not matched_edges:
        return None, _fallback_community_ids_from_consumer(
            consumer_api, communities, api_to_community, exclude_fallback_cids
        )

    unresolved_edges = [
        edge for edge in matched_edges if edge.get("id_field") not in available_fields
    ]
    if not unresolved_edges:
        return None, []

    candidate_ops: list[str] = []
    for edge in unresolved_edges:
        producer_op = str(edge.get("producer_op", ""))
        parts = _split_op_id(producer_op)
        if parts is None or parts in selected_api_keys:
            continue
        candidate_ops.append(producer_op)
    candidate_apis, _ = _collect_api_specs_from_ops(
        sorted(candidate_ops), exclude=selected_api_keys
    )
    fallback_cids = sorted(
        set(_fallback_community_ids_from_edges(unresolved_edges, exclude_fallback_cids))
        | set(
            _fallback_community_ids_from_consumer(
                consumer_api, communities, api_to_community, exclude_fallback_cids
            )
        )
    )
    if not candidate_apis:
        return None, fallback_cids

    relevant_unresolved_params = sorted(
        {
            str(edge.get("id_field", "")).strip()
            for edge in unresolved_edges
            if str(edge.get("id_field", "")).strip()
        }
    )
    return (
        {
            "consumer_op": consumer_op,
            "relevant_unresolved_params": relevant_unresolved_params,
            "evidence_pairs": sorted(
                unresolved_edges,
                key=lambda edge: (
                    str(edge.get("producer_op", "")),
                    str(edge.get("id_field", "")),
                    str(edge.get("source", "")),
                ),
            ),
            "candidate_apis": candidate_apis,
            "fallback_community_ids": fallback_cids,
        },
        [],
    )


def _build_api_dependency_requests(
    *,
    selected_consumers: list[dict],
    selected_apis: list[dict],
    dependency_index: dict[str, list[dict]],
    communities: dict[str, dict],
    api_to_community: dict[tuple[str, str], str],
    exclude_fallback_cids: set[str] | None = None,
) -> tuple[list[dict], list[str]]:
    available_field_counts = _build_available_field_counts(selected_apis)
    selected_api_keys = {
        (str(api.get("app_name", "")), str(api.get("api_name", "")))
        for api in selected_apis
    }
    requests: list[dict] = []
    fallback_cids: set[str] = set()
    for consumer_api in sorted(selected_consumers, key=_api_key_from_spec):
        request, consumer_fallback = _make_api_dependency_request(
            consumer_api=consumer_api,
            dependency_index=dependency_index,
            available_field_counts=available_field_counts,
            selected_api_keys=selected_api_keys,
            communities=communities,
            api_to_community=api_to_community,
            exclude_fallback_cids=exclude_fallback_cids or set(),
        )
        if request is not None:
            requests.append(request)
        else:
            fallback_cids.update(consumer_fallback)
    requests.sort(key=lambda request: request["consumer_op"])
    return requests, sorted(fallback_cids)


def _collect_new_community_candidates(
    community_ids: list[str],
    communities: dict[str, dict],
    all_filtered: dict[str, dict],
) -> list[dict]:
    exclude = {
        (str(api.get("app_name", "")), str(api.get("api_name", "")))
        for api in all_filtered.values()
    }
    return _collect_api_specs_from_communities(
        community_ids, communities, exclude=exclude
    )


def _simple_candidate(api: dict) -> dict:
    return {
        "app": str(api.get("app_name", "")),
        "name": str(api.get("api_name", "")),
        "description": str(api.get("description", "")),
        "method": str(api.get("method", "")).upper(),
    }


async def search_apis_by_community_routing(
    task_instruction: str,
    *,
    planned_apps: list[str] | None = None,
    community_layer: bool = True,
    dependency_expansion: bool = True,
    global_pool: bool = False,
    use_guidance: bool = True,
    forward_dependency_expansion: bool = False,
    model_cfg: ModelConfig | None = None,
    io_trace: list[dict] | None = None,
) -> dict:
    # `community_layer` / `dependency_expansion` are the two retrieval-ablation
    # switches (wired by finding/ablation_finders.py). Both default True → the
    # full community-routing pipeline (Variant A), byte-for-byte unchanged.
    #   community_layer=False  → skip the community-select LLM; use every
    #                            community of the planned apps (app-scope pool).
    #   dependency_expansion=False → stop after the single seed_filter pass
    #                            (no producer-API expansion).
    resolved_model_cfg = model_cfg or model_config_from_env()
    communities = _load_communities()
    dependency_index = _load_api_dependency_index()
    forward_dependency_index = (
        _load_forward_dependency_index() if forward_dependency_expansion else {}
    )
    app_context = _load_file(_APP_CONTEXT_PATH)
    behavior_guidelines = _load_file(_BEHAVIOR_GUIDELINES_PATH)
    if not use_guidance:
        # Guidance-ablation (impl "community_noguide"): blank out BOTH the
        # train-induced behavior_guidelines and the app_context hint, in
        # community_select AND seed_filter, to measure how much of the finder's
        # selection accuracy rests on that semi-manually induced guidance vs the
        # model + official API specs alone.
        app_context = ""
        behavior_guidelines = ""
    api_to_community = _build_api_to_community_map()
    usage_metrics = _empty_usage_metrics()
    retry_metrics = _empty_retry_metrics()

    all_apps = sorted(
        {
            str(community.get("app", ""))
            for community in communities.values()
            if community.get("app")
        }
    )
    valid_apps, invalid_apps, app_fallback_used = _apps_from_planner(
        planned_apps, all_apps
    )

    # ── Stage-1 app-select ────────────────────────────────────────────────────
    # Fires only when the planner gave NO usable app (app=None/"all"/invalid).
    # Instead of letting community_select flood over every app's communities (the
    # 986aa4e 8-app slow/429 path), a thin LLM picks the app(s) this milestone
    # needs, then we scope community_select to them. [] = pure-compute milestone
    # (no API) → surface nothing.
    app_select_trace = None
    if community_layer and app_fallback_used:
        app_sel, asu, asr = await _call_llm_json_for_step(
            step="app_select",
            user_prompt=APP_SELECT_USER_PROMPT.format(
                task_instruction=task_instruction,
                app_catalog=_build_app_catalog(communities),
            ),
            sys_prompt=APP_SELECT_SYS_PROMPT,
            model_cfg=resolved_model_cfg,
            io_trace=io_trace,
        )
        _accumulate_usage_metrics(usage_metrics, asu)
        _accumulate_retry_metrics(retry_metrics, asr)
        raw_apps = app_sel.get("apps") if isinstance(app_sel, dict) else None
        picked = [a for a in (raw_apps or []) if isinstance(a, str) and a in all_apps]
        valid_apps = picked  # scope to the picked app(s)
        app_select_trace = {"step": "app_select", "selected_apps": picked}
        if not picked:
            # Pure-compute milestone (no app API needed) → surface nothing. Must
            # return here: an EMPTY app_filter is treated as "no filter" by
            # _build_community_text and would flood every community.
            return {
                "matched_apps": [],
                "fallback_used": app_fallback_used,
                "selected_communities": [],
                "candidate_apis": [],
                "candidate_count": 0,
                "llm_rounds": 1,
                "llm_call_attempts": 1 + int(retry_metrics.get("retry_count") or 0),
                **usage_metrics,
                **retry_metrics,
                "dependency_rounds": 0,
                "routing_trace": [
                    {
                        "step": "planned_app_filter",
                        "planned_apps": planned_apps or [],
                        "matched_apps": [],
                        "invalid_apps": invalid_apps,
                        "fallback_used": True,
                    },
                    app_select_trace,
                ],
            }

    if community_layer:
        community_user_prompt = COMMUNITY_SELECT_USER_PROMPT.format(
            task_instruction=task_instruction,
            community_text=_build_community_text(
                communities, app_filter=set(valid_apps)
            ),
        )
        community_sys_prompt = COMMUNITY_SELECT_SYS_PROMPT.format(
            behavior_guidelines=behavior_guidelines,
        )
        community_result, usage, retry = await _call_llm_json_for_step(
            step="community_select",
            user_prompt=community_user_prompt,
            sys_prompt=community_sys_prompt,
            model_cfg=resolved_model_cfg,
            io_trace=io_trace,
        )
        _accumulate_usage_metrics(usage_metrics, usage)
        _accumulate_retry_metrics(retry_metrics, retry)
        # Graded scoring: keep communities whose LLM score >= threshold. A
        # high-relevance community is no longer dropped by a conservative binary
        # pick (the failure 552869a/634f342 showed). Accept {"scores": {...}} or
        # a bare {community_id: score} map for robustness.
        raw_scores = community_result.get("scores")
        if not isinstance(raw_scores, dict):
            raw_scores = {
                k: v for k, v in community_result.items() if isinstance(v, (int, float))
            }
        min_score = active_config().retrieval.community_select_min_score
        scores = {
            cid: sc
            for cid, sc in raw_scores.items()
            if isinstance(cid, str) and isinstance(sc, (int, float))
        }
        valid_ids = [
            cid
            for cid, sc in sorted(scores.items(), key=lambda kv: -kv[1])
            if cid in communities and sc >= min_score
        ]
        invalid_ids = sorted({cid for cid in scores if cid not in communities})
        base_llm_rounds = 1
        community_select_trace = {
            "step": "community_select",
            "selected_communities": valid_ids,
            "invalid_communities": invalid_ids,
            "scores": scores,
            "min_score": min_score,
        }
    else:
        # No-community ablation (Variants B / B2): skip the community-select LLM
        # and take EVERY community of the planned apps, so the seed_filter LLM
        # sees the whole app's API surface at once (app-scope selection). No LLM
        # call here → base_llm_rounds = 0. The trace keeps the same schema with
        # an explicit skipped=True marker (no silent field drop).
        if global_pool:
            # Global no-narrowing ablation (impl "global_seed_filter"): ignore
            # planned_apps entirely and feed the seed_filter LLM the WHOLE
            # user-facing API surface — every community of every app (all 454
            # ops) — to measure whether selection degrades at full scale. This
            # is the "no retrieval narrowing" condition; the seed_filter still
            # runs, only its candidate pool is global.
            valid_ids = sorted(communities.keys())
        elif app_fallback_used:
            # No app assigned (compute milestone) → do NOT fall back to ALL
            # apps' communities; that would feed all 454 ops to seed_filter — a
            # huge prompt, and ruinous under full-spec rendering. Empty valid_ids
            # → the early-return below yields candidate_apis=[]. Community mode
            # reaches the same [] via community_select returning nothing, so the
            # variants stay consistent on no-app compute milestones.
            valid_ids = []
        else:
            valid_apps_set = set(valid_apps)
            valid_ids = sorted(
                community_id
                for community_id, community in communities.items()
                if str(community.get("app", "")) in valid_apps_set
            )
        invalid_ids = []
        base_llm_rounds = 0
        community_select_trace = {
            "step": "community_select",
            "skipped": True,
            "selected_communities": valid_ids,
            "invalid_communities": invalid_ids,
        }

    route_trace: list[dict] = [
        {
            "step": "planned_app_filter",
            "planned_apps": planned_apps or [],
            "matched_apps": valid_apps,
            "invalid_apps": invalid_apps,
            "fallback_used": app_fallback_used,
        },
    ]
    if app_select_trace is not None:
        route_trace.append(app_select_trace)
    route_trace.append(community_select_trace)

    if not valid_ids:
        return {
            "matched_apps": valid_apps,
            "fallback_used": app_fallback_used,
            "selected_communities": [],
            "candidate_apis": [],
            "candidate_count": 0,
            "llm_rounds": base_llm_rounds,
            "llm_call_attempts": base_llm_rounds
            + int(retry_metrics.get("retry_count") or 0),
            **usage_metrics,
            **retry_metrics,
            "dependency_rounds": 0,
            "routing_trace": route_trace,
        }

    llm_rounds = base_llm_rounds
    dependency_rounds = 0
    all_filtered: dict[str, dict] = {}
    processed_fallback_cids: set[str] = set(valid_ids)

    seed_candidates = _collect_new_community_candidates(
        valid_ids, communities, all_filtered
    )
    route_trace.append(
        {
            "step": "seed_candidates",
            "community_count": len(valid_ids),
            "candidate_count": len(seed_candidates),
        }
    )
    if not seed_candidates:
        return {
            "matched_apps": valid_apps,
            "fallback_used": app_fallback_used,
            "selected_communities": valid_ids,
            "candidate_apis": [],
            "candidate_count": 0,
            "llm_rounds": llm_rounds,
            "llm_call_attempts": llm_rounds
            + int(retry_metrics.get("retry_count") or 0),
            **usage_metrics,
            **retry_metrics,
            "dependency_rounds": 0,
            "routing_trace": route_trace,
        }

    llm_rounds += 1
    seed_selected, usage, retry = await _select_generic_candidates(
        task_instruction=task_instruction,
        candidates=seed_candidates,
        selected_apis=list(all_filtered.values()),
        behavior_guidelines=behavior_guidelines,
        app_context=app_context,
        model_cfg=resolved_model_cfg,
        io_trace=io_trace,
        trace_step="seed_filter",
    )
    _accumulate_usage_metrics(usage_metrics, usage)
    _accumulate_retry_metrics(retry_metrics, retry)
    frontier = _merge_selected_apis(all_filtered, seed_selected)
    route_trace.append(
        {
            "step": "seed_filter",
            "selected_api_keys": sorted(
                _api_key_from_spec(api) for api in seed_selected
            ),
        }
    )
    if not frontier:
        if app_fallback_used:
            # No app was assigned to this milestone (planned_apps empty → app
            # fallback to ALL apps). With no real app selection, the backstop
            # below would dump seed_candidates[:40] = the alphabetically-first
            # ops (amazon-heavy garbage) for a milestone that is almost always
            # pure compute (operates on a prior variable, needs no API). Return
            # [] cleanly instead. This removes a variant-dependent confound from
            # retrieval ablations: app-scope dumped 40 wrong-app APIs here while
            # community dumped fewer, so the BACKSTOP — not the finder under
            # study — decided the outcome (2026-06-06: 9dabbc9/552869a got 40
            # amazon APIs on venmo compute milestones).
            route_trace.append(
                {
                    "step": "empty_filter_backstop",
                    "skipped": True,
                    "reason": "app_fallback_compute_milestone",
                    "seed_candidate_count": len(seed_candidates),
                }
            )
            return {
                "matched_apps": valid_apps,
                "fallback_used": app_fallback_used,
                "selected_communities": valid_ids,
                "candidate_apis": [],
                "candidate_count": 0,
                "llm_rounds": llm_rounds,
                "llm_call_attempts": llm_rounds
                + int(retry_metrics.get("retry_count") or 0),
                **usage_metrics,
                **retry_metrics,
                "dependency_rounds": 0,
                "routing_trace": route_trace,
            }
        # Empty-filter coverage backstop (see _EMPTY_FILTER_BACKSTOP_CAP). The
        # filter rejected every op in the selected communities; rather than
        # returning candidate_apis=[] (executor then guesses → MAX_CYCLES), fall
        # back to the selected communities' own ops so the executor has the
        # right community's APIs to work with.
        backstop = [
            _simple_candidate(api)
            for api in seed_candidates[:_EMPTY_FILTER_BACKSTOP_CAP]
        ]
        route_trace.append(
            {
                "step": "empty_filter_backstop",
                "backstop_candidate_count": len(backstop),
                "seed_candidate_count": len(seed_candidates),
            }
        )
        return {
            "matched_apps": valid_apps,
            "fallback_used": app_fallback_used,
            "selected_communities": valid_ids,
            "candidate_apis": backstop,
            "candidate_count": len(backstop),
            "llm_rounds": llm_rounds,
            "llm_call_attempts": llm_rounds
            + int(retry_metrics.get("retry_count") or 0),
            **usage_metrics,
            **retry_metrics,
            "dependency_rounds": 0,
            "routing_trace": route_trace,
        }

    if not dependency_expansion:
        # No-dependency ablation (Variants B2 / A-no-dep): stop after the single
        # seed_filter pass. all_filtered already holds the seed-selected APIs
        # (merged above), so emptying the frontier skips producer-API expansion
        # while leaving final_specs = the seed selection. dependency_rounds stays
        # 0 and no dependency_round trace entries are appended.
        route_trace.append({"step": "dependency_expansion", "skipped": True})
        frontier = []

    while frontier:
        requests, initial_fallback_cids = _build_api_dependency_requests(
            selected_consumers=frontier,
            selected_apis=list(all_filtered.values()),
            dependency_index=dependency_index,
            communities=communities,
            api_to_community=api_to_community,
            exclude_fallback_cids=processed_fallback_cids,
        )
        forward_fallback_cids: list[str] = []
        if forward_dependency_expansion:
            # Forward expansion: surface the communities of EXACT-edge consumers of
            # the selected ops (producer -> "act on what it produced" next-step APIs),
            # merged into this round's fallback set for the existing LLM filter to prune.
            forward_fallback_cids = _forward_fallback_community_ids(
                frontier,
                forward_dependency_index,
                api_to_community,
                processed_fallback_cids,
            )
            if forward_fallback_cids:
                initial_fallback_cids = sorted(
                    set(initial_fallback_cids) | set(forward_fallback_cids)
                )
        if not requests and not initial_fallback_cids:
            break
        dependency_rounds += 1
        next_frontier: list[dict] = []
        pending_fallback_cids = set(initial_fallback_cids)
        round_trace = {
            "step": "dependency_round",
            "round": dependency_rounds,
            "request_count": len(requests),
            "initial_fallback_communities": sorted(initial_fallback_cids),
            "forward_fallback_communities": forward_fallback_cids,
            "exact_selected": [],
            "fallback_selected": [],
        }

        if requests:
            # ONE batched LLM call per layer (was: one call per consumer). The
            # prompt keeps each consumer's candidate prerequisites grouped under
            # it; the union of selected producers is merged in.
            llm_rounds += 1
            selected, usage, retry = await _select_api_dependency_candidates_batched(
                task_instruction=task_instruction,
                requests=requests,
                model_cfg=resolved_model_cfg,
                io_trace=io_trace,
                trace_step=f"dependency_round_{dependency_rounds}:batched",
            )
            _accumulate_usage_metrics(usage_metrics, usage)
            _accumulate_retry_metrics(retry_metrics, retry)
            newly_added = _merge_selected_apis(all_filtered, selected)
            next_frontier.extend(newly_added)
            round_trace["exact_selected"].append(
                {
                    "consumers": sorted(
                        _display_op_id(r["consumer_op"]) for r in requests
                    ),
                    "selected_api_keys": sorted(
                        _api_key_from_spec(api) for api in newly_added
                    ),
                }
            )
            # a consumer whose candidates were ALL rejected still falls back to
            # its producer communities (preserves the per-consumer fallback path)
            selected_keys = {_api_key_from_spec(api) for api in selected}
            for request in requests:
                if not any(
                    _api_key_from_spec(c) in selected_keys
                    for c in request["candidate_apis"]
                ):
                    pending_fallback_cids.update(
                        request.get("fallback_community_ids", [])
                    )

        fallback_cids = sorted(
            community_id
            for community_id in pending_fallback_cids
            if community_id not in processed_fallback_cids
        )
        if fallback_cids:
            processed_fallback_cids.update(fallback_cids)
            fallback_candidates = _collect_new_community_candidates(
                fallback_cids,
                communities,
                all_filtered,
            )
            if fallback_candidates:
                llm_rounds += 1
                fallback_selected, usage, retry = await _select_generic_candidates(
                    task_instruction=task_instruction,
                    candidates=fallback_candidates,
                    selected_apis=list(all_filtered.values()),
                    behavior_guidelines=behavior_guidelines,
                    app_context=app_context,
                    model_cfg=resolved_model_cfg,
                    io_trace=io_trace,
                    trace_step=f"dependency_round_{dependency_rounds}:fallback_filter",
                )
                _accumulate_usage_metrics(usage_metrics, usage)
                _accumulate_retry_metrics(retry_metrics, retry)
                newly_added = _merge_selected_apis(all_filtered, fallback_selected)
                next_frontier.extend(newly_added)
                round_trace["fallback_selected"] = sorted(
                    _api_key_from_spec(api) for api in newly_added
                )
                round_trace["fallback_communities"] = fallback_cids
        route_trace.append(round_trace)
        frontier = next_frontier

    final_specs = list(all_filtered.values())
    return {
        "matched_apps": valid_apps,
        "fallback_used": app_fallback_used,
        "selected_communities": valid_ids,
        "candidate_apis": [_simple_candidate(api) for api in final_specs],
        "candidate_count": len(final_specs),
        "llm_rounds": llm_rounds,
        "llm_call_attempts": llm_rounds + int(retry_metrics.get("retry_count") or 0),
        **usage_metrics,
        **retry_metrics,
        "dependency_rounds": dependency_rounds,
        "routing_trace": route_trace,
    }


__all__ = [
    "search_apis_by_community_routing",
    "_api_key_from_spec",
    "_build_api_dependency_requests",
    "_build_available_field_counts",
    "_make_api_dependency_request",
]
