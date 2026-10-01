from __future__ import annotations

import json
import os
import re
import textwrap
from collections import defaultdict
from pathlib import Path
from pprint import pformat

from google.adk.tools import FunctionTool

from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder
from adk_appworld_agent.appworld.holder import AppWorldClientHolder
from adk_appworld_agent.observability.log_files import raw_log_detail_enabled
from adk_appworld_agent.subagents.executor.guardrails import (
    unwrap_printed_program_source,
    validate_generated_code,
)

_API_CALL_RE = re.compile(
    r"apis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_AUTH_EXEMPT_APPS = {"supervisor", "api_docs"}
_LOGIN_METHODS = {"login"}
EXECUTOR_CANDIDATE_APIS_ENV = "EXECUTOR_CANDIDATE_APIS_JSON"
EXECUTOR_PRIOR_VARIABLES_ENV = "EXECUTOR_PRIOR_VARIABLES_JSON"
EXECUTOR_TASK_DATETIME_ENV = "EXECUTOR_TASK_DATETIME"
EXECUTOR_SANDBOX_LOG_PATH_ENV = "EXECUTOR_SANDBOX_LOG_PATH"
EXECUTOR_GUARDRAILS_ENV = "EXECUTOR_STAGE2_GUARDRAILS"
EXECUTOR_SANDBOX_FINALIZE_MARKER = "__CODEX_SANDBOX_FINALIZE__:"
_SANDBOX_TRACE_MARKER = "__CODEX_SANDBOX_TRACE__:"


def _collect_app_references(code: str) -> dict[str, set[str]]:
    references: dict[str, set[str]] = defaultdict(set)
    for app_name, api_name in _API_CALL_RE.findall(code):
        references[app_name].add(api_name)
    return dict(references)


def _apps_requiring_prefetch(references: dict[str, set[str]]) -> list[str]:
    apps: list[str] = []
    for app_name, methods in references.items():
        if app_name in _AUTH_EXEMPT_APPS:
            continue
        if methods and methods.issubset(_LOGIN_METHODS):
            continue
        apps.append(app_name)
    return sorted(apps)


def _prefetch_tokens(code: str) -> dict[str, str]:
    auth_manager = AppWorldAuthHolder.get_auth_manager()
    references = _collect_app_references(code)
    tokens: dict[str, str] = {}
    for app_name in _apps_requiring_prefetch(references):
        token = auth_manager.get_access_token(app_name)
        if isinstance(token, str) and token:
            tokens[app_name] = token
    return tokens


def _python_literal(value: object) -> str:
    return pformat(value, sort_dicts=True)


def _load_candidate_apis_from_env() -> list[dict]:
    raw = os.getenv(EXECUTOR_CANDIDATE_APIS_ENV)
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    normalized = [
        _normalize_candidate_api(item) for item in parsed if isinstance(item, dict)
    ]
    return [item for item in normalized if item is not None]


def _normalize_candidate_api(item: dict) -> dict | None:
    app = item.get("app") or item.get("app_name")
    name = item.get("name") or item.get("api_name")
    if not isinstance(app, str) or not isinstance(name, str) or not app or not name:
        return None
    normalized = dict(item)
    normalized["app"] = app
    normalized["name"] = name
    normalized.setdefault("app_name", app)
    normalized.setdefault("api_name", name)
    return normalized


def _load_prior_variables_from_env() -> dict[str, dict]:
    raw = os.getenv(EXECUTOR_PRIOR_VARIABLES_ENV)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        name: value
        for name, value in parsed.items()
        if isinstance(name, str) and isinstance(value, dict)
    }


def _load_task_datetime_from_env() -> str:
    raw = os.getenv(EXECUTOR_TASK_DATETIME_ENV)
    return raw if isinstance(raw, str) else ""


def _stage2_guardrails_enabled() -> bool:
    return os.getenv(EXECUTOR_GUARDRAILS_ENV) == "1"


def _sandbox_log_path() -> Path | None:
    explicit = os.getenv(EXECUTOR_SANDBOX_LOG_PATH_ENV)
    if explicit:
        return Path(explicit)
    executor_log_path = os.getenv("EXECUTOR_LOG_PATH")
    if not executor_log_path:
        return None
    return Path(executor_log_path).with_name("sandbox_api_calls.jsonl")


def _append_sandbox_trace(*, code: str, api_calls: list[dict]) -> str | None:
    if not api_calls:
        return None
    path = _sandbox_log_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "app_references": {
            app_name: sorted(api_names)
            for app_name, api_names in sorted(_collect_app_references(code).items())
        },
        "api_calls": api_calls,
    }
    if raw_log_detail_enabled():
        record["code"] = code
    with path.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path)


def sandbox_trace_byte_offset() -> int:
    """Current byte size of sandbox_api_calls.jsonl, or 0 if not set / not yet created.

    Used by controller to snapshot file size BEFORE an EXECUTE phase so it can
    read only the new records appended DURING that phase.
    """
    path = _sandbox_log_path()
    if path is None or not path.exists():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def read_sandbox_trace_since(start_offset: int) -> list[dict]:
    """Read sandbox_api_calls.jsonl records appended since `start_offset` bytes.

    Returns parsed records (each = `{app_references, api_calls, [code]}`).
    Empty list if file missing, no new bytes, or all new content malformed.
    """
    path = _sandbox_log_path()
    if path is None or not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size <= start_offset:
        return []
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(start_offset)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except OSError:
        return []
    return records


def summarize_trace_for_prompt(
    api_calls: list[dict],
    *,
    max_calls: int = 20,
    max_item_field_chars: int = 200,
) -> list[dict]:
    """Normalize api_calls into a compact list[dict] for state.history storage.

    Drops access_token + sensitive kwargs, keeps app/api_name/kwargs/
    result_shape/result_items. Passes `result_items` through unchanged
    (the structure already comes pre-truncated from sandbox-side
    `_bounded_result_items`) — the downstream render layer
    (data_summarizer helper) handles smart compression at prompt-render
    time.

    Defaults are intentionally loose. Sandbox already caps at
    `_runtime_max_list_items` per call (10) with per-field string
    truncation; this function just sanitizes + passes through.
    """
    out: list[dict] = []
    if not isinstance(api_calls, list):
        return out
    truncated_calls = len(api_calls) > max_calls
    for call in api_calls[:max_calls]:
        if not isinstance(call, dict):
            continue
        kwargs = call.get("kwargs") or {}
        if isinstance(kwargs, dict):
            kwargs_clean = {k: v for k, v in kwargs.items() if k != "access_token"}
        else:
            kwargs_clean = kwargs
        entry: dict[str, object] = {
            "app": call.get("app"),
            "api_name": call.get("api_name"),
            "kwargs": kwargs_clean,
        }
        status = call.get("status")
        if status and status != "ok":
            entry["status"] = status
            err = call.get("error")
            if err:
                entry["error"] = str(err)[:max_item_field_chars]
        shape = call.get("result_shape")
        if isinstance(shape, dict):
            entry["result_shape"] = shape
        # Pass result_items through unchanged — render layer
        # (data_summarizer.summarize_data_for_llm) handles compression.
        ri = call.get("result_items")
        if ri is not None:
            entry["result_items"] = ri
        out.append(entry)
    if truncated_calls:
        out.append({"_more_calls": len(api_calls) - max_calls})
    return out


def _split_sandbox_trace(raw_output: str) -> tuple[str, list[dict]]:
    if not raw_output:
        return "", []
    lines = raw_output.splitlines()
    visible_lines: list[str] = []
    api_calls: list[dict] = []
    finalize_args: dict | None = None
    for line in lines:
        if line.startswith(EXECUTOR_SANDBOX_FINALIZE_MARKER):
            payload = line[len(EXECUTOR_SANDBOX_FINALIZE_MARKER) :].strip()
            if payload:
                try:
                    parsed = json.loads(payload)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    finalize_args = parsed
            continue
        if line.startswith(_SANDBOX_TRACE_MARKER):
            payload = line[len(_SANDBOX_TRACE_MARKER) :].strip()
            if not payload:
                continue
            try:
                parsed = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, list):
                api_calls = [item for item in parsed if isinstance(item, dict)]
            continue
        visible_lines.append(line)
    visible_output = "\n".join(visible_lines).strip()
    if finalize_args is not None:
        marker = EXECUTOR_SANDBOX_FINALIZE_MARKER + json.dumps(
            finalize_args, ensure_ascii=False
        )
        visible_output = f"{visible_output}\n{marker}".strip()
    return visible_output, api_calls


def _build_sandbox_prelude(
    *,
    profile: dict,
    passwords: dict[str, str],
    tokens: dict[str, str],
    candidate_apis: list[dict],
    prior_variables: dict[str, dict],
    task_datetime: str,
) -> str:
    # Embed prior_variables as a JSON STRING literal (parsed at sandbox
    # runtime via json.loads) instead of pformat'd dict literal. For large
    # variables (e.g. 88KB JSON of 237 transactions), pformat output runs
    # to thousands of lines of nested Python literals which libcst's
    # safety_guard fails to parse, causing EXECUTOR_DID_NOT_FINALIZE on
    # any milestone that consumes such variables. A single string literal,
    # however long, is one token to libcst — no recursive-parse issue.
    _prior_vars_json_literal = repr(json.dumps(prior_variables, ensure_ascii=False))
    return f"""import json as _runtime_json
_runtime_profile = {_python_literal(profile)}
_runtime_passwords = {_python_literal(passwords)}
_runtime_tokens = {_python_literal(tokens)}
_runtime_candidate_apis = {_python_literal(candidate_apis)}
_runtime_prior_variables = _runtime_json.loads({_prior_vars_json_literal})
_runtime_task_datetime = {_python_literal(task_datetime)}
_api_doc_schema_cache = {{}}
_runtime_api_call_log = []
_runtime_finalize_calls = []
_runtime_sandbox_trace_marker = {_python_literal(_SANDBOX_TRACE_MARKER)}
_runtime_sandbox_finalize_marker = {_python_literal(EXECUTOR_SANDBOX_FINALIZE_MARKER)}
_runtime_raw_log_detail = {_python_literal(raw_log_detail_enabled())}
_runtime_max_list_items = 10
_runtime_max_bytes_per_rec = 16000
_runtime_max_str_field_chars = 150
_runtime_max_bytes_total = 200000
_runtime_trace_total_bytes = [0]
import datetime as _runtime_datetime
if "_appworld_raw_apis" not in globals():
    _appworld_raw_apis = apis
_original_apis = _appworld_raw_apis
_candidate_apis_by_app = {{}}
for _item in _runtime_candidate_apis:
    if not isinstance(_item, dict):
        continue
    _app_name = _item.get("app")
    if isinstance(_app_name, str):
        _candidate_apis_by_app.setdefault(_app_name, []).append(dict(_item))
_candidate_api_names_by_app = {{}}
for _app_name, _items in _candidate_apis_by_app.items():
    _names = sorted(
        {{
            _entry.get("name")
            for _entry in _items
            if isinstance(_entry, dict) and isinstance(_entry.get("name"), str)
        }}
    )
    _candidate_api_names_by_app[_app_name] = _names


def _cache_key(app_name, api_name):
    return f"{{app_name}}.{{api_name}}"


def _preview_value(value, *, limit=240):
    text = repr(_redact_value(value))
    if len(text) > limit:
        return text[:limit] + f"...<+{{len(text) - limit}}c>"
    return text


def _result_shape(value):
    if value is None:
        return {{"type": "none"}}
    if isinstance(value, list):
        shape = {{
            "type": "list",
            "count_hint": len(value),
        }}
        dict_keys = set()
        for item in value[:3]:
            if isinstance(item, dict):
                dict_keys.update(str(key) for key in item.keys())
        if dict_keys:
            shape["keys"] = sorted(dict_keys)
        return shape
    if isinstance(value, dict):
        return {{
            "type": "object",
            "keys": sorted(str(key) for key in value.keys()),
        }}
    return {{
        "type": type(value).__name__,
    }}


def _redact_value(value):
    if isinstance(value, dict):
        safe = {{}}
        for key, item in value.items():
            if key == "access_token" and isinstance(item, str):
                safe[key] = "<redacted>"
                continue
            safe[key] = _redact_value(item)
        return safe
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_value(item) for item in value)
    return value


def _sanitize_kwargs(kwargs):
    safe = {{}}
    for key, value in kwargs.items():
        if key == "access_token" and isinstance(value, str):
            safe[key] = "<redacted>"
            continue
        safe[key] = _preview_value(value, limit=120)
    return safe


def _truncate_str_fields(item, *, max_chars):
    if isinstance(item, dict):
        out = {{}}
        for k, v in item.items():
            if isinstance(v, str) and len(v) > max_chars:
                out[k] = v[:max_chars] + "…"
            elif isinstance(v, dict):
                out[k] = _truncate_str_fields(v, max_chars=max_chars)
            elif isinstance(v, list):
                out[k] = [_truncate_str_fields(x, max_chars=max_chars) if isinstance(x, (dict, list)) else (x[:max_chars] + "…" if isinstance(x, str) and len(x) > max_chars else x) for x in v]
            else:
                out[k] = v
        return out
    if isinstance(item, list):
        return [_truncate_str_fields(x, max_chars=max_chars) for x in item]
    if isinstance(item, str) and len(item) > max_chars:
        return item[:max_chars] + "…"
    return item


def _bounded_result_items(value):
    if isinstance(value, list):
        return {{
            "_list_total": len(value),
            "_truncated": len(value) > _runtime_max_list_items,
            "items": [
                _truncate_str_fields(_redact_value(item), max_chars=_runtime_max_str_field_chars)
                for item in value[:_runtime_max_list_items]
            ],
        }}
    if isinstance(value, (dict, str, list)):
        return _truncate_str_fields(_redact_value(value), max_chars=_runtime_max_str_field_chars)
    return _redact_value(value)


def _attach_result_items(entry, result):
    if _runtime_trace_total_bytes[0] >= _runtime_max_bytes_total:
        entry["result_items"] = {{"_global_truncated": True}}
        return
    items = _bounded_result_items(result)
    try:
        blob = _runtime_json.dumps(items, default=str, ensure_ascii=False)
    except Exception as exc:
        entry["result_items"] = {{"_serialization_failed": True, "_error": str(exc)}}
        return
    if len(blob) > _runtime_max_bytes_per_rec:
        entry["result_items"] = {{
            "_oversize": True,
            "_size_bytes": len(blob),
            "_type": type(result).__name__,
        }}
        _runtime_trace_total_bytes[0] += 120
        return
    entry["result_items"] = items
    _runtime_trace_total_bytes[0] += len(blob)


def _record_api_call(kind, *, app_name, api_name, kwargs=None, status, result=None, error=None, meta=None):
    entry = {{
        "kind": kind,
        "app": app_name,
        "api_name": api_name,
        "kwargs": _sanitize_kwargs(kwargs or {{}}),
        "status": status,
    }}
    if result is not None:
        if _runtime_raw_log_detail:
            entry["result_preview"] = _preview_value(result)
        entry["result_shape"] = _result_shape(_redact_value(result))
        _attach_result_items(entry, result)
    if error is not None:
        entry["error"] = str(error)
    if meta:
        entry["meta"] = dict(meta)
    _runtime_api_call_log.append(entry)


def _runtime_finalize(answer, milestone_done, summary, variables_json):
    args = {{
        "answer": str(answer),
        "milestone_done": bool(milestone_done),
        "summary": str(summary),
        "variables_json": str(variables_json),
    }}
    _runtime_finalize_calls.append(args)
    return {{"ok": True}}


class _RuntimeFinalizer:
    def finalize(self, *, answer, milestone_done, summary, variables_json):
        return _runtime_finalize(
            answer=answer,
            milestone_done=milestone_done,
            summary=summary,
            variables_json=variables_json,
        )


def _candidate_api_names(app_name):
    return _candidate_api_names_by_app.get(app_name, [])


def _validate_candidate_api(app_name, api_name):
    if _runtime_candidate_apis and app_name not in _candidate_apis_by_app:
        raise Exception(
            f"Finder candidate_apis does not include {{app_name}}.{{api_name}}. "
            f"Valid apps: {{', '.join(sorted(_candidate_apis_by_app))}}"
        )
    names = _candidate_api_names(app_name)
    if names and api_name not in names:
        raise Exception(
            f"Finder candidate_apis does not include {{app_name}}.{{api_name}}. "
            f"Valid APIs: {{', '.join(names)}}"
        )


def _get_api_doc(app_name, api_name):
    _validate_candidate_api(app_name, api_name)
    key = _cache_key(app_name, api_name)
    if key not in _api_doc_schema_cache:
        _api_doc_schema_cache[key] = _original_apis.api_docs.show_api_doc(
            app_name=app_name,
            api_name=api_name,
        )
    return _api_doc_schema_cache[key]


def _parameter_names(api_doc):
    if not isinstance(api_doc, dict):
        return set()
    names = set()
    for parameter in api_doc.get("parameters", []):
        if isinstance(parameter, dict):
            name = parameter.get("name")
        else:
            name = parameter
        if isinstance(name, str):
            names.add(name)
    return names


def _needs_access_token(app_name, api_name):
    return "access_token" in _parameter_names(_get_api_doc(app_name, api_name))


class _WrappedApiDocs:
    def __init__(self, raw_api_docs):
        self._raw_api_docs = raw_api_docs

    def show_api_descriptions(self, *, app_name):
        candidates = _candidate_apis_by_app.get(app_name)
        if candidates:
            result = [dict(item) for item in candidates]
            _record_api_call(
                "api_docs.show_api_descriptions",
                app_name=app_name,
                api_name="show_api_descriptions",
                kwargs={{"app_name": app_name}},
                status="ok",
                result=result,
                meta={{"source": "finder_candidates"}},
            )
            return result
        try:
            result = self._raw_api_docs.show_api_descriptions(app_name=app_name)
            _record_api_call(
                "api_docs.show_api_descriptions",
                app_name=app_name,
                api_name="show_api_descriptions",
                kwargs={{"app_name": app_name}},
                status="ok",
                result=result,
                meta={{"source": "runtime_api_docs"}},
            )
            return result
        except Exception as exc:
            _record_api_call(
                "api_docs.show_api_descriptions",
                app_name=app_name,
                api_name="show_api_descriptions",
                kwargs={{"app_name": app_name}},
                status="error",
                error=exc,
                meta={{"source": "runtime_api_docs"}},
            )
            raise

    def show_api_doc(self, *, app_name, api_name):
        try:
            result = _get_api_doc(app_name, api_name)
            _record_api_call(
                "api_docs.show_api_doc",
                app_name=app_name,
                api_name=api_name,
                kwargs={{"app_name": app_name, "api_name": api_name}},
                status="ok",
                result=result,
            )
            return result
        except Exception as exc:
            _record_api_call(
                "api_docs.show_api_doc",
                app_name=app_name,
                api_name=api_name,
                kwargs={{"app_name": app_name, "api_name": api_name}},
                status="error",
                error=exc,
            )
            raise

    def __getattr__(self, name):
        return getattr(self._raw_api_docs, name)


class _WrappedApp:
    def __init__(self, app_name, raw_app):
        self._app_name = app_name
        self._raw_app = raw_app

    def __getattr__(self, api_name):
        _validate_candidate_api(self._app_name, api_name)
        raw_api = getattr(self._raw_app, api_name)
        if not callable(raw_api):
            return raw_api

        def _call(*args, **kwargs):
            final_kwargs = dict(kwargs)
            injected_access_token = False
            overrode_caller_access_token = False
            if _needs_access_token(self._app_name, api_name):
                token = _runtime_tokens.get(self._app_name)
                if token:
                    # Always use the runtime token. If the caller didn't pass
                    # one (or passed None), record as injection; if the caller
                    # passed a different non-empty value (hallucinated email /
                    # fake placeholder string), record as override and replace.
                    caller_token = final_kwargs.get("access_token")
                    if "access_token" not in final_kwargs or caller_token is None:
                        injected_access_token = True
                    elif caller_token != token:
                        overrode_caller_access_token = True
                    final_kwargs["access_token"] = token
            try:
                result = raw_api(*args, **final_kwargs)
                _record_api_call(
                    "app_api",
                    app_name=self._app_name,
                    api_name=api_name,
                    kwargs=final_kwargs,
                    status="ok",
                    result=result,
                    meta={{"injected_access_token": injected_access_token, "overrode_caller_access_token": overrode_caller_access_token}},
                )
                return result
            except Exception as exc:
                _record_api_call(
                    "app_api",
                    app_name=self._app_name,
                    api_name=api_name,
                    kwargs=final_kwargs,
                    status="error",
                    error=exc,
                    meta={{"injected_access_token": injected_access_token, "overrode_caller_access_token": overrode_caller_access_token}},
                )
                raise

        return _call


class _ApisProxy:
    def __getattr__(self, app_name):
        if app_name == "default_api":
            return default_api
        raw_app = getattr(_original_apis, app_name)
        if app_name == "api_docs":
            return _WrappedApiDocs(raw_app)
        if app_name in {_python_literal(_AUTH_EXEMPT_APPS)}:
            return raw_app
        return _WrappedApp(app_name, raw_app)


profile = dict(_runtime_profile)
passwords = dict(_runtime_passwords)
tokens = dict(_runtime_tokens)
candidate_apis = list(_runtime_candidate_apis)
prior_variables = dict(_runtime_prior_variables)
prior_variable_values = {{}}
for _name, _record in prior_variables.items():
    if not isinstance(_record, dict):
        continue
    _value_json = _record.get("value_json")
    if not isinstance(_value_json, str) or not _value_json.strip():
        continue
    try:
        prior_variable_values[_name] = _runtime_json.loads(_value_json)
    except Exception:
        pass
candidate_api_names_by_app = {{
    _app_name: list(_names)
    for _app_name, _names in _candidate_api_names_by_app.items()
}}
task_datetime = _runtime_task_datetime
task_datetime_dt = (
    _runtime_datetime.datetime.fromisoformat(_runtime_task_datetime)
    if _runtime_task_datetime
    else None
)
task_date = task_datetime_dt.date() if task_datetime_dt is not None else None
api_doc_schema_cache = _api_doc_schema_cache
finalize = _runtime_finalize
default_api = _RuntimeFinalizer()
apis = _ApisProxy()
"""


def _build_sandbox_script(
    code: str,
    *,
    profile: dict,
    passwords: dict[str, str],
    tokens: dict[str, str],
    candidate_apis: list[dict],
    prior_variables: dict[str, dict],
    task_datetime: str,
) -> str:
    prelude = _build_sandbox_prelude(
        profile=profile,
        passwords=passwords,
        tokens=tokens,
        candidate_apis=candidate_apis,
        prior_variables=prior_variables,
        task_datetime=task_datetime,
    )
    indented_code = "\n".join(
        f"    {line}" if line else "" for line in code.splitlines()
    )
    return (
        f"{prelude}\n"
        "try:\n"
        f"{indented_code}\n"
        # An apis.* call that RAISED is an OBSERVABLE outcome (e.g. a 422
        # "insufficient balance"), already recorded by the proxy as
        # status='error'. The AppWorld sandbox would otherwise return ONLY the
        # traceback and DISCARD this stdout — so api_trace ends up [] and the
        # planner misreads it as "no API call was made" (the 2c544f9 / 9ef798c
        # root: a real, actionable error becomes invisible). Catch it here so
        # execute_python returns stdout normally: emit a STRUCTURED error result
        # (committed value carries the real error) and let the finally print the
        # trace marker (which now includes the failed call). A PURE code bug
        # (SyntaxError / NameError / no failed api call recorded) re-raises, so
        # the traceback path + inner repair loop are unchanged.
        "except Exception as _exc:\n"
        # Only swallow a failed REAL app API call (kind='app_api') — i.e. the app
        # itself rejected the request. Guardrail / api-doc rejections (kind=
        # 'api_docs.*', e.g. "candidate_apis does not include X") and pure code
        # bugs must still propagate to the traceback + inner-repair path.
        "    if _runtime_api_call_log and _runtime_api_call_log[-1].get('status') == 'error' and _runtime_api_call_log[-1].get('kind') == 'app_api':\n"
        "        _failed = _runtime_api_call_log[-1]\n"
        "        _err = str(_failed.get('error'))\n"
        "        print(_runtime_json.dumps({'value': None, 'summary': 'API ' + str(_failed.get('app')) + '.' + str(_failed.get('api_name')) + ' rejected the call: ' + _err, 'error': _err, 'description': 'The API call failed with this error -- an observable outcome to act on, not a code bug.', 'answer': 'null'}, ensure_ascii=False))\n"
        "    else:\n"
        "        raise\n"
        "finally:\n"
        "    if _runtime_finalize_calls:\n"
        "        print(_runtime_sandbox_finalize_marker + _runtime_json.dumps(_runtime_finalize_calls[-1], ensure_ascii=False))\n"
        "    print(_runtime_sandbox_trace_marker + _runtime_json.dumps(_runtime_api_call_log, ensure_ascii=False))\n"
        # Restore unproxied apis so the next milestone's auth prefetch (which
        # goes through AppWorld's persistent execute_function) isn't blocked
        # by THIS milestone's candidate-api whitelist.
        "    apis = _original_apis\n"
    )


_BROKEN_NEWLINE_LITERAL_PATTERN = re.compile(r"(['\"])\n\1")
_ESCAPED_SOURCE_QUOTE_PATTERN = re.compile(r"(?<!\\)\\(['\"])")


def repair_broken_newline_escapes(code: str) -> str:
    """Repair `'<real-newline>'` left behind in extracted Python source.

    Two distinct callers introduce this pattern:

    1. Gemini Flash tool-call serialization sometimes lowers the JSON string
       ``'\\n'`` (the two-character escape sequence) to a real newline
       character before our process sees it.

    2. ``unwrap_printed_program_source`` extracts the inner string of a
       ``program = "..."; print(program)`` wrapper via ``ast.value.value``;
       Python's AST decoder evaluates string-literal escape sequences, so
       a correctly-escaped ``split('\\n')`` inside the wrapper string
       becomes ``split('<real-newline>')`` after extraction. This is the
       a30375d_2 / 7847649_2 / d194965_2 / 0d01c76_2 cluster from full56
       v1 (~4-5 task ✗ NOT_FINALIZE attributable to this single pattern).

    Either way the result is invalid Python: ``SyntaxError: unterminated
    string literal``. Replace the ``'<real-newline>'`` /
    ``"<real-newline>"`` pattern with proper escape sequences so the code
    compiles. The sanitizer must run AFTER any AST extraction step, not
    just on the raw model output, because the extraction itself is what
    creates the broken bytes for case (2).
    """
    return _BROKEN_NEWLINE_LITERAL_PATTERN.sub(r"\1\\n\1", code)


def repair_escaped_source_quotes(code: str) -> str:
    """Return a candidate repair for JSON-style source quote escaping.

    Gemini tool-call arguments can arrive with source delimiters escaped as if
    the code were still inside a JSON string, e.g. ``directory_path=\'~/x\'``.
    The backslash is outside any string literal, so Python raises
    ``SyntaxError: unexpected character after line continuation character``.

    This function is intentionally only a candidate transform: valid Python can
    also contain escaped quotes inside string literals. Callers should prefer
    the original source whenever it compiles.
    """
    return _ESCAPED_SOURCE_QUOTE_PATTERN.sub(r"\1", code)


def _compiles_as_python(code: str) -> bool:
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError:
        return False
    return True


_SMART_QUOTE_TRANSLATION = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def repair_smart_quotes(code: str) -> str:
    """ASCII-normalize Unicode curly quotes (a Gemini emission artifact that makes
    the program unparseable when they appear as string delimiters)."""
    return code.translate(_SMART_QUOTE_TRANSLATION)


def repair_generated_source_artifacts(code: str) -> str:
    """Normalize deterministic serialization artifacts in generated code."""
    newline_repaired = repair_broken_newline_escapes(code)
    if _compiles_as_python(newline_repaired):
        return newline_repaired

    quote_repaired = repair_broken_newline_escapes(
        repair_escaped_source_quotes(newline_repaired)
    )
    if _compiles_as_python(quote_repaired):
        return quote_repaired

    # Last tier: curly-quote delimiters (U+2018/2019/201C/201D). Applied only when
    # the code does not otherwise compile, so string CONTENT keeps its original
    # characters whenever the program is already valid.
    smart_repaired = repair_broken_newline_escapes(
        repair_escaped_source_quotes(repair_smart_quotes(newline_repaired))
    )
    if _compiles_as_python(smart_repaired):
        return smart_repaired

    # Final tier: trailing-garbage recovery. gemini on some servings appends
    # junk AFTER the program's required final print(json.dumps(...)) line — an
    # extra ')', a curly-quote fence, markdown prose, non-ASCII tokens. Rather
    # than pattern-match each variant, trim to the largest compilable prefix
    # using the compiler's OWN reported error position. Reached only when
    # nothing above compiled, so valid code is never trimmed.
    salvaged = _extract_compilable_program(smart_repaired)
    if _compiles_as_python(salvaged):
        return salvaged
    return newline_repaired


def _is_trailing_garbage(suffix: str) -> bool:
    """True when a trimmed-off suffix is clearly junk — a curly-quote/backtick
    fence, non-ASCII tokens, markdown prose, or only redundant closing
    punctuation/whitespace — rather than a real (merely mis-indented or buggy)
    statement that the repair loop should see and fix. This is what keeps the
    salvage from masking genuine errors."""
    s = suffix.strip()
    if not s:
        return True
    if any(ord(ch) > 127 for ch in s) or any(ch in "`‘’“”" for ch in s):
        return True
    if "**" in s or s.lstrip().startswith("#"):
        return True
    if all(ch in ")]},; \t\n" for ch in s):
        return True
    return False


def _extract_compilable_program(code: str) -> str:
    """Recover the valid program when trailing GARBAGE was appended after it.

    Iteratively truncates at the compiler's reported SyntaxError position until
    the prefix compiles, then salvages ONLY IF the removed suffix is garbage
    (`_is_trailing_garbage`). If the removed part looks like real code — e.g. a
    genuinely mis-indented statement — the original is returned unchanged so the
    repair loop still sees the real error. Only meaningful for non-compiling
    input; the caller gates on that, so valid code is never touched."""
    cur = code
    for _ in range(12):
        try:
            compile(cur, "<generated>", "exec")
            break
        except SyntaxError as exc:
            lines = cur.split("\n")
            ln = exc.lineno or len(lines)
            if not (1 <= ln <= len(lines)):
                return code
            col = (
                exc.offset
                if (exc.offset and exc.offset >= 1)
                else len(lines[ln - 1]) + 1
            )
            abs_off = sum(len(line) + 1 for line in lines[: ln - 1]) + (col - 1)
            if abs_off >= len(cur):
                abs_off = len(cur) - 1
            nxt = cur[:abs_off] if abs_off > 0 else ""
            if not nxt.strip() or nxt == cur:
                return code
            cur = nxt
    else:
        return code
    if _is_trailing_garbage(code[len(cur) :]):
        return cur
    return code


# Backwards-compat alias for old internal callers / tests.
_repair_broken_newline_escapes = repair_broken_newline_escapes


# A code fence is a run of >=3 backticks OR curly-quote glyphs. gemini-2.5-flash
# on some Vertex servings emits a CURLY-QUOTE fence (e.g. ``‘‘‘import json``)
# instead of a backtick fence — the wrapped program is intact, only the delimiter
# is wrong, so a backtick-only strip left the leading ``‘‘‘`` in place and the
# code failed to compile. Treat both delimiters as fences. (Observed 2026-07-05
# after a project switch: 0/12 vs 2/4 tasks; the code body itself was correct.)
_FENCE_CHARS = "`‘’“”"
_LEADING_FENCE = re.compile(
    rf"^[{re.escape(_FENCE_CHARS)}]{{3,}}[ \t]*([A-Za-z]+[ \t]*\n)?"
)
_ANY_FENCE = re.compile(rf"[{re.escape(_FENCE_CHARS)}]{{3,}}")


def clean_execute_python_code(raw: str) -> str:
    """Normalize model-generated execute_python code before sandbox execution."""
    stripped = raw.strip()
    lead = _LEADING_FENCE.match(stripped)
    if lead:
        # Strip the opening fence (+ optional ``python``/``json`` language word),
        # then drop the closing fence and any trailing prose/garbage after it.
        inner = stripped[lead.end() :]
        trail = _ANY_FENCE.search(inner)
        if trail:
            inner = inner[: trail.start()]
        body = textwrap.dedent(inner).strip()
        return repair_generated_source_artifacts(body)
    # Triple-quote fence: gemini also wraps code in ``'''python … '''`` or
    # ``\"\"\"python … \"\"\"`` (straight triple-quotes standing in for ```).
    # These delimiters are ALSO legal Python (docstrings), so require a language
    # tag right after the opening fence (a docstring is virtually never exactly
    # ``'''python\n``) AND accept the unwrap only if it compiles — a real
    # leading docstring compiles as-is and is never reached here.
    tq = re.match(r"^('''|\"\"\")(?:python|json|tool_code|text)[ \t]*\n", stripped)
    if tq:
        inner = stripped[tq.end() :]
        idx = inner.rfind(tq.group(1))
        if idx != -1:
            inner = inner[:idx]
        candidate = textwrap.dedent(inner).strip()
        # Accept the unwrap ONLY if it compiles. A ``'''python … '''`` fence
        # unwraps to runnable code; a genuine leading docstring unwraps to prose
        # that does not compile, so the original is kept untouched.
        if candidate and _compiles_as_python(candidate):
            return repair_generated_source_artifacts(candidate)
    # Dedent BEFORE stripping. Stripping first removes the first line's leading
    # whitespace, which prevents textwrap.dedent from detecting the common
    # indent shared by every line — a frequent Gemini tool-call quirk where
    # every line is prefixed with one space.
    body = textwrap.dedent(raw).strip()
    return repair_generated_source_artifacts(body)


def execute_python(code: str) -> str:
    """Execute Python code in AppWorld with deterministic auth injection."""
    code = clean_execute_python_code(code)
    auth_manager = AppWorldAuthHolder.get_auth_manager()
    profile = auth_manager.get_profile()
    candidate_apis = _load_candidate_apis_from_env()
    if _stage2_guardrails_enabled():
        code = unwrap_printed_program_source(code)
        # AST extraction inside unwrap collapses correctly-escaped
        # ``'\n'`` into a real newline; re-escape before validation/run
        # so we don't blow up on a self-inflicted SyntaxError.
        code = repair_generated_source_artifacts(code)
        validate_generated_code(
            code,
            candidate_apis=candidate_apis,
            allowed_profile_keys=set(profile.keys()),
        )
    client = AppWorldClientHolder.get_client()
    passwords = auth_manager.get_passwords()
    tokens = _prefetch_tokens(code)
    prior_variables = _load_prior_variables_from_env()
    task_datetime = _load_task_datetime_from_env()
    script = _build_sandbox_script(
        code,
        profile=profile,
        passwords=passwords,
        tokens=tokens,
        candidate_apis=candidate_apis,
        prior_variables=prior_variables,
        task_datetime=task_datetime,
    )
    raw_output = client.execute_python(script)
    visible_output, api_calls = _split_sandbox_trace(raw_output)
    _append_sandbox_trace(code=code, api_calls=api_calls)
    return visible_output


async def _execute_python_tool(code: str) -> str:
    """Async tool entry. Runs the (sync) execute_python on the RPC client's
    DEDICATED worker thread via run_in_executor, so the synchronous zerorpc
    round-trip never blocks the shared asyncio event loop. Blocking the loop
    inline is what previously starved the loop and made the next finder request
    hang to its deadline (project_finder_429_client_bug). Falls back to inline
    for non-RPC (fake/test) clients that have no worker executor.
    """
    import asyncio

    client = AppWorldClientHolder.get_client()
    executor = getattr(client, "rpc_executor", None)
    if executor is None:
        return execute_python(code)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, execute_python, code)


# Preserve the tool name the model is instructed to call (ADK derives the tool
# name from func.__name__); the sync `execute_python` stays for direct/test use.
_execute_python_tool.__name__ = "execute_python"
_execute_python_tool.__doc__ = execute_python.__doc__

EXECUTE_PYTHON_TOOL = FunctionTool(_execute_python_tool)


def finalize(
    answer: str,
    milestone_done: bool,
    summary: str,
    variables_json: str,
) -> dict:
    """Signal the current milestone is complete and hand typed results to the controller.

    Call this exactly ONCE when the milestone is done. Do not emit plain text
    as a termination signal.

    Args:
        answer: Final answer for the task (for query tasks) or "null" (for action tasks).
        milestone_done: True if the milestone's intent was accomplished.
        summary: One-sentence description of what was accomplished.
        variables_json: JSON array of named variables, e.g.
            '[{"name":"wife_email","value_json":"\\"sarah@ex.com\\"","description":"Email address for the user's wife."}]'
            Each entry has: name (snake_case), value_json (JSON-stringified full
            value), and description. Use "[]" if nothing worth naming.
    """
    return {"ok": True}


FINALIZE_TOOL = FunctionTool(finalize)


def submit_final(
    value: object,
    summary: str,
    description: str,
    answer: str,
) -> dict:
    """Submit the milestone result via a schema-enforced tool call.

    Call this EXACTLY ONCE after execute_python returns, with the
    structured result the orchestrator should commit. The four args
    correspond to the same four-field contract previously printed as
    `json.dumps(...)` inside the executed code. Passing them through a
    tool call instead removes the "wrap-in-string then exec" failure
    mode that Gemini Flash periodically falls into.

    Args:
        value: The raw data the next milestone reads. Pass as a NATIVE
            Python value (list / dict / scalar / None) — do NOT
            json-stringify it. For mutation-only milestones whose
            outcome is the side effect itself, pass `None`.
        summary: One sentence describing what this milestone DID.
        description: One sentence describing the STRUCTURE of `value`
            (field names + shape). Empty string is allowed when
            `value` is a primitive scalar.
        answer: The terminal answer string for query terminal
            milestones, OR the four-character literal `"null"` for
            mutation terminal milestones and intermediate milestones.
    """
    return {"ok": True}


SUBMIT_FINAL_TOOL = FunctionTool(submit_final)

__all__ = [
    "EXECUTE_PYTHON_TOOL",
    "FINALIZE_TOOL",
    "SUBMIT_FINAL_TOOL",
    "_apps_requiring_prefetch",
    "_load_candidate_apis_from_env",
    "_load_task_datetime_from_env",
    "_build_sandbox_script",
    "_collect_app_references",
    "EXECUTOR_CANDIDATE_APIS_ENV",
    "EXECUTOR_PRIOR_VARIABLES_ENV",
    "EXECUTOR_SANDBOX_LOG_PATH_ENV",
    "EXECUTOR_TASK_DATETIME_ENV",
    "clean_execute_python_code",
    "execute_python",
    "finalize",
    "repair_escaped_source_quotes",
    "repair_generated_source_artifacts",
    "submit_final",
]
