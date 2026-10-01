from __future__ import annotations

import contextlib
import io
import os

import pytest

from adk_appworld_agent.appworld.auth_holder import AppWorldAuthHolder
from adk_appworld_agent.appworld.holder import AppWorldClientHolder
from adk_appworld_agent.subagents.executor.appworld_tools import (
    EXECUTOR_CANDIDATE_APIS_ENV,
    EXECUTOR_GUARDRAILS_ENV,
    EXECUTOR_PRIOR_VARIABLES_ENV,
    EXECUTOR_SANDBOX_LOG_PATH_ENV,
    EXECUTOR_TASK_DATETIME_ENV,
    execute_python,
)


class _FakeAuthManager:
    def __init__(self) -> None:
        self.token_calls: list[str] = []

    def get_profile(self) -> dict:
        return {"email": "clmiller@gmail.com", "phone_number": "2125442118"}

    def get_passwords(self) -> dict[str, str]:
        return {"venmo": "venmo-pass", "phone": "phone-pass"}

    def get_access_token(self, app_name: str) -> str | None:
        self.token_calls.append(app_name)
        return {"venmo": "venmo-token", "phone": "phone-token"}.get(app_name)


class _FakeApiDocs:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.description_calls: list[str] = []

    def show_api_descriptions(self, *, app_name: str):
        self.description_calls.append(app_name)
        if app_name == "venmo":
            return [
                {"name": "search_friends", "description": "find friends"},
                {"name": "send_payment", "description": "send money"},
            ]
        raise AssertionError(app_name)

    def show_api_doc(self, *, app_name: str, api_name: str):
        self.calls.append((app_name, api_name))
        if (app_name, api_name) == ("venmo", "search_friends"):
            return {
                "parameters": [
                    {"name": "access_token"},
                    {"name": "page_index"},
                ]
            }
        if (app_name, api_name) == ("supervisor", "show_profile"):
            return {"parameters": []}
        raise AssertionError((app_name, api_name))


class _FakeSupervisor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def show_profile(self, **kwargs):
        self.calls.append(("show_profile", dict(kwargs)))
        return {"name": "Supervisor"}


class _FakeVenmo:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def search_friends(self, **kwargs):
        self.calls.append(("search_friends", dict(kwargs)))
        return dict(kwargs)


class _FakePhone:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def search_contacts(self, **kwargs):
        self.calls.append(("search_contacts", dict(kwargs)))
        return dict(kwargs)


class _FakeApis:
    def __init__(self) -> None:
        self.api_docs = _FakeApiDocs()
        self.supervisor = _FakeSupervisor()
        self.venmo = _FakeVenmo()
        self.phone = _FakePhone()


class _SandboxClient:
    def __init__(self, apis: _FakeApis) -> None:
        self.apis = apis
        self.scripts: list[str] = []
        self.namespace = {"apis": apis}

    def execute_python(self, script: str) -> str:
        self.scripts.append(script)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(script, self.namespace, self.namespace)
        return stdout.getvalue().strip()


def _set_tool_runtime():
    apis = _FakeApis()
    client = _SandboxClient(apis)
    auth = _FakeAuthManager()
    AppWorldClientHolder.set_client(client)
    AppWorldAuthHolder.set_auth_manager(auth)
    return apis, client, auth


def _reset_tool_runtime() -> None:
    AppWorldAuthHolder.reset()
    AppWorldClientHolder.reset()
    os.environ.pop(EXECUTOR_CANDIDATE_APIS_ENV, None)
    os.environ.pop(EXECUTOR_GUARDRAILS_ENV, None)
    os.environ.pop(EXECUTOR_PRIOR_VARIABLES_ENV, None)
    os.environ.pop(EXECUTOR_SANDBOX_LOG_PATH_ENV, None)
    os.environ.pop(EXECUTOR_TASK_DATETIME_ENV, None)


def test_execute_python_passthrough_for_supervisor_and_api_docs():
    apis, _, auth = _set_tool_runtime()

    try:
        output = execute_python(
            "print(apis.supervisor.show_profile())\n"
            "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='search_friends'))"
        )
    finally:
        _reset_tool_runtime()

    assert auth.token_calls == []
    assert apis.supervisor.calls == [("show_profile", {})]
    assert apis.api_docs.calls == [("venmo", "search_friends")]
    assert "Supervisor" in output
    assert "access_token" in output


def test_execute_python_injects_access_token_for_authenticated_api():
    apis, _, auth = _set_tool_runtime()

    try:
        output = execute_python("print(apis.venmo.search_friends(page_index=0))")
    finally:
        _reset_tool_runtime()

    assert auth.token_calls == ["venmo"]
    assert apis.api_docs.calls == [("venmo", "search_friends")]
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]
    assert "venmo-token" in output


def test_execute_python_overrides_caller_supplied_access_token():
    """Proxy must overwrite a bogus caller-supplied token with the runtime one.

    Without this override, the code agent occasionally writes
    `cognizant_access_token = "<email>"` (or any hallucinated string) and
    passes it via `access_token=`. AppWorld 401-cascades and the milestone
    fails. The proxy is the structural fix: trust the runtime, not the LLM.
    """
    apis, _, auth = _set_tool_runtime()

    try:
        output = execute_python(
            "print(apis.venmo.search_friends(page_index=0, access_token='manual-token'))"
        )
    finally:
        _reset_tool_runtime()

    assert auth.token_calls == ["venmo"]
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]
    assert "venmo-token" in output


def test_execute_python_noop_when_caller_supplies_correct_access_token():
    apis, _, auth = _set_tool_runtime()

    try:
        output = execute_python(
            "print(apis.venmo.search_friends(page_index=0, access_token='venmo-token'))"
        )
    finally:
        _reset_tool_runtime()

    assert auth.token_calls == ["venmo"]
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]
    assert "venmo-token" in output


def test_execute_python_replaces_caller_supplied_none_access_token():
    apis, _, auth = _set_tool_runtime()

    try:
        output = execute_python(
            "print(apis.venmo.search_friends(page_index=0, access_token=None))"
        )
    finally:
        _reset_tool_runtime()

    assert auth.token_calls == ["venmo"]
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]
    assert "venmo-token" in output


def test_execute_python_keeps_raw_apis_across_repeated_calls():
    apis, _, _ = _set_tool_runtime()

    try:
        first = execute_python(
            "apis.api_docs.show_api_doc(app_name='venmo', api_name='search_friends')"
        )
        second = execute_python("print(apis.venmo.search_friends(page_index=1))")
    finally:
        _reset_tool_runtime()

    assert first == ""
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 1, "access_token": "venmo-token"})
    ]
    assert "venmo-token" in second


def test_execute_python_exposes_candidate_api_metadata_and_filters_descriptions():
    apis, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app":"venmo","name":"search_friends","description":"find friends"}]'
    )

    try:
        output = execute_python(
            "print(candidate_api_names_by_app)\n"
            "print(apis.api_docs.show_api_descriptions(app_name='venmo'))"
        )
    finally:
        _reset_tool_runtime()

    assert apis.api_docs.description_calls == []
    assert "search_friends" in output
    assert "send_payment" not in output


def test_execute_python_accepts_full_spec_candidate_shape_for_allow_list():
    apis, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app_name":"venmo","api_name":"search_friends",'
        '"parameters":[{"name":"access_token"},{"name":"page_index"}],'
        '"response_schemas":{"success":[]}}]'
    )

    try:
        output = execute_python(
            "print(candidate_api_names_by_app)\n"
            "print(apis.venmo.search_friends(page_index=0))"
        )
    finally:
        _reset_tool_runtime()

    assert "search_friends" in output
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]


def test_execute_python_rejects_api_doc_lookup_outside_candidate_set():
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app":"venmo","name":"search_friends","description":"find friends"}]'
    )

    try:
        with pytest.raises(
            Exception, match="Finder candidate_apis does not include venmo.send_payment"
        ):
            execute_python(
                "apis.api_docs.show_api_doc(app_name='venmo', api_name='send_payment')"
            )
    finally:
        _reset_tool_runtime()


def test_execute_python_rejects_direct_app_call_outside_candidate_set():
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app_name":"venmo","api_name":"search_friends","description":"find friends"}]'
    )

    try:
        with pytest.raises(
            Exception, match="Finder candidate_apis does not include venmo.send_payment"
        ):
            execute_python("apis.venmo.send_payment()")
    finally:
        _reset_tool_runtime()


def test_execute_python_rejects_direct_app_call_for_non_candidate_app():
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app_name":"venmo","api_name":"search_friends","description":"find friends"}]'
    )

    try:
        with pytest.raises(
            Exception,
            match="Finder candidate_apis does not include phone.search_contacts",
        ):
            execute_python("apis.phone.search_contacts()")
    finally:
        _reset_tool_runtime()


def test_execute_python_normalizes_code_before_sandbox_execution():
    _, _, _ = _set_tool_runtime()

    try:
        fenced = execute_python("```python\n    print('cleaned fence')\n```")
        indented = execute_python("    print('cleaned indent')")
    finally:
        _reset_tool_runtime()

    assert fenced == "cleaned fence"
    assert indented == "cleaned indent"


def test_execute_python_exposes_task_datetime_helpers():
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_TASK_DATETIME_ENV] = "2023-05-18T12:00:00"

    try:
        output = execute_python(
            "print(task_datetime)\n"
            "print(task_datetime_dt.isoformat())\n"
            "print(task_date.isoformat())"
        )
    finally:
        _reset_tool_runtime()

    lines = output.splitlines()
    assert lines == [
        "2023-05-18T12:00:00",
        "2023-05-18T12:00:00",
        "2023-05-18",
    ]


def test_execute_python_exposes_prior_variable_values():
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_PRIOR_VARIABLES_ENV] = (
        '{"wife_email":{"name":"wife_email","description":"Sarah email",'
        '"value_json":"\\"sarah@ex.com\\""},'
        '"missing":{"name":"missing","description":"missing value","value_json":""}}'
    )

    try:
        output = execute_python(
            "print(prior_variables['wife_email']['description'])\n"
            "print(prior_variable_values['wife_email'])\n"
            "print('missing' in prior_variable_values)"
        )
    finally:
        _reset_tool_runtime()

    assert output.splitlines() == ["Sarah email", "sarah@ex.com", "False"]


def test_execute_python_exposes_finalize_inside_code_as_marker():
    _, _, _ = _set_tool_runtime()

    try:
        output = execute_python(
            "finalize(answer='null', milestone_done=True, summary='x', variables_json='[]')"
        )
    finally:
        _reset_tool_runtime()

    assert "answer" in output
    assert "milestone_done" in output
    assert "__CODEX_SANDBOX_FINALIZE__:" in output


def test_execute_python_writes_sandbox_api_trace(tmp_path):
    _, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_SANDBOX_LOG_PATH_ENV] = str(
        tmp_path / "sandbox_api_calls.jsonl"
    )

    try:
        output = execute_python(
            "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='search_friends'))\n"
            "print(apis.venmo.search_friends(page_index=0))"
        )
    finally:
        _reset_tool_runtime()

    trace_path = tmp_path / "sandbox_api_calls.jsonl"
    assert trace_path.exists()
    content = trace_path.read_text(encoding="utf-8")
    assert "api_docs.show_api_doc" in content
    assert "search_friends" in content
    assert '"result_shape"' in content
    assert '"result_preview"' not in content
    assert '"code"' not in content
    assert '"type": "object"' in content
    assert '"page_index"' in content
    assert "venmo-token" not in content
    assert "venmo-token" in output


def test_execute_python_raised_api_call_surfaces_structured_error_and_trace(tmp_path):
    """FIX-VIS: when an apis.* call RAISES (e.g. a 422), the error must become an
    OBSERVABLE outcome — execute_python returns a STRUCTURED error result (committed
    value carries the error) AND the failed call appears in the trace with
    status='error' — instead of the sandbox discarding stdout and leaving
    api_trace=[] (the 2c544f9 / 9ef798c root)."""
    apis, _, _ = _set_tool_runtime()
    os.environ[EXECUTOR_SANDBOX_LOG_PATH_ENV] = str(
        tmp_path / "sandbox_api_calls.jsonl"
    )

    def _boom(**kwargs):
        raise Exception(
            'Response status code is 422:\n{"message":"insufficient balance"}'
        )

    apis.venmo.search_friends = _boom

    try:
        output = execute_python(
            "result = apis.venmo.search_friends(page_index=0)\nprint(result)"
        )
    finally:
        _reset_tool_runtime()

    # execute_python returned NORMALLY (did not propagate the raise) with a
    # structured error result the planner can read.
    assert "insufficient balance" in output
    assert "rejected the call" in output
    # the failed call is captured in the trace as status='error'
    trace = (tmp_path / "sandbox_api_calls.jsonl").read_text(encoding="utf-8")
    assert "search_friends" in trace
    assert '"status": "error"' in trace
    assert "insufficient balance" in trace


def test_execute_python_pure_code_bug_still_raises(tmp_path):
    """FIX-VIS gating: a PURE code bug (no failed api call recorded) must still
    propagate — keeping the traceback path + inner-repair loop — NOT be swallowed
    as a structured api error."""
    _set_tool_runtime()
    os.environ[EXECUTOR_SANDBOX_LOG_PATH_ENV] = str(
        tmp_path / "sandbox_api_calls.jsonl"
    )
    try:
        with pytest.raises(Exception):
            execute_python("raise ValueError('pure code bug, no api call')")
    finally:
        _reset_tool_runtime()


def test_execute_python_guardrails_compile_before_sandbox():
    _, client, _ = _set_tool_runtime()
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"

    try:
        with pytest.raises(Exception, match="failed Python compile"):
            execute_python("if True print('bad syntax')")
    finally:
        _reset_tool_runtime()

    assert client.scripts == []


def test_execute_python_guardrails_reject_unknown_api_kwarg_before_sandbox():
    _, client, _ = _set_tool_runtime()
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app_name":"venmo","api_name":"search_friends",'
        '"parameters":[{"name":"access_token"},{"name":"page_index"}],'
        '"response_schemas":{"success":[]}}]'
    )

    try:
        with pytest.raises(Exception, match="unknown keyword 'cognizant_id'"):
            execute_python("apis.venmo.search_friends(cognizant_id='x')")
    finally:
        _reset_tool_runtime()

    assert client.scripts == []


def test_execute_python_guardrails_reject_profile_key_before_sandbox():
    _, client, _ = _set_tool_runtime()
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"

    try:
        with pytest.raises(Exception, match="profile\\['cognizant_id'\\]"):
            execute_python("print(profile['cognizant_id'])")
    finally:
        _reset_tool_runtime()

    assert client.scripts == []


def test_execute_python_guardrails_unwrap_printed_program_source_before_sandbox():
    _, client, _ = _set_tool_runtime()
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"

    try:
        output = execute_python(
            'cognizant_code = """\n'
            "import json\n"
            "result = apis.venmo.search_friends(page_index=0)\n"
            'print(json.dumps({"value": result}))\n'
            '"""\n'
            "print(cognizant_code)"
        )
    finally:
        _reset_tool_runtime()

    assert len(client.scripts) == 1
    assert '{"value":' in output


def test_execute_python_guardrails_allow_valid_code():
    apis, client, _ = _set_tool_runtime()
    os.environ[EXECUTOR_GUARDRAILS_ENV] = "1"
    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app_name":"venmo","api_name":"search_friends",'
        '"parameters":[{"name":"access_token"},{"name":"page_index"}],'
        '"response_schemas":{"success":[]}}]'
    )

    try:
        output = execute_python(
            "print(profile['email'])\nprint(apis.venmo.search_friends(page_index=0))"
        )
    finally:
        _reset_tool_runtime()

    assert len(client.scripts) == 1
    assert apis.venmo.calls == [
        ("search_friends", {"page_index": 0, "access_token": "venmo-token"})
    ]
    assert "clmiller@gmail.com" in output


def test_execute_python_restores_unproxied_apis_after_script_runs():
    """Subsequent execute_function calls (auth prefetch for the next milestone)
    must see the raw `apis`, not the previous milestone's _ApisProxy."""

    apis, client, _ = _set_tool_runtime()

    os.environ[EXECUTOR_CANDIDATE_APIS_ENV] = (
        '[{"app":"venmo","name":"search_friends"}]'
    )

    try:
        execute_python("print(apis.venmo.search_friends(page_index=0))")
    finally:
        _reset_tool_runtime()

    # AppWorld's persistent interpreter (modelled by client.namespace) must
    # have the unproxied raw apis bound to the global `apis` after the
    # script's finally clause ran. Without this, the next milestone's
    # auth-prefetch login would be blocked by THIS milestone's candidate-api
    # whitelist.
    assert client.namespace["apis"] is apis
    assert client.namespace["apis"] is client.namespace["_original_apis"]
