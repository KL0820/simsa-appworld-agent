"""Gevent-safety + event-loop guards for the AppWorld RPC client (2026-06-23).

zerorpc's gevent hub is THREAD-AFFINE, and the sync RPC must NOT block the shared
asyncio loop (blocking it poisoned the next finder call — project_finder_429_client_bug).
So: ONE dedicated worker thread owns the client; every call runs on it; the
execute_python tool is ASYNC and offloads onto that worker so the loop stays free.
These pin the invariants; a future regression that puts the client back on the
loop / on a multi-worker pool fails here.
"""

from __future__ import annotations

import inspect
import sys
import threading
import types


def _install_fake_zerorpc(monkeypatch, records: dict) -> None:
    fake = types.ModuleType("zerorpc")

    class _FakeClient:
        def __init__(self, timeout: int = 30) -> None:
            records["init_thread"] = threading.current_thread()

        def connect(self, addr: str) -> None:
            records["connect_thread"] = threading.current_thread()

        def execute_python(self, script: str) -> str:
            records.setdefault("exec_threads", []).append(threading.current_thread())
            return "OUT"

        def execute_function(self, *a):  # noqa: ANN001
            return None

        def init_world(self, *a):  # noqa: ANN001
            records.setdefault("init_world_threads", []).append(
                threading.current_thread()
            )

        def close_world(self):
            pass

    fake.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zerorpc", fake)


def test_rpc_client_single_worker_and_thread_affine(monkeypatch):
    from adk_appworld_agent.appworld import client as client_mod

    records: dict = {}
    _install_fake_zerorpc(monkeypatch, records)
    c = client_mod.AppWorldRpcClient("tcp://127.0.0.1:65500")
    try:
        # gevent hub affinity REQUIRES exactly one worker
        assert c.rpc_executor._max_workers == 1
        # client was created/connected on the worker thread, NOT the caller thread
        worker = records["connect_thread"]
        assert worker is not threading.current_thread()
        assert records["init_thread"] is worker
        # every call runs on that same single worker thread
        c.execute_python("a")
        c.execute_python("b")
        c.init_world("t", "exp")
        assert records["exec_threads"] and all(
            t is worker for t in records["exec_threads"]
        )
        assert all(t is worker for t in records["init_world_threads"])
    finally:
        c.shutdown()


def test_execute_python_tool_is_async_and_named():
    from adk_appworld_agent.subagents.executor.appworld_tools import (
        EXECUTE_PYTHON_TOOL,
        _execute_python_tool,
        execute_python,
    )

    # the TOOL must be a coroutine (so ADK awaits it -> loop free during the RPC)
    assert inspect.iscoroutinefunction(_execute_python_tool)
    # tool name the model is told to call must be preserved
    assert _execute_python_tool.__name__ == "execute_python"
    assert EXECUTE_PYTHON_TOOL.name == "execute_python"
    # the sync entry point is still a plain function (tests/direct callers use it)
    assert not inspect.iscoroutinefunction(execute_python)
