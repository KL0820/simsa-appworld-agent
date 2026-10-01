from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any


class AppWorldRpcClient:
    """Thin zerorpc wrapper that satisfies `seams.AppWorldClient`, made
    gevent-safe and event-loop-friendly.

    zerorpc is gevent-based and its hub is THREAD-AFFINE: the client must be
    created on, and every call issued from, ONE single thread. So this class owns
    a dedicated single-worker thread, creates the zerorpc client ON it, and
    marshals every call to it.

    Critically, the HOT path (execute_python, called many times during EXECUTE)
    is offloaded via `rpc_executor` + asyncio.run_in_executor so the shared
    asyncio event loop is NOT blocked by the synchronous RPC round-trip. Blocking
    the loop with the inline sync call is what previously starved the shared loop
    and made the NEXT finder request hang to its deadline (the 2-day finder
    "429"/timeout bug — see project_finder_429_client_bug; root = commit 3c74aed
    transport timeout + this inline sync RPC both degrading the shared loop).

    Subagents never see zerorpc directly — they call through this object via the
    `AppWorldClient` Protocol.
    """

    def __init__(self, addr: str, *, timeout: int = 30) -> None:
        self._addr = addr
        self._timeout = timeout
        # ONE worker thread owns the gevent hub + the zerorpc client.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="appworld-rpc"
        )
        self._worker_thread: threading.Thread | None = None
        self._rpc: Any = None
        # create + connect ON the worker thread so its gevent hub lives there.
        self._executor.submit(self._connect).result()

    def _connect(self) -> None:
        import zerorpc

        self._worker_thread = threading.current_thread()
        self._rpc = zerorpc.Client(timeout=self._timeout)
        self._rpc.connect(self._addr)

    def _run(self, fn: Any, *args: Any) -> Any:
        # If we are already ON the worker thread (e.g. the async execute_python
        # tool offloaded the whole sync call here via run_in_executor), run inline
        # to avoid a nested-submit deadlock on the single worker. Otherwise marshal
        # to the worker thread and block for the result (gevent hub affinity).
        if threading.current_thread() is self._worker_thread:
            return fn(*args)
        return self._executor.submit(fn, *args).result()

    @property
    def rpc_executor(self) -> ThreadPoolExecutor:
        """The single-worker executor that owns the gevent zerorpc client. The
        async execute_python tool offloads onto THIS executor (run_in_executor)
        so the RPC runs on the hub-owning thread while the event loop stays free.
        """
        return self._executor

    def call(self, app: str, function: str, /, **kwargs: Any) -> Any:
        params = {k: v for k, v in kwargs.items() if v is not None}
        return self._run(self._rpc.execute_function, app, function, params)

    def init_world(self, task_id: str, experiment_name: str) -> None:
        self._run(self._rpc.init_world, task_id, experiment_name)

    def close_world(self) -> None:
        self._run(self._rpc.close_world)

    def get_task_instruction(self) -> str:
        return self._run(self._rpc.get_task_instruction)

    def submit_answer(self, answer: str) -> Any:
        return self._run(self._rpc.submit_answer, answer)

    def execute_python(self, script: str) -> str:
        return str(self._run(self._rpc.execute_python, script)).strip()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
