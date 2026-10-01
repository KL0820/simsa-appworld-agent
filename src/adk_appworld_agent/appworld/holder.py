from __future__ import annotations

from adk_appworld_agent.appworld.client import AppWorldRpcClient


class AppWorldClientHolder:
    """Process-wide accessor for the AppWorld zerorpc client.

    Subagents reach the client through this holder via the AppWorldClient seam,
    so they stay decoupled from zerorpc and from how the client was wired up.
    """

    _client: AppWorldRpcClient | None = None

    @classmethod
    def set_client(cls, client: AppWorldRpcClient) -> None:
        cls._client = client

    @classmethod
    def get_client(cls) -> AppWorldRpcClient:
        if cls._client is None:
            raise RuntimeError(
                "AppWorldClientHolder: client not initialized — call set_client() before running subagents."
            )
        return cls._client

    @classmethod
    def reset(cls) -> None:
        cls._client = None
