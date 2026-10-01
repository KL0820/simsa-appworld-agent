from __future__ import annotations

from adk_appworld_agent.appworld.auth import AppWorldAuthManager


class AppWorldAuthHolder:
    """Process-wide accessor for the current task-run auth manager."""

    _auth_manager: AppWorldAuthManager | None = None

    @classmethod
    def set_auth_manager(cls, auth_manager: AppWorldAuthManager) -> None:
        cls._auth_manager = auth_manager

    @classmethod
    def get_auth_manager(cls) -> AppWorldAuthManager:
        if cls._auth_manager is None:
            raise RuntimeError(
                "AppWorldAuthHolder: auth manager not initialized — call set_auth_manager() before running tools."
            )
        return cls._auth_manager

    @classmethod
    def reset(cls) -> None:
        cls._auth_manager = None
