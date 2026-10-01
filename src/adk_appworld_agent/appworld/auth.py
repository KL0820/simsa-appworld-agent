from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from typing import Any

from adk_appworld_agent.seams import AppWorldClient


class AppWorldAuthManager:
    """Deterministic AppWorld auth helper scoped to one task run."""

    def __init__(self, client: AppWorldClient) -> None:
        self._client = client
        self._profile: dict[str, Any] | None = None
        self._passwords: dict[str, str] | None = None
        self._tokens: dict[str, str] = {}

    def get_profile(self) -> dict[str, Any]:
        if self._profile is None:
            raw_profile = self._deserialize_payload(
                self._client.call("supervisor", "show_profile")
            )
            if not isinstance(raw_profile, Mapping):
                raise RuntimeError(
                    "supervisor.show_profile returned a non-mapping payload"
                )
            self._profile = dict(raw_profile)
        return dict(self._profile)

    def get_passwords(self) -> dict[str, str]:
        if self._passwords is None:
            raw_passwords = self._deserialize_payload(
                self._client.call("supervisor", "show_account_passwords")
            )
            self._passwords = self._normalize_passwords(raw_passwords)
        return dict(self._passwords)

    def get_access_token(self, app_name: str) -> str | None:
        if app_name in self._tokens:
            return self._tokens[app_name]

        passwords = self.get_passwords()
        password = passwords.get(app_name)
        if password is None:
            return None

        username = self._resolve_username(app_name)
        login_result = self._deserialize_payload(
            self._client.call(
                app_name,
                "login",
                username=username,
                password=password,
            )
        )
        if not isinstance(login_result, Mapping):
            preview = repr(login_result)
            if len(preview) > 200:
                preview = preview[:200] + f"...<+{len(preview) - 200} chars>"
            raise RuntimeError(
                f"{app_name}.login returned a non-mapping payload "
                f"(type={type(login_result).__name__}): {preview}"
            )

        access_token = login_result.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"{app_name}.login did not return a usable access_token")

        self._tokens[app_name] = access_token
        return access_token

    def clear(self) -> None:
        self._profile = None
        self._passwords = None
        self._tokens = {}

    def _resolve_username(self, app_name: str) -> str:
        profile = self.get_profile()
        key = "phone_number" if app_name == "phone" else "email"
        username = profile.get(key)
        if not isinstance(username, str) or not username:
            raise RuntimeError(
                f"supervisor profile missing required {key!r} for {app_name} login"
            )
        return username

    @staticmethod
    def _normalize_passwords(raw_passwords: Any) -> dict[str, str]:
        if isinstance(raw_passwords, Mapping):
            return {
                str(account_name): str(password)
                for account_name, password in raw_passwords.items()
                if isinstance(account_name, str) and password is not None
            }

        if not isinstance(raw_passwords, list):
            raise RuntimeError(
                "supervisor.show_account_passwords returned an unexpected payload"
            )

        normalized: dict[str, str] = {}
        for item in raw_passwords:
            if not isinstance(item, Mapping):
                continue
            account_name = item.get("account_name")
            password = item.get("password")
            if isinstance(account_name, str) and password is not None:
                normalized[account_name] = str(password)
        return normalized

    @staticmethod
    def _deserialize_payload(payload: Any) -> Any:
        if not isinstance(payload, str):
            return payload

        stripped = payload.strip()
        if not stripped:
            return payload

        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(stripped)
            except Exception:
                continue
        return payload
