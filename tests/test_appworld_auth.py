from __future__ import annotations

from adk_appworld_agent.appworld.auth import AppWorldAuthManager


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def call(self, app: str, function: str, /, **kwargs):
        self.calls.append((app, function, dict(kwargs)))
        if (app, function) == ("supervisor", "show_profile"):
            return {
                "email": "clmiller@gmail.com",
                "phone_number": "2125442118",
            }
        if (app, function) == ("supervisor", "show_account_passwords"):
            return [
                {"account_name": "phone", "password": "phone-pass"},
                {"account_name": "venmo", "password": "venmo-pass"},
            ]
        if function == "login":
            return {"access_token": f"{app}-token"}
        raise AssertionError((app, function, kwargs))


def test_auth_manager_uses_phone_number_for_phone_login():
    client = _FakeClient()
    auth = AppWorldAuthManager(client)

    token = auth.get_access_token("phone")

    assert token == "phone-token"
    assert client.calls[-1] == (
        "phone",
        "login",
        {"username": "2125442118", "password": "phone-pass"},
    )


def test_auth_manager_uses_email_and_caches_tokens():
    client = _FakeClient()
    auth = AppWorldAuthManager(client)

    first = auth.get_access_token("venmo")
    second = auth.get_access_token("venmo")

    assert first == "venmo-token"
    assert second == "venmo-token"
    login_calls = [call for call in client.calls if call[:2] == ("venmo", "login")]
    assert login_calls == [
        (
            "venmo",
            "login",
            {"username": "clmiller@gmail.com", "password": "venmo-pass"},
        )
    ]


def test_auth_manager_clear_resets_task_scoped_cache():
    client = _FakeClient()
    auth = AppWorldAuthManager(client)

    auth.get_access_token("venmo")
    auth.clear()
    auth.get_access_token("venmo")

    login_calls = [call for call in client.calls if call[:2] == ("venmo", "login")]
    assert len(login_calls) == 2
