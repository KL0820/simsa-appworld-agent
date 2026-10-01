from __future__ import annotations

from adk_appworld_agent.appworld.world_context import (
    build_world_run_name,
    close_world_safely,
)


def test_build_world_run_name_is_task_scoped():
    name = build_world_run_name(
        experiment_name="community find",
        run_name="20260420_182713_375688",
        task_id="3d9a636_2",
    )

    assert name == "community_find__20260420_182713_375688__3d9a636_2"


def test_close_world_safely_swallows_client_errors():
    class _BrokenClient:
        def __init__(self) -> None:
            self.calls = 0

        def close_world(self) -> None:
            self.calls += 1
            raise RuntimeError("close failed")

    client = _BrokenClient()
    close_world_safely(client)  # type: ignore[arg-type]

    assert client.calls == 1
