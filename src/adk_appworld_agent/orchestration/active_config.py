"""Process-wide active RunConfig — set ONCE at run bootstrap, read by the
deep module-level helpers that can't receive a RunConfig through their call
chain (retry ladders, the genai HTTP client, api-spec/data paths).

This is a set-once typed singleton, NOT ad-hoc global mutation: a run loads
its one experiment config file, calls `set_active_config(cfg)` at startup,
and the same `cfg` is dumped to artifacts. Subagent-level knobs still flow
through the RunConfig passed into each subagent factory; this singleton only
serves helpers below that boundary.

When unset (tests, ad-hoc imports) `active_config()` returns schema defaults,
which equal the historical env/hardcoded values — so unconfigured code keeps
its previous behaviour.
"""

from __future__ import annotations

from adk_appworld_agent.orchestration.run_config import RunConfig

_active: RunConfig | None = None
_default: RunConfig | None = None


def set_active_config(config: RunConfig | None) -> None:
    """Install the run's resolved config (call once at bootstrap). Passing
    None clears it (used by tests to restore defaults)."""
    global _active
    _active = config


def active_config() -> RunConfig:
    """The active RunConfig, or schema defaults when unset."""
    global _default
    if _active is not None:
        return _active
    if _default is None:
        _default = RunConfig()
    return _default


__all__ = ["active_config", "set_active_config"]
