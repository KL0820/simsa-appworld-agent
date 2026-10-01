from __future__ import annotations

from dataclasses import dataclass, field

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.observability.metrics import zero_metrics
from adk_appworld_agent.orchestration.cache.spec import CacheSpec
from adk_appworld_agent.orchestration.cache.store import SubagentCacheStore


@dataclass
class CachePolicy:
    """Per-subagent read/write gates.

    ``read`` and ``write`` are sets of ``CacheSpec.subagent_name`` values that are
    allowed to hit / populate the cache. Empty set means "none"; pass ``None``
    to allow all known subagent_names.
    """

    read: set[str] | None = field(default_factory=set)
    write: set[str] | None = field(default_factory=set)
    version_override: str | None = None

    def can_read(self, subagent_name: str) -> bool:
        return self.read is None or subagent_name in self.read

    def can_write(self, subagent_name: str) -> bool:
        return self.write is None or subagent_name in self.write

    def effective_version(self, spec: CacheSpec) -> str:
        return self.version_override or spec.version

    @classmethod
    def disabled(cls) -> "CachePolicy":
        return cls(read=set(), write=set())

    @classmethod
    def read_all(cls) -> "CachePolicy":
        return cls(read=None, write=set())

    @classmethod
    def read_write_all(cls) -> "CachePolicy":
        return cls(read=None, write=None)


@dataclass
class CacheHit:
    envelope: SubagentEnvelope
    source: str


class CacheLayer:
    """Wraps a store + policy. Subagents consult this through the invoker."""

    def __init__(self, *, store: SubagentCacheStore, policy: CachePolicy) -> None:
        self.store = store
        self.policy = policy

    def lookup(self, spec: CacheSpec, subagent_input: SubagentInput) -> CacheHit | None:
        if not self.policy.can_read(spec.subagent_name):
            return None
        key = spec.key_fn(subagent_input)
        version = self.policy.effective_version(spec)
        hit = self.store.get(spec.subagent_name, key, version)
        if hit is None:
            return None
        envelope, meta = hit
        source = str(meta.get("source") or "subagent_cache")
        return CacheHit(envelope=envelope, source=source)

    def save(
        self,
        spec: CacheSpec,
        subagent_input: SubagentInput,
        envelope: SubagentEnvelope,
        *,
        source: str,
    ) -> None:
        if not self.policy.can_write(spec.subagent_name):
            return
        if not spec.replayable(envelope):
            return
        key = spec.key_fn(subagent_input)
        version = self.policy.effective_version(spec)
        meta = {
            "source": source,
            "task_id": subagent_input.task_context.task_id,
            "phase": subagent_input.phase.value,
        }
        self.store.put(spec.subagent_name, key, version, envelope, meta=meta)

    def adapt_replay(
        self,
        spec: CacheSpec,
        hit: CacheHit,
        subagent_input: SubagentInput,
    ) -> SubagentEnvelope:
        envelope = hit.envelope
        if spec.adapt_replay is not None:
            return spec.adapt_replay(envelope, subagent_input, hit.source)
        return _default_adapt_replay(envelope, subagent_input, hit.source)


def _default_adapt_replay(
    envelope: SubagentEnvelope, subagent_input: SubagentInput, source: str
) -> SubagentEnvelope:
    next_env = envelope.model_copy(deep=True)
    next_env.attempt = subagent_input.attempt
    warning = f"replayed from {source}"
    if warning not in next_env.warnings:
        next_env.warnings = [*next_env.warnings, warning]
    metrics = replay_metrics(next_env.metrics)
    if isinstance(next_env.payload, dict):
        next_env.payload = {
            **next_env.payload,
            "cache_source": source,
            "metrics": metrics,
        }
    next_env.metrics = metrics
    return next_env


def replay_metrics(metrics: dict | None) -> dict:
    """Return metrics for the current cache replay, not the original run."""
    original = dict(metrics or {})
    replayed = zero_metrics()
    replayed["replayed"] = True
    if original:
        replayed["cached_wall_ms"] = int(original.get("wall_ms") or 0)
        replayed["cached_llm_calls"] = int(original.get("llm_calls") or 0)
        replayed["cached_total_tokens"] = int(original.get("total_tokens") or 0)
    return replayed
