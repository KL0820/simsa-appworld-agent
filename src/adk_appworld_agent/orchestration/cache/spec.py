from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)

KeyFn = Callable[[SubagentInput], str]
ReplayableFn = Callable[[SubagentEnvelope], bool]
AdaptReplayFn = Callable[[SubagentEnvelope, SubagentInput, str], SubagentEnvelope]


def default_replayable(envelope: SubagentEnvelope) -> bool:
    return envelope.status == SubagentStatus.SUCCEEDED


def canonical_key(parts: dict[str, Any]) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheSpec:
    subagent_name: str
    version: str
    key_fn: KeyFn
    replayable: ReplayableFn = default_replayable
    adapt_replay: AdaptReplayFn | None = None
