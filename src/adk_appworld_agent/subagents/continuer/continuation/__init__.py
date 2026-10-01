"""Continuation agent package.

Public construction APIs are re-exported here. History-render helpers remain
available for focused unit tests; new implementation code should import private
helpers from :mod:`.agent` directly.
"""

from .agent import (
    ContinuationPlannerSubagent,
    DeterministicContinuationStub,
    build_continuation_planner_subagent,
    build_continuation_stub_subagent,
)
from .agent import (
    _build_history_block as _build_history_block,
)
from .agent import (
    _render_cycle_oneliner as _render_cycle_oneliner,
)
from .agent import (
    _render_history_entry as _render_history_entry,
)
from .agent import (
    _summarize_value_for_oneliner as _summarize_value_for_oneliner,
)
from .prompts import CONTINUATION_SYSTEM_PROMPT as CONTINUATION_SYSTEM_PROMPT

__all__ = [
    "ContinuationPlannerSubagent",
    "DeterministicContinuationStub",
    "build_continuation_planner_subagent",
    "build_continuation_stub_subagent",
]
