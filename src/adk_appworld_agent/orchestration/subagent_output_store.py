from __future__ import annotations

from adk_appworld_agent.contracts.subagent_output import SubagentEnvelope
from adk_appworld_agent.orchestration.state import Phase

SUBAGENT_OUTPUTS_KEY = "subagent_outputs"


class SubagentOutputStore:
    def read_all(self, session_state: dict) -> dict[str, list[SubagentEnvelope]]:
        raw = session_state.get(SUBAGENT_OUTPUTS_KEY, {}) or {}
        parsed: dict[str, list[SubagentEnvelope]] = {}
        for phase_name, items in raw.items():
            parsed[phase_name] = [
                SubagentEnvelope.model_validate(item) for item in (items or [])
            ]
        return parsed

    def read(self, session_state: dict, phase: Phase | str) -> SubagentEnvelope | None:
        phase_key = phase.value if isinstance(phase, Phase) else phase
        all_outputs = self.read_all(session_state)
        outputs = all_outputs.get(phase_key, [])
        return outputs[-1] if outputs else None

    def read_latest(
        self, session_state: dict, phase: Phase | str
    ) -> SubagentEnvelope | None:
        return self.read(session_state, phase)

    def next_attempt(self, session_state: dict, phase: Phase) -> int:
        latest_outputs = self.read_all(session_state).get(phase.value, [])
        return len(latest_outputs) + 1

    def append_delta(
        self, session_state: dict, envelope: SubagentEnvelope
    ) -> dict[str, dict[str, list[dict]]]:
        current = {
            phase_name: [item.model_dump(mode="json") for item in items]
            for phase_name, items in self.read_all(session_state).items()
        }
        current.setdefault(envelope.phase.value, [])
        current[envelope.phase.value].append(envelope.model_dump(mode="json"))
        return {SUBAGENT_OUTPUTS_KEY: current}
