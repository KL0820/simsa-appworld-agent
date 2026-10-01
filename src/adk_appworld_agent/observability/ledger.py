from __future__ import annotations

from pydantic import BaseModel

from adk_appworld_agent.orchestration.state import RunState

LEDGER_KEY = "execution_ledger"


class ExecutionLedgerEntry(BaseModel):
    state_before: RunState
    state_after: RunState
    phase_decision: str
    failure_code: str | None = None
    policy_source: str | None = None


class ExecutionLedgerStore:
    def read(self, session_state: dict) -> list[ExecutionLedgerEntry]:
        raw_entries = session_state.get(LEDGER_KEY, []) or []
        return [ExecutionLedgerEntry.model_validate(item) for item in raw_entries]

    def append_delta(
        self, session_state: dict, entry: ExecutionLedgerEntry
    ) -> dict[str, list[dict]]:
        current = [item.model_dump(mode="json") for item in self.read(session_state)]
        current.append(entry.model_dump(mode="json"))
        return {LEDGER_KEY: current}
