from __future__ import annotations

from adk_appworld_agent.orchestration.state import RunState, TaskContext

RUN_STATE_KEY = "run_state"


class RunStateRepository:
    def exists(self, session_state: dict) -> bool:
        return (
            RUN_STATE_KEY in session_state and session_state[RUN_STATE_KEY] is not None
        )

    def load(
        self,
        session_state: dict,
        *,
        default_instruction: str = "",
    ) -> RunState:
        raw_state = session_state.get(RUN_STATE_KEY)
        if raw_state:
            return RunState.model_validate(raw_state)
        return RunState(task_context=TaskContext(instruction=default_instruction))

    def save_delta(self, state: RunState) -> dict[str, dict]:
        return {RUN_STATE_KEY: state.model_dump(mode="json")}
