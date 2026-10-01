from __future__ import annotations

import asyncio
import json

from adk_appworld_agent.contracts.final_answer import FinalAnswerOutput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.contracts.submission import SubmissionCandidate
from adk_appworld_agent.orchestration.orchestrator import Orchestrator
from adk_appworld_agent.orchestration.run_config import ModelConfig
from adk_appworld_agent.orchestration.state import Phase, RunState, TaskContext
from adk_appworld_agent.submission.final_answer_extractor import (
    _build_extractor_llm_agent,
    _build_user_prompt,
    _parse_final_answer_response,
    build_executor_output_text,
    extract_appworld_final_answer,
)
from tests.fixtures import load_executor_failure_fixture


def _model_cfg() -> ModelConfig:
    return ModelConfig(
        name="gemini-2.5-flash", temperature=0.0, top_p=1.0, top_k=1, seed=0
    )


def _run(coro):
    return asyncio.run(coro)


# ── build_executor_output_text ────────────────────────────────────────────────


def test_build_executor_output_text_includes_value_summary_answer():
    """Restores the answer line — the extractor LLM needs it to disambiguate
    action ("null" literal) from query (natural-language answer). Removing
    the answer line was the 325d6ec_2 regression: the extractor lost the
    action discriminator and started type-inferring queries from value
    dicts. Python None on the answer key is rendered as the four-character
    string "null" (the JSON serialisation the planner / executor prompt
    targets)."""
    text = build_executor_output_text(
        {
            "value": 620.0,
            "answer": "620.00",
            "summary": "You have received 620.00 on Venmo this month.",
        }
    )
    assert "summary: You have received 620.00 on Venmo this month." in text
    assert "value: 620.0" in text
    assert "answer: 620.00" in text


def test_build_executor_output_text_renders_null_answer_for_action():
    """`answer: null` is the action-task discriminator — the extractor sees
    the literal and propagates `null` instead of refining a query answer
    from value. Blank summary is still skipped (no useful signal)."""
    text = build_executor_output_text(
        {"value": [1, 2, 3], "answer": "null", "summary": ""}
    )
    assert "answer: null" in text
    assert "summary:" not in text
    assert "value: [1, 2, 3]" in text


def test_build_executor_output_text_normalises_python_none_answer_to_null():
    """Defensive: when Gemini emits JSON `null` (decoded back to Python
    `None`) instead of the literal string `"null"`, the rendered line
    still says `answer: null` so the extractor sees the same
    discriminator."""
    text = build_executor_output_text(
        {"value": {"sent": True}, "answer": None, "summary": "Sent."}
    )
    assert "answer: null" in text
    assert "answer: None" not in text


def test_build_executor_output_text_handles_none_input():
    assert build_executor_output_text(None) == "null"
    assert build_executor_output_text({}) == "null"


def test_build_executor_output_text_truncates_huge_value():
    big = {"value": list(range(2000)), "answer": "null", "summary": ""}
    text = build_executor_output_text(big)
    assert "(truncated)" in text


# ── _build_user_prompt ────────────────────────────────────────────────────────


def test_user_prompt_includes_task_instruction_and_executor_output():
    prompt = _build_user_prompt(
        task_instruction="How much have I received?",
        executor_output_text="value: 620.0",
    )
    assert "How much have I received?" in prompt
    assert "value: 620.0" in prompt
    # cuga's template uses the literal placeholders
    assert "user_intent" in prompt
    assert "system_answer" in prompt


# ── _parse_final_answer_response ──────────────────────────────────────────────


def test_parser_accepts_schema_conformant_json():
    out = _parse_final_answer_response(
        json.dumps(
            {"thoughts": ["x"], "final_answer": "620", "final_answer_type": "float"}
        )
    )
    assert isinstance(out, FinalAnswerOutput)
    assert out.final_answer == "620"
    assert out.final_answer_type == "float"


def test_parser_returns_none_on_missing_field():
    out = _parse_final_answer_response(
        json.dumps({"thoughts": [], "final_answer": "x"})
    )
    assert out is None


def test_parser_returns_none_on_invalid_type_literal():
    out = _parse_final_answer_response(
        json.dumps(
            {"thoughts": [], "final_answer": "x", "final_answer_type": "boolean"}
        )
    )
    assert out is None


def test_parser_returns_none_on_blank_input():
    assert _parse_final_answer_response("") is None
    assert _parse_final_answer_response("   ") is None


def test_parser_returns_none_on_invalid_json():
    assert _parse_final_answer_response("not json") is None


# ── _build_extractor_llm_agent ────────────────────────────────────────────────


def test_extractor_llm_agent_uses_output_schema_and_model_cfg():
    agent = _build_extractor_llm_agent(_model_cfg())
    assert agent.name == "appworld_final_answer_extractor"
    assert agent.output_schema is FinalAnswerOutput
    # Instruction must contain cuga's prompt anchors
    assert "user_intent" in agent.instruction
    assert "final_answer_type" in agent.instruction
    cfg = agent.generate_content_config
    assert cfg.temperature == 0.0
    assert cfg.seed == 0


# ── extract_appworld_final_answer (mocked LLM session) ────────────────────────


def _patched_session(
    monkeypatch, raw_text: str | None, *, raise_exc: Exception | None = None
):
    captured: dict = {}

    async def fake_session(*, inner_agent, user_prompt):
        captured["user_prompt"] = user_prompt
        captured["inner_agent_name"] = getattr(inner_agent, "name", None)
        if raise_exc is not None:
            raise raise_exc
        return raw_text or ""

    monkeypatch.setattr(
        "adk_appworld_agent.submission.final_answer_extractor._run_extractor_session",
        fake_session,
    )
    return captured


def test_extractor_returns_parsed_output_on_valid_response(monkeypatch):
    captured = _patched_session(
        monkeypatch,
        json.dumps(
            {
                "thoughts": ["mocked"],
                "final_answer": "620",
                "final_answer_type": "float",
            }
        ),
    )
    out = _run(
        extract_appworld_final_answer(
            task_instruction="How much money have I received this month?",
            last_executor_output={
                "value": 620.0,
                "answer": None,
                "summary": "received 620",
            },
            model_cfg=_model_cfg(),
        )
    )
    assert out is not None
    assert out.final_answer == "620"
    assert out.final_answer_type == "float"
    assert "How much money have I received" in captured["user_prompt"]
    assert "summary: received 620" in captured["user_prompt"]
    assert captured["inner_agent_name"] == "appworld_final_answer_extractor"


def test_extractor_returns_none_on_session_exception(monkeypatch):
    _patched_session(monkeypatch, None, raise_exc=RuntimeError("rate limit"))
    out = _run(
        extract_appworld_final_answer(
            task_instruction="t",
            last_executor_output={"value": 1},
            model_cfg=_model_cfg(),
        )
    )
    assert out is None


def test_extractor_returns_none_on_blank_response(monkeypatch):
    _patched_session(monkeypatch, "")
    out = _run(
        extract_appworld_final_answer(
            task_instruction="t",
            last_executor_output={"value": 1},
            model_cfg=_model_cfg(),
        )
    )
    assert out is None


# ── fixture-driven semantic checks ────────────────────────────────────────────


def test_extractor_handles_value_null_fixtures_when_llm_succeeds(monkeypatch):
    """For value-null fixtures (mode A), simulate LLM picking the bare value."""
    cases = load_executor_failure_fixture("answer_norm_value_null")
    for case in cases:
        expected_type = case["expected_after_fix"].get("answer_type", "str")
        if expected_type == "null":
            continue  # negative covered separately in orchestrator hook tests
        expected = case["expected_after_fix"]["submission_answer"]
        _patched_session(
            monkeypatch,
            json.dumps(
                {
                    "thoughts": ["mocked"],
                    "final_answer": expected,
                    "final_answer_type": expected_type,
                }
            ),
        )
        out = _run(
            extract_appworld_final_answer(
                task_instruction=case["data"].get(
                    "milestone_intent", "What is the value?"
                ),
                last_executor_output=case["data"]["stdout_json"],
                model_cfg=_model_cfg(),
            )
        )
        assert out is not None, f"extractor must succeed for {case['source']}"
        assert out.final_answer == expected


def test_extractor_handles_verbose_sentence_fixtures_when_llm_succeeds(monkeypatch):
    """For verbose-sentence fixtures (mode B), simulate LLM extracting the bare value."""
    cases = load_executor_failure_fixture("answer_norm_verbose_sentence")
    for case in cases:
        expected = case["expected_after_fix"]["submission_answer"]
        _patched_session(
            monkeypatch,
            json.dumps(
                {
                    "thoughts": ["mocked"],
                    "final_answer": expected,
                    "final_answer_type": "str",
                }
            ),
        )
        out = _run(
            extract_appworld_final_answer(
                task_instruction=case["data"].get("milestone_intent", "Question?"),
                last_executor_output={
                    "value": None,
                    "answer": case["data"]["answer"],
                    "summary": "",
                },
                model_cfg=_model_cfg(),
            )
        )
        assert out is not None, f"extractor must succeed for {case['source']}"
        assert out.final_answer == expected


# ── orchestrator hook integration ─────────────────────────────────────────────


def _orch() -> Orchestrator:
    return Orchestrator(submitter_provider=lambda: None)


def _exec_output_with_stdout_json(stdout_json: dict) -> SubagentEnvelope:
    return SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="executor_subagent_code_plan_execute",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={"code_execute": {"stdout_json": stdout_json}},
    )


def _candidate_default() -> SubmissionCandidate:
    return SubmissionCandidate(
        answer="null", task_type_hint="action", answer_type="null", source="executor"
    )


def _session_state_with_instruction(instruction: str) -> dict:
    state = RunState(task_context=TaskContext(task_id="t", instruction=instruction))
    return {"run_state": state.model_dump(mode="json")}


def _patched_orchestrator_extractor(monkeypatch, returned: FinalAnswerOutput | None):
    async def fake(**kwargs):
        return returned

    monkeypatch.setattr(
        "adk_appworld_agent.orchestration.orchestrator.extract_appworld_final_answer",
        fake,
    )


def test_hook_overrides_candidate_with_extractor_query_answer(monkeypatch):
    _patched_orchestrator_extractor(
        monkeypatch,
        FinalAnswerOutput(thoughts=[], final_answer="620", final_answer_type="float"),
    )
    new_candidate, payload = _run(
        _orch()._apply_final_answer_extractor(
            session_state=_session_state_with_instruction("How much did I receive?"),
            exec_output=_exec_output_with_stdout_json(
                {"value": 620.0, "answer": None, "summary": "ok"}
            ),
            candidate=_candidate_default(),
        )
    )
    assert new_candidate.answer == "620"
    assert new_candidate.answer_type == "float"
    assert new_candidate.task_type_hint == "query"
    assert new_candidate.source == "final_answer_extractor"
    assert payload["status"] == "applied"


def test_hook_maps_NA_to_null_action_candidate(monkeypatch):
    _patched_orchestrator_extractor(
        monkeypatch,
        FinalAnswerOutput(thoughts=[], final_answer="N/A", final_answer_type="str"),
    )
    new_candidate, payload = _run(
        _orch()._apply_final_answer_extractor(
            session_state=_session_state_with_instruction("Send a text message."),
            exec_output=_exec_output_with_stdout_json(
                {"value": {"sent": True}, "answer": "null", "summary": ""}
            ),
            candidate=_candidate_default(),
        )
    )
    assert new_candidate.answer == "null"
    assert new_candidate.answer_type == "null"
    assert new_candidate.task_type_hint == "action"
    assert new_candidate.source == "final_answer_extractor"
    assert payload["status"] == "applied"


def test_hook_falls_back_when_extractor_returns_none(monkeypatch):
    _patched_orchestrator_extractor(monkeypatch, None)
    candidate = _candidate_default()
    new_candidate, payload = _run(
        _orch()._apply_final_answer_extractor(
            session_state=_session_state_with_instruction("Question?"),
            exec_output=_exec_output_with_stdout_json(
                {"value": 1, "answer": "null", "summary": ""}
            ),
            candidate=candidate,
        )
    )
    # Untouched candidate
    assert new_candidate is candidate
    assert payload["status"] == "fallback"


def test_hook_returns_unchanged_when_no_stdout_json(monkeypatch):
    fake_called = {"called": False}

    async def fake(**kwargs):
        fake_called["called"] = True
        return None

    monkeypatch.setattr(
        "adk_appworld_agent.orchestration.orchestrator.extract_appworld_final_answer",
        fake,
    )

    candidate = _candidate_default()
    exec_output = SubagentEnvelope(
        phase=Phase.EXECUTE,
        subagent_name="executor_subagent_code_plan_execute",
        attempt=1,
        status=SubagentStatus.SUCCEEDED,
        payload={"code_execute": {}},  # no stdout_json
    )
    new_candidate, payload = _run(
        _orch()._apply_final_answer_extractor(
            session_state=_session_state_with_instruction("Question?"),
            exec_output=exec_output,
            candidate=candidate,
        )
    )
    assert new_candidate is candidate
    assert payload is None
    # Extractor must NOT have been called
    assert fake_called["called"] is False


# ── F4b: extractor prompt contract — action intent → "null", not "N/A" ────────


def test_extractor_system_prompt_uses_literal_null_not_NA():
    """The extractor used to emit `final_answer="N/A"` for action intent,
    which the AppWorld evaluator rejects (it expects the literal `"null"`
    matching the SubmissionCandidate normaliser). F4b 2026-05-31 aligns
    the extractor's action output with the rest of the pipeline."""
    from pathlib import Path

    prompt = Path(
        "src/adk_appworld_agent/submission/prompts/final_answer_appworld_system.txt"
    ).read_text(encoding="utf-8")
    # Action examples must produce literal `null`, not the legacy "N/A".
    assert '"final_answer": "null"' in prompt
    # Legacy "N/A" must NOT appear in any few-shot example output
    # (a single mention inside the negative-list explanation is OK; ban it
    # only from the JSON example outputs).
    na_lines = [
        line
        for line in prompt.splitlines()
        if line.strip().startswith('"final_answer": "N/A"')
    ]
    assert na_lines == [], f"action example still outputs N/A: {na_lines[:2]}"


def test_extractor_prompt_has_navigation_with_stop_rule():
    """The 325d6ec_2 root after F4: even when the executor correctly
    emits `answer: null`, the extractor was re-classifying navigation-
    with-stop-condition as Single Value Intent (it saw the stopping
    entity in `value`). F4b adds an explicit rule + a few-shot so the
    extractor recognises the pattern and propagates null."""
    from pathlib import Path

    prompt = Path(
        "src/adk_appworld_agent/submission/prompts/final_answer_appworld_system.txt"
    ).read_text(encoding="utf-8")
    assert "Navigation / loop with a stop condition" in prompt
    assert "loop machinery" in prompt
    assert "until" in prompt.lower()
    # task_instruction-anchored rule (mirrors F4 executor-side rule)
    assert "Anchor your classification on the `user_intent` literally" in prompt
    # No task-specific entity from real eval tasks
    assert "Phantom Pain" not in prompt
    assert "Keep going to the next song" not in prompt


def test_extractor_prompt_preserves_rescue_path_for_query_intent():
    """Caveat from baseline audit: 7 baseline tasks were passing only
    because the extractor RESCUED a wrongly-null executor answer
    (executor emitted answer=null but task was actually a query, so the
    extractor pulled the value out as the answer). The F4b prompt must
    NOT break that rescue path — i.e. when user_intent is interrogative,
    the extractor must still extract the value, not propagate null.
    Verified here by asserting the Single Value / Calculation rules are
    still present and not subordinated to the action rule."""
    from pathlib import Path

    prompt = Path(
        "src/adk_appworld_agent/submission/prompts/final_answer_appworld_system.txt"
    ).read_text(encoding="utf-8")
    assert "Single Value Intent" in prompt
    assert "Calculation Intent" in prompt
    # Interrogative cues that distinguish query from action — without
    # these the anchor rule is one-sided.
    for q in ["what", "which", "how many", "tell me", "give me"]:
        assert q in prompt.lower()
