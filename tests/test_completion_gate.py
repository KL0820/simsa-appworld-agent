from __future__ import annotations

from adk_appworld_agent.contracts.submission import (
    CompletionStatus,
    SubmissionCandidate,
)
from adk_appworld_agent.orchestration.completion_gate import CompletionGate


class _FakeSubmitter:
    def __init__(self, *, result=None, raises: Exception | None = None):
        self.result = result
        self.raises = raises
        self.calls: list[str] = []

    def submit_answer(self, answer: str):
        self.calls.append(answer)
        if self.raises is not None:
            raise self.raises
        return self.result


def test_gate_blocks_when_candidate_missing():
    gate = CompletionGate()
    submitter = _FakeSubmitter()

    decision = gate.verify_and_submit(None, submitter)

    assert decision.status == CompletionStatus.BLOCKED
    assert decision.block_reason == "NO_SUBMISSION_CANDIDATE"
    assert submitter.calls == []


def test_gate_normalizes_query_numeric_answers():
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 1\nNum Failed Tests : 0\nNum Total Tests : 1"
        }
    )
    candidate = SubmissionCandidate(
        answer=" 42 ",
        task_type_hint="query",
        answer_type="int",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.SUBMITTED
    assert decision.submitted_answer == "42"
    assert submitter.calls == ["42"]
    assert decision.extra == {
        "passed": 1,
        "failed": 0,
        "total": 1,
        "report_parsed": True,
    }


def test_gate_strips_newlines_from_submitted_answer():
    """AppWorld RPC `submit_answer` silently substitutes any answer
    containing a raw newline with `<<NOT_GIVEN>>` (a30375d_2 2026-05-31).
    Gate must collapse internal newlines to spaces before submission."""
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 3\nNum Failed Tests : 0\nNum Total Tests : 3"
        }
    )
    candidate = SubmissionCandidate(
        answer="Your dreams are the blueprints for your reality.\n by Unknown",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.SUBMITTED
    # Internal `\n` between sentence and ` by Unknown` collapsed to space.
    assert decision.submitted_answer == (
        "Your dreams are the blueprints for your reality.  by Unknown"
    )
    assert "\n" not in submitter.calls[0], (
        f"submitter received unescaped newline: {submitter.calls[0]!r}"
    )


def test_gate_escapes_apostrophes_from_submitted_answer():
    """Same root as the newline case (a30375d_2 2026-05-31 preflight): the
    AppWorld RPC `submit_answer` interpolates the answer into a single-
    quoted Python f-string literal — `f\"apis.supervisor.complete_task(answer='{answer}')\"`.
    A raw apostrophe closes the literal early, the rest is invalid
    Python, the call silently no-ops, and the task answer stays at the
    server's `<<NOT_GIVEN>>` default. Gate must backslash-escape `'` so
    the server-side interpolation parses to the original string."""
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 3\nNum Failed Tests : 0\nNum Total Tests : 3"
        }
    )
    candidate = SubmissionCandidate(
        answer="Your time is limited, don't waste it living someone else's life.",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.SUBMITTED
    # Each `'` is replaced with `\\'` (two source chars: backslash, quote).
    assert decision.submitted_answer == (
        "Your time is limited, don\\'t waste it living someone else\\'s life."
    )
    # Sanity: when the server interpolates submitted_answer into a
    # single-quoted Python literal, the literal must be valid Python.
    code = f"x = '{decision.submitted_answer}'"
    namespace: dict = {}
    exec(code, namespace)
    assert namespace["x"] == candidate.answer, (
        f"round-trip lost the original; got {namespace['x']!r}"
    )


def test_gate_escapes_backslash_before_quote():
    """Order matters: must escape `\\` BEFORE `'`. Otherwise the `\\'`
    produced by quote-escape would itself get backslash-escaped to
    `\\\\'`, which on the server parses as `\\` + closing quote — wrong."""
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 1\nNum Failed Tests : 0\nNum Total Tests : 1"
        }
    )
    # Pathological answer: literal backslash AND apostrophe.
    candidate = SubmissionCandidate(
        answer="a\\b'c",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.SUBMITTED
    code = f"x = '{decision.submitted_answer}'"
    namespace: dict = {}
    exec(code, namespace)
    assert namespace["x"] == candidate.answer


def test_gate_normalizes_action_null_answers():
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 2\nNum Failed Tests : 0\nNum Total Tests : 2"
        }
    )
    candidate = SubmissionCandidate(
        answer="",
        task_type_hint="action",
        answer_type="null",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.SUBMITTED
    assert decision.submitted_answer == "null"
    assert submitter.calls == ["null"]


def test_gate_blocks_when_evaluator_reports_failures():
    gate = CompletionGate()
    submitter = _FakeSubmitter(
        result={
            "result": "Num Passed Tests : 1\nNum Failed Tests : 2\nNum Total Tests : 3"
        }
    )
    candidate = SubmissionCandidate(
        answer="hello",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.BLOCKED
    assert decision.block_reason == "COMPLETION_EVAL_FAILED"
    assert decision.extra == {
        "passed": 1,
        "failed": 2,
        "total": 3,
        "report_parsed": True,
    }


def test_gate_blocks_when_report_is_unparseable():
    gate = CompletionGate()
    submitter = _FakeSubmitter(result={"result": "stub eval ok"})
    candidate = SubmissionCandidate(
        answer="hi",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.BLOCKED
    assert decision.block_reason == "COMPLETION_REPORT_UNPARSEABLE"
    assert decision.extra == {
        "passed": None,
        "failed": None,
        "total": None,
        "report_parsed": False,
    }


def test_gate_reports_submitter_exception_as_block():
    gate = CompletionGate()
    submitter = _FakeSubmitter(raises=RuntimeError("rpc down"))
    candidate = SubmissionCandidate(
        answer="hi",
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )

    decision = gate.verify_and_submit(candidate, submitter)

    assert decision.status == CompletionStatus.BLOCKED
    assert decision.block_reason == "SUBMIT_CALL_FAILED"
    assert decision.extra["error"] == "rpc down"
