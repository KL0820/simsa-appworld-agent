from __future__ import annotations

import re
from typing import Any, Protocol

from adk_appworld_agent.contracts.submission import (
    CompletionDecision,
    CompletionStatus,
    SubmissionCandidate,
)

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PASS_RE = re.compile(r"Num Passed Tests\s*:\s*(\d+)")
_FAIL_RE = re.compile(r"Num Failed Tests\s*:\s*(\d+)")
_TOTAL_RE = re.compile(r"Num Total\s*Tests\s*:\s*(\d+)")


class _Submitter(Protocol):
    def submit_answer(self, answer: str) -> Any: ...


def _evaluation_summary(
    *,
    passed: int | None = None,
    failed: int | None = None,
    total: int | None = None,
    report_parsed: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    summary = {
        "passed": passed,
        "failed": failed,
        "total": total,
        "report_parsed": report_parsed,
    }
    summary.update(extra)
    return summary


class CompletionGate:
    """Normalizes answer contracts and verifies evaluator success."""

    def verify_and_submit(
        self,
        candidate: SubmissionCandidate | None,
        submitter: _Submitter,
    ) -> CompletionDecision:
        if candidate is None:
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                block_reason="NO_SUBMISSION_CANDIDATE",
            )

        if not isinstance(candidate.answer, str):
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                block_reason="CANDIDATE_ANSWER_NOT_STRING",
                candidate=candidate,
            )

        try:
            normalized_answer = self._normalize_answer(candidate)
        except ValueError as exc:
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                block_reason="CANDIDATE_ANSWER_NORMALIZATION_FAILED",
                candidate=candidate,
                extra=_evaluation_summary(error=str(exc)),
            )

        try:
            report = submitter.submit_answer(normalized_answer)
        except Exception as exc:
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                block_reason="SUBMIT_CALL_FAILED",
                candidate=candidate,
                submitted_answer=normalized_answer,
                extra=_evaluation_summary(error=str(exc)),
            )

        return self._decision_from_report(
            candidate=candidate,
            normalized_answer=normalized_answer,
            report=report,
        )

    @staticmethod
    def _normalize_answer(candidate: SubmissionCandidate) -> str:
        # The AppWorld RPC `submit_answer` interpolates the answer into a
        # SINGLE-QUOTED Python literal on the server side:
        #     self.world.execute(
        #         f"apis.supervisor.complete_task(answer='{answer}')"
        #     )
        # so any raw single-quote / backslash / newline in the answer
        # breaks the string literal → SyntaxError → complete_task is
        # never called → the task's `answer` field stays at its default
        # `"<<NOT_GIVEN>>"`. We saw this in two flavors on a30375d_2:
        #   - 2026-05-30 baseline: `\n by Unknown` suffix (newline)
        #   - 2026-05-31 preflight: `don't` / `else's` (apostrophe)
        # Pre-escape so the server-side interpolation parses to the
        # original answer. Eval uses normalize_text=True so collapsed
        # whitespace doesn't affect substring matching.
        raw = candidate.answer.strip()
        escaped = (
            raw.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", " ")
            .replace("\r", " ")
        )

        if candidate.answer_type == "null":
            return "null"
        if candidate.answer_type == "int":
            return str(int(escaped))
        if candidate.answer_type == "float":
            return str(float(escaped))
        return escaped

    def _decision_from_report(
        self,
        *,
        candidate: SubmissionCandidate,
        normalized_answer: str,
        report: Any,
    ) -> CompletionDecision:
        report_text = self._extract_report_text(report)
        counts = self._extract_eval_counts(report_text)

        if counts is not None:
            passed, failed, total = counts
            summary = _evaluation_summary(
                passed=passed,
                failed=failed,
                total=total,
                report_parsed=True,
            )
            if failed == 0:
                return CompletionDecision(
                    status=CompletionStatus.SUBMITTED,
                    submitted_answer=normalized_answer,
                    evaluation_report=report,
                    candidate=candidate,
                    extra=summary,
                )
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                submitted_answer=normalized_answer,
                evaluation_report=report,
                block_reason="COMPLETION_EVAL_FAILED",
                candidate=candidate,
                extra=summary,
            )

        clean_report = _ANSI_RE.sub("", report_text)
        if "Num Failed Tests : 0" in clean_report:
            return CompletionDecision(
                status=CompletionStatus.SUBMITTED,
                submitted_answer=normalized_answer,
                evaluation_report=report,
                candidate=candidate,
                extra=_evaluation_summary(report_parsed=False),
            )

        fail_count = self._extract_fail_count(clean_report)
        if fail_count is not None and fail_count > 0:
            return CompletionDecision(
                status=CompletionStatus.BLOCKED,
                submitted_answer=normalized_answer,
                evaluation_report=report,
                block_reason="COMPLETION_EVAL_FAILED",
                candidate=candidate,
                extra=_evaluation_summary(failed=fail_count, report_parsed=False),
            )

        return CompletionDecision(
            status=CompletionStatus.BLOCKED,
            submitted_answer=normalized_answer,
            evaluation_report=report,
            block_reason="COMPLETION_REPORT_UNPARSEABLE",
            candidate=candidate,
            extra=_evaluation_summary(report_parsed=False),
        )

    @staticmethod
    def _extract_report_text(report: Any) -> str:
        if isinstance(report, dict):
            for key in ("result", "report", "message"):
                value = report.get(key)
                if value is not None:
                    return str(value)
        return str(report or "")

    @staticmethod
    def _extract_eval_counts(report_text: str) -> tuple[int, int, int] | None:
        if not report_text:
            return None
        clean_report = _ANSI_RE.sub("", report_text)
        passed_match = _PASS_RE.search(clean_report)
        failed_match = _FAIL_RE.search(clean_report)
        total_match = _TOTAL_RE.search(clean_report)
        if not (passed_match and failed_match and total_match):
            return None
        return (
            int(passed_match.group(1)),
            int(failed_match.group(1)),
            int(total_match.group(1)),
        )

    @staticmethod
    def _extract_fail_count(report_text: str) -> int | None:
        failed_match = _FAIL_RE.search(report_text)
        if failed_match is None:
            return None
        return int(failed_match.group(1))
