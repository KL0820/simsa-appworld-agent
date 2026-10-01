"""Result & submission building for the code_plan_execute executor.

Parses Stage-2 stdout, classifies caught-failures, and builds ExecutorResult
/ SubmissionCandidate. Extracted from agent.py in the 2026-06-22
behavior-preserving split."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution import _CodeExecuteRun

import json
from typing import Any

from adk_appworld_agent.contracts.code_plan import CodePlanOutput
from adk_appworld_agent.contracts.executor_result import ExecutorResult, MemoryVariable
from adk_appworld_agent.contracts.submission import SubmissionCandidate
from adk_appworld_agent.subagents.executor.appworld_tools import (
    EXECUTOR_SANDBOX_FINALIZE_MARKER,
)
from adk_appworld_agent.subagents.executor.code_plan_execute._primitives import (
    _FLOAT_RE,
    _INT_RE,
)


def _parse_execute_stdout(
    raw_stdout: str,
) -> tuple[dict | None, dict | None, str | None]:
    """Parse Stage 2 stdout.

    Primary: last JSON line with "value" key → (stdout_json, None, None).
    Fallback: FINALIZE_MARKER line → (None, finalize_args, None).
    Fail: (None, None, error_str).
    """
    lines = (raw_stdout or "").splitlines()

    # Primary: scan in reverse for last valid JSON with "value" key
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except (TypeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        if "value" in obj:
            return obj, None, None
        if "error" in obj and "value" not in obj:
            return None, None, f"code reported error: {obj['error']}"

    # Fallback: FINALIZE_MARKER
    for line in reversed(lines):
        if line.startswith(EXECUTOR_SANDBOX_FINALIZE_MARKER):
            payload = line[len(EXECUTOR_SANDBOX_FINALIZE_MARKER) :].strip()
            try:
                finalize_args = json.loads(payload)
            except (TypeError, ValueError):
                continue
            if isinstance(finalize_args, dict):
                return None, finalize_args, None

    return None, None, "no valid JSON stdout with 'value' key and no finalize marker"


# ── ExecutorResult builder ────────────────────────────────────────────────────


def _build_executor_result(
    *,
    exec_run: _CodeExecuteRun,
    plan: CodePlanOutput,
    milestone_id: str,
) -> tuple[ExecutorResult, SubmissionCandidate, bool]:
    stdout_json = exec_run.stdout_json
    finalize_args = exec_run.finalize_args
    submit_final_args = exec_run.submit_final_args

    # Path Z: structured tool call args (preferred when present). The
    # submit_final ADK tool takes (value, summary, description, answer)
    # via the function-call schema — bypassing the print(json.dumps())
    # codegen path that periodically triggered the cognizant_* wrap-in-
    # string failure mode (fix_batch_a_full 2026-05-31). The shape mirrors
    # Path A's stdout_json so we route through the same code by
    # synthesising a `stdout_json` substitute.
    if submit_final_args is not None and stdout_json is None:
        stdout_json = {
            "value": submit_final_args.get("value"),
            "summary": submit_final_args.get("summary") or "",
            "description": submit_final_args.get("description") or "",
            "answer": submit_final_args.get("answer"),
        }

    # Path A: clean JSON stdout with "value" key
    if stdout_json is not None:
        raw_value = stdout_json.get("value")
        summary = str(stdout_json.get("summary") or "")
        # milestone_done resolution: the executor's explicit `milestone_done`
        # field (typed contract) when present; otherwise default to done.
        explicit_done = stdout_json.get("milestone_done")
        if isinstance(explicit_done, bool):
            milestone_done_flag = explicit_done
        else:
            milestone_done_flag = True
        try:
            value_json = json.dumps(raw_value, ensure_ascii=False)
        except (TypeError, ValueError):
            value_json = json.dumps(str(raw_value), ensure_ascii=False)
        # MemoryVariable.description dual-channel: prefer the runtime
        # description the executor wrote in stdout, fallback to the
        # planner's schema description. Either is non-empty (the planner
        # description has min_length=30; the runtime one can be empty in
        # which case we use the planner's).
        exec_description = stdout_json.get("description")
        if isinstance(exec_description, str) and exec_description.strip():
            mem_description = exec_description.strip()
        else:
            mem_description = plan.output_variable.description
        variable = MemoryVariable(
            name=plan.output_variable.name,
            value_json=value_json,
            description=mem_description,
            source_milestone_id=milestone_id,
        )
        # SubmissionCandidate routing: prefer the executor's explicit
        # `answer` key (4-field unified contract). It disambiguates
        # action ("null" literal) from query (natural-language answer)
        # without forcing the submission layer to type-infer from value.
        # Falls back to value-type inference only when the key is
        # completely missing — that path covers legacy / non-conformant
        # outputs and shouldn't be hit on a planner trained with the
        # 4-field rule.
        candidate = _candidate_from_answer_field(
            stdout_json=stdout_json,
            raw_value=raw_value,
        )
        result = ExecutorResult(
            answer=candidate.answer,
            milestone_done=milestone_done_flag,
            summary=summary,
            variables=[variable],
        )
        return result, candidate, True

    # Path B: FINALIZE_MARKER fallback
    if finalize_args is not None:
        result = _parse_finalize_args(finalize_args, milestone_id)
        return result, _infer_submission_candidate(result.answer), True

    # Path C: failure
    result = ExecutorResult(answer="", milestone_done=False, summary="", variables=[])
    return result, _infer_submission_candidate(""), False


def _parse_finalize_args(args: dict, milestone_id: str) -> ExecutorResult:
    answer = args.get("answer", "")
    if not isinstance(answer, str):
        answer = str(answer)
    milestone_done = bool(args.get("milestone_done", False))
    summary = args.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)
    raw_variables = args.get("variables_json", "[]")
    parsed_variables: list[MemoryVariable] = []
    if isinstance(raw_variables, str) and raw_variables.strip():
        try:
            items = json.loads(raw_variables)
        except (TypeError, ValueError):
            items = []
        if isinstance(items, dict):
            items = [items]
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                try:
                    mv = MemoryVariable.model_validate(item)
                except Exception:
                    continue
                if not mv.value_json.strip():
                    continue
                if mv.source_milestone_id == "" and milestone_id:
                    mv = mv.model_copy(update={"source_milestone_id": milestone_id})
                parsed_variables.append(mv)
    return ExecutorResult(
        answer=answer,
        milestone_done=milestone_done,
        summary=summary,
        variables=parsed_variables,
    )


def _value_as_answer_string(value: Any) -> str:
    """Render the raw value as a fallback answer string for ExecutorResult.

    Used before final_answer_extractor runs; the extractor overrides
    SubmissionCandidate.answer for the actual submission. Kept short
    (10k char cap) for log readability.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        rendered = json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(value)
    return rendered[:10000]


def _candidate_from_answer_field(
    *, stdout_json: dict, raw_value: Any
) -> SubmissionCandidate:
    """Route to action vs query using the executor's explicit `answer` key.

    Contract (planner + executor prompt):
    - `answer == "null"` (the four-character string) -> action / intermediate.
    - `answer` is any other non-empty string -> query with that answer.
    - `answer is None` or the key is missing entirely -> fallback to
      `_infer_submission_candidate_from_value`. We also tolerate Python
      `None` here even though the prompt says "literal `null`" because
      Gemini occasionally serialises the Python literal `None` to JSON
      `null`, which `json.loads` lifts back to `None`.

    The submission layer downstream (final_answer_extractor) may still
    rewrite `answer` for query tasks; this routing only fixes the
    `task_type_hint` / `answer_type` so action tasks don't get
    type-inferred into queries (the 325d6ec_2 regression).
    """
    if "answer" not in stdout_json:
        return _infer_submission_candidate_from_value(raw_value)
    answer_obj = stdout_json.get("answer")
    if answer_obj is None:
        return _infer_submission_candidate_from_value(raw_value)
    answer_str = str(answer_obj).strip()
    if answer_str == "" or answer_str.lower() == "null":
        return SubmissionCandidate(
            answer="null",
            task_type_hint="action",
            answer_type="null",
            source="executor",
        )
    # Non-null answer string -> query path. answer_type stays "str" here;
    # the extractor refines int / float when the bare answer is a number.
    return SubmissionCandidate(
        answer=answer_str,
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )


def _infer_submission_candidate_from_value(value: Any) -> SubmissionCandidate:
    """Build a SubmissionCandidate by inspecting the raw value's type.

    Type maps onto SubmissionCandidate.answer_type's Literal:
    null / int / float / str. bool, list, dict all serialize to str
    (their JSON repr is the displayed answer) — the extractor will
    typically override this for query tasks anyway.
    """
    answer_str = _value_as_answer_string(value)
    if value is None or answer_str.strip().lower() == "null" or answer_str == "":
        return SubmissionCandidate(
            answer="null",
            task_type_hint="action",
            answer_type="null",
            source="executor",
        )
    # bool must come before int (isinstance(True, int) is True). bool maps
    # to str because the contract Literal doesn't include "bool".
    if isinstance(value, bool):
        return SubmissionCandidate(
            answer=answer_str,
            task_type_hint="query",
            answer_type="str",
            source="executor",
        )
    if isinstance(value, int):
        return SubmissionCandidate(
            answer=answer_str,
            task_type_hint="query",
            answer_type="int",
            source="executor",
        )
    if isinstance(value, float):
        return SubmissionCandidate(
            answer=answer_str,
            task_type_hint="query",
            answer_type="float",
            source="executor",
        )
    # list / dict / str all fall through to str — the answer field carries
    # the JSON-serialized representation. Evaluator submission-cast logic
    # handles list/dict from the answer string.
    return SubmissionCandidate(
        answer=answer_str,
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )


def _infer_submission_candidate(final_response: str) -> SubmissionCandidate:
    """Legacy string-based variant — still used by FINALIZE_MARKER (Path B)
    + empty failure (Path C) where we only have a free-text answer.
    """
    answer = (final_response or "").strip()
    normalized = answer.lower()
    if not answer or normalized == "null":
        return SubmissionCandidate(
            answer="null",
            task_type_hint="action",
            answer_type="null",
            source="executor",
        )
    if _INT_RE.fullmatch(answer):
        return SubmissionCandidate(
            answer=answer,
            task_type_hint="query",
            answer_type="int",
            source="executor",
        )
    if _FLOAT_RE.fullmatch(answer):
        return SubmissionCandidate(
            answer=answer,
            task_type_hint="query",
            answer_type="float",
            source="executor",
        )
    return SubmissionCandidate(
        answer=answer,
        task_type_hint="query",
        answer_type="str",
        source="executor",
    )


# ── LLM runners ──────────────────────────────────────────────────────────────
