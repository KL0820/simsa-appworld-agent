from __future__ import annotations

import asyncio
import json

import pytest

from adk_appworld_agent.contracts.code_plan import CodePlanOutput, VariableSpec
from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import SubagentStatus
from adk_appworld_agent.observability.subagent_logs import (
    io_record_from_subagent_output,
    render_subagent_io_markdown,
)
from adk_appworld_agent.orchestration.state import Phase, TaskContext
from adk_appworld_agent.subagents.executor.appworld_tools import (
    EXECUTOR_SANDBOX_FINALIZE_MARKER,
)
from adk_appworld_agent.subagents.executor.code_plan_execute import (
    CODE_EXECUTOR_SYSTEM_PROMPT,
    CODE_PLANNER_SYSTEM_PROMPT,
    _build_code_execute_prompt,
    _build_code_execute_repair_prompt,
    _build_code_plan_prompt,
    _build_executor_result,
    _clean_code_arg,
    _CodeExecuteRun,
    _extract_tool_calls,
    _no_execute_python_call_error,
    _parse_code_plan,
    _parse_execute_stdout,
    _should_repair_code_execute_run,
    build_code_plan_execute_subagent,
)
from adk_appworld_agent.subagents.registry import (
    available_subagent_impls,
    build_subagent,
)

# ── Schema / prompt unit tests ────────────────────────────────────────────────


def test_code_plan_output_contract_validates_plan_and_variable_name():
    output = CodePlanOutput(
        plan_steps=["Do work."],
        construct_step="Build result_dict.",
        print_step="Print json.dumps(result_dict).",
        output_variable=VariableSpec(
            name=" result ",
            description="Result variable holding the milestone output value.",
        ),
    )

    assert output.output_variable.name == "result"
    assert output.plan_steps == ["Do work."]
    assert output.numbered_steps() == [
        "Do work.",
        "Build result_dict.",
        "Print json.dumps(result_dict).",
    ]

    valid_spec = VariableSpec(
        name="x",
        description="Generic placeholder variable for contract tests.",
    )

    # plan_steps blank
    with pytest.raises(ValueError):
        CodePlanOutput(
            plan_steps=[""],
            construct_step="x",
            print_step="y",
            output_variable=valid_spec,
        )
    # variable name blank
    with pytest.raises(ValueError):
        CodePlanOutput(
            plan_steps=["Do work."],
            construct_step="x",
            print_step="y",
            output_variable=VariableSpec(
                name=" ",
                description="Generic placeholder variable for contract tests.",
            ),
        )
    # construct_step blank — was previously possible (final 2 steps lived
    # inside the plan list) and is the regression that schema-driven
    # validation locks down (a30375d_2 / 9dabbc9_2 M4 in smoke v3).
    with pytest.raises(ValueError):
        CodePlanOutput(
            plan_steps=["Do work."],
            construct_step="",
            print_step="y",
            output_variable=valid_spec,
        )
    # print_step blank
    with pytest.raises(ValueError):
        CodePlanOutput(
            plan_steps=["Do work."],
            construct_step="x",
            print_step="",
            output_variable=valid_spec,
        )


def test_code_plan_prompt_contains_task_milestone_apis_and_prior_variables():
    subagent_input = _sample_execute_input()

    prompt = _build_code_plan_prompt(subagent_input)

    assert '"task_id": "task_1"' in prompt
    assert '"intent": "Find useful data."' in prompt
    # full spec enrichment replaces simple {name} with {api_name, parameters, response_schemas}
    assert '"api_name": "show_transactions"' in prompt
    assert '"parameters"' in prompt
    # prior_variables_preview is now appended as a markdown section AFTER the
    # JSON payload (Entry 7 — avoids json.dumps escaping \n in the markdown).
    assert "## Prior milestone variables" in prompt
    assert "prior value" in prompt
    # Should NOT be inside the JSON payload (would get \n escape bug)
    assert '"prior_variables_preview"' not in prompt


def test_code_plan_prompt_skips_cold_start_rationale_placeholder():
    """`cold-start: rough plan` (and other controller-internal sentinels) are
    placeholders that carry NO actionable guidance. Rendering them adds a
    `## Continuation rationale` section + meta-instruction attention cost
    on cycle 0, despite there being no real guidance. The block builder
    must skip these placeholders so the c0 m0 prompt structurally matches
    the no-rationale path."""
    base = _sample_execute_input()
    for placeholder in (
        "cold-start: rough plan",
        "fallback: no milestones to run",
        "fallback: retry active milestone",
    ):
        meta = dict(base.metadata)
        meta["latest_continuation_rationale"] = placeholder
        si = base.model_copy(update={"metadata": meta})
        prompt = _build_code_plan_prompt(si)
        assert "## Continuation rationale" not in prompt, (
            f"placeholder {placeholder!r} should not render rationale section"
        )


def test_code_plan_prompt_renders_real_rationale():
    """Non-placeholder rationale text (LLM-generated or framework_override
    with actionable content) MUST still render — the cleanup only filters
    known no-op sentinels."""
    base = _sample_execute_input()
    real = "filter by sender before pagination; previous attempt skipped this"
    meta = dict(base.metadata)
    meta["latest_continuation_rationale"] = real
    si = base.model_copy(update={"metadata": meta})
    prompt = _build_code_plan_prompt(si)
    assert "## Continuation rationale" in prompt
    assert real in prompt


def test_code_planner_system_prompt_teaches_choosing_among_overlapping_apis():
    """Universal API-selection rule: when candidate_apis has overlapping
    semantics, the planner must compare descriptions against milestone
    wording and prefer the candidate whose semantics align most specifically
    (e.g. scope-defining API over generic + client-side filter; bulk API
    over per-item × N). This is metacognitive — a "pause and compare"
    nudge, not a forced behavior, so it benefits any task with overlapping
    candidates and is silent when candidates are independent.

    Layer-level phrasing only per `feedback_no_task_specific_in_prompts`."""
    prompt = CODE_PLANNER_SYSTEM_PROMPT
    assert "Choosing among overlapping candidate APIs" in prompt
    overlap_section = prompt.split("Choosing among overlapping candidate APIs", 1)[1]
    overlap_section = overlap_section.split("##", 1)[0]
    # core directive: read each candidate's description, don't default to familiar
    assert (
        "do NOT default" in overlap_section
        or "do not default" in overlap_section.lower()
    )
    assert "description" in overlap_section.lower()
    # scope/constraint preference
    assert "scope" in overlap_section.lower() or "constraint" in overlap_section.lower()
    # bulk vs per-item alternative
    assert "per-item" in overlap_section.lower() or "bulk" in overlap_section.lower()
    # ranking position is approximate
    assert "ranking" in overlap_section.lower() or "ranked" in overlap_section.lower()


def test_code_planner_system_prompt_teaches_response_schema_reading():
    """Universal compound-retrieval rule: when a bulk candidate API's
    `response_schemas.success` already lists the attribute the milestone
    needs alongside the item identifier, planner should use that bulk
    call alone instead of layering a per-item detail-API call on top.
    The rule is metacognitive: read each candidate's response_schemas
    keys upfront before committing to a plan.

    Universal benefit:
    - Tasks with bulk-or-detail overlap → rule guides to bulk
    - Tasks where the bulk API lacks the field → rule explicitly
      authorizes per-item detail
    - Tasks with a single API → rule is silent (no comparison needed)

    Layer-level only per `feedback_no_task_specific_in_prompts`."""
    prompt = CODE_PLANNER_SYSTEM_PROMPT
    assert "Reading response_schemas for compound retrieval" in prompt
    section = prompt.split("Reading response_schemas for compound retrieval", 1)[1]
    section = section.split("##", 1)[0]
    # core directive: check response_schemas before committing
    assert "response_schemas.success" in section
    assert "per-item detail" in section.lower() or "per-item" in section.lower()
    # explicit "bulk over per-item" preference
    assert (
        "bulk" in section.lower()
        or "one call" in section.lower()
        or "ONE call" in section
    )
    # explicit clause that per-item is still valid when bulk lacks fields
    assert (
        "does NOT carry" in section
        or "does not carry" in section.lower()
        or ("justified only when" in section.lower() and "fields" in section.lower())
    )
    # read upfront, not after the plan
    assert "upfront" in section.lower() or "before writing" in section.lower()


def test_code_planner_system_prompt_rules_have_explicit_conditional_openings():
    """Both new rule sections added by code_planner_input_v1 must open with
    an explicit "applies only when ..." clause so the planner doesn't waste
    attention on rules whose triggering section is absent (notably c0 m0,
    where neither `## Continuation rationale` nor `## Prior milestone
    variables` exists)."""
    prompt = CODE_PLANNER_SYSTEM_PROMPT
    rat_header = "## Reading the Continuation rationale section"
    trunc_header = "## Reading truncated distribution / sample summaries"
    assert rat_header in prompt
    assert trunc_header in prompt

    # Normalize linebreaks in the prompt body — the source string is
    # hard-wrapped for readability, but the conditional opening must be
    # detectable regardless of where the wrap happens.
    def _norm(s: str) -> str:
        return " ".join(s.split())

    # Rationale section runs until the next top-level rule header
    rationale_section = _norm(prompt.split(rat_header, 1)[1].split(trunc_header, 1)[0])
    assert "applies ONLY when" in rationale_section
    assert "skip this rule" in rationale_section
    # Truncation section runs until the final "Respond ONLY" closing
    trunc_section = _norm(prompt.split(trunc_header, 1)[1].split("Respond ONLY", 1)[0])
    assert "applies ONLY when" in trunc_section
    assert "skip this rule" in trunc_section


def test_code_planner_system_prompt_has_cuga_style_planning_constraints():
    prompt = CODE_PLANNER_SYSTEM_PROMPT

    assert "task_instruction" in prompt
    assert "exact parameter names from the schema" in prompt
    # access_token / login concepts must NOT appear in the planner prompt —
    # the proxy auto-injects tokens, and showing the LLM these strings only
    # invites hallucinated `cognizant_access_token = "..."` patterns.
    assert "access_token" not in prompt
    assert "login" not in prompt
    assert "Explain data flow" in prompt
    assert "Describe loops explicitly" in prompt
    assert "pagination" in prompt
    # Print contract — axis C alignment: planner + executor both target
    # {value, summary}; variable_name / description / answer are NOT in
    # the print payload (variable_name + description live in
    # output_variable schema; answer was legacy and is now derived).
    assert "construct_step" in prompt
    assert "print_step" in prompt
    assert '"value"' in prompt
    assert '"summary"' in prompt
    assert "variable_name" not in prompt or "variable_name and description" in prompt
    # Schema-driven completeness — construct/print are now their own fields.
    # State-changing milestones must call the action API directly. The
    # previous "Include conditional branches" rule trained Gemini Flash to
    # wrap actions in `if found else cannot_complete` and the executor
    # walked the do-nothing branch, leaving venmo / phone state unchanged
    # (smoke 233903: 024c982_2 1/7, 9dabbc9_2 1/9, 3d9a636_2 2/5).
    assert "Action-milestone rule" in prompt
    assert "Read-milestone rule" in prompt
    assert "do nothing" in prompt
    assert "Include conditional branches for missing" not in prompt
    # The previous "flag it as a blocker" rule pulled in the opposite
    # direction from the action rule. The two together let the planner
    # treat any runtime-might-be-None value as a blocker reason and write
    # do-nothing branches anyway. The rule is now scoped to schema-level
    # blockers (no candidate API can produce the required input) and
    # explicitly disclaims the runtime-None case.
    assert (
        "If a required parameter cannot be satisfied by prior variables or task data,"
        not in prompt
    )
    assert "no candidate API will produce" in prompt
    assert "defensive" in prompt or "Defensive" in prompt


def test_code_planner_prompt_carries_axis_c_description_enrichment():
    """axis C Phase 1 fix — the planner prompt MUST require
    output_variable.description to (a) spell out derived shape (not raw API
    shape), (b) copy noun qualifier / time window / selection conditions
    verbatim from milestone wording, (c) reinforce that description is a
    STRUCTURED schema field (not printed). Without these, post-axis-C
    description writes too sparse → 4-loss cluster (0a9d82a, 9dabbc9,
    425a494, 3b8fb7a) per docs/notes/2026-05-15_axisC_fix_options.md.

    Assert only LAYER-LEVEL phrases per `feedback_no_task_specific_in_prompts` —
    no specific task instruction strings."""
    prompt = CODE_PLANNER_SYSTEM_PROMPT
    # (a) derived shape rule
    assert "derived shape" in prompt.lower() or "shape of the data" in prompt.lower()
    assert "not the shape of the raw API" in prompt or "raw API response" in prompt
    # (b) verbatim qualifier carry-over
    assert "noun qualifier" in prompt
    assert "time window" in prompt
    assert "selection condition" in prompt
    assert "verbatim" in prompt or "same wording" in prompt
    # (c) STRUCTURED schema field, separate from the runtime description
    # the executor emits in stdout. The dual-channel rule says the runtime
    # description (in print_step's JSON) is the preferred MemoryVariable
    # description if non-empty; otherwise the planner's schema description
    # is used.
    assert "STRUCTURED schema field" in prompt
    assert "runtime" in prompt.lower()
    assert "distinct from" in prompt or "separate" in prompt.lower()
    # (d) action-milestone value=null allowance (axis C 325d6ec class)
    assert "mutation" in prompt
    assert "side effect" in prompt
    assert "null" in prompt


def test_code_executor_system_prompt_constraints():
    prompt = CODE_EXECUTOR_SYSTEM_PROMPT

    assert "execute_python" in prompt
    assert "EXACTLY ONCE" in prompt
    assert "json.dumps" in prompt
    assert "{{" not in prompt
    assert "}}" not in prompt
    assert "printing a Python program string" in prompt
    assert "put the actual program directly" in prompt
    assert "finalize" in prompt
    assert "Do NOT call" in prompt


def test_code_planner_prompt_carries_4field_print_contract():
    """All four print-fields (value, summary, description, answer) must be
    named in the planner prompt as the unified print contract. The planner
    has to teach the executor what shape to emit, and any partial mention
    (e.g. only {value, summary}) is what produced the axis_abc_full 325d6ec_2
    regression: when the prompt dropped the `answer` field, action tasks
    lost their `"null"` discriminator and the submission layer started
    type-inferring queries from value dicts."""
    prompt = CODE_PLANNER_SYSTEM_PROMPT
    assert '"value"' in prompt
    assert '"summary"' in prompt
    assert '"description"' in prompt
    assert '"answer"' in prompt
    # The mandatory-four rule must be stated explicitly so the planner
    # cannot omit any single key as an "optional" extension.
    assert "four-field print contract" in prompt or "All four" in prompt
    # The answer-field rule must distinguish query / mutation / intermediate
    # — without these the planner cannot decide whether to write a string
    # or the "null" literal.
    assert "mutation" in prompt
    assert '"null"' in prompt
    assert "intermediate" in prompt.lower()


def test_answer_field_rule_anchored_on_task_instruction_not_milestone():
    """The 325d6ec_2 regression (newgraph_full56 2026-05-30): milestone
    wording mentions identifying "the first downloaded song", planner emits
    that song's title as `answer`, eval expects literal `"null"` because the
    task_instruction is purely imperative ("Keep going to the next song
    until you reach …"). Without anchoring the classification on the
    task_instruction's question form (not the milestone wording), Gemini
    keeps mis-classifying loop-stopper entities as the user's requested
    answer.

    Assert layer-level phrasing only — no task strings."""
    for prompt_name, prompt in (
        ("planner", CODE_PLANNER_SYSTEM_PROMPT),
        ("executor", CODE_EXECUTOR_SYSTEM_PROMPT),
    ):
        # Classification anchored on task_instruction, not milestone wording.
        assert "task_instruction" in prompt, prompt_name
        # Interrogative cue list — the layer-level signal for query.
        # At least three of these question words must be enumerated so the
        # rule has teeth beyond a single example.
        interrogatives = ["what", "which", "how many", "list", "tell me", "give me"]
        present = sum(1 for q in interrogatives if q in prompt.lower())
        assert present >= 3, (
            f"{prompt_name} prompt enumerates {present}/6 interrogatives; "
            f"need ≥3 to make the query-form rule concrete"
        )
        # Imperative-verb cluster — also layer-level (no specific app).
        # At least three action verbs must be enumerated.
        imperatives = [
            "send",
            "create",
            "update",
            "delete",
            "play",
            "advance",
            "move",
            "navigate",
            "reset",
            "mark",
            "toggle",
        ]
        present = sum(1 for v in imperatives if v in prompt.lower())
        assert present >= 5, (
            f"{prompt_name} prompt enumerates {present}/11 imperatives; "
            f"need ≥5 to cover non-CRUD mutation classes"
        )
        # The loop-stopper carve-out: the rule must explicitly say that an
        # entity identified to satisfy an "until / when" condition is NOT
        # the user's answer. This is the exact 325d6ec_2 trap.
        assert "until" in prompt.lower(), prompt_name
        assert (
            "loop machinery" in prompt.lower()
            or "loop's stopping criterion" in prompt.lower()
            or "stopping criterion" in prompt.lower()
        ), prompt_name


def test_code_executor_prompt_carries_4field_submit_final_contract():
    """Z (2026-05-31): the 4-field contract (value / summary / description
    / answer) is now expressed as `submit_final(...)` tool call args
    instead of `print(json.dumps({...}))` JSON keys. The keyword args
    must all appear so the LlmAgent knows the schema. A drift between
    planner and executor on these four fields is the failure mode
    bb50c8a tried to fix and that this revision keeps fixed."""
    prompt = CODE_EXECUTOR_SYSTEM_PROMPT
    # Tool call form: submit_final(value=..., summary=..., description=..., answer=...)
    assert "submit_final" in prompt
    assert "value=" in prompt
    assert "summary=" in prompt
    assert "description=" in prompt
    assert "answer=" in prompt
    # The instruction must call out submit_final as a separate tool call
    # after execute_python, not optional decoration.
    assert "Call `submit_final`" in prompt or "call `submit_final`" in prompt
    # Mutation vs query routing inside the executor:
    assert "mutation" in prompt.lower()
    assert "query" in prompt.lower()


def test_planner_and_executor_prompts_have_no_task_specific_terms():
    """Per feedback_no_task_specific_in_prompts — neither prompt may reference
    a specific test task's entity, instruction quote, or operation phrase.

    Bare app names (venmo / spotify / phone / file_system / ...) are NOT
    forbidden because they appear in legitimate API-surface notation
    (`apis.<app>.<method>`) and in layer-level noun-disambiguation
    examples. What we forbid is the next layer up: concrete entity nouns
    and instruction-quote phrases lifted from specific eval tasks.
    """
    forbidden = [
        # Concrete entity nouns lifted from specific past failures
        "Athens trip",
        "Phantom Pain",
        "limited-screen-time",
        "downloaded song",
        "housing bill",
        # Operation phrases lifted from specific instructions
        "make private Venmo",
        "Send a payment THEN return",
        "longest limited-screen-time",
        "Keep going to the next song",
        # Task IDs that are not in the §2a stable smoke set
        "0a9d82a_2",
        "9dabbc9_2",
        "325d6ec_2",
        "3b8fb7a_2",
        "425a494_2",
    ]
    for prompt_name, prompt in (
        ("planner", CODE_PLANNER_SYSTEM_PROMPT),
        ("executor", CODE_EXECUTOR_SYSTEM_PROMPT),
    ):
        for term in forbidden:
            assert term.lower() not in prompt.lower(), (
                f"{prompt_name} prompt contains task-specific term {term!r}"
            )


def test_clean_code_arg_unwraps_printed_program_source():
    code = _clean_code_arg(
        'cognizant_code = """\n'
        "import json\n"
        "result = apis.venmo.search_friends(page_index=0)\n"
        'print(json.dumps({"value": 1, "answer": "null"}))\n'
        '"""\n'
        "print(cognizant_code)"
    )

    assert "cognizant_code" not in code
    assert 'print(json.dumps({"value": 1, "answer": "null"}))' in code


def test_clean_code_arg_dedents_uniform_leading_whitespace():
    # Gemini sometimes prefixes every line with a leading space inside the
    # tool-call code argument. Dedent has to run before the outer strip so the
    # common prefix is detected from line 1, otherwise compile() raises
    # IndentationError.
    raw = (
        " import json\n"
        " result = []\n"
        " while True:\n"
        "    result.append(1)\n"
        "    break\n"
        ' print(json.dumps({"value": result}))'
    )
    code = _clean_code_arg(raw)

    compile(code, "<test>", "exec")
    assert not code.startswith(" ")


def test_clean_code_arg_preserves_genuine_indent_errors_for_repair():
    # Truly inconsistent indentation (no uniform common prefix) must still
    # surface so the repair loop can react to it.
    raw = "import json\n  print('bad indent')"
    code = _clean_code_arg(raw)

    assert code == raw


def test_clean_code_arg_repairs_broken_newline_string_literal():
    # Gemini Flash function-call serialization sometimes lowers the
    # two-character escape `\n` inside a single-character string literal
    # to a real newline, leaving behind `'<real-newline>'` which Python
    # cannot compile (SyntaxError: unterminated string literal). Surfaced
    # on a30375d_2 M2 in smoke v4 with 10 identical repair attempts.
    raw = "lines = note_content.split('\n')\nfor line in lines:\n    print(line)"

    code = _clean_code_arg(raw)

    # The pattern must compile after cleaning.
    compile(code, "<test>", "exec")
    assert "split('\\n')" in code
    # And the same input idempotent when run twice (the repair shouldn't
    # double-escape an already-correct `\n`).
    code_again = _clean_code_arg(code)
    assert code_again == code


def test_clean_code_arg_repairs_escaped_source_quote_delimiters():
    # Gemini tool-call args can leak JSON-style escaping into the Python source
    # itself: directory_path=\'...\' is invalid because the backslash sits
    # outside any string literal. The cleaner should recover the intended
    # Python source before guardrail validation.
    raw = (
        "import json\n\n"
        "meeting_files = apis.file_system.show_directory("
        "directory_path=\\'~/documents/work/meetings_files/\\', "
        "entry_type=\\'files\\', recursive=False)\n"
        "print(json.dumps({"
        '\\"value\\": meeting_files, '
        '\\"summary\\": \\"ok\\", '
        '\\"description\\": \\"listed\\", '
        '\\"answer\\": \\"null\\"'
        "}))"
    )

    code = _clean_code_arg(raw)

    compile(code, "<test>", "exec")
    assert "directory_path='~/documents/work/meetings_files/'" in code
    assert "\\'" not in code


def test_clean_code_arg_preserves_valid_escaped_quote_inside_string():
    raw = "title = 'Nostalgia\\'s Hold'\nprint(title)"

    code = _clean_code_arg(raw)

    assert code == raw
    compile(code, "<test>", "exec")


def test_clean_code_arg_keeps_real_newlines_in_multiline_strings():
    # A genuine multi-line string literal (real newlines spanning more than
    # one character of payload) is left alone; only the single-character
    # ``'<newline>'`` / ``"<newline>"`` shape is repaired.
    raw = "doc = '''\n    first\n    second\n'''\nprint(doc)"

    code = _clean_code_arg(raw)

    assert "'''" in code
    compile(code, "<test>", "exec")


def test_clean_code_arg_repairs_unwrap_introduced_newline_corruption():
    # Reproduces the a30375d_2 / 7847649_2 / d194965_2 / 0d01c76_2 cluster
    # from full56 v1: the LLM wrapped the program in
    # ``cognizant_code = """..."""; print(cognizant_code)``. The inner
    # program is correctly escaped (``split('\\n')`` is 4 chars: quote,
    # backslash, n, quote). ``unwrap_printed_program_source`` extracts the
    # inner string via ast.value.value, which decodes the escape into a
    # real newline character — the resulting code has ``split('<NL>')``
    # which Python rejects as ``unterminated string literal``. Without
    # post-unwrap repair, the executor's repair loop trips on the same
    # SyntaxError every attempt and the milestone never finalises.
    raw = (
        'cognizant_code = """\n'
        "import json\n"
        'note_content = prior_variable_values["note"]["value"]\n'
        "lines = note_content.split('\\n')\n"
        'print(json.dumps({"value": lines}))\n'
        '"""\n'
        "print(cognizant_code)"
    )

    code = _clean_code_arg(raw)

    # cognizant_code wrapper is gone (unwrap fired).
    assert "cognizant_code" not in code
    # The escape sequence inside the inner program survived intact.
    assert "split('\\n')" in code
    compile(code, "<test>", "exec")


def test_code_execute_prompt_includes_full_allowed_api_specs():
    subagent_input = _sample_execute_input()
    plan = CodePlanOutput(
        plan_steps=["Call the allowed API."],
        construct_step="Construct result_dict.",
        print_step="Print the final JSON.",
        output_variable=VariableSpec(
            name="result",
            description="Result value holding the API response payload.",
        ),
    )

    prompt = _build_code_execute_prompt(subagent_input, plan)

    assert "Allowed API specs (authoritative JSON):" in prompt
    assert '"api_name": "show_transactions"' in prompt
    assert '"parameters": [' in prompt
    assert '"response_schemas": {' in prompt
    assert "venmo.show_transactions(" not in prompt


def test_enriched_candidate_apis_strip_access_token_param():
    """The proxy auto-injects access_token; exposing it to the code agent only
    invites hallucinated `cognizant_access_token = "..."` patterns that
    401-cascade. Both planner and executor prompts must never contain it.
    """
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        _enrich_candidate_apis,
    )

    enriched = _enrich_candidate_apis([{"app": "venmo", "name": "show_transactions"}])
    assert enriched, "venmo.show_transactions spec should resolve"
    params = enriched[0].get("parameters")
    assert isinstance(params, list)
    param_names = [p.get("name") for p in params if isinstance(p, dict)]
    assert "access_token" not in param_names
    # other real params should survive
    assert any(name and name != "access_token" for name in param_names)


def test_code_plan_prompt_never_mentions_access_token():
    subagent_input = _sample_execute_input()
    prompt = _build_code_plan_prompt(subagent_input)
    assert "access_token" not in prompt


def test_code_execute_prompt_never_mentions_access_token():
    subagent_input = _sample_execute_input()
    plan = CodePlanOutput(
        plan_steps=["Call the allowed API."],
        construct_step="Construct result_dict.",
        print_step="Print the final JSON.",
        output_variable=VariableSpec(
            name="result",
            description="Result value holding the API response payload.",
        ),
    )
    prompt = _build_code_execute_prompt(subagent_input, plan)
    assert "access_token" not in prompt


def test_parse_code_plan_validates_schema():
    plan, parsed_json, parse_error = _parse_code_plan(
        '{"plan_steps":["Inspect requests."],'
        '"construct_step":"Build result_dict.",'
        '"print_step":"Print json.dumps(result_dict).",'
        '"output_variable":{"name":"payment_request",'
        '"description":"Payment request entity awaiting approval decision."}}'
    )

    assert parse_error is None
    assert plan is not None
    assert parsed_json == plan.model_dump(mode="json")

    # Empty plan_steps fails schema.
    plan, parsed_json, parse_error = _parse_code_plan('{"plan_steps":[]}')

    assert plan is None
    assert parsed_json is None
    assert parse_error

    # Missing construct_step / print_step also fails schema — this is the
    # whole point of the field split: a plan that forgets the trailing
    # steps must surface as a parse_error so the repair loop retries,
    # not slip through with a silently incomplete plan.
    plan, parsed_json, parse_error = _parse_code_plan(
        '{"plan_steps":["Inspect requests."],'
        '"output_variable":{"name":"payment_request","description":"r"}}'
    )

    assert plan is None
    assert parse_error


# ── _parse_execute_stdout ─────────────────────────────────────────────────────


def test_parse_execute_stdout_success():
    raw = 'debug line\n{"value": 42, "answer": "42", "summary": "done"}'
    stdout_json, finalize_args, parse_error = _parse_execute_stdout(raw)
    assert stdout_json == {"value": 42, "answer": "42", "summary": "done"}
    assert finalize_args is None
    assert parse_error is None


def test_parse_execute_stdout_last_json_wins():
    raw = '{"value": 1, "answer": "1", "summary": "first"}\n{"value": 2, "answer": "2", "summary": "second"}'
    stdout_json, _, _ = _parse_execute_stdout(raw)
    assert stdout_json is not None
    assert stdout_json["value"] == 2


def test_parse_execute_stdout_ignores_non_value_json():
    raw = '{"some": "data"}\n{"value": 99, "answer": "99", "summary": "ok"}'
    stdout_json, finalize_args, parse_error = _parse_execute_stdout(raw)
    assert stdout_json is not None
    assert stdout_json["value"] == 99


def test_parse_execute_stdout_error_key():
    raw = '{"error": "something went wrong"}'
    stdout_json, finalize_args, parse_error = _parse_execute_stdout(raw)
    assert stdout_json is None
    assert finalize_args is None
    assert parse_error is not None
    assert "error" in parse_error


def test_parse_execute_stdout_finalize_marker_fallback():
    finalize_payload = json.dumps(
        {
            "answer": "null",
            "milestone_done": True,
            "summary": "done via finalize",
            "variables_json": "[]",
        }
    )
    raw = f"some print output\n{EXECUTOR_SANDBOX_FINALIZE_MARKER}{finalize_payload}"
    stdout_json, finalize_args, parse_error = _parse_execute_stdout(raw)
    assert stdout_json is None
    assert finalize_args is not None
    assert finalize_args["answer"] == "null"
    assert parse_error is None


def test_parse_execute_stdout_failure():
    stdout_json, finalize_args, parse_error = _parse_execute_stdout("no json here")
    assert stdout_json is None
    assert finalize_args is None
    assert parse_error is not None


def test_parse_execute_stdout_empty():
    stdout_json, finalize_args, parse_error = _parse_execute_stdout("")
    assert stdout_json is None
    assert finalize_args is None
    assert parse_error is not None


def test_build_executor_result_preserves_zero_answer_via_value_fallback():
    """When the executor's stdout omits the `answer` key, the routing path
    falls back to value-type inference. A scalar `value: 0` must produce
    candidate.answer="0", task_type_hint="query", answer_type="int" — the
    "preserves zero" contract that the original test pinned down. The new
    4-field rule prefers an explicit `answer` key when present (see the
    answer_key_drives_submission test), but missing-key resilience is
    still required."""
    plan = CodePlanOutput(
        plan_steps=["Count matching records."],
        construct_step="Construct result_dict.",
        print_step="Print the final JSON.",
        output_variable=VariableSpec(
            name="matching_count",
            description="Integer count of records matching the filter.",
        ),
    )
    exec_run = _CodeExecuteRun(
        stdout_json={"value": 0, "summary": "No matches."},
        finalize_args=None,
        raw_stdout='{"value": 0, "summary": "No matches."}',
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert finalize_called is True
    assert result.answer == "0"
    assert candidate.answer == "0"
    assert candidate.task_type_hint == "query"
    assert candidate.answer_type == "int"


def test_build_executor_result_answer_key_drives_submission():
    """4-field contract: when the executor emits an explicit `answer` key,
    that value (not the type-inferred shape of `value`) drives both
    SubmissionCandidate.answer and ExecutorResult.answer. This is the
    fix for 325d6ec_2 — under the axis C 2-field contract, action tasks
    whose value happened to be a dict got type-inferred into queries and
    the extractor returned the wrong final answer."""
    plan = CodePlanOutput(
        plan_steps=["Count matching records."],
        construct_step="Construct result_dict.",
        print_step="Print the final JSON.",
        output_variable=VariableSpec(
            name="matching_count",
            description="Integer count of records matching the filter.",
        ),
    )
    exec_run = _CodeExecuteRun(
        stdout_json={"value": 5, "answer": "I found five items", "summary": "Found 5."},
        finalize_args=None,
        raw_stdout='{"value": 5, "answer": "I found five items", "summary": "Found 5."}',
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert finalize_called is True
    assert candidate.task_type_hint == "query"
    # Explicit answer string flows through unchanged.
    assert result.answer == "I found five items"
    assert candidate.answer == "I found five items"


def test_build_executor_result_answer_null_routes_to_action():
    """Action / intermediate milestones write the four-character string
    literal "null" into the answer field. The submission layer must route
    that to task_type_hint="action" and submission_answer="null" — the
    extractor then propagates the null without type-inferring a query
    answer from value."""
    plan = CodePlanOutput(
        plan_steps=["Send text"],
        construct_step="Construct result_dict.",
        print_step="Print result",
        output_variable=VariableSpec(
            name="send_result",
            description="Status payload from the send_text_message call.",
        ),
    )
    exec_run = _CodeExecuteRun(
        stdout_json={
            "value": {"status": "success", "message_id": 16793},
            "summary": "Sent successfully.",
            "description": "Status dict with status and message_id fields.",
            "answer": "null",
        },
        finalize_args=None,
        raw_stdout="",
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert finalize_called is True
    assert candidate.task_type_hint == "action"
    assert candidate.answer == "null"
    assert candidate.answer_type == "null"
    assert result.answer == "null"


def test_build_executor_result_missing_answer_key_falls_back_to_type_infer():
    """When the executor's stdout has no `answer` key at all (legacy /
    non-conformant output), the routing falls back to type-inferring from
    value. value=[1,2,3] -> answer="[1, 2, 3]", task_type_hint="query",
    answer_type="str". The fallback shouldn't disappear silently — it's
    a safety net for outputs that drift from the 4-field contract."""
    plan = CodePlanOutput(
        plan_steps=["Collect items."],
        construct_step="Construct result_dict.",
        print_step="Print the final JSON.",
        output_variable=VariableSpec(
            name="items",
            description="List of items collected for the milestone.",
        ),
    )
    exec_run = _CodeExecuteRun(
        stdout_json={"value": [1, 2, 3], "summary": "Got three."},
        finalize_args=None,
        raw_stdout='{"value": [1, 2, 3], "summary": "Got three."}',
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert finalize_called is True
    assert candidate.task_type_hint == "query"
    assert candidate.answer_type == "str"
    assert result.answer == "[1, 2, 3]"


def test_build_executor_result_memory_variable_description_from_stdout():
    """Dual-channel description: when the executor emits a non-empty
    `description` in stdout, that runtime description wins over the
    planner's schema description on MemoryVariable. The runtime LLM has
    seen the actual value shape, so its description is more accurate."""
    plan = CodePlanOutput(
        plan_steps=["Read items."],
        construct_step="Construct result_dict.",
        print_step="Print.",
        output_variable=VariableSpec(
            name="items",
            description="Schema description from planner — generic baseline.",
        ),
    )
    exec_run = _CodeExecuteRun(
        stdout_json={
            "value": [{"id": 1, "title": "x"}],
            "summary": "Read.",
            "description": "Specific runtime description naming id and title fields.",
            "answer": "null",
        },
        finalize_args=None,
        raw_stdout="",
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, _candidate, _f = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert len(result.variables) == 1
    assert result.variables[0].description == (
        "Specific runtime description naming id and title fields."
    )


def test_build_executor_result_memory_variable_description_falls_back_to_plan():
    """Dual-channel description: when the executor's runtime `description`
    is missing or empty, MemoryVariable falls back to the planner's
    schema description (which has min_length=30 enforced by commit 1)."""
    plan = CodePlanOutput(
        plan_steps=["Read items."],
        construct_step="Construct result_dict.",
        print_step="Print.",
        output_variable=VariableSpec(
            name="items",
            description="Planner schema description, used as a fallback.",
        ),
    )
    exec_run = _CodeExecuteRun(
        # No description key in stdout
        stdout_json={"value": [1, 2], "summary": "Read.", "answer": "null"},
        finalize_args=None,
        raw_stdout="",
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    result, _c, _f = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )

    assert result.variables[0].description == (
        "Planner schema description, used as a fallback."
    )

    # Whitespace-only runtime description also falls back.
    exec_run.stdout_json["description"] = "   "
    result2, _c2, _f2 = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )
    assert result2.variables[0].description == (
        "Planner schema description, used as a fallback."
    )


def _make_exec_run(stdout_json: dict) -> _CodeExecuteRun:
    return _CodeExecuteRun(
        stdout_json=stdout_json,
        finalize_args=None,
        raw_stdout=json.dumps(stdout_json),
        code="",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )


def test_build_executor_result_respects_explicit_milestone_done_false():
    plan = CodePlanOutput(
        plan_steps=["Try"],
        construct_step="Construct result_dict.",
        print_step="Print",
        output_variable=VariableSpec(
            name="status",
            description="Status dictionary indicating attempt outcome.",
        ),
    )
    exec_run = _make_exec_run(
        {
            "value": {"ok": True},
            "answer": "did the thing",
            "summary": "completed",
            "milestone_done": False,
        }
    )

    result, _c, _f = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )
    assert result.milestone_done is False


def test_build_executor_result_explicit_milestone_done_true_overrides_keyword():
    # Edge case: a legitimate "no rows found" summary that contains a keyword.
    # The explicit flag wins over the heuristic — opt-out is always available.
    plan = CodePlanOutput(
        plan_steps=["Search"],
        construct_step="Construct result_dict.",
        print_step="Print",
        output_variable=VariableSpec(
            name="results",
            description="Search results list (may be empty when nothing matches).",
        ),
    )
    exec_run = _make_exec_run(
        {
            "value": {"results": []},
            "answer": "null",
            "summary": "Could not retrieve any matching rows; the table was empty.",
            "milestone_done": True,
        }
    )

    result, _c, _f = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )
    assert result.milestone_done is True


def test_build_executor_result_clean_success_stays_done():
    plan = CodePlanOutput(
        plan_steps=["Send"],
        construct_step="Construct result_dict.",
        print_step="Print",
        output_variable=VariableSpec(
            name="status",
            description="Status payload from the successful send_text_message call.",
        ),
    )
    exec_run = _make_exec_run(
        {
            "value": {"status": "success", "text_message_id": 16793},
            "answer": "null",
            "summary": "Text message sent successfully to partner.",
        }
    )

    result, _c, _f = _build_executor_result(
        exec_run=exec_run,
        plan=plan,
        milestone_id="m1",
    )
    assert result.milestone_done is True


def test_code_execute_repair_prompt_includes_previous_code_and_diagnostics():
    failed_run = _CodeExecuteRun(
        stdout_json=None,
        finalize_args=None,
        raw_stdout="traceback text",
        code="print('bad')",
        model_input_raw={},
        raw_text="",
        parse_error="generated code failed Python compile",
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[{"name": "execute_python", "args": {"code": "print('bad')"}}],
        event_diagnostics=[{"role": "model"}],
        usage={},
    )

    prompt = _build_code_execute_repair_prompt(
        base_prompt="base instructions",
        failed_run=failed_run,
        next_attempt=2,
        max_attempts=4,
    )

    assert "base instructions" in prompt
    assert "Repair required" in prompt
    assert "attempt 2 of 4" in prompt
    assert "Rewrite the complete Python program" in prompt
    assert "print('bad')" in prompt
    assert "generated code failed Python compile" in prompt
    assert "traceback text" in prompt


def test_code_execute_repair_only_runs_after_code_exists():
    with_code = _CodeExecuteRun(
        stdout_json=None,
        finalize_args=None,
        raw_stdout="",
        code="print('bad')",
        model_input_raw={},
        raw_text="",
        parse_error="bad code",
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )
    no_code = with_code.__class__(
        stdout_json=None,
        finalize_args=None,
        raw_stdout="",
        code=None,
        model_input_raw={},
        raw_text="",
        parse_error="provider timeout",
        llm_raised="provider timeout",
        tool_call_count=0,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
    )

    assert _should_repair_code_execute_run(with_code) is True
    assert _should_repair_code_execute_run(no_code) is False


def test_no_execute_python_call_error_is_explicit():
    error = _no_execute_python_call_error("", [{"parts": []}])

    assert "execute_python tool call was not observed" in error
    assert "event_diagnostics" in error


def test_extract_tool_calls_records_name_and_args():
    event = type(
        "Event",
        (),
        {
            "content": type(
                "Content",
                (),
                {
                    "parts": [
                        type(
                            "Part",
                            (),
                            {
                                "function_call": type(
                                    "FunctionCall",
                                    (),
                                    {
                                        "name": "execute_python",
                                        "args": {"code": "print('hello')"},
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()
        },
    )()

    assert _extract_tool_calls(event) == [
        {"name": "execute_python", "args": {"code": "print('hello')"}}
    ]


# ── Z: submit_final tool path ─────────────────────────────────────────────────


def _z_plan() -> CodePlanOutput:
    return CodePlanOutput(
        plan_steps=["Do work."],
        construct_step="Construct result.",
        print_step="Submit via submit_final.",
        output_variable=VariableSpec(
            name="result_var",
            description="Generic result holding the milestone output value.",
        ),
    )


def test_build_executor_result_path_z_submit_final_drives_commit():
    """Z: when submit_final_args is present and stdout_json is None
    (executor's Python code didn't print json.dumps), the orchestrator
    must commit the milestone using the submit_final tool args. This is
    the bypass that retires the cognizant_* wrap-in-string failure mode
    (fix_batch_a_full 2026-05-31)."""
    exec_run = _CodeExecuteRun(
        stdout_json=None,
        finalize_args=None,
        raw_stdout="",
        code="result = apis.spotify.show_current_song()",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[
            {"name": "execute_python", "args": {"code": "..."}},
            {
                "name": "submit_final",
                "args": {
                    "value": {"track_id": 42, "title": "Reverie"},
                    "summary": "Got current song.",
                    "description": "Dict with track_id and title.",
                    "answer": "Reverie",
                },
            },
        ],
        event_diagnostics=[],
        usage={},
        submit_final_args={
            "value": {"track_id": 42, "title": "Reverie"},
            "summary": "Got current song.",
            "description": "Dict with track_id and title.",
            "answer": "Reverie",
        },
    )
    result, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run, plan=_z_plan(), milestone_id="m0"
    )
    assert finalize_called is True
    assert result.answer == "Reverie"
    assert candidate.answer == "Reverie"
    assert candidate.task_type_hint == "query"
    # value committed to MemoryVariable should round-trip the dict via JSON
    committed = json.loads(result.variables[0].value_json)
    assert committed == {"track_id": 42, "title": "Reverie"}


def test_build_executor_result_path_z_null_answer_routes_to_action():
    """Z + F4 contract: submit_final(answer="null", value=None) on an
    action / mutation milestone must route to action submission, not
    type-infer a query answer from value=None."""
    exec_run = _CodeExecuteRun(
        stdout_json=None,
        finalize_args=None,
        raw_stdout="",
        code="apis.spotify.next_song()",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
        submit_final_args={
            "value": None,
            "summary": "Advanced to next track.",
            "description": "",
            "answer": "null",
        },
    )
    _, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run, plan=_z_plan(), milestone_id="m0"
    )
    assert finalize_called is True
    assert candidate.answer == "null"
    assert candidate.task_type_hint == "action"
    assert candidate.answer_type == "null"


def test_build_executor_result_stdout_json_still_works_when_submit_final_absent():
    """Backward compat: if Gemini ignores Z and emits the legacy
    print(json.dumps({...})) shape with no submit_final call, Path A
    (stdout_json) still works. This guards the migration window —
    older cache entries / regression-test fixtures keep passing."""
    exec_run = _CodeExecuteRun(
        stdout_json={
            "value": 7,
            "summary": "Counted seven.",
            "description": "Integer count.",
            "answer": "7",
        },
        finalize_args=None,
        raw_stdout='{"value": 7, "summary": "Counted seven.", "answer": "7"}',
        code="...",
        model_input_raw={},
        raw_text="",
        parse_error=None,
        llm_raised=None,
        tool_call_count=1,
        tool_calls=[],
        event_diagnostics=[],
        usage={},
        # submit_final_args left at default (None)
    )
    _, candidate, finalize_called = _build_executor_result(
        exec_run=exec_run, plan=_z_plan(), milestone_id="m0"
    )
    assert finalize_called is True
    assert candidate.answer == "7"


def test_executor_prompt_demands_two_tool_calls_in_order():
    """The Z contract: prompt must explicitly require execute_python
    then submit_final, both EXACTLY ONCE. Single-call habits drift the
    LlmAgent back into either the print(json.dumps()) pattern or
    forgetting to commit the structured result."""
    prompt = CODE_EXECUTOR_SYSTEM_PROMPT
    assert "TWO tool calls" in prompt or "two tool calls" in prompt.lower()
    assert "execute_python" in prompt
    assert "submit_final" in prompt
    # Both must be required, not just "available"
    assert "EXACTLY ONCE" in prompt
    # Don't let the prompt accidentally re-instate the print(json.dumps())
    # contract — that's the wrap-in-string trigger.
    assert "Your code MUST end with a print statement" not in prompt


# ── Deterministic end-to-end ──────────────────────────────────────────────────


def test_code_plan_execute_deterministic_test_mode_emits_executor_contract():
    subagent = build_code_plan_execute_subagent(
        use_llm_code_plan=False,
        use_llm_code_execute=False,
    )
    subagent_input = _sample_execute_input()

    async def _run_once():
        items = []
        async for envelope in subagent.run_subagent(subagent_input, None):
            items.append(envelope)
        return items

    envelopes = asyncio.run(_run_once())

    assert len(envelopes) == 1
    envelope = envelopes[0]
    assert envelope.status == SubagentStatus.SUCCEEDED
    assert envelope.phase == Phase.EXECUTE
    assert envelope.subagent_name == "executor_subagent_code_plan_execute"
    assert envelope.payload["finalize_called"] is True
    assert envelope.payload["milestone_done"] is True
    assert envelope.payload["tool_call_count"] == 0
    assert envelope.payload["submission_candidate"]["source"] == "executor"
    assert envelope.payload["code_plan"]["output_variable"]["name"] == "m1_result"
    variables = envelope.payload["executor_result"]["variables"]
    assert variables[0]["name"] == "m1_result"
    assert variables[0]["source_milestone_id"] == "m1"
    assert json.loads(variables[0]["value_json"])  # value_json is valid JSON
    # io + metrics now live on envelope.io / envelope.metrics, not payload.
    assert envelope.io["input"]["model_calls"][0]["step"] == "code_plan"
    assert envelope.io["input"]["model_calls"][1]["step"] == "code_execute"
    assert envelope.metrics["llm_calls"] == 0

    envelope.io["output"]["model_calls"][1]["code"] = (
        "import json\n"
        'print(json.dumps({"value": {"ok": True}, "answer": "null", "summary": "done"}))'
    )
    envelope.io["output"]["model_calls"][1]["raw_stdout"] = (
        '{"value": {"ok": true}, "answer": "null", "summary": "done"}'
    )
    envelope.io["output"]["model_calls"][1]["tool_calls"] = [
        {"name": "execute_python", "args": {"code": "print('hello')"}}
    ]
    envelope.io["output"]["model_calls"][1]["event_diagnostics"] = [
        {"role": "model", "parts": [{"function_call": {"name": "execute_python"}}]}
    ]
    record = io_record_from_subagent_output(
        Phase.EXECUTE.value, envelope.model_dump(mode="json")
    )
    rendered = "\n".join(render_subagent_io_markdown(record))
    assert "[1] code_plan" in rendered
    assert "[2] code_execute" in rendered
    assert "Tool calls:" in rendered
    assert '"name": "execute_python"' in rendered
    assert '"code": "print' in rendered
    assert "execute_python code:" in rendered
    assert 'print(json.dumps({"value": {"ok": True}' in rendered
    assert "execute_python stdout:" in rendered
    assert '"summary": "done"' in rendered
    assert "Event diagnostics:" in rendered
    assert '"function_call"' in rendered
    assert "executor_subagent_code_plan_execute" in rendered


# ── Builder tests ─────────────────────────────────────────────────────────────


def test_code_plan_execute_builder_defaults_to_real_code_planner_and_executor():
    subagent = build_code_plan_execute_subagent()

    assert subagent.use_llm_code_plan is True
    assert subagent.use_llm_code_execute is True
    assert subagent.inner_agent is not None
    assert subagent.inner_executor_agent is not None
    # ANY mode forces model to call execute_python; break-on-result prevents infinite loops
    config = subagent.inner_executor_agent.generate_content_config
    assert config is not None
    assert config.tool_config is not None


def test_build_code_plan_execute_subagent_skeleton_mode():
    subagent = build_code_plan_execute_subagent(
        use_llm_code_plan=False,
        use_llm_code_execute=False,
    )
    assert subagent.inner_agent is None
    assert subagent.inner_executor_agent is None
    assert subagent.use_llm_code_plan is False
    assert subagent.use_llm_code_execute is False


def test_registry_offers_code_plan_execute_impl():
    assert "code_plan_execute" in available_subagent_impls(Phase.EXECUTE)

    subagent = build_subagent(Phase.EXECUTE, "code_plan_execute")

    assert subagent.name == "executor_subagent_code_plan_execute"
    assert subagent.phase == Phase.EXECUTE
    assert getattr(subagent, "use_llm_code_plan") is True
    assert getattr(subagent, "use_llm_code_execute") is True


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sample_execute_input() -> SubagentInput:
    return SubagentInput(
        phase=Phase.EXECUTE,
        attempt=1,
        task_context=TaskContext(
            task_id="task_1",
            instruction="Do the task.",
            task_datetime="2023-05-18T12:00:00",
        ),
        metadata={
            "milestone_id": "m1",
            "milestone_intent": "Find useful data.",
            "milestone_index": 0,
            "milestone_total": 2,
            "candidate_apis": [{"app": "venmo", "name": "show_transactions"}],
            "prior_variables_preview": "prior value",
        },
    )


def test_llm_stall_timeout_follows_unified_knob(monkeypatch):
    """The executor stall timer reads the unified retry.llm_call_timeout_s, so
    ONE config knob governs both the finder deadline and the executor stall."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RunConfig
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        EXECUTOR_LLM_STALL_TIMEOUT_ENV,
        _llm_stall_timeout_seconds,
    )

    monkeypatch.delenv(EXECUTOR_LLM_STALL_TIMEOUT_ENV, raising=False)
    c = RunConfig()
    set_active_config(c)
    assert _llm_stall_timeout_seconds() == c.retry.llm_call_timeout_s == 150.0

    # changing the single unified knob moves the executor stall timer too
    set_active_config(RunConfig(retry={"llm_call_timeout_s": 90.0}))
    assert _llm_stall_timeout_seconds() == 90.0


@pytest.fixture(autouse=True)
def _reset_active_config_cpe():
    """Stall timeout now comes from the active RunConfig; restore defaults
    after any test that installs one."""
    yield
    from adk_appworld_agent.orchestration.active_config import set_active_config

    set_active_config(None)


def test_llm_stall_timeout_nonpositive_falls_back():
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RunConfig
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        _llm_stall_timeout_seconds,
    )

    # 0 / negative fall back to the default — never bound the stream at <=0.
    for bad in (0.0, -1.0):
        set_active_config(RunConfig(retry={"llm_call_timeout_s": bad}))
        assert _llm_stall_timeout_seconds() == 60.0


# ── Stall guard behavior ──────────────────────────────────────────────────────


class _StalledAsyncGen:
    """Async generator that never yields — used to simulate Gemini stream stall."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(3600)
        raise StopAsyncIteration  # unreachable

    async def aclose(self):
        return None


class _RunnerWithStream:
    def __init__(self, stream):
        self._stream = stream

    def run_async(self, **_):
        return self._stream


class _FakeSessionService:
    async def create_session(self, **_):
        return None


def test_executor_stream_stall_sets_stalled_marker(monkeypatch):
    """If the LLM stream produces no event within the stall timeout, the run
    must come back marked stalled with llm_raised populated — so the emit
    layer can route it to EXECUTOR_LLM_STALLED instead of EXECUTOR_LLM_RAISED."""
    from adk_appworld_agent.orchestration.active_config import set_active_config
    from adk_appworld_agent.orchestration.run_config import RunConfig
    from adk_appworld_agent.subagents.executor import code_plan_execute as cpe

    set_active_config(RunConfig(retry={"llm_call_timeout_s": 0.05}))

    runner = _RunnerWithStream(_StalledAsyncGen())
    session_service = _FakeSessionService()

    run = asyncio.run(
        cpe._run_code_executor_attempt(
            runner=runner,
            session_service=session_service,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            prompt="p",
            model_input_raw={"model": "test"},
            attempt_index=0,
        )
    )

    assert run.stalled is True
    assert run.llm_raised is not None
    assert "stalled" in run.llm_raised.lower()
    assert "0.05s" in run.llm_raised
    assert run.code is None
    assert run.tool_call_count == 0


class _RaisingAsyncGen:
    """Async generator that raises an exception immediately."""

    def __init__(self, exc):
        self._exc = exc
        self._called = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._called:
            raise StopAsyncIteration
        self._called = True
        raise self._exc

    async def aclose(self):
        return None


def test_executor_stream_exception_is_not_stall(monkeypatch):
    """A non-timeout exception (e.g. 429) must NOT set stalled=True — it
    should still route to EXECUTOR_LLM_RAISED via existing path."""
    from adk_appworld_agent.subagents.executor import code_plan_execute as cpe

    monkeypatch.setenv(cpe.EXECUTOR_LLM_STALL_TIMEOUT_ENV, "5")

    runner = _RunnerWithStream(_RaisingAsyncGen(RuntimeError("429 RESOURCE_EXHAUSTED")))
    session_service = _FakeSessionService()

    run = asyncio.run(
        cpe._run_code_executor_attempt(
            runner=runner,
            session_service=session_service,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            prompt="p",
            model_input_raw={"model": "test"},
            attempt_index=0,
        )
    )

    assert run.stalled is False
    assert run.llm_raised is not None
    assert "429" in run.llm_raised


# ── Z loop-guard: empty-code execute_python defense ───────────────────────────


class _ToolCallEvent:
    """Event stub carrying a function_call part — minimal shape needed by
    _extract_tool_calls."""

    def __init__(self, name: str, args: dict):
        self.content = type(
            "Content",
            (),
            {
                "parts": [
                    type(
                        "Part",
                        (),
                        {
                            "function_call": type(
                                "FunctionCall",
                                (),
                                {
                                    "name": name,
                                    "args": args,
                                },
                            )()
                        },
                    )()
                ]
            },
        )()
        self.actions = None
        self.usage_metadata = None


class _ScriptedEventStream:
    """Emit a fixed sequence of events then end via StopAsyncIteration.
    Used to script execute_python tool-call patterns into the executor loop."""

    def __init__(self, events: list):
        self._events = events
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._events):
            raise StopAsyncIteration
        ev = self._events[self._i]
        self._i += 1
        return ev

    async def aclose(self):
        return None


def test_executor_stream_breaks_on_consecutive_empty_execute_python_calls():
    """Z loop-guard regression: when Gemini repeatedly calls
    `execute_python(code="")` it never advances the milestone and the
    per-event stall timer never fires (each call resets it). f323bae_2
    (fix_batch_a_z_full 2026-05-31) showed 173 consecutive empty calls
    before the 40-min wall timeout killed the task. The executor stream
    must break on the 2nd consecutive empty call and surface stream_error
    so the repair loop can take over (or the task fails fast)."""
    from adk_appworld_agent.subagents.executor import code_plan_execute as cpe

    # 3 consecutive empty execute_python calls — guard should fire on the 2nd
    events = [
        _ToolCallEvent("execute_python", {"code": ""}),
        _ToolCallEvent("execute_python", {"code": ""}),
        _ToolCallEvent("execute_python", {"code": ""}),
    ]
    runner = _RunnerWithStream(_ScriptedEventStream(events))
    session_service = _FakeSessionService()

    run = asyncio.run(
        cpe._run_code_executor_attempt(
            runner=runner,
            session_service=session_service,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            prompt="p",
            model_input_raw={"model": "test"},
            attempt_index=0,
        )
    )

    # Guard fires on 2nd empty call — we should NOT have processed all 3
    assert run.stalled is False
    assert run.parse_error is not None
    assert "empty-code loop" in run.parse_error
    # tool_call_count records the empty calls observed before break
    assert run.tool_call_count >= 2
    # code never got populated — Gemini never emitted real code
    assert run.code is None


def test_executor_stream_real_call_then_empty_does_not_falsely_trigger_loop_guard():
    """Real-code call followed by a SINGLE empty call must NOT trigger the
    guard — Gemini sometimes emits a stray empty tool call after a real one
    and we shouldn't kill the run for that. Guard requires CONSECUTIVE
    empties (≥2) to fire."""
    from adk_appworld_agent.subagents.executor import code_plan_execute as cpe

    events = [
        _ToolCallEvent("execute_python", {"code": "print('ok')"}),
        _ToolCallEvent("execute_python", {"code": ""}),  # single stray empty
        _ToolCallEvent(
            "submit_final",
            {
                "value": 1,
                "summary": "done",
                "description": "test",
                "answer": "1",
            },
        ),
    ]
    runner = _RunnerWithStream(_ScriptedEventStream(events))
    session_service = _FakeSessionService()

    run = asyncio.run(
        cpe._run_code_executor_attempt(
            runner=runner,
            session_service=session_service,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            prompt="p",
            model_input_raw={"model": "test"},
            attempt_index=0,
        )
    )

    # Guard NOT triggered — real call broke the streak
    assert run.parse_error != "" if run.parse_error else True
    assert run.parse_error is None or "empty-code loop" not in str(run.parse_error)
    # submit_final captured
    assert run.submit_final_args == {
        "value": 1,
        "summary": "done",
        "description": "test",
        "answer": "1",
    }


def test_executor_stream_breaks_on_excessive_real_execute_python_calls():
    """Defense against Gemini emitting 10+ real `execute_python` calls in
    one attempt (e.g. trying to incrementally probe APIs). After
    _MAX_TOTAL_EXECUTE_CALLS the loop must break to bound token cost."""
    from adk_appworld_agent.subagents.executor import code_plan_execute as cpe

    # 7 real execute_python calls — guard limit is 6
    events = [
        _ToolCallEvent("execute_python", {"code": f"print({i})"}) for i in range(7)
    ]
    runner = _RunnerWithStream(_ScriptedEventStream(events))
    session_service = _FakeSessionService()

    run = asyncio.run(
        cpe._run_code_executor_attempt(
            runner=runner,
            session_service=session_service,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            prompt="p",
            model_input_raw={"model": "test"},
            attempt_index=0,
        )
    )

    assert run.parse_error is not None
    assert "execute_python budget" in run.parse_error
    # Budget guard runs AFTER capturing the code, so `code` should hold the
    # last real call's code (the one that tripped the limit) for debugging.
    assert run.code is not None


# ── Repair-loop transient short-circuit (Commit B) ────────────────────────────


def _make_canned_run(
    *, llm_raised: str | None = None, stalled: bool = False, success: bool = False
) -> "object":
    """Build a fake _CodeExecuteRun for repair_loop tests."""
    from adk_appworld_agent.subagents.executor.code_plan_execute import _CodeExecuteRun

    return _CodeExecuteRun(
        stdout_json={"value": "ok", "answer": "x", "summary": "done"}
        if success
        else None,
        finalize_args=None,
        raw_stdout="",
        code=None,
        model_input_raw={"model": "test"},
        raw_text="",
        parse_error=llm_raised,
        llm_raised=llm_raised,
        tool_call_count=0,
        tool_calls=[],
        event_diagnostics=[],
        usage={
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thoughts_tokens": 0,
        },
        stalled=stalled,
    )


def test_repair_loop_transient_short_circuits_with_backoff(monkeypatch):
    """When an attempt fails transient (429 or stall), repair_loop must:
    - sleep with exponential backoff (16, 32, 64, 128)
    - re-run the SAME attempt_index (no repair prompt)
    - not consume the repair budget
    Then on a successful retry, return.
    """
    from adk_appworld_agent.subagents.executor.code_plan_execute import execution as cpe

    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(cpe.asyncio, "sleep", _fake_sleep)

    # Sequence: 429, stall, success
    runs = [
        _make_canned_run(llm_raised="429 RESOURCE_EXHAUSTED"),
        _make_canned_run(stalled=True, llm_raised="LLM stalled (no event for 60s)"),
        _make_canned_run(success=True),
    ]
    attempt_indices_seen: list[int] = []

    async def _fake_attempt(*, attempt_index: int, **_):
        attempt_indices_seen.append(attempt_index)
        return runs[len(attempt_indices_seen) - 1]

    monkeypatch.setattr(cpe, "_run_code_executor_attempt", _fake_attempt)

    result = asyncio.run(
        cpe._run_code_executor_repair_loop(
            runner=None,
            session_service=None,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            base_prompt="p",
            base_model_input_raw={"model": "test"},
        )
    )

    # All 3 attempts ran at attempt_index=0 (no repair-budget burn)
    assert attempt_indices_seen == [0, 0, 0]
    # 429 → unified rate-limit ladder rung 1 (4s); stall → transient
    # backoff rung 1 (16s, separate ladder); 3rd attempt succeeds, no sleep after.
    # The two budgets are independent counters now.
    assert sleeps == [4, 16]
    # Final result is the success run
    assert result.stdout_json is not None


def test_repair_loop_transient_budget_exhausted_falls_through(monkeypatch):
    """After the 4 transient retries are used up, additional transient
    failures must NOT keep retrying — they fall through to the normal
    failure path (which propagates EXECUTOR_LLM_STALLED to the controller)."""
    from adk_appworld_agent.subagents.executor.code_plan_execute import execution as cpe

    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(cpe.asyncio, "sleep", _fake_sleep)

    # Always stall — 5 calls = initial + 4 backoff retries (then falls through)
    attempt_indices_seen: list[int] = []

    async def _fake_attempt(*, attempt_index: int, **_):
        attempt_indices_seen.append(attempt_index)
        return _make_canned_run(stalled=True, llm_raised="LLM stalled")

    monkeypatch.setattr(cpe, "_run_code_executor_attempt", _fake_attempt)

    result = asyncio.run(
        cpe._run_code_executor_repair_loop(
            runner=None,
            session_service=None,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            base_prompt="p",
            base_model_input_raw={"model": "test"},
        )
    )

    # 1 initial + 4 retries = 5 attempts, all at attempt_index=0
    assert attempt_indices_seen == [0, 0, 0, 0, 0]
    assert sleeps == [16, 32, 64, 128]
    # Final result is the stalled run — to be routed by emit logic
    assert result.stalled is True


def test_repair_loop_non_transient_failure_uses_repair_budget(monkeypatch):
    """A non-transient failure (e.g. SyntaxError where the model produced
    code but it didn't parse) must still go through the existing repair
    flow — building a repair prompt and bumping attempt_index."""
    from adk_appworld_agent.subagents.executor.code_plan_execute import execution as cpe

    sleeps: list[float] = []

    async def _fake_sleep(d: float) -> None:
        sleeps.append(d)

    monkeypatch.setattr(cpe.asyncio, "sleep", _fake_sleep)

    from adk_appworld_agent.subagents.executor.code_plan_execute import _CodeExecuteRun

    def _repairable_run() -> _CodeExecuteRun:
        # code present, no llm_raised → should hit _should_repair_code_execute_run==True
        return _CodeExecuteRun(
            stdout_json=None,
            finalize_args=None,
            raw_stdout="",
            code="x = 1\n",
            model_input_raw={"model": "test"},
            raw_text="",
            parse_error="some parse error",
            llm_raised=None,
            tool_call_count=1,
            tool_calls=[],
            event_diagnostics=[],
            usage={
                "llm_calls": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "thoughts_tokens": 0,
            },
            stalled=False,
        )

    attempt_indices_seen: list[int] = []
    runs = [_repairable_run(), _repairable_run(), _repairable_run(), _repairable_run()]

    async def _fake_attempt(*, attempt_index: int, **_):
        attempt_indices_seen.append(attempt_index)
        return runs[attempt_index]

    monkeypatch.setattr(cpe, "_run_code_executor_attempt", _fake_attempt)

    asyncio.run(
        cpe._run_code_executor_repair_loop(
            runner=None,
            session_service=None,
            app_name="t",
            user_id="u",
            subagent_input=SubagentInput(
                phase=Phase.EXECUTE,
                attempt=1,
                task_context=TaskContext(
                    task_id="t", instruction="i", task_datetime="2024-01-01T00:00:00"
                ),
                metadata={},
            ),
            base_prompt="p",
            base_model_input_raw={"model": "test"},
        )
    )

    # Each repair attempt advances attempt_index
    assert attempt_indices_seen == [0, 1, 2, 3]
    # No transient sleeps
    assert sleeps == []


def test_prior_returns_render_real_data_for_retry():
    """Option A (observe-before-act in the code-writer's own context): on retry,
    the prior-attempts block shows the REAL returns the last attempt observed
    (actual field names / string values), from api_trace.result_items — not a
    'see continuation history' pointer. No re-fetch, no extra LLM pass."""
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        _format_prior_attempts_block,
        _render_prior_returns,
    )

    api_trace = [
        {
            "app": "venmo",
            "api_name": "show_transactions",
            "status": "ok",
            "result_items": {
                "items": [
                    {"description": "Bill for Electricity", "amount": 32},
                    {"description": "For Phone Bill", "amount": 40},
                    {"description": "Bill for Electricity", "amount": 25},
                ]
            },
        },
    ]
    rendered = _render_prior_returns(api_trace)
    assert rendered is not None
    assert "venmo.show_transactions returned" in rendered
    assert "Bill for Electricity" in rendered  # the real value, not the task literal

    block = _format_prior_attempts_block(
        [{"agent_output": {"api_trace": api_trace, "success": False}, "success": False}]
    )
    assert "Observed API returns from your last attempt" in block
    assert "Bill for Electricity" in block
    assert "see continuation history" not in block  # the old lossy pointer is gone


def test_prior_returns_none_for_local_compute():
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        _render_prior_returns,
    )

    assert _render_prior_returns([]) is None
    # errored calls are skipped (no usable return to show)
    assert (
        _render_prior_returns([{"app": "x", "api_name": "y", "status": "error"}])
        is None
    )


def test_self_assess_flag_plumbs_default_on():
    # Default flipped ON 2026-06-22: self_assess is the executor->continuation
    # diagnosis channel the convergence redesign (§3g) honors. Still ablatable
    # by explicitly passing executor_self_assess=False.
    from adk_appworld_agent.orchestration.run_config import RunConfig
    from adk_appworld_agent.subagents.executor.code_plan_execute import (
        build_code_plan_execute_subagent,
    )

    assert build_code_plan_execute_subagent(run_config=RunConfig()).self_assess is True
    assert (
        build_code_plan_execute_subagent(
            run_config=RunConfig(executor_self_assess=False)
        ).self_assess
        is False
    )


def test_self_assess_observed_render_surfaces_real_key():
    """The self-assess observed view is grounded in the REAL api returns
    (full result_items), so it surfaces the actual content key the executor
    must match — e.g. the underscore key it kept missing (0a9d82a)."""
    from adk_appworld_agent.subagents.executor.code_plan_execute.agent import (
        _render_observed_from_records,
    )

    recs = [
        {
            "api_calls": [
                {
                    "app": "simple_note",
                    "api_name": "show_note",
                    "status": "ok",
                    "result_items": {
                        "items": [{"content": "limited_screen_time_to_1_hr: yes"}]
                    },
                },
            ]
        },
    ]
    out = _render_observed_from_records(recs)
    assert out is not None
    assert "simple_note.show_note" in out
    assert "limited_screen_time_to_1_hr" in out  # the real key, grounded
    assert _render_observed_from_records([]) is None
