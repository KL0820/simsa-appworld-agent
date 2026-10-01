"""Central configuration constants for the agent system.

Single source of truth for all numeric thresholds, caps, regex patterns,
and sensitive field names used across the subagent / orchestration layers.

Other modules MUST import from here instead of defining magic numbers.
Editing this file is how you tune the system.

Sections are grouped by concern; add new constants to the matching section.
"""

from __future__ import annotations

# ── Continuation rationale ─────────────────────────────────────────────
# Hard cap on continuation rationale length. Used as:
#   - schema max_length on ContinuationDecision.rationale (producer side)
#   - reader-side safety truncation on every consumer that renders rationale
# Producer prompt asks for ≤RATIONALE_TARGET_LENGTH target (soft guidance
# via §5b); HARD cap above is a safety buffer for occasional overshoot.
RATIONALE_MAX_LENGTH = 1000
RATIONALE_TARGET_LENGTH = 300

# ── Continuation RETRY-without-revise loop guard ───────────────────────
# After this many consecutive RETRY decisions on the same active milestone
# with revised_milestones=None, framework injects a last_framework_override
# instructing the LLM to either emit revised_milestones, ADVANCE, or ABORT
# on the next round. Set to a number that gives the LLM a couple of free
# attempts to recover but bounds the wasted-cycles class.
RETRY_WITHOUT_REVISE_LIMIT = 3

# ── Per-milestone cycle budget (hard backstop) ─────────────────────────
# Total PLAN cycles a single active milestone may consume before the
# framework forces progress (ADVANCE to the next milestone accepting the
# best-so-far committed value, or SUBMIT if it is the last milestone with a
# successful EXECUTE). Guard 4 above only catches RETRY-WITHOUT-revise
# streaks; a loop that REVISES the milestone intent every cycle slips past
# it (042a9fc_2: 12 revise cycles on one milestone burned the whole wall
# clock, downstream milestones never ran). This caps any one milestone so a
# code-gen-hard subtask can no longer starve the rest of the task. Set
# generously: legitimate multi-try milestones rarely exceed ~3 cycles, so 5
# only bites genuinely stuck ones, while staying well under max_cycles (15).
PER_MILESTONE_CYCLE_LIMIT = 5

# ── Cycle history rendering ────────────────────────────────────────────
# How many most-recent cycles render in full detail in continuation's
# history block. Older cycles condense to a one-liner each.
MAX_HISTORY_CYCLES = 2

# ── Prior variables rendering ──────────────────────────────────────────
# Cap on the markdown summary string produced by variable_store.summary().
# All readers (continuation / code_planner / code_executor) see this same
# truncated string — no per-reader extra cap.
PRIOR_VARIABLES_MAX_LENGTH = 5000

# Each variable's individual value preview cap (used inside _summary_block
# when falling back to raw JSON for shapes the helper doesn't recurse into).
VARIABLE_PREVIEW_MAX_LENGTH = 5000

# ── stdout / failure rendering in cycle history ────────────────────────
# Per-field string cap in `stdout (4-field):` block of EXECUTE-success
# render (scalar / string case in _format_stdout_field).
STDOUT_FIELD_PREVIEW_MAX_LENGTH = 500

# stdout_excerpt cap in EXECUTE-FAILED entry render (continuation sees
# this as the failure evidence — generous so traceback / error message
# isn't cut mid-line).
STDOUT_EXCERPT_MAX_LENGTH = 1500

# code_planner's prior_attempts block — per-attempt fields are
# summarized aggressively because the block lists every retry attempt
# on this milestone, so total length grows with retry count.
PRIOR_ATTEMPT_STDOUT_EXCERPT_MAX_LENGTH = 300
PRIOR_ATTEMPT_PARSE_ERROR_MAX_LENGTH = 200
PRIOR_ATTEMPT_CODE_EXCERPT_MAX_LENGTH = 700
PRIOR_ATTEMPT_SUMMARY_MAX_LENGTH = 200
PRIOR_ATTEMPT_VALUE_PREVIEW_MAX_LENGTH = 400

# Executor code preserved in FAILED EXECUTE diag (used by both the
# producer when building diag dict and the renderer when re-emitting
# the code block to the LLM).
EXECUTOR_CODE_DIAG_MAX_LENGTH = 2000

# parse_error / llm_raised producer-side caps in FAILED EXECUTE diag
# (renderer prints them straight; single source of truth on producer).
PARSE_ERROR_MAX_LENGTH = 500
LLM_RAISED_MAX_LENGTH = 500

# Framework override note appended to PLAN entry render (rare debug
# field, kept short to avoid drowning the rationale).
FRAMEWORK_OVERRIDE_PREVIEW_MAX_LENGTH = 120

# Fallback rough_planner milestone task cap — when rough_planner LLM
# fails and we synthesize a single-task plan from the raw instruction.
# The resulting milestone goes through every downstream subagent so the
# cap is LLM-facing despite being on the fallback path.
FALLBACK_MILESTONE_TASK_MAX_LENGTH = 500

# variable_store derives a "keys" field for dict / list[dict] variables,
# capped at this many to keep the variable summary block bounded.
VARIABLE_KEYS_PREVIEW_MAX_COUNT = 20

# Cap on executor output text passed to final_answer_extractor's prompt
# (separate LLM phase from continuation_planner; concerns the submission
# step, not cycle history).
FINAL_ANSWER_TEXT_MAX_LENGTH = 4000

# ── Data summarization helper (api_trace, prior_variables value) ───────
# Categorical distribution: show top-K + "and N more" sentinel
DISTRIBUTION_TOP_K = 5

# Sample rendering
SAMPLE_COUNT = 3  # rows per item summary
SAMPLE_FIELD_MAX_CHARS = 80  # per-field value truncation in sample lines

# Header display (call list before "and N more" sentinel)
HEADER_DISPLAY_CAP = 3

# Helper-internal preview cap for primitive / dict-field rendering
# (summarize_dict field values, primitive list samples).
HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH = 200

# ── Field classification thresholds ────────────────────────────────────
# Field named *_id AND ≥ this fraction unique → "id" class (skip distribution).
ID_UNIQUE_RATIO_THRESHOLD = 0.9

# int/float field: n_unique > n_items * frac → numeric_range (else categorical).
CATEGORICAL_NUMERIC_MAX_UNIQUE_FRAC = 0.5

# Floor for small-sample numeric: ≤ this unique values → categorical
# regardless of fraction (prevents 3-row data with 2 unique from going range).
CATEGORICAL_NUMERIC_MIN_UNIQUE = 2

# String field with avg length > this → free_form (skip distribution);
# below threshold → categorical even when 100% unique.
FREE_FORM_AVG_LEN_THRESHOLD = 60

# ── Regex / patterns ───────────────────────────────────────────────────
ID_FIELD_PATTERN = r"_id$|^id$"
ISO_DATETIME_PATTERN = r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}:\d{2})?"

# ── Sensitive / hidden fields (kwargs filtering) ───────────────────────
KWARGS_HIDDEN_FIELDS = frozenset({"access_token"})


__all__ = [
    "RATIONALE_MAX_LENGTH",
    "RATIONALE_TARGET_LENGTH",
    "RETRY_WITHOUT_REVISE_LIMIT",
    "PER_MILESTONE_CYCLE_LIMIT",
    "MAX_HISTORY_CYCLES",
    "PRIOR_VARIABLES_MAX_LENGTH",
    "VARIABLE_PREVIEW_MAX_LENGTH",
    "STDOUT_FIELD_PREVIEW_MAX_LENGTH",
    "STDOUT_EXCERPT_MAX_LENGTH",
    "PRIOR_ATTEMPT_STDOUT_EXCERPT_MAX_LENGTH",
    "PRIOR_ATTEMPT_PARSE_ERROR_MAX_LENGTH",
    "PRIOR_ATTEMPT_CODE_EXCERPT_MAX_LENGTH",
    "PRIOR_ATTEMPT_SUMMARY_MAX_LENGTH",
    "PRIOR_ATTEMPT_VALUE_PREVIEW_MAX_LENGTH",
    "EXECUTOR_CODE_DIAG_MAX_LENGTH",
    "PARSE_ERROR_MAX_LENGTH",
    "LLM_RAISED_MAX_LENGTH",
    "FRAMEWORK_OVERRIDE_PREVIEW_MAX_LENGTH",
    "FALLBACK_MILESTONE_TASK_MAX_LENGTH",
    "VARIABLE_KEYS_PREVIEW_MAX_COUNT",
    "FINAL_ANSWER_TEXT_MAX_LENGTH",
    "DISTRIBUTION_TOP_K",
    "SAMPLE_COUNT",
    "SAMPLE_FIELD_MAX_CHARS",
    "HEADER_DISPLAY_CAP",
    "HELPER_PRIMITIVE_PREVIEW_MAX_LENGTH",
    "ID_UNIQUE_RATIO_THRESHOLD",
    "CATEGORICAL_NUMERIC_MAX_UNIQUE_FRAC",
    "CATEGORICAL_NUMERIC_MIN_UNIQUE",
    "FREE_FORM_AVG_LEN_THRESHOLD",
    "ID_FIELD_PATTERN",
    "ISO_DATETIME_PATTERN",
    "KWARGS_HIDDEN_FIELDS",
]
