# Executor failure fixtures

Deterministic ground-truth fail cases for the F1-F6 flow fixes. Source =
the post-Fix-1 full56 run at `logs/plan_finder_executor_fixes_full/20260505_020924_311443/`.

Each fixture JSON entry has the same envelope:

```json
{
  "source": { "task_id": "...", "milestone_index": 0, "log_run": "...", "synthetic": false, "note": "..." },
  "data": { ... fix-specific payload ... },
  "expected_after_fix": { ... assertion-ready fields ... },
  "note": "<optional context>"
}
```

`synthetic: true` marks a case I constructed (not from logs) to defend
against regex / heuristic over-fit — adversarial coverage for the regex
boundary, year-like numbers, action-vs-query branches, etc.

## Files

| File | Fix targeted | # entries | Purpose |
|---|---|---:|---|
| `plan_give_up.json` | F1 | 5 | CodePlanOutput with give-up phrases (e.g. "cannot be completed"); F1 must reject these at parse time |
| `wrapper_anti_pattern.json` | F4 + F5 | 2 (1 real + 1 synthetic negative) | The `cognizant_code = """..."""\nprint(<var>)` anti-pattern; F4 must inject diagnostic in repair prompt; F5 must classify it as a wrapper |
| `answer_norm_value_null.json` | F2 mode A | 3 (2 real + 1 synthetic negative) | stdout_json with numeric `value` but `answer="null"` (or None) — F2 should backfill answer from value, EXCEPT for action milestones |
| `answer_norm_verbose_sentence.json` | F2 mode B | 9 (4 real + 5 synthetic adversarial) | stdout_json where `answer` is a long sentence containing the actual numeric answer; F2 should extract the bare number with regex, defending against year-like numbers, multi-numeric sentences, no-numeric answers |

## Source-task map

| task_id | M | fix | found in |
|---|---:|---|---|
| 270f1ff_2 | 1 | F1 | give-up plan |
| 2d9f728_2 | 4 | F1 | give-up plan |
| 6f4b9a5_2 | 2 | F1 | give-up plan ("**BLOCKER**: no music database APIs") |
| 986aa4e_2 | 1 | F1 | give-up plan ("task cannot be completed") |
| 32616b5_2 | 2 | F1 | give-up plan ("expense info missing") |
| 9dabbc9_2 | 2 | F4 + F5 | cognizant_code wrapper code |
| 21abae1_2 | 1 | F2 (mode A) | value=620.0, answer=None |
| bde252e_2 | ? | F2 (mode A) | value=9, answer="null" |
| 7847649_2 | 1 | F2 (mode B) | answer="There are 3 activities..." |
| 166f4ff_2 | 1 | F2 (mode B) | answer="$833.00..." |
| afc4005_2 | 1 | F2 (mode B) | answer="30 minutes" |
| dac78d9_2 | 1 | F2 (mode B) | answer="8 Venmo friends since October 1st, 2022" |

## Loading

```python
from tests.fixtures import load_executor_failure_fixture

cases = load_executor_failure_fixture("plan_give_up")
for case in cases:
    plan_data = case["data"]
    expected = case["expected_after_fix"]
    ...
```

## Maintenance

If a fixture's source task is later reproduced under a different log run
(e.g. after a reproducer regression), do NOT edit the fixture in place —
add a new entry with the new `source.log_run`. The old log path keeps
historical attribution.
