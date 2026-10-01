"""Curly-quote code fences (gemini emits ‘‘‘ instead of ```).

The fence strip must (a) recover the intact program wrapped in a curly fence,
(b) leave every kind of valid code untouched — it only fires when the code
LITERALLY starts with a >=3 run of fence glyphs, which valid Python never does.
"""

from __future__ import annotations

import ast

from adk_appworld_agent.subagents.executor.appworld_tools import (
    clean_execute_python_code,
)


def _compiles(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_recovers_curly_fenced_program():
    raw = "‘‘‘import json\n\nx = {c['email'] for c in data}\nprint(json.dumps({'value': list(x)}))\n‘‘‘junk1. **Extract**"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "‘" not in out and "junk" not in out
    assert "import json" in out and "c['email']" in out


def test_recovers_curly_fence_with_lang_tag():
    raw = "‘‘‘python\nimport json\nprint(json.dumps({'value': 1}))\n‘‘‘"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert out.startswith("import json")


def test_backtick_fence_still_stripped():
    raw = "```python\nimport json\nprint(json.dumps({'value': 5}))\n```"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "`" not in out and out.startswith("import json")


# --- false-positive guards: valid code must pass through untouched -----------


def test_plain_code_untouched():
    raw = "import json\nresult = [i for i in range(3)]\nprint(json.dumps({'value': result}))"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "result" in out and "range(3)" in out


def test_string_with_backticks_not_truncated():
    raw = 'x = "```"\nprint(x)'
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "print(x)" in out  # trailing content NOT dropped


def test_string_with_curly_quotes_untouched():
    raw = "msg = 'I‘m ‘‘‘ here'\nprint(msg)"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "print(msg)" in out


def test_docstring_untouched():
    raw = '"""module doc"""\nimport json\nprint(1)'
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert "import json" in out and "print(1)" in out


# --- trailing-garbage recovery (extra ')', fence, prose, non-ascii after code) ---

from adk_appworld_agent.subagents.executor.appworld_tools import (
    repair_generated_source_artifacts,
)


def test_recovers_extra_trailing_paren():
    raw = 'import json\nv=[1,2]\nprint(json.dumps({"value": v})))'
    out = repair_generated_source_artifacts(raw)
    assert _compiles(out) and "print(json.dumps" in out


def test_recovers_trailing_fence_and_nonascii():
    raw = 'import json\nx=1\nprint(json.dumps({"value": x}))\n‘‘‘))ोबर1. **Extract**'
    out = repair_generated_source_artifacts(raw)
    assert _compiles(out) and "‘" not in out and "ोबर" not in out


def test_valid_code_not_trimmed_by_recovery():
    raw = 'x = "func())"\nprint(x)'
    out = repair_generated_source_artifacts(raw)
    assert out.strip() == raw.strip()
