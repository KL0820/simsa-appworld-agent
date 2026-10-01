"""Triple-quote code fences: gemini sometimes wraps code in a straight
triple-quote fence with a language tag (e.g. '''python … ''') instead of a
backtick fence. Observed live on 042a9fc_2. Because '''/\"\"\" are legal Python
(docstrings), the strip requires a language tag AND accepts the unwrap only if
it compiles — a genuine leading docstring is kept untouched."""

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


def test_recovers_triple_single_quote_complete_fence():
    raw = "'''python\nimport json\nprint(json.dumps({'value': 1}))\n'''"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert out.startswith("import json")
    assert "python" not in out.split("\n")[0]


def test_recovers_triple_double_quote_fence_with_trailing_junk():
    raw = '"""python\nimport json\nx = [1]\nprint(json.dumps({\'value\': x}))\n"""junk'
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert out.startswith("import json")


def test_recovers_triple_quote_fence_without_close():
    raw = "'''python\nimport json\nprint(json.dumps({'value': 2}))"
    out = clean_execute_python_code(raw)
    assert _compiles(out)
    assert out.startswith("import json")


def test_leading_docstring_without_lang_tag_untouched():
    raw = "'''module docstring'''\nimport json\nprint(json.dumps({'value': 3}))"
    out = clean_execute_python_code(raw)
    assert out.strip() == raw.strip()


def test_midline_triple_quote_string_untouched():
    raw = "sql = '''\nSELECT *\n'''\nprint(sql)"
    out = clean_execute_python_code(raw)
    assert out.strip() == raw.strip()


def test_leading_triple_quote_prose_docstring_untouched():
    # '''python then PROSE (not code) then close, followed by real code — a
    # genuine docstring-first program; unwrap yields prose that fails to compile
    # so the original is preserved.
    raw = "'''python\nmodule notes here\n'''\nimport json\nprint(json.dumps({'value': 5}))"
    out = clean_execute_python_code(raw)
    assert out.strip() == raw.strip()
