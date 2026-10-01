from __future__ import annotations

import ast
from dataclasses import dataclass


class GeneratedCodeValidationError(ValueError):
    """Raised when Stage 2 generated code violates deterministic contracts."""


@dataclass(frozen=True)
class ApiSpecIndex:
    allowed_apis: dict[tuple[str, str], set[str]]


def validate_generated_code(
    code: str,
    *,
    candidate_apis: list[dict],
    allowed_profile_keys: set[str],
) -> None:
    """Validate generated Stage 2 code before it reaches the AppWorld sandbox."""
    tree = _parse_python(code)
    _validate_not_printing_program_source(tree)
    _validate_api_calls(tree, _build_api_spec_index(candidate_apis))
    _validate_profile_keys(tree, allowed_profile_keys)


def unwrap_printed_program_source(code: str) -> str:
    """Unwrap the common Stage 2 error: assigning code to a string and printing it."""
    tree = _parse_python(code)
    body = getattr(tree, "body", [])
    if len(body) != 2:
        return code
    assignment, print_expr = body
    if not isinstance(assignment, ast.Assign):
        return code
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name):
        return code
    if not isinstance(assignment.value, ast.Constant) or not isinstance(
        assignment.value.value, str
    ):
        return code
    assigned_name = assignment.targets[0].id
    if not _prints_name(print_expr, assigned_name):
        return code
    inner_code = assignment.value.value
    if not _looks_like_python_program(inner_code):
        return code
    return inner_code.strip()


def _parse_python(code: str) -> ast.AST:
    try:
        return ast.parse(code)
    except SyntaxError as exc:
        detail = f"{exc.__class__.__name__}: {exc.msg}"
        if exc.lineno is not None:
            detail += f" at line {exc.lineno}"
        raise GeneratedCodeValidationError(
            f"generated code failed Python compile: {detail}"
        ) from exc


def _build_api_spec_index(candidate_apis: list[dict]) -> ApiSpecIndex:
    allowed_apis: dict[tuple[str, str], set[str]] = {}
    for item in candidate_apis:
        if not isinstance(item, dict):
            continue
        app = item.get("app") or item.get("app_name")
        name = item.get("name") or item.get("api_name")
        if not isinstance(app, str) or not isinstance(name, str):
            continue
        allowed_apis[(app, name)] = _parameter_names(item)
    return ApiSpecIndex(allowed_apis=allowed_apis)


def _parameter_names(api_spec: dict) -> set[str]:
    names: set[str] = set()
    for parameter in api_spec.get("parameters", []):
        if isinstance(parameter, dict):
            name = parameter.get("name")
        else:
            name = parameter
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _validate_api_calls(tree: ast.AST, index: ApiSpecIndex) -> None:
    if not index.allowed_apis:
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        api_ref = _api_call_ref(node)
        if api_ref is None:
            continue
        app_name, api_name = api_ref
        if app_name in {"api_docs", "supervisor"}:
            continue
        if (app_name, api_name) not in index.allowed_apis:
            # Enumerate the valid candidates in the rejection so the executor
            # can self-correct to a real API instead of looping on a
            # hallucinated/wrong name (634f342_2: continuation invented the
            # plural `add_songs_to_playlist`; the real `add_song_to_playlist`
            # was in candidate_apis the whole time). Steering the repair to the
            # actual allow-list is a deterministic control-plane fix — it does
            # NOT touch continuation (cf. the reverted strip-hallucinated-apis
            # approach) and leaks no op names beyond the executor's own list.
            available = ", ".join(
                sorted(f"apis.{a}.{n}" for (a, n) in index.allowed_apis)
            )
            _raise_at(
                node,
                f"generated code calls apis.{app_name}.{api_name}, which is NOT "
                "an available API (not in the finder candidate_apis). Call ONLY "
                f"one of these available APIs: {available}",
            )
        allowed_kwargs = index.allowed_apis[(app_name, api_name)]
        if node.args:
            _raise_at(
                node,
                f"generated code calls apis.{app_name}.{api_name} with positional "
                "arguments; use explicit keyword arguments from the API spec",
            )
        for keyword in node.keywords:
            if keyword.arg is None:
                _raise_at(
                    node,
                    f"generated code calls apis.{app_name}.{api_name} with **kwargs; "
                    "guardrails require explicit API keyword names",
                )
            if keyword.arg not in allowed_kwargs:
                _raise_at(
                    node,
                    f"generated code calls apis.{app_name}.{api_name} with unknown "
                    f"keyword '{keyword.arg}'. Allowed keywords: "
                    f"{', '.join(sorted(allowed_kwargs)) or '(none)'}",
                )


def _validate_not_printing_program_source(tree: ast.AST) -> None:
    source_like_names = _source_like_string_assignments(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_plain_print_call(node):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in source_like_names:
                _raise_at(
                    node,
                    "generated code prints a Python program string instead of "
                    "executing the program directly",
                )


def _source_like_string_assignments(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(
            node.value.value, str
        ):
            continue
        if not _looks_like_python_program(node.value.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _looks_like_python_program(value: str) -> bool:
    if "\n" not in value:
        return False
    source_markers = ("apis.", "prior_variable_values", "task_datetime", "profile[")
    result_markers = ("print(json.dumps", "json.dumps({")
    return any(marker in value for marker in source_markers) and any(
        marker in value for marker in result_markers
    )


def _is_plain_print_call(node: ast.Call) -> bool:
    return isinstance(node.func, ast.Name) and node.func.id == "print"


def _prints_name(node: ast.AST, name: str) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and _is_plain_print_call(node.value)
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == name
    )


def _api_call_ref(node: ast.Call) -> tuple[str, str] | None:
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    api_name = func.attr
    app_node = func.value
    if not isinstance(app_node, ast.Attribute):
        return None
    app_name = app_node.attr
    root = app_node.value
    if not isinstance(root, ast.Name) or root.id != "apis":
        return None
    return app_name, api_name


def _validate_profile_keys(tree: ast.AST, allowed_profile_keys: set[str]) -> None:
    if not allowed_profile_keys:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_profile_name(node.value):
            key = _literal_string_key(node.slice)
            if key is None:
                _raise_at(node, "generated code reads profile[...] with a dynamic key")
            if key not in allowed_profile_keys:
                _raise_at(
                    node,
                    f"generated code reads profile['{key}'], which is not one of "
                    f"the allowed profile keys: {', '.join(sorted(allowed_profile_keys))}",
                )
        if isinstance(node, ast.Call) and _is_profile_get_call(node):
            key = _first_literal_string_arg(node)
            if key is None:
                _raise_at(
                    node, "generated code calls profile.get(...) with a dynamic key"
                )
            if key not in allowed_profile_keys:
                _raise_at(
                    node,
                    f"generated code reads profile.get('{key}'), which is not one of "
                    f"the allowed profile keys: {', '.join(sorted(allowed_profile_keys))}",
                )


def _is_profile_name(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "profile"


def _is_profile_get_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and _is_profile_name(node.func.value)
    )


def _literal_string_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _first_literal_string_arg(node: ast.Call) -> str | None:
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _raise_at(node: ast.AST, message: str) -> None:
    line = getattr(node, "lineno", None)
    if line is not None:
        message = f"{message} at line {line}"
    raise GeneratedCodeValidationError(message)


__all__ = [
    "GeneratedCodeValidationError",
    "unwrap_printed_program_source",
    "validate_generated_code",
]
