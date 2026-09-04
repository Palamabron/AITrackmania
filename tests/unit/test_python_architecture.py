from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PYTHON_ROOTS = (
    REPOSITORY / "trackmaniarl",
    REPOSITORY / "tests",
    REPOSITORY / "scripts",
    REPOSITORY / "docs" / "diagrams",
)
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
FORBIDDEN_COMMENT_MARKERS = (
    "# type:" + " ignore",
    "# no" + "qa",
    "TO" + "DO",
    "FIX" + "ME",
)
BANNER_COMMENT = re.compile(r"^\s*#\s*(?:-{3,}|={3,})")


def _python_files() -> Iterator[Path]:
    for root in PYTHON_ROOTS:
        yield from root.rglob("*.py")


def _function_nodes(path: Path) -> Iterator[FunctionNode]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _caller_parameters(node: FunctionNode) -> list[ast.arg]:
    positional = [*node.args.posonlyargs, *node.args.args]
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    parameters = [*positional, *node.args.kwonlyargs]
    if node.args.vararg is not None:
        parameters.append(node.args.vararg)
    if node.args.kwarg is not None:
        parameters.append(node.args.kwarg)
    return parameters


def _boolean_default_names(node: FunctionNode) -> set[str]:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaults = node.args.defaults
    defaulted = positional[-len(defaults) :] if defaults else []
    paired = zip(defaulted, defaults, strict=True)
    names = {argument.arg for argument, value in paired if _is_boolean_constant(value)}
    names.update(
        argument.arg
        for argument, value in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True)
        if _is_boolean_constant(value)
    )
    return names


def _is_boolean_constant(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and isinstance(value.value, bool)


def _has_boolean_annotation(argument: ast.arg) -> bool:
    return _is_boolean_annotation(argument.annotation)


def _is_boolean_annotation(annotation: ast.expr | None) -> bool:
    if isinstance(annotation, ast.Name):
        return annotation.id == "bool"
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_boolean_annotation(annotation.left) or _is_boolean_annotation(annotation.right)
    return False


def _location(path: Path, node: FunctionNode) -> str:
    return f"{path.relative_to(REPOSITORY)}:{node.lineno}:{node.name}"


def _forbidden_comment_markers(line: str) -> list[str]:
    matches = [marker for marker in FORBIDDEN_COMMENT_MARKERS if marker in line]
    if BANNER_COMMENT.match(line):
        matches.append("banner comment")
    return matches


def test_functions_are_strictly_shorter_than_twenty_lines() -> None:
    violations = {
        _location(path, node): node.end_lineno - node.lineno + 1
        for path in _python_files()
        for node in _function_nodes(path)
        if node.end_lineno is not None and node.end_lineno - node.lineno + 1 >= 20
    }

    assert violations == {}


def test_functions_accept_at_most_three_caller_arguments() -> None:
    violations = {
        _location(path, node): len(_caller_parameters(node))
        for path in _python_files()
        for node in _function_nodes(path)
        if len(_caller_parameters(node)) > 3
    }

    assert violations == {}


def test_functions_do_not_use_direct_boolean_flags() -> None:
    violations = {
        _location(path, node): argument.arg
        for path in _python_files()
        for node in _function_nodes(path)
        for argument in _caller_parameters(node)
        if argument.arg in _boolean_default_names(node) or _has_boolean_annotation(argument)
    }

    assert violations == {}


def test_source_has_no_suppressions_or_placeholder_comments() -> None:
    violations = {
        f"{path.relative_to(REPOSITORY)}:{line_number}": marker
        for path in _python_files()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for marker in _forbidden_comment_markers(line)
    }

    assert violations == {}


def test_exception_handlers_are_explicit_and_nonempty() -> None:
    violations = {
        f"{path.relative_to(REPOSITORY)}:{handler.lineno}": "invalid exception handler"
        for path in _python_files()
        for handler in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(handler, ast.ExceptHandler)
        and (
            handler.type is None
            or all(isinstance(statement, ast.Pass) for statement in handler.body)
        )
    }

    assert violations == {}
