from __future__ import annotations

import html
import inspect
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from trackmaniarl.core.runtime import import_symbol, resolve_run, validate_resolved_run
from trackmaniarl.core.spec import (
    ActorExecutionSpec,
    ComponentsSpec,
    DistributedSpec,
    EvaluationMapSpec,
    EvaluationSuiteSpec,
    RunSpec,
    TrainingSpec,
)
from trackmaniarl.trackmania.environment_config import TrackmaniaEnvironmentConfig

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "readme" / "examples"
MARKDOWN_FILES = (
    ROOT / "README.md",
    *sorted((ROOT / "readme").glob("*.md")),
    ROOT / "docs" / "diagrams" / "README.md",
)
YAML_EXAMPLES = tuple(sorted((*EXAMPLES.glob("*.yaml"), *EXAMPLES.glob("*.yml"))))
FULL_RUN_EXAMPLES = tuple(path for path in YAML_EXAMPLES if ".fragment." not in path.name)
RUN_SPEC_KEYS = frozenset({"api_version", "run_id", "components"})
INLINE_LINK = re.compile(r"!?\[[^]]*]\(\s*(<[^>]+>|[^)\s]+)")
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^]]+]:\s*(<[^>]+>|\S+)")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
YAML_FENCE = re.compile(r"^```ya?ml\s*\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
RUNTIME_INJECTIONS = frozenset(
    {"base_dir", "model", "pipeline", "request", "run_dir", "run_id", "seed"}
)
RUNTIME_CONFIG_MODELS = (
    RunSpec,
    ComponentsSpec,
    TrainingSpec,
    DistributedSpec,
    ActorExecutionSpec,
    EvaluationSuiteSpec,
    EvaluationMapSpec,
    TrackmaniaEnvironmentConfig,
)
IMPLICIT_SDK_ENV_KEYS = frozenset({"GEMINI_API_KEY"})
CREDENTIAL_ENV_KEY = re.compile(r'["\']([A-Z][A-Z0-9_]*(?:_API_KEY|_TOKEN))["\']')


@dataclass(frozen=True, slots=True)
class _MarkdownDestination:
    value: str
    line: int


def _example_id(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _load_yaml_mapping(path: Path) -> dict[object, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{_example_id(path)} must contain a YAML mapping"
    return payload


def _component_specs(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        class_path = value.get("class_path")
        if isinstance(class_path, str) and class_path.startswith("trackmaniarl."):
            yield value
        for item in value.values():
            yield from _component_specs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _component_specs(item)


def _assert_component_kwargs(spec: Mapping[str, Any], location: str) -> None:
    class_path = str(spec["class_path"])
    parameters = inspect.signature(import_symbol(class_path)).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return
    kwargs = spec.get("kwargs", {})
    assert isinstance(kwargs, Mapping), f"{location}: {class_path} kwargs must be a mapping"
    unknown = set(kwargs) - set(parameters)
    missing = {
        name
        for name, item in parameters.items()
        if item.default is inspect.Parameter.empty
        and item.kind not in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        and name not in RUNTIME_INJECTIONS
        and name not in kwargs
    }
    assert not unknown, f"{location}: {class_path} has unknown kwargs {sorted(unknown)}"
    assert not missing, f"{location}: {class_path} is missing required kwargs {sorted(missing)}"


def _assert_markdown_component_examples() -> None:
    for path in MARKDOWN_FILES:
        markdown = path.read_text(encoding="utf-8")
        for match in YAML_FENCE.finditer(markdown):
            payload = yaml.safe_load(match.group(1))
            line_number = markdown[: match.start()].count("\n") + 2
            location = f"{path.relative_to(ROOT).as_posix()}:{line_number}"
            for spec in _component_specs(payload):
                _assert_component_kwargs(spec, location)


def _documented_table_fields(reference: str) -> set[str]:
    field_cells = "\n".join(
        line.split("|", maxsplit=2)[1]
        for line in reference.splitlines()
        if line.lstrip().startswith("|")
    )
    code_spans = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", field_cells)
    return {name for span in code_spans for name in span.split(".")}


def _assert_runtime_config_fields_are_documented() -> None:
    reference = "\n".join(
        (ROOT / "readme" / name).read_text(encoding="utf-8")
        for name in ("configuration.md", "rewards.md")
    )
    documented = _documented_table_fields(reference)
    missing = {
        model.__name__: sorted(name for name in model.model_fields if name not in documented)
        for model in RUNTIME_CONFIG_MODELS
    }
    assert not {name: fields for name, fields in missing.items() if fields}


def _credential_env_keys() -> set[str]:
    sources = (
        *sorted((ROOT / "trackmaniarl").rglob("*.py")),
        *sorted((ROOT / "scripts").glob("*.py")),
    )
    discovered = {
        key
        for path in sources
        for key in CREDENTIAL_ENV_KEY.findall(path.read_text(encoding="utf-8"))
    }
    return discovered | set(IMPLICIT_SDK_ENV_KEYS)


def _env_example_keys() -> set[str]:
    lines = (ROOT / ".env-example").read_text(encoding="utf-8").splitlines()
    return {
        line.partition("=")[0].strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#") and "=" in line
    }


def _content_lines(markdown: str) -> list[tuple[int, str]]:
    content: list[tuple[int, str]] = []
    fence_marker = ""
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        match = FENCE.match(line)
        if match is not None:
            marker = match.group(1)[0]
            if not fence_marker:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = ""
            continue
        if not fence_marker:
            content.append((line_number, line))
    return content


def _markdown_destinations(markdown: str) -> tuple[_MarkdownDestination, ...]:
    destinations: list[_MarkdownDestination] = []
    for line_number, line in _content_lines(markdown):
        for match in INLINE_LINK.finditer(line):
            destinations.append(_MarkdownDestination(match.group(1).strip("<>"), line_number))
        reference = REFERENCE_LINK.match(line)
        if reference is not None:
            destinations.append(_MarkdownDestination(reference.group(1).strip("<>"), line_number))
    return tuple(destinations)


def _heading_slug(heading: str) -> str:
    text = re.sub(r"!?\[([^]]*)]\([^)]+\)", r"\1", heading)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).replace("`", "")
    text = re.sub(r"\s+#+\s*$", "", text)
    text = "".join(
        character for character in text.lower() if character.isalnum() or character in " _-"
    )
    return re.sub(r"\s", "-", text)


def _markdown_anchors(path: Path) -> frozenset[str]:
    counts: dict[str, int] = {}
    anchors: set[str] = set()
    for _, line in _content_lines(path.read_text(encoding="utf-8")):
        match = HEADING.match(line)
        if match is None:
            continue
        base = _heading_slug(match.group(1))
        duplicate = counts.get(base, 0)
        counts[base] = duplicate + 1
        anchors.add(base if duplicate == 0 else f"{base}-{duplicate}")
    return frozenset(anchors)


def _has_exact_case(path: Path) -> bool:
    cursor = ROOT
    for part in path.relative_to(ROOT).parts:
        if part not in {child.name for child in cursor.iterdir()}:
            return False
        cursor /= part
    return True


def _check_destination(source: Path, destination: _MarkdownDestination) -> str | None:
    parsed = urlsplit(destination.value)
    if parsed.scheme or parsed.netloc:
        return None
    relative_path = unquote(parsed.path)
    target = source if not relative_path else (source.parent / relative_path).resolve()
    label = f"{source.relative_to(ROOT).as_posix()}:{destination.line}"
    if not target.is_relative_to(ROOT):
        return f"{label}: link escapes the repository: {destination.value}"
    if not target.exists() or not _has_exact_case(target):
        return f"{label}: missing or case-mismatched target: {destination.value}"
    if parsed.fragment and target.suffix.lower() == ".md":
        anchor = unquote(parsed.fragment)
        if anchor not in _markdown_anchors(target):
            return f"{label}: missing Markdown anchor #{anchor} in {target.relative_to(ROOT)}"
    return None


def test_yaml_example_classification_is_explicit() -> None:
    assert YAML_EXAMPLES, "readme/examples must contain at least one YAML example"
    for path in YAML_EXAMPLES:
        payload = _load_yaml_mapping(path)
        if ".fragment." not in path.name:
            assert payload.keys() >= RUN_SPEC_KEYS, (
                f"{_example_id(path)} is not a complete RunSpec; "
                "name intentional snippets *.fragment.yaml"
            )
    _assert_markdown_component_examples()
    _assert_runtime_config_fields_are_documented()


def test_env_example_covers_user_managed_credentials() -> None:
    assert _env_example_keys() == _credential_env_keys()


@pytest.mark.parametrize("path", FULL_RUN_EXAMPLES, ids=_example_id)
def test_full_yaml_example_resolves_and_completes_validation(path: Path, tmp_path: Path) -> None:
    spec = RunSpec.from_yaml(path)
    isolated = spec.model_copy(update={"artifacts_dir": tmp_path / "artifacts"})
    run = resolve_run(isolated, base_dir=path.parent)
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()

    assert metrics
    assert all(isinstance(value, float) for value in metrics.values())


def test_repository_relative_markdown_links_and_anchors_resolve() -> None:
    failures = [
        failure
        for source in MARKDOWN_FILES
        for destination in _markdown_destinations(source.read_text(encoding="utf-8"))
        if (failure := _check_destination(source, destination)) is not None
    ]

    assert not failures, "\n" + "\n".join(failures)
