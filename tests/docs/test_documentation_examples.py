from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
import yaml

from trackmaniarl.core.runtime import resolve_run, validate_resolved_run
from trackmaniarl.core.spec import RunSpec

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
