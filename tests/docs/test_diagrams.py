from __future__ import annotations

import json
import runpy
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DIAGRAMS = ROOT / "docs" / "diagrams"
EXPECTED_STEMS = (
    "checkpoint-resume",
    "demonstration-timing",
    "distributed-security",
    "imitation-learning",
    "model-composition",
    "replay-sequence",
    "reward-decomposition",
    "runtime-architecture",
    "trackmania-integration",
)
with patch.object(sys, "path", [str(ROOT), *sys.path]):
    RENDERER = runpy.run_module("docs.diagrams.render", run_name="diagram_renderer")
BUILD_SCENE = cast(Callable[[dict[str, Any]], dict[str, Any]], RENDERER["build_scene"])
RENDER_SVG = cast(Callable[[dict[str, Any]], str], RENDERER["render_svg"])
PREVIEW_DOCUMENT = cast(
    Callable[[dict[str, Any], str, dict[str, Any], str], object], RENDERER["PreviewDocument"]
)
RENDER_HTML = cast(Callable[[object], str], RENDERER["render_html"])


def _spec(stem: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((DIAGRAMS / f"{stem}.spec.json").read_text(encoding="utf-8")),
    )


def test_diagram_set_is_deliberate() -> None:
    stems = tuple(
        path.name.removesuffix(".spec.json") for path in sorted(DIAGRAMS.glob("*.spec.json"))
    )

    assert stems == EXPECTED_STEMS


def _assert_valid_spec(stem: str) -> None:
    spec = _spec(stem)
    groups = (spec["zones"], spec["nodes"], spec["edges"], spec.get("notes", []))
    semantic_ids = [item["id"] for group in groups for item in group]
    node_ids = {node["id"] for node in spec["nodes"]}

    assert len(semantic_ids) == len(set(semantic_ids))
    assert all(edge["from"] in node_ids and edge["to"] in node_ids for edge in spec["edges"])
    assert spec["width"] > 0
    assert spec["height"] > 0
    for edge in spec["edges"]:
        points = edge.get("points")
        assert points is None or (
            len(points) >= 2
            and all(
                len(point) == 2 and all(isinstance(value, int | float) for value in point)
                for point in points
            )
        )


def test_diagram_specs_have_valid_references_and_unique_ids() -> None:
    for stem in EXPECTED_STEMS:
        _assert_valid_spec(stem)


def _assert_previews_match_spec(stem: str) -> None:
    spec = _spec(stem)
    scene = BUILD_SCENE(spec)
    svg = RENDER_SVG(spec)

    assert json.loads((DIAGRAMS / f"{stem}.excalidraw").read_text(encoding="utf-8")) == scene
    assert (DIAGRAMS / f"{stem}-preview.svg").read_text(encoding="utf-8") == svg
    preview = PREVIEW_DOCUMENT(spec, svg, scene, stem)
    assert (DIAGRAMS / f"{stem}-preview.html").read_text(encoding="utf-8") == RENDER_HTML(preview)
    element_ids = [element["id"] for element in scene["elements"]]
    assert len(element_ids) == len(set(element_ids))


def test_committed_editable_and_web_previews_match_specs() -> None:
    for stem in EXPECTED_STEMS:
        _assert_previews_match_spec(stem)


def _assert_png_matches_canvas(stem: str) -> None:
    spec = _spec(stem)
    data = (DIAGRAMS / f"{stem}-preview.png").read_bytes()

    assert data[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    assert struct.unpack(">II", data[16:24]) == (spec["width"], spec["height"])


def test_png_previews_match_canvases() -> None:
    for stem in EXPECTED_STEMS:
        _assert_png_matches_canvas(stem)
