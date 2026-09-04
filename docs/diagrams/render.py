"""Render the repository's architecture diagram specifications."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from docs.diagrams.render_common import PALETTE, Bounds, TextLayout
from docs.diagrams.render_common import edge_label_position as _edge_label_position
from docs.diagrams.render_common import edge_points as _edge_points
from docs.diagrams.render_common import wrap as _wrap
from docs.diagrams.render_preview import PreviewDocument, render_html, render_svg

ROOT = Path(__file__).resolve().parent
ELEMENT_STYLE_DEFAULTS: dict[str, Any] = {
    "angle": 0,
    "strokeColor": "#1f2937",
    "backgroundColor": "transparent",
    "fillStyle": "solid",
    "strokeWidth": 2,
    "strokeStyle": "solid",
    "roughness": 1,
    "opacity": 100,
}
ELEMENT_FRAME_DEFAULTS: dict[str, Any] = {
    "frameId": None,
    "roundness": {"type": 3},
}
ELEMENT_VERSION_DEFAULTS: dict[str, Any] = {
    "version": 1,
}
ELEMENT_DELETION_DEFAULTS: dict[str, Any] = {
    "isDeleted": False,
}
ELEMENT_FINAL_DEFAULTS: dict[str, Any] = {
    "updated": 1,
    "link": None,
    "locked": False,
}


def _nonce(item_id: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{item_id}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_646 + 1


def _base(kind: str, item_id: str, bounds: Bounds) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": kind,
        "x": bounds.x,
        "y": bounds.y,
        "width": bounds.width,
        "height": bounds.height,
        **ELEMENT_STYLE_DEFAULTS,
        "groupIds": [],
        **ELEMENT_FRAME_DEFAULTS,
        "seed": _nonce(item_id, "seed"),
        **ELEMENT_VERSION_DEFAULTS,
        "versionNonce": _nonce(item_id, "version"),
        **ELEMENT_DELETION_DEFAULTS,
        "boundElements": [],
        **ELEMENT_FINAL_DEFAULTS,
    }


def _text(item_id: str, label: str, layout: TextLayout) -> dict[str, Any]:
    wrapped = _wrap(label, layout.width, layout.size)
    lines = wrapped.count("\n") + 1
    bounds = Bounds(layout.x, layout.y, layout.width, lines * layout.size * 1.25)
    element = _base("text", item_id, bounds)
    element.update(_text_properties(label, wrapped, layout))
    return element


def _text_properties(label: str, wrapped: str, layout: TextLayout) -> dict[str, Any]:
    return {
        "strokeColor": layout.color,
        "fontSize": layout.size,
        "fontFamily": 1,
        "text": wrapped,
        "textAlign": layout.align,
        "verticalAlign": "middle",
        "containerId": None,
        "originalText": label,
        "autoResize": False,
        "lineHeight": 1.25,
    }


def _zone_frame(zone: dict[str, Any], stroke: str, fill: str) -> dict[str, Any]:
    bounds = Bounds(zone["x"], zone["y"], zone["w"], zone["h"])
    frame = _base("rectangle", f"zone-{zone['id']}", bounds)
    frame.update(
        {
            "strokeColor": stroke,
            "backgroundColor": fill,
            "strokeStyle": "dashed",
            "strokeWidth": 1.5,
            "opacity": 32,
        }
    )
    return frame


def _zone_elements(zone: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[zone["color"]]
    layout = TextLayout(
        zone["x"] + 18,
        zone["y"] + 14,
        zone["w"] - 36,
        19,
        stroke,
        "left",
    )
    title = _text(f"zone-{zone['id']}-label", zone["label"], layout)
    return [_zone_frame(zone, stroke, fill), title]


def _node_box(node: dict[str, Any], stroke: str, fill: str) -> dict[str, Any]:
    bounds = Bounds(node["x"], node["y"], node["w"], node["h"])
    box = _base(node.get("shape", "rectangle"), node["id"], bounds)
    box.update({"strokeColor": stroke, "backgroundColor": fill, "strokeWidth": 2.5})
    return box


def _node_content_position(node: dict[str, Any]) -> tuple[int, float]:
    title_lines = _wrap(node["label"], node["w"] - 24, 20).count("\n") + 1
    detail = node.get("detail", "")
    detail_lines = _wrap(detail, node["w"] - 28, 18).count("\n") + 1 if detail else 0
    detail_height = 5 + detail_lines * 22.5 if detail_lines else 0
    content_height = title_lines * 25 + detail_height
    title_y = node["y"] + max(7, (node["h"] - content_height) / 2)
    return title_lines, title_y


def _node_title(node: dict[str, Any], title_y: float) -> dict[str, Any]:
    layout = TextLayout(node["x"] + 12, title_y, node["w"] - 24, 20, "#111827", "center")
    return _text(f"{node['id']}-title", node["label"], layout)


def _node_detail(node: dict[str, Any], title_lines: int, title_y: float) -> list[dict[str, Any]]:
    detail = node.get("detail")
    if not detail:
        return []
    layout = TextLayout(
        node["x"] + 14,
        title_y + title_lines * 25 + 5,
        node["w"] - 28,
        18,
        "#475569",
        "center",
    )
    return [_text(f"{node['id']}-detail", detail, layout)]


def _node_elements(node: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[node["color"]]
    title_lines, title_y = _node_content_position(node)
    elements = [_node_box(node, stroke, fill), _node_title(node, title_y)]
    return elements + _node_detail(node, title_lines, title_y)


def _arrow_element(edge: dict[str, Any], points: list[list[float]], stroke: str) -> dict[str, Any]:
    start_x, start_y = points[0]
    relative = [[x - start_x, y - start_y] for x, y in points]
    bounds = Bounds(start_x, start_y, points[-1][0] - start_x, points[-1][1] - start_y)
    arrow = _base("arrow", edge["id"], bounds)
    arrow.update(
        {
            "strokeColor": stroke,
            "strokeWidth": 2.5,
            "strokeStyle": edge.get("style", "solid"),
            "points": relative,
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
        }
    )
    return arrow


def _edge_label(
    edge: dict[str, Any], points: list[list[float]], stroke: str
) -> list[dict[str, Any]]:
    label = edge.get("label")
    if not label:
        return []
    label_x, label_y = edge.get("label_at", _edge_label_position(points))
    layout = TextLayout(label_x - 90, label_y - 28, 180, 18, stroke, "center")
    return [_text(f"{edge['id']}-label", label, layout)]


def _edge_elements(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    points = _edge_points(edge, nodes)
    stroke = PALETTE[edge.get("color", "slate")][0]
    return [_arrow_element(edge, points, stroke), *_edge_label(edge, points, stroke)]


def _note_elements(note: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[note.get("color", "slate")]
    bounds = Bounds(note["x"], note["y"], note["w"], note["h"])
    box = _base("rectangle", note["id"], bounds)
    box.update({"strokeColor": stroke, "backgroundColor": fill, "strokeWidth": 1.5})
    layout = TextLayout(note["x"] + 14, note["y"] + 12, note["w"] - 28, 18, "#1f2937", "left")
    return [
        box,
        _text(f"{note['id']}-text", note["text"], layout),
    ]


def _scene_elements(spec: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    title_layout = TextLayout(55, 28, spec["width"] - 110, 32, "#111827", "left")
    subtitle_layout = TextLayout(55, 76, spec["width"] - 110, 18, "#64748b", "left")
    elements.extend(
        [
            _text("title", spec["title"], title_layout),
            _text("subtitle", spec["subtitle"], subtitle_layout),
        ]
    )
    for zone in spec["zones"]:
        elements.extend(_zone_elements(zone))
    for edge in spec["edges"]:
        elements.extend(_edge_elements(edge, nodes))
    for node in spec["nodes"]:
        elements.extend(_node_elements(node))
    for note in spec.get("notes", []):
        elements.extend(_note_elements(note))
    return elements


def build_scene(spec: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: node for node in spec["nodes"]}
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": _scene_elements(spec, nodes),
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }


def render_one(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    stem = spec_path.name.removesuffix(".spec.json")
    scene = build_scene(spec)
    svg = render_svg(spec)
    (ROOT / f"{stem}.excalidraw").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / f"{stem}-preview.svg").write_text(svg, encoding="utf-8")
    (ROOT / f"{stem}-preview.html").write_text(
        render_html(PreviewDocument(spec, svg, scene, stem)), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.specs or sorted(ROOT.glob("*.spec.json"))
    for path in paths:
        render_one(path if path.is_absolute() else ROOT / path)


if __name__ == "__main__":
    main()
