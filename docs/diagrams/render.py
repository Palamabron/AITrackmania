"""Render the repository's architecture diagram specifications."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PALETTE = {
    "blue": ("#1864ab", "#dbeafe"),
    "green": ("#2b8a3e", "#dcfce7"),
    "orange": ("#d97706", "#ffedd5"),
    "purple": ("#7048e8", "#ede9fe"),
    "red": ("#c92a2a", "#fee2e2"),
    "slate": ("#475569", "#f1f5f9"),
    "cyan": ("#087f8c", "#cffafe"),
}


def _nonce(item_id: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{item_id}:{purpose}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_646 + 1


def _base(
    kind: str, item_id: str, x: float, y: float, width: float, height: float
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": kind,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": "#1f2937",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": {"type": 3},
        "seed": _nonce(item_id, "seed"),
        "version": 1,
        "versionNonce": _nonce(item_id, "version"),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }


def _wrap(label: str, width: float, size: int) -> str:
    line_width = max(8, int(width / (size * 0.55)))
    return "\n".join(
        wrapped
        for paragraph in label.split("\n")
        for wrapped in textwrap.wrap(paragraph, line_width, break_long_words=False)
    )


def _text(
    item_id: str,
    label: str,
    x: float,
    y: float,
    width: float,
    size: int,
    color: str,
    align: str = "center",
) -> dict[str, Any]:
    wrapped = _wrap(label, width, size)
    lines = wrapped.count("\n") + 1
    element = _base("text", item_id, x, y, width, lines * size * 1.25)
    element.update(
        {
            "strokeColor": color,
            "fontSize": size,
            "fontFamily": 1,
            "text": wrapped,
            "textAlign": align,
            "verticalAlign": "middle",
            "containerId": None,
            "originalText": label,
            "autoResize": False,
            "lineHeight": 1.25,
        }
    )
    return element


def _zone_elements(zone: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[zone["color"]]
    frame = _base("rectangle", f"zone-{zone['id']}", zone["x"], zone["y"], zone["w"], zone["h"])
    frame.update(
        {
            "strokeColor": stroke,
            "backgroundColor": fill,
            "strokeStyle": "dashed",
            "strokeWidth": 1.5,
            "opacity": 32,
        }
    )
    title = _text(
        f"zone-{zone['id']}-label",
        zone["label"],
        zone["x"] + 18,
        zone["y"] + 14,
        zone["w"] - 36,
        19,
        stroke,
        "left",
    )
    return [frame, title]


def _node_elements(node: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[node["color"]]
    shape = node.get("shape", "rectangle")
    box = _base(shape, node["id"], node["x"], node["y"], node["w"], node["h"])
    box.update({"strokeColor": stroke, "backgroundColor": fill, "strokeWidth": 2.5})
    title_lines = _wrap(node["label"], node["w"] - 24, 20).count("\n") + 1
    detail_lines = (
        _wrap(node.get("detail", ""), node["w"] - 28, 18).count("\n") + 1
        if node.get("detail")
        else 0
    )
    content_height = title_lines * 25 + (5 + detail_lines * 22.5 if detail_lines else 0)
    title_y = node["y"] + max(7, (node["h"] - content_height) / 2)
    title = _text(
        f"{node['id']}-title", node["label"], node["x"] + 12, title_y, node["w"] - 24, 20, "#111827"
    )
    elements = [box, title]
    if detail := node.get("detail"):
        elements.append(
            _text(
                f"{node['id']}-detail",
                detail,
                node["x"] + 14,
                title_y + title_lines * 25 + 5,
                node["w"] - 28,
                18,
                "#475569",
            )
        )
    return elements


def _edge_points(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[list[float]]:
    if points := edge.get("points"):
        return points
    source = nodes[edge["from"]]
    target = nodes[edge["to"]]
    return [
        [source["x"] + source["w"], source["y"] + source["h"] / 2],
        [target["x"], target["y"] + target["h"] / 2],
    ]


def _edge_label_position(points: list[list[float]]) -> tuple[float, float]:
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    if start_y == end_y:
        return ((start_x + end_x) / 2, start_y - 18)
    if start_x == end_x:
        return (start_x + 100, (start_y + end_y) / 2)
    point = points[len(points) // 2]
    return (point[0], point[1])


def _edge_elements(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    points = _edge_points(edge, nodes)
    start_x, start_y = points[0]
    relative = [[x - start_x, y - start_y] for x, y in points]
    stroke = PALETTE[edge.get("color", "slate")][0]
    arrow = _base(
        "arrow", edge["id"], start_x, start_y, points[-1][0] - start_x, points[-1][1] - start_y
    )
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
    elements = [arrow]
    if label := edge.get("label"):
        label_x, label_y = edge.get("label_at", _edge_label_position(points))
        elements.append(
            _text(f"{edge['id']}-label", label, label_x - 90, label_y - 28, 180, 18, stroke)
        )
    return elements


def _note_elements(note: dict[str, Any]) -> list[dict[str, Any]]:
    stroke, fill = PALETTE[note.get("color", "slate")]
    box = _base("rectangle", note["id"], note["x"], note["y"], note["w"], note["h"])
    box.update({"strokeColor": stroke, "backgroundColor": fill, "strokeWidth": 1.5})
    return [
        box,
        _text(
            f"{note['id']}-text",
            note["text"],
            note["x"] + 14,
            note["y"] + 12,
            note["w"] - 28,
            18,
            "#1f2937",
            "left",
        ),
    ]


def build_scene(spec: dict[str, Any]) -> dict[str, Any]:
    nodes = {node["id"]: node for node in spec["nodes"]}
    elements: list[dict[str, Any]] = []
    elements.append(
        _text("title", spec["title"], 55, 28, spec["width"] - 110, 32, "#111827", "left")
    )
    elements.append(
        _text("subtitle", spec["subtitle"], 55, 76, spec["width"] - 110, 18, "#64748b", "left")
    )
    for zone in spec["zones"]:
        elements.extend(_zone_elements(zone))
    for edge in spec["edges"]:
        elements.extend(_edge_elements(edge, nodes))
    for node in spec["nodes"]:
        elements.extend(_node_elements(node))
    for note in spec.get("notes", []):
        elements.extend(_note_elements(note))
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
        "files": {},
    }


def _svg_text(
    label: str, x: float, y: float, width: float, size: int, color: str, align: str = "middle"
) -> str:
    anchor = {"middle": "middle", "start": "start"}[align]
    text_x = x + width / 2 if align == "middle" else x
    lines = _wrap(label, width, size).split("\n")
    spans = "".join(
        f'<tspan x="{text_x}" dy="{0 if index == 0 else size * 1.25}">{html.escape(line)}</tspan>'
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{text_x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{size}" '
        f'fill="{color}">{spans}</text>'
    )


def render_svg(spec: dict[str, Any]) -> str:
    nodes = {node["id"]: node for node in spec["nodes"]}
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{spec["width"]}" '
            f'height="{spec["height"]}" viewBox="0 0 {spec["width"]} {spec["height"]}">'
        ),
        (
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" '
            'refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" '
            'fill="context-stroke"/></marker><filter id="shadow" x="-20%" y="-20%" '
            'width="140%" height="140%"><feDropShadow dx="0" dy="2" stdDeviation="3" '
            'flood-opacity=".14"/></filter></defs>'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(spec["title"], 55, 61, spec["width"] - 110, 32, "#111827", "start"),
        _svg_text(spec["subtitle"], 55, 99, spec["width"] - 110, 18, "#64748b", "start"),
    ]
    for zone in spec["zones"]:
        stroke, fill = PALETTE[zone["color"]]
        parts.append(
            f'<rect x="{zone["x"]}" y="{zone["y"]}" width="{zone["w"]}" '
            f'height="{zone["h"]}" rx="14" fill="{fill}" fill-opacity=".35" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="8 7"/>'
        )
        parts.append(
            _svg_text(
                zone["label"], zone["x"] + 18, zone["y"] + 29, zone["w"] - 36, 19, stroke, "start"
            )
        )
    for edge in spec["edges"]:
        points = _edge_points(edge, nodes)
        stroke = PALETTE[edge.get("color", "slate")][0]
        dash = ' stroke-dasharray="8 7"' if edge.get("style") == "dashed" else ""
        joined = " ".join(f"{x},{y}" for x, y in points)
        parts.append(
            f'<polyline points="{joined}" fill="none" stroke="{stroke}" '
            f'stroke-width="2.5"{dash} marker-end="url(#arrow)"/>'
        )
        if label := edge.get("label"):
            label_x, label_y = edge.get("label_at", _edge_label_position(points))
            parts.append(
                f'<rect x="{label_x - 88}" y="{label_y - 22}" width="176" '
                'height="25" rx="5" fill="#ffffff" fill-opacity=".92"/>'
            )
            parts.append(_svg_text(label, label_x - 88, label_y - 5, 176, 18, stroke))
    for node in spec["nodes"]:
        stroke, fill = PALETTE[node["color"]]
        if node.get("shape") == "diamond":
            points = " ".join(
                (
                    f"{node['x'] + node['w'] / 2},{node['y']}",
                    f"{node['x'] + node['w']},{node['y'] + node['h'] / 2}",
                    f"{node['x'] + node['w'] / 2},{node['y'] + node['h']}",
                    f"{node['x']},{node['y'] + node['h'] / 2}",
                )
            )
            parts.append(
                f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
                'stroke-width="2.5" filter="url(#shadow)"/>'
            )
        else:
            parts.append(
                f'<rect x="{node["x"]}" y="{node["y"]}" width="{node["w"]}" '
                f'height="{node["h"]}" rx="12" fill="{fill}" stroke="{stroke}" '
                'stroke-width="2.5" filter="url(#shadow)"/>'
            )
        title_lines = _wrap(node["label"], node["w"] - 24, 20).count("\n") + 1
        detail_lines = (
            _wrap(node.get("detail", ""), node["w"] - 28, 18).count("\n") + 1
            if node.get("detail")
            else 0
        )
        content_height = title_lines * 25 + (5 + detail_lines * 22.5 if detail_lines else 0)
        content_top = node["y"] + max(7, (node["h"] - content_height) / 2)
        parts.append(
            _svg_text(
                node["label"], node["x"] + 12, content_top + 20, node["w"] - 24, 20, "#111827"
            )
        )
        if detail := node.get("detail"):
            parts.append(
                _svg_text(
                    detail,
                    node["x"] + 14,
                    content_top + title_lines * 25 + 23,
                    node["w"] - 28,
                    18,
                    "#475569",
                )
            )
    for note in spec.get("notes", []):
        stroke, fill = PALETTE[note.get("color", "slate")]
        parts.append(
            f'<rect x="{note["x"]}" y="{note["y"]}" width="{note["w"]}" '
            f'height="{note["h"]}" rx="10" fill="{fill}" stroke="{stroke}" '
            'stroke-width="1.5"/>'
        )
        parts.append(
            _svg_text(
                note["text"], note["x"] + 14, note["y"] + 31, note["w"] - 28, 18, "#1f2937", "start"
            )
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_html(spec: dict[str, Any], svg: str, scene: dict[str, Any], stem: str) -> str:
    encoded = html.escape(json.dumps(scene, ensure_ascii=False))
    style = (
        "body{margin:0;background:#e2e8f0;font-family:system-ui}main{padding:24px}"
        ".canvas{max-width:1600px;margin:auto;background:white;box-shadow:0 12px 40px #0f172a33}"
        "svg{display:block;width:100%;height:auto}button{position:fixed;right:24px;bottom:24px;"
        "padding:12px 18px;border:0;border-radius:9px;background:#1864ab;color:white;"
        "font-weight:700;cursor:pointer}"
    )
    script = (
        "document.getElementById('download').onclick=()=>{"
        "const data=JSON.parse(document.getElementById('scene').textContent);"
        "const a=document.createElement('a');"
        "a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],"
        "{type:'application/json'}));"
        f"a.download='{stem}.excalidraw';a.click();}};"
    )
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>{html.escape(spec['title'])}</title><style>{style}</style></head>",
            f'<body><main><div class="canvas">{svg}</div></main>',
            '<button id="download">Download editable diagram</button>',
            f'<script id="scene" type="application/json">{encoded}</script>',
            f"<script>{script}</script></body></html>",
        )
    )


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
        render_html(spec, svg, scene, stem), encoding="utf-8"
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
