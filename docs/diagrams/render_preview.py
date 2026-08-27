from __future__ import annotations

import html
import json
from dataclasses import dataclass
from typing import Any

from docs.diagrams.render_common import (
    PALETTE,
    TextLayout,
    edge_label_position,
    edge_points,
    wrap,
)

HTML_STYLE = (
    "body{margin:0;background:#e2e8f0;font-family:system-ui}main{padding:24px}"
    ".canvas{max-width:1600px;margin:auto;background:white;box-shadow:0 12px 40px #0f172a33}"
    "svg{display:block;width:100%;height:auto}button{position:fixed;right:24px;bottom:24px;"
    "padding:12px 18px;border:0;border-radius:9px;background:#1864ab;color:white;"
    "font-weight:700;cursor:pointer}"
)


@dataclass(frozen=True, slots=True)
class PreviewDocument:
    spec: dict[str, Any]
    svg: str
    scene: dict[str, Any]
    stem: str


def _svg_text(label: str, layout: TextLayout) -> str:
    anchor = {"middle": "middle", "start": "start"}[layout.align]
    text_x = layout.x + layout.width / 2 if layout.align == "middle" else layout.x
    lines = wrap(label, layout.width, layout.size).split("\n")
    spans = "".join(
        f'<tspan x="{text_x}" dy="{0 if index == 0 else layout.size * 1.25}">'
        f"{html.escape(line)}</tspan>"
        for index, line in enumerate(lines)
    )
    return (
        f'<text x="{text_x}" y="{layout.y}" text-anchor="{anchor}" '
        f'font-family="Inter,Arial,sans-serif" font-size="{layout.size}" '
        f'fill="{layout.color}">{spans}</text>'
    )


def render_svg(spec: dict[str, Any]) -> str:
    nodes = {node["id"]: node for node in spec["nodes"]}
    parts = _svg_header(spec)
    for zone in spec["zones"]:
        parts.extend(_svg_zone(zone))
    for edge in spec["edges"]:
        parts.extend(_svg_edge(edge, nodes))
    for node in spec["nodes"]:
        parts.extend(_svg_node(node))
    for note in spec.get("notes", []):
        parts.extend(_svg_note(note))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _svg_header(spec: dict[str, Any]) -> list[str]:
    title = TextLayout(55, 61, spec["width"] - 110, 32, "#111827", "start")
    subtitle = TextLayout(55, 99, spec["width"] - 110, 18, "#64748b", "start")
    return [
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
        _svg_text(spec["title"], title),
        _svg_text(spec["subtitle"], subtitle),
    ]


def _svg_zone(zone: dict[str, Any]) -> list[str]:
    stroke, fill = PALETTE[zone["color"]]
    layout = TextLayout(zone["x"] + 18, zone["y"] + 29, zone["w"] - 36, 19, stroke, "start")
    return [
        (
            f'<rect x="{zone["x"]}" y="{zone["y"]}" width="{zone["w"]}" '
            f'height="{zone["h"]}" rx="14" fill="{fill}" fill-opacity=".35" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="8 7"/>'
        ),
        _svg_text(zone["label"], layout),
    ]


def _svg_edge(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[str]:
    points = edge_points(edge, nodes)
    stroke = PALETTE[edge.get("color", "slate")][0]
    dash = ' stroke-dasharray="8 7"' if edge.get("style") == "dashed" else ""
    joined = " ".join(f"{x},{y}" for x, y in points)
    parts = [
        f'<polyline points="{joined}" fill="none" stroke="{stroke}" '
        f'stroke-width="2.5"{dash} marker-end="url(#arrow)"/>'
    ]
    if label := edge.get("label"):
        label_x, label_y = edge.get("label_at", edge_label_position(points))
        parts.append(
            f'<rect x="{label_x - 88}" y="{label_y - 22}" width="176" '
            'height="25" rx="5" fill="#ffffff" fill-opacity=".92"/>'
        )
        layout = TextLayout(label_x - 88, label_y - 5, 176, 18, stroke, "middle")
        parts.append(_svg_text(label, layout))
    return parts


def _svg_node_position(node: dict[str, Any]) -> tuple[int, float]:
    title_lines = wrap(node["label"], node["w"] - 24, 20).count("\n") + 1
    detail = node.get("detail", "")
    detail_lines = wrap(detail, node["w"] - 28, 18).count("\n") + 1 if detail else 0
    detail_height = 5 + detail_lines * 22.5 if detail_lines else 0
    content_height = title_lines * 25 + detail_height
    content_top = node["y"] + max(7, (node["h"] - content_height) / 2)
    return title_lines, content_top


def _svg_node_text(node: dict[str, Any]) -> list[str]:
    title_lines, content_top = _svg_node_position(node)
    title_layout = TextLayout(
        node["x"] + 12, content_top + 20, node["w"] - 24, 20, "#111827", "middle"
    )
    parts = [_svg_text(node["label"], title_layout)]
    detail = node.get("detail")
    if not detail:
        return parts
    detail_layout = TextLayout(
        node["x"] + 14,
        content_top + title_lines * 25 + 23,
        node["w"] - 28,
        18,
        "#475569",
        "middle",
    )
    return [*parts, _svg_text(detail, detail_layout)]


def _svg_node(node: dict[str, Any]) -> list[str]:
    stroke, fill = PALETTE[node["color"]]
    return [_svg_node_shape(node, stroke, fill), *_svg_node_text(node)]


def _svg_node_shape(node: dict[str, Any], stroke: str, fill: str) -> str:
    if node.get("shape") == "diamond":
        points = " ".join(
            (
                f"{node['x'] + node['w'] / 2},{node['y']}",
                f"{node['x'] + node['w']},{node['y'] + node['h'] / 2}",
                f"{node['x'] + node['w'] / 2},{node['y'] + node['h']}",
                f"{node['x']},{node['y'] + node['h'] / 2}",
            )
        )
        return (
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" '
            'stroke-width="2.5" filter="url(#shadow)"/>'
        )
    return (
        f'<rect x="{node["x"]}" y="{node["y"]}" width="{node["w"]}" '
        f'height="{node["h"]}" rx="12" fill="{fill}" stroke="{stroke}" '
        'stroke-width="2.5" filter="url(#shadow)"/>'
    )


def _svg_note(note: dict[str, Any]) -> list[str]:
    stroke, fill = PALETTE[note.get("color", "slate")]
    layout = TextLayout(note["x"] + 14, note["y"] + 31, note["w"] - 28, 18, "#1f2937", "start")
    return [
        (
            f'<rect x="{note["x"]}" y="{note["y"]}" width="{note["w"]}" '
            f'height="{note["h"]}" rx="10" fill="{fill}" stroke="{stroke}" '
            'stroke-width="1.5"/>'
        ),
        _svg_text(note["text"], layout),
    ]


def _download_script(stem: str) -> str:
    return (
        "document.getElementById('download').onclick=()=>{"
        "const data=JSON.parse(document.getElementById('scene').textContent);"
        "const a=document.createElement('a');"
        "a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],"
        "{type:'application/json'}));"
        f"a.download='{stem}.excalidraw';a.click();}};"
    )


def render_html(preview: PreviewDocument) -> str:
    encoded = html.escape(json.dumps(preview.scene, ensure_ascii=False))
    script = _download_script(preview.stem)
    return "\n".join(
        (
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            f"<title>{html.escape(preview.spec['title'])}</title>"
            f"<style>{HTML_STYLE}</style></head>",
            f'<body><main><div class="canvas">{preview.svg}</div></main>',
            '<button id="download">Download editable diagram</button>',
            f'<script id="scene" type="application/json">{encoded}</script>',
            f"<script>{script}</script></body></html>",
        )
    )
