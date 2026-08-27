from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, cast

PALETTE = {
    "blue": ("#1864ab", "#dbeafe"),
    "green": ("#2b8a3e", "#dcfce7"),
    "orange": ("#d97706", "#ffedd5"),
    "purple": ("#7048e8", "#ede9fe"),
    "red": ("#c92a2a", "#fee2e2"),
    "slate": ("#475569", "#f1f5f9"),
    "cyan": ("#087f8c", "#cffafe"),
}


@dataclass(frozen=True, slots=True)
class Bounds:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class TextLayout:
    x: float
    y: float
    width: float
    size: int
    color: str
    align: str


def wrap(label: str, width: float, size: int) -> str:
    line_width = max(8, int(width / (size * 0.55)))
    return "\n".join(
        wrapped
        for paragraph in label.split("\n")
        for wrapped in textwrap.wrap(paragraph, line_width, break_long_words=False)
    )


def edge_points(edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> list[list[float]]:
    if points := edge.get("points"):
        return cast(list[list[float]], points)
    source = nodes[edge["from"]]
    target = nodes[edge["to"]]
    return [
        [source["x"] + source["w"], source["y"] + source["h"] / 2],
        [target["x"], target["y"] + target["h"] / 2],
    ]


def edge_label_position(points: list[list[float]]) -> tuple[float, float]:
    start_x, start_y = points[0]
    end_x, end_y = points[-1]
    if start_y == end_y:
        return ((start_x + end_x) / 2, start_y - 18)
    if start_x == end_x:
        return (start_x + 100, (start_y + end_y) / 2)
    point = points[len(points) // 2]
    return (point[0], point[1])
