from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


@dataclass(frozen=True, slots=True)
class SeriesStyle:
    color: str
    label: str
    size: float
    alpha: float = 0.7


@dataclass(frozen=True, slots=True)
class PlotData:
    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    recorded_count: int
    extension_left: np.ndarray
    extension_right: np.ndarray
    compare_center: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class PlotRequest:
    data: PlotData
    title: str
    renderer: str


@dataclass(frozen=True, slots=True)
class PlotSegments:
    reward_center: np.ndarray
    extension_center: np.ndarray


def _scatter3d(ax: Any, points: np.ndarray, style: SeriesStyle) -> None:
    if len(points) == 0:
        return
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=style.color,
        marker="o",
        s=style.size,
        label=style.label,
        alpha=style.alpha,
    )


def _plotly_trace(points: np.ndarray, style: SeriesStyle) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=style.label,
        marker={"size": style.size, "color": style.color, "opacity": style.alpha},
    )


def _plotly_line(points: np.ndarray, style: SeriesStyle) -> go.Scatter3d:
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines+markers",
        name=style.label,
        line={"color": style.color, "width": style.size},
        marker={"size": 2, "color": style.color, "opacity": 0.85},
    )


def _segments(data: PlotData) -> PlotSegments:
    recorded = min(data.recorded_count, len(data.center))
    return PlotSegments(data.center[:recorded], data.center[recorded:])


def _plotly_figure(request: PlotRequest, segments: PlotSegments) -> go.Figure:
    traces = _plotly_traces(request.data, segments)
    plotly_fig = go.Figure(data=traces)
    plotly_fig.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectratio": {"x": 1, "y": 0.01, "z": 1},
        },
        title=request.title,
    )
    return plotly_fig


def show_plots(request: PlotRequest) -> None:
    segments = _segments(request.data)
    _show_matplotlib(request, segments)
    _plotly_figure(request, segments).show(renderer=request.renderer)


def _show_matplotlib(request: PlotRequest, segments: PlotSegments) -> None:
    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")
    if request.data.compare_center is None:
        _plot_track_points(ax, request.data, segments)
    else:
        _plot_comparison(ax, request.data)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(request.title)
    ax.legend(loc="best")
    plt.axis("equal")
    plt.show()


def _matplotlib_boundaries(data: PlotData) -> list[tuple[np.ndarray, SeriesStyle]]:
    return [
        (data.left, SeriesStyle("0.55", f"left ({len(data.left)})", 2)),
        (data.right, SeriesStyle("0.35", f"right ({len(data.right)})", 2)),
    ]


def _matplotlib_centers(
    segments: PlotSegments,
) -> list[tuple[np.ndarray, SeriesStyle]]:
    return [
        (
            segments.reward_center,
            SeriesStyle("tab:blue", f"reward/center ({len(segments.reward_center)})", 8),
        ),
        (
            segments.extension_center,
            SeriesStyle(
                "tab:red", f"lidar extension center ({len(segments.extension_center)})", 10
            ),
        ),
    ]


def _matplotlib_extensions(data: PlotData) -> list[tuple[np.ndarray, SeriesStyle]]:
    left = SeriesStyle("tab:pink", f"lidar extension left ({len(data.extension_left)})", 6)
    right = SeriesStyle("tab:orange", f"lidar extension right ({len(data.extension_right)})", 6)
    return [(data.extension_left, left), (data.extension_right, right)]


def _plot_track_points(ax: Any, data: PlotData, segments: PlotSegments) -> None:
    series = [
        *_matplotlib_boundaries(data),
        *_matplotlib_centers(segments),
        *_matplotlib_extensions(data),
    ]
    for points, style in series:
        _scatter3d(ax, points, style)


def _matplotlib_line(ax: Any, points: np.ndarray, style: SeriesStyle) -> None:
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=style.color,
        linewidth=2.0,
        label=style.label,
        alpha=0.9,
    )


def _plot_comparison(ax: Any, data: PlotData) -> None:
    compare_center = data.compare_center
    if compare_center is None:
        raise ValueError("comparison center is required")
    before = SeriesStyle("tab:orange", f"before smooth ({len(compare_center)})", 6, 0.35)
    after = SeriesStyle("tab:blue", f"after smooth ({len(data.center)})", 6, 0.35)
    _matplotlib_line(ax, compare_center, before)
    _matplotlib_line(ax, data.center, after)
    _scatter3d(ax, compare_center, SeriesStyle(before.color, "_nolegend_", 6, 0.35))
    _scatter3d(ax, data.center, SeriesStyle(after.color, "_nolegend_", 6, 0.35))


def _plotly_comparison(data: PlotData) -> list[go.Scatter3d]:
    compare_center = data.compare_center
    if compare_center is None:
        raise ValueError("comparison center is required")
    before = SeriesStyle("#f59e0b", f"before smooth ({len(compare_center)})", 4)
    after = SeriesStyle("#3b82f6", f"after smooth ({len(data.center)})", 4)
    return [_plotly_line(compare_center, before), _plotly_line(data.center, after)]


def _plotly_boundaries(data: PlotData) -> list[tuple[np.ndarray, SeriesStyle]]:
    return [
        (data.left, SeriesStyle("#888888", f"left ({len(data.left)})", 2, 0.75)),
        (data.right, SeriesStyle("#555555", f"right ({len(data.right)})", 2, 0.75)),
    ]


def _plotly_centers(segments: PlotSegments) -> list[tuple[np.ndarray, SeriesStyle]]:
    return [
        (
            segments.reward_center,
            SeriesStyle("#3b82f6", f"reward/center ({len(segments.reward_center)})", 4, 0.75),
        ),
        (
            segments.extension_center,
            SeriesStyle(
                "#ef4444", f"lidar extension center ({len(segments.extension_center)})", 5, 0.75
            ),
        ),
    ]


def _plotly_extensions(data: PlotData) -> list[tuple[np.ndarray, SeriesStyle]]:
    left = SeriesStyle("#f472b6", f"lidar extension left ({len(data.extension_left)})", 4, 0.75)
    right = SeriesStyle("#f59e0b", f"lidar extension right ({len(data.extension_right)})", 4, 0.75)
    return [(data.extension_left, left), (data.extension_right, right)]


def _plotly_traces(data: PlotData, segments: PlotSegments) -> list[go.Scatter3d]:
    if data.compare_center is not None:
        return _plotly_comparison(data)
    traces: list[go.Scatter3d] = []
    series = [
        *_plotly_boundaries(data),
        *_plotly_centers(segments),
        *_plotly_extensions(data),
    ]
    for points, style in series:
        if len(points):
            traces.append(_plotly_trace(points, style))
    return traces
