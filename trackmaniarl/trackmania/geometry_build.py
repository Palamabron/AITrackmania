"""Boundary recording cleanup and geometry asset persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.geometry import (
    GEOMETRY_ASSET_VERSION,
    _clean_boundary,
    _extend_open_finish,
    _is_closed_loop,
    _pair_opposite_boundary,
    _resample,
    _resample_matching,
    _segment_lengths,
    _smooth_polyline,
    file_sha256,
)
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest


@dataclass(frozen=True, slots=True)
class _GeometryLines:
    left: np.ndarray
    right: np.ndarray
    center: np.ndarray
    recorded_count: int


def build(request: GeometryBuildRequest) -> Path:
    _validate_build_request(request)
    lines = _recorded_geometry(request)
    lines = _extended_geometry(request, lines)
    _save_geometry(request, lines)
    return request.output


def _validate_build_request(request: GeometryBuildRequest) -> None:
    if not request.map_uid:
        raise ValueError("map_uid is required")
    if not np.isfinite(request.spacing_m) or request.spacing_m <= 0.0:
        raise ValueError("spacing_m must be finite and positive")
    if request.smooth_window < 1 or request.smooth_window % 2 == 0:
        raise ValueError("smooth_window must be a positive odd integer")
    if request.lookahead_points < 0:
        raise ValueError("lookahead_points must be non-negative")


def _recorded_geometry(request: GeometryBuildRequest) -> _GeometryLines:
    left = _clean_boundary(np.load(request.left_recording))
    right = _clean_boundary(np.load(request.right_recording))
    left_length = float(_segment_lengths(left).sum())
    count = max(2, round(left_length / request.spacing_m) + 1)
    left = _resample(left, count)
    right = _pair_opposite_boundary(left, right)
    lines = _GeometryLines(left, right, (left + right) / 2.0, 0)
    lines = _resample_center(lines, request.spacing_m)
    lines = _smooth_geometry(lines, request.smooth_window)
    _validate_paired_geometry(lines)
    return _GeometryLines(lines.left, lines.right, lines.center, len(lines.center))


def _resample_center(lines: _GeometryLines, spacing_m: float) -> _GeometryLines:
    center_length = float(_segment_lengths(lines.center).sum())
    count = max(2, round(center_length / spacing_m) + 1)
    values = _resample_matching((lines.left, lines.right, lines.center), count)
    return _GeometryLines(*values, lines.recorded_count)


def _smooth_geometry(lines: _GeometryLines, window: int) -> _GeometryLines:
    if window == 1:
        return lines
    left = _smooth_polyline(lines.left, window)
    right = _smooth_polyline(lines.right, window)
    center = (left + right) / 2.0
    values = _resample_matching((left, right, center), len(center))
    return _GeometryLines(*values, lines.recorded_count)


def _validate_paired_geometry(lines: _GeometryLines) -> None:
    widths = np.linalg.norm(lines.left - lines.right, axis=1)
    if not np.isfinite(lines.center).all() or float(np.median(widths)) <= 0.1:
        raise ValueError("paired boundaries are degenerate or overlap")


def _extended_geometry(request: GeometryBuildRequest, lines: _GeometryLines) -> _GeometryLines:
    left, right, center = lines.left, lines.right, lines.center
    if request.lookahead_points > 0 and not _is_closed_loop(center):
        left, right, center = _extend_open_finish(
            (left, right, center), request.lookahead_points, request.spacing_m
        )
    return _GeometryLines(left, right, center, lines.recorded_count)


def _save_geometry(request: GeometryBuildRequest, lines: _GeometryLines) -> None:
    request.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        request.output,
        version=np.asarray(GEOMETRY_ASSET_VERSION),
        map_uid=np.asarray(request.map_uid),
        map_sha256=np.asarray(file_sha256(request.map_path)),
        left=lines.left,
        center=lines.center.astype(np.float32),
        right=lines.right,
        spacing_m=np.asarray(request.spacing_m, dtype=np.float32),
        smooth_window=np.asarray(request.smooth_window, dtype=np.int32),
        recorded_count=np.asarray(lines.recorded_count, dtype=np.int32),
        left_sha256=np.asarray(file_sha256(request.left_recording)),
        right_sha256=np.asarray(file_sha256(request.right_recording)),
    )
