"""Geometry-derived lidar and telemetry feature calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

import trackmaniarl.trackmania.lidar_geometry_projection as lidar_geometry_projection


def telemetry(pipeline: Any, observation: Any) -> np.ndarray:
    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    if values.shape != (len(pipeline.source_fields),):
        raise ValueError(
            f"lidar source schema {pipeline.schema_version} requires 33 fields, got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("lidar telemetry contains non-finite values")
    return values


def scale_telemetry(pipeline: Any, values: np.ndarray) -> np.ndarray:
    return lidar_geometry_projection.scale_telemetry(pipeline, values)


def local_lidar(pipeline: Any, values: np.ndarray, nearest: int) -> tuple[np.ndarray, np.ndarray]:
    return lidar_geometry_projection.local_lidar(pipeline, values, nearest)


def geometry_cumulative_distance(pipeline: Any) -> np.ndarray:
    return np.asarray(
        pipeline._line_cumulative_distance(pipeline.geometry.center), dtype=np.float64
    )


def line_cumulative_distance(points: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.empty(len(points), dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(distances, out=cumulative[1:])
    return cumulative


def guidance_lookahead_line(pipeline: Any) -> np.ndarray:
    guidance = (
        pipeline._expert_guidance_line
        if pipeline._expert_guidance_line is not None
        else pipeline._reference_line
    )
    if len(guidance) != pipeline.geometry.recorded_count:
        raise ValueError("guidance line length must match recorded geometry")
    return np.concatenate((guidance, pipeline.geometry.center[pipeline.geometry.recorded_count :]))


def nearest_progress_index(pipeline: Any, position: np.ndarray, race_time_ms: float) -> int:
    start = max(0, pipeline._progress_index - pipeline.nearest_backward_points)
    stop = min(
        len(pipeline.geometry.center),
        pipeline._progress_index + pipeline.nearest_forward_points + 1,
    )
    distances = np.sum((pipeline.geometry.center[start:stop] - position) ** 2, axis=1)
    nearest = start + int(np.argmin(distances))
    if pipeline.limit_progress_by_kinematics and pipeline._last_race_time_ms is not None:
        nearest = _kinematically_reachable(pipeline, nearest, race_time_ms)
    pipeline._last_race_time_ms = race_time_ms
    pipeline._progress_index = max(pipeline._progress_index, nearest)
    return int(pipeline._progress_index)


def _kinematically_reachable(pipeline: Any, nearest: int, race_time_ms: float) -> int:
    elapsed_s = max(0.0, (race_time_ms - pipeline._last_race_time_ms) / 1_000.0)
    current_distance = float(pipeline._cumulative_distance[pipeline._progress_index])
    distance_limit = current_distance + pipeline.max_speed_mps * min(
        elapsed_s, pipeline.max_time_delta_s
    )
    reachable = int(
        np.searchsorted(pipeline._cumulative_distance, distance_limit, side="right") - 1
    )
    return int(min(nearest, max(reachable, pipeline._progress_index)))


def unit_horizontal(vector: np.ndarray) -> np.ndarray:
    horizontal = np.asarray(vector, dtype=np.float32).copy()
    horizontal[1] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-5:
        raise ValueError("track-relative vector has no horizontal component")
    return horizontal / norm


def horizontal_heading(pipeline: Any, direction: np.ndarray) -> np.ndarray:
    horizontal = np.asarray(direction, dtype=np.float32).copy()
    horizontal[1] = 0.0
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-5:
        return np.asarray(pipeline._last_heading, dtype=np.float32)
    heading = horizontal / norm
    pipeline._last_heading = heading
    return heading


def track_relative(pipeline: Any, values: np.ndarray, geometry_index: int) -> np.ndarray:
    return lidar_geometry_projection.track_relative(pipeline, values, geometry_index)


def pace_features(pipeline: Any, values: np.ndarray, geometry_index: int) -> np.ndarray:
    assert pipeline.pace_profile is not None
    index = min(geometry_index, len(pipeline._reference_line) - 1)
    reference_time_s = pipeline.pace_profile.time_at_index(index)
    time_debt_s = float(values[3]) / 1_000.0 - reference_time_s
    features = [np.clip(time_debt_s / pipeline.pace_debt_clip_s, -1.0, 1.0)]
    distance = pipeline._reference_cumulative_distance[index]
    for offset in pipeline.reference_speed_offsets_m:
        target = min(distance + offset, pipeline._reference_cumulative_distance[-1])
        future = int(np.searchsorted(pipeline._reference_cumulative_distance, target, side="left"))
        features.append(
            np.clip(
                pipeline.pace_profile.speed_at_index(future) / pipeline.max_speed_mps,
                0.0,
                1.0,
            )
        )
    return np.asarray(features, dtype=np.float32)


def dynamic_features(pipeline: Any, values: np.ndarray) -> np.ndarray:
    return lidar_geometry_projection.dynamic_features(pipeline, values)


def goal_features(pipeline: Any, values: np.ndarray, geometry_index: int) -> np.ndarray:
    return lidar_geometry_projection.goal_features(pipeline, values, geometry_index)
