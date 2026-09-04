"""Lidar projections derived from geometry and telemetry state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class _LocalFrame:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray


@dataclass(frozen=True, slots=True)
class _Lookahead:
    indices: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True, slots=True)
class _TrackFrame:
    center: np.ndarray
    index: int
    position: np.ndarray
    tangent: np.ndarray
    right: np.ndarray
    forward: np.ndarray
    velocity: np.ndarray
    half_width: float


@dataclass(frozen=True, slots=True)
class _DynamicSample:
    race_time_ms: float
    local_velocity: np.ndarray
    yaw: float


@dataclass(frozen=True, slots=True)
class _GoalFrame:
    relative: np.ndarray
    left_relative: np.ndarray
    right_relative: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    tangent: np.ndarray
    distance: float
    remaining: float
    half_width: float


def scale_telemetry(pipeline: Any, values: np.ndarray) -> np.ndarray:
    velocity = _velocity_features(pipeline, values)
    selected = _scaled_values(pipeline, values, velocity)
    if pipeline.include_control_inputs:
        selected.extend((values[30], values[31], values[32]))
    return np.clip(np.asarray(selected, dtype=np.float32), -1.0, 1.0)


def _velocity_features(pipeline: Any, values: np.ndarray) -> tuple[float, float, float]:
    if not pipeline.local_velocity_features:
        return float(values[7]), float(values[8]), float(values[9])
    forward = pipeline._horizontal_heading(values[10:13])
    right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
    velocity = values[7:10]
    return (
        float(np.dot(velocity, forward)),
        float(velocity[1]),
        float(np.dot(velocity, right)),
    )


def _scaled_values(
    pipeline: Any, values: np.ndarray, velocity: tuple[float, float, float]
) -> list[np.float32 | float]:
    velocity_scale = pipeline.velocity_to_mps_scale / pipeline.max_speed_mps
    timing = [values[0] / 100.0, values[1] / 10.0, values[2], values[3] / 60_000.0]
    motion = [
        velocity[0] * velocity_scale,
        velocity[1] * velocity_scale,
        velocity[2] * velocity_scale,
        values[16] * velocity_scale,
        values[17] / 10_000.0,
        values[18] / 6.0,
    ]
    contact = [*values[19:23], values[27] / 4.0, values[28] / 1_000.0, values[29]]
    return [*timing, *motion, *contact]


def local_lidar(pipeline: Any, values: np.ndarray, nearest: int) -> tuple[np.ndarray, np.ndarray]:
    frame = _local_frame(pipeline, values)
    lookahead = _lookahead(pipeline, nearest)
    channels = [_boundary_channels(pipeline, frame, lookahead.indices)]
    if pipeline.include_racing_line_channels:
        channels.append(_racing_line_channels(pipeline, frame, lookahead.indices))
    if pipeline.include_finish_channels:
        channels.append(_finish_channels(pipeline, lookahead.indices))
    local = np.concatenate(channels, axis=0)
    local *= lookahead.mask[None, :]
    return np.clip(local, -1.0, 1.0).astype(np.float32), lookahead.mask


def _local_frame(pipeline: Any, values: np.ndarray) -> _LocalFrame:
    forward = pipeline._horizontal_heading(values[10:13])
    # OpenPlanet uses X/Z coordinates, so this preserves the established car-local axes.
    right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
    return _LocalFrame(values[4:7], forward, right)


def _lookahead(pipeline: Any, nearest: int) -> _Lookahead:
    indices = nearest + np.arange(1, pipeline.samples_per_side + 1)
    mask = (indices < len(pipeline.geometry.center)).astype(np.float32)
    indices = np.clip(indices, 0, len(pipeline.geometry.center) - 1)
    return _Lookahead(indices, mask)


def _boundary_channels(pipeline: Any, frame: _LocalFrame, indices: np.ndarray) -> np.ndarray:
    left = pipeline.geometry.left[indices] - frame.position
    right = pipeline.geometry.right[indices] - frame.position
    channels = np.stack(
        (
            left @ frame.right,
            left @ frame.forward,
            right @ frame.right,
            right @ frame.forward,
        ),
        axis=0,
    )
    return np.asarray(channels / float(pipeline.max_distance_m))


def _racing_line_channels(pipeline: Any, frame: _LocalFrame, indices: np.ndarray) -> np.ndarray:
    relative = pipeline._lookahead_line[indices] - frame.position
    channels = np.stack((relative @ frame.right, relative @ frame.forward), axis=0)
    return np.asarray(channels / float(pipeline.max_distance_m))


def _finish_channels(pipeline: Any, indices: np.ndarray) -> np.ndarray:
    finish_index = pipeline.geometry.recorded_count - 1
    distance = (indices.astype(np.float32) - finish_index) / 2.0
    finish_marker = np.exp(-0.5 * np.square(distance))
    virtual_marker = (indices >= pipeline.geometry.recorded_count).astype(np.float32)
    return np.stack((finish_marker, virtual_marker), axis=0)


def track_relative(pipeline: Any, values: np.ndarray, geometry_index: int) -> np.ndarray:
    frame = _track_frame(pipeline, values, geometry_index)
    relative = np.asarray(_track_values(pipeline, frame), dtype=np.float32)
    return np.clip(relative, -1.0, 1.0)


def _track_frame(pipeline: Any, values: np.ndarray, geometry_index: int) -> _TrackFrame:
    center, index, tangent, right = _track_geometry(pipeline, geometry_index)
    width = float(np.linalg.norm(pipeline.geometry.left[index] - pipeline.geometry.right[index]))
    return _TrackFrame(
        center,
        index,
        values[4:7],
        tangent,
        right,
        pipeline._horizontal_heading(values[10:13]),
        values[7:10] * pipeline.velocity_to_mps_scale,
        max(0.5 * width, 1.0),
    )


def _track_geometry(
    pipeline: Any, geometry_index: int
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    center = (
        pipeline.geometry.racing_line
        if pipeline.use_racing_line
        else pipeline.geometry.reward_center
    )
    index = min(geometry_index, len(center) - 1)
    before, after = max(0, index - 1), min(len(center) - 1, index + 1)
    tangent = pipeline._unit_horizontal(center[after] - center[before])
    right = np.asarray([tangent[2], 0.0, -tangent[0]], dtype=np.float32)
    return center, index, tangent, right


def _track_values(pipeline: Any, frame: _TrackFrame) -> list[float]:
    offset = frame.position - frame.center[frame.index]
    return [
        frame.index / max(1, pipeline.geometry.recorded_count - 1),
        float(np.dot(offset, frame.right)) / frame.half_width,
        float(np.dot(frame.forward, frame.right)),
        float(np.dot(frame.forward, frame.tangent)),
        float(np.dot(frame.velocity, frame.tangent)) / pipeline.max_speed_mps,
        float(np.dot(frame.velocity, frame.right)) / pipeline.max_speed_mps,
    ]


def dynamic_features(pipeline: Any, values: np.ndarray) -> np.ndarray:
    sample = _dynamic_sample(pipeline, values)
    elapsed_s = _dynamic_elapsed(pipeline, sample.race_time_ms)
    features = _dynamic_delta(pipeline, sample, elapsed_s)
    _store_dynamic_sample(pipeline, sample)
    return features


def _dynamic_sample(pipeline: Any, values: np.ndarray) -> _DynamicSample:
    forward = pipeline._horizontal_heading(values[10:13])
    right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
    velocity = values[7:10] * pipeline.velocity_to_mps_scale
    local_velocity = np.asarray(
        [np.dot(velocity, forward), np.dot(velocity, right)], dtype=np.float32
    )
    return _DynamicSample(
        float(values[3]), local_velocity, float(np.arctan2(forward[2], forward[0]))
    )


def _dynamic_elapsed(pipeline: Any, race_time_ms: float) -> float:
    previous = pipeline._last_dynamic_race_time_ms
    return 0.0 if previous is None else max(0.0, (race_time_ms - previous) / 1_000.0)


def _dynamic_delta(pipeline: Any, sample: _DynamicSample, elapsed_s: float) -> np.ndarray:
    if elapsed_s <= 1e-4 or elapsed_s > pipeline.max_time_delta_s:
        return np.zeros(4, dtype=np.float32)
    acceleration = (sample.local_velocity - pipeline._last_dynamic_velocity) / elapsed_s
    previous_yaw = pipeline._last_dynamic_yaw
    yaw_delta = 0.0 if previous_yaw is None else sample.yaw - previous_yaw
    yaw_delta = float(np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta)))
    return np.asarray(
        [
            np.clip(elapsed_s / 0.05, 0.0, 1.0),
            np.clip(yaw_delta / elapsed_s / 4.0, -1.0, 1.0),
            np.clip(acceleration[0] / 40.0, -1.0, 1.0),
            np.clip(acceleration[1] / 40.0, -1.0, 1.0),
        ],
        dtype=np.float32,
    )


def _store_dynamic_sample(pipeline: Any, sample: _DynamicSample) -> None:
    pipeline._last_dynamic_race_time_ms = sample.race_time_ms
    pipeline._last_dynamic_velocity = sample.local_velocity
    pipeline._last_dynamic_yaw = sample.yaw


def goal_features(pipeline: Any, values: np.ndarray, geometry_index: int) -> np.ndarray:
    frame = _goal_frame(pipeline, values, geometry_index)
    projections = _goal_projections(pipeline, frame)
    distances = _goal_distances(pipeline, frame)
    return np.asarray([*projections, *distances], dtype=np.float32)


def _goal_frame(pipeline: Any, values: np.ndarray, geometry_index: int) -> _GoalFrame:
    finish_index, center, left, right_edge = _finish_edges(pipeline)
    position = values[4:7]
    forward = pipeline._horizontal_heading(values[10:13])
    right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
    relative = center - position
    remaining = _remaining_distance(pipeline, min(geometry_index, finish_index))
    tangent = pipeline._unit_horizontal(pipeline._reference_line[-1] - pipeline._reference_line[-2])
    return _GoalFrame(
        relative,
        left - position,
        right_edge - position,
        forward,
        right,
        tangent,
        max(float(np.linalg.norm(relative)), 1e-6),
        max(0.0, remaining),
        0.5 * float(np.linalg.norm(left - right_edge)),
    )


def _finish_edges(pipeline: Any) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    index = pipeline.geometry.recorded_count - 1
    return (
        index,
        pipeline.geometry.center[index],
        pipeline.geometry.left[index],
        pipeline.geometry.right[index],
    )


def _remaining_distance(pipeline: Any, index: int) -> float:
    remaining = pipeline._reference_cumulative_distance[-1]
    remaining -= pipeline._reference_cumulative_distance[index]
    return float(remaining)


def _goal_projections(pipeline: Any, frame: _GoalFrame) -> list[np.floating[Any] | float]:
    vectors = (frame.relative, frame.left_relative, frame.right_relative)
    return [
        np.clip(float(np.dot(vector, axis)) / pipeline.max_distance_m, -1.0, 1.0)
        for vector in vectors
        for axis in (frame.right, frame.forward)
    ]


def _goal_distances(pipeline: Any, frame: _GoalFrame) -> list[np.floating[Any] | float]:
    lateral = float(np.dot(frame.relative, frame.right))
    longitudinal = float(np.dot(frame.relative, frame.forward))
    total_distance = pipeline._reference_cumulative_distance[-1]
    return [
        np.clip(np.log1p(frame.distance) / np.log1p(total_distance), 0.0, 1.0),
        np.clip(lateral / frame.distance, -1.0, 1.0),
        np.clip(longitudinal / frame.distance, -1.0, 1.0),
        np.clip(float(np.dot(frame.tangent, frame.right)), -1.0, 1.0),
        np.clip(float(np.dot(frame.tangent, frame.forward)), -1.0, 1.0),
        np.clip(frame.half_width / 20.0, 0.0, 1.0),
        np.clip(frame.remaining / pipeline.max_distance_m, 0.0, 1.0),
        float(frame.remaining <= pipeline.max_distance_m),
    ]
