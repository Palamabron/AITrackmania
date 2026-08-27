"""Reference lap timing projected onto a TrackMania trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class GeometryAsset(Protocol):
    map_uid: str
    sha256: str


@dataclass(frozen=True, slots=True)
class _PaceData:
    values: np.ndarray
    points: np.ndarray
    finish_time_s: float
    velocity_scale: float


@dataclass(frozen=True, slots=True)
class PaceDemonstrationRequest:
    path: str | Path
    geometry: GeometryAsset
    trajectory: np.ndarray
    velocity_to_mps_scale: float = 0.001


@dataclass(frozen=True, slots=True)
class PaceFrameRequest:
    frames: np.ndarray
    trajectory: np.ndarray
    finish_time_s: float
    velocity_to_mps_scale: float = 0.001


@dataclass(frozen=True, slots=True)
class ReferencePaceProfile:
    """Reference elapsed time for each monotonic trajectory index."""

    reference_times_s: np.ndarray
    reference_speeds_mps: np.ndarray | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.reference_times_s, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("reference pace must contain at least two finite times")
        if np.any(np.diff(values) < 0.0):
            raise ValueError("reference pace times must be monotonic")
        object.__setattr__(self, "reference_times_s", values)
        if self.reference_speeds_mps is not None:
            speeds = np.asarray(self.reference_speeds_mps, dtype=np.float64)
            if speeds.shape != values.shape or not np.isfinite(speeds).all():
                raise ValueError("reference speeds must match reference pace times")
            object.__setattr__(self, "reference_speeds_mps", np.maximum(speeds, 0.0))

    @classmethod
    def from_demonstration(cls, request: PaceDemonstrationRequest) -> ReferencePaceProfile:
        frames, finish_time_s = _load_pace_reference(request.path, request.geometry)
        return cls.from_frames(
            PaceFrameRequest(
                frames,
                request.trajectory,
                finish_time_s,
                request.velocity_to_mps_scale,
            )
        )

    @classmethod
    def from_frames(cls, request: PaceFrameRequest) -> ReferencePaceProfile:
        values = np.asarray(request.frames, dtype=np.float32)
        points = np.asarray(request.trajectory, dtype=np.float32)
        data = _PaceData(values, points, request.finish_time_s, request.velocity_to_mps_scale)
        _validate_pace_inputs(data)
        race_times_s = _validated_race_times(values, request.finish_time_s)
        indices = _monotonic_projection(values[:, 4:7], points[:, :3], race_times_s)
        _validate_reaches_end(indices, len(points))
        sampled_indices, first_samples = np.unique(indices, return_index=True)
        profile = _interpolated_times(points, sampled_indices, race_times_s[first_samples])
        profile[-1] = request.finish_time_s
        speeds = _interpolated_speeds(data, sampled_indices, first_samples)
        return cls(profile, speeds)

    def time_at_index(self, index: int) -> float:
        bounded = min(max(index, 0), len(self.reference_times_s) - 1)
        return float(self.reference_times_s[bounded])

    def speed_at_index(self, index: int) -> float:
        if self.reference_speeds_mps is None:
            return 0.0
        bounded = min(max(index, 0), len(self.reference_speeds_mps) - 1)
        return float(self.reference_speeds_mps[bounded])


def _load_pace_reference(path: str | Path, geometry: GeometryAsset) -> tuple[np.ndarray, float]:
    with np.load(Path(path), allow_pickle=False) as archive:
        _validate_reference_archive(archive.files, "pace")
        if str(archive["map_uid"].item()) != geometry.map_uid:
            raise ValueError("pace reference map UID does not match the configured geometry")
        if str(archive["geometry_sha256"].item()) != geometry.sha256:
            raise ValueError("pace reference geometry hash does not match the configured geometry")
        frames = np.asarray(archive["frames"], dtype=np.float32)
        return frames, float(archive["finish_time_s"].item())


def _validate_pace_inputs(data: _PaceData) -> None:
    values, points = data.values, data.points
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 7:
        raise ValueError("pace reference frames must have shape (frames >= 2, fields >= 7)")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("pace trajectory must have shape (points >= 2, coordinates >= 3)")
    if data.finish_time_s <= 0.0 or not np.isfinite(data.finish_time_s):
        raise ValueError("pace reference finish time must be positive and finite")
    if not np.isfinite(data.velocity_scale) or data.velocity_scale <= 0.0:
        raise ValueError("velocity_to_mps_scale must be finite and positive")
    if not np.isfinite(points).all():
        raise ValueError("pace trajectory must contain finite points")


def _validated_race_times(values: np.ndarray, finish_time_s: float) -> np.ndarray:
    race_times_s = values[:, 3].astype(np.float64) / 1_000.0
    if not np.isfinite(values).all() or np.any(np.diff(race_times_s) <= 0.0):
        raise ValueError("pace reference frames contain invalid race times")
    if np.any(values[:-1, 2]) or not bool(values[-1, 2]):
        raise ValueError("pace reference must end with exactly one finish frame")
    if abs(float(race_times_s[-1]) - finish_time_s) > 0.05:
        raise ValueError("pace reference finish time does not match its final frame")
    race_times_s[-1] = finish_time_s
    if np.any(np.diff(race_times_s) <= 0.0):
        raise ValueError("pace reference finish metadata is not monotonic")
    return race_times_s


def _validate_reaches_end(indices: np.ndarray, point_count: int) -> None:
    if indices[-1] < point_count - 2:
        raise ValueError("pace reference does not reach the end of the configured trajectory")


def _interpolated_times(
    points: np.ndarray, sampled_indices: np.ndarray, sampled_times: np.ndarray
) -> np.ndarray:
    targets = np.arange(len(points), dtype=np.float64)
    values = np.interp(targets, sampled_indices, sampled_times)
    return np.asarray(values, dtype=np.float64)


def _interpolated_speeds(
    data: _PaceData,
    sampled_indices: np.ndarray,
    first_samples: np.ndarray,
) -> np.ndarray | None:
    if data.values.shape[1] <= 16:
        return None
    targets = np.arange(len(data.points), dtype=np.float64)
    sampled_speeds = data.values[first_samples, 16] * data.velocity_scale
    return _smooth(np.interp(targets, sampled_indices, sampled_speeds))


def _validate_reference_archive(files: list[str], name: str) -> None:
    required = {"map_uid", "geometry_sha256", "frames", "finish_time_s"}
    missing = required - set(files)
    if missing:
        raise ValueError(f"{name} reference is missing keys: {sorted(missing)}")


def demonstration_guidance_line(
    path: str | Path,
    geometry: GeometryAsset,
    trajectory: np.ndarray,
) -> np.ndarray:
    """Align a complete expert lap to monotonic geometry indices."""

    frames, finish_time_s = _load_guidance_reference(path, geometry)
    points = np.asarray(trajectory, dtype=np.float32)
    indices = _guidance_projection(frames, points, finish_time_s)
    if indices[0] > 1 or indices[-1] < len(points) - 2:
        raise ValueError("guidance reference does not contain a complete lap")
    return _interpolate_guidance_positions(frames[:, 4:7], indices, len(points))


def _load_guidance_reference(path: str | Path, geometry: GeometryAsset) -> tuple[np.ndarray, float]:
    with np.load(Path(path), allow_pickle=False) as archive:
        _validate_reference_archive(archive.files, "guidance")
        if str(archive["map_uid"].item()) != geometry.map_uid:
            raise ValueError("guidance reference map UID does not match the configured geometry")
        if str(archive["geometry_sha256"].item()) != geometry.sha256:
            raise ValueError(
                "guidance reference geometry hash does not match the configured geometry"
            )
        return (
            np.asarray(archive["frames"], dtype=np.float32),
            float(archive["finish_time_s"].item()),
        )


def _guidance_projection(
    frames: np.ndarray, points: np.ndarray, finish_time_s: float
) -> np.ndarray:
    if frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] < 7:
        raise ValueError("guidance reference frames must have shape (frames >= 2, fields >= 7)")
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("guidance trajectory must have shape (points >= 2, coordinates >= 3)")
    if not np.isfinite(frames).all() or not np.isfinite(points).all():
        raise ValueError("guidance reference frames and trajectory must be finite")
    if finish_time_s <= 0.0 or not np.isfinite(finish_time_s):
        raise ValueError("guidance reference finish time must be positive and finite")
    race_times_s = frames[:, 3].astype(np.float64) / 1_000.0
    if np.any(np.diff(race_times_s) < 0.0):
        raise ValueError("guidance reference race times must be monotonic")
    return _monotonic_projection(frames[:, 4:7], points[:, :3], race_times_s)


def _interpolate_guidance_positions(
    positions: np.ndarray, indices: np.ndarray, point_count: int
) -> np.ndarray:
    sampled_indices = np.unique(indices)
    sampled_positions = np.stack(
        [positions[indices == index].mean(axis=0) for index in sampled_indices]
    )
    target_indices = np.arange(point_count, dtype=np.float64)
    return np.stack(
        [
            np.interp(target_indices, sampled_indices, sampled_positions[:, coordinate])
            for coordinate in range(3)
        ],
        axis=1,
    ).astype(np.float32)


def _monotonic_projection(
    positions: np.ndarray, trajectory: np.ndarray, race_times_s: np.ndarray
) -> np.ndarray:
    indices = np.empty(len(positions), dtype=np.intp)
    cumulative_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))]
    previous = 0
    for frame_index, position in enumerate(positions):
        elapsed_s = (
            0.0 if frame_index == 0 else race_times_s[frame_index] - race_times_s[frame_index - 1]
        )
        max_distance = cumulative_distance[previous] + 100.0 * max(0.0, elapsed_s) + 2.0
        stop = min(
            len(trajectory),
            int(np.searchsorted(cumulative_distance, max_distance, side="right")),
        )
        distances = np.sum((trajectory[previous:stop] - position) ** 2, axis=1)
        previous += int(np.argmin(distances))
        indices[frame_index] = previous
    return indices


def _smooth(values: np.ndarray, window: int = 9) -> np.ndarray:
    if len(values) < window:
        return np.asarray(values, dtype=np.float64)
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")
