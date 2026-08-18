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
    def from_demonstration(
        cls,
        path: str | Path,
        geometry: GeometryAsset,
        trajectory: np.ndarray,
    ) -> ReferencePaceProfile:
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            required = {"map_uid", "geometry_sha256", "frames", "finish_time_s"}
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"pace reference is missing keys: {sorted(missing)}")
            if str(archive["map_uid"].item()) != geometry.map_uid:
                raise ValueError("pace reference map UID does not match the configured geometry")
            if str(archive["geometry_sha256"].item()) != geometry.sha256:
                raise ValueError(
                    "pace reference geometry hash does not match the configured geometry"
                )
            frames = np.asarray(archive["frames"], dtype=np.float32)
            finish_time_s = float(archive["finish_time_s"].item())
        return cls.from_frames(frames, trajectory, finish_time_s=finish_time_s)

    @classmethod
    def from_frames(
        cls,
        frames: np.ndarray,
        trajectory: np.ndarray,
        *,
        finish_time_s: float,
    ) -> ReferencePaceProfile:
        values = np.asarray(frames, dtype=np.float32)
        points = np.asarray(trajectory, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 7:
            raise ValueError("pace reference frames must have shape (frames >= 2, fields >= 7)")
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
            raise ValueError("pace trajectory must have shape (points >= 2, coordinates >= 3)")
        if finish_time_s <= 0.0 or not np.isfinite(finish_time_s):
            raise ValueError("pace reference finish time must be positive and finite")
        race_times_s = values[:, 3].astype(np.float64) / 1_000.0
        if not np.isfinite(values).all() or np.any(np.diff(race_times_s) < 0.0):
            raise ValueError("pace reference frames contain invalid race times")
        indices = _monotonic_projection(values[:, 4:7], points[:, :3], race_times_s)
        if indices[-1] < len(points) - 2:
            raise ValueError("pace reference does not reach the end of the configured trajectory")
        sampled_indices, first_samples = np.unique(indices, return_index=True)
        sampled_times = race_times_s[first_samples]
        profile = np.interp(
            np.arange(len(points), dtype=np.float64), sampled_indices, sampled_times
        )
        profile[-1] = finish_time_s
        sampled_speeds = values[first_samples, 16] if values.shape[1] > 16 else None
        speeds = (
            None
            if sampled_speeds is None
            else _smooth(
                np.interp(
                    np.arange(len(points), dtype=np.float64),
                    sampled_indices,
                    sampled_speeds,
                )
            )
        )
        return cls(np.maximum.accumulate(profile), speeds)

    def time_at_index(self, index: int) -> float:
        bounded = min(max(index, 0), len(self.reference_times_s) - 1)
        return float(self.reference_times_s[bounded])

    def speed_at_index(self, index: int) -> float:
        if self.reference_speeds_mps is None:
            return 0.0
        bounded = min(max(index, 0), len(self.reference_speeds_mps) - 1)
        return float(self.reference_speeds_mps[bounded])


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
