"""Small asset-recording utilities for first-party TrackMania projects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

import numpy as np

from trackmaniarl.trackmania.telemetry import DEFAULT_POSITION_INDICES, TelemetryFrame


class TelemetryReader(Protocol):
    """Supplies validated telemetry frames for asset recording."""

    def read(self) -> TelemetryFrame: ...


@dataclass(frozen=True, slots=True)
class BoundaryRecordingRequest:
    output: Path
    client: TelemetryReader
    max_duration_s: float = 600.0
    minimum_spacing_m: float = 0.25
    finish_index: int = 2
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES
    status: Callable[[str], None] | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryRecordingRequest:
    output: Path
    client: TelemetryReader
    samples: int = 2_000
    sample_interval_s: float = 0.0
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES


@dataclass(slots=True)
class _BoundaryProgress:
    points: list[np.ndarray] = field(default_factory=list)
    active_run: bool = False
    initial_position: np.ndarray | None = None
    last_position: np.ndarray | None = None
    started_at: float = field(default_factory=monotonic)
    sample_index: int = 0


@dataclass(frozen=True, slots=True)
class _BoundarySample:
    frame: TelemetryFrame
    position: np.ndarray
    finished: bool
    moved_m: float


class _RecordingState(Enum):
    WAITING = auto()
    RECORDING = auto()
    FINISHED = auto()


def record_trajectory(request: TrajectoryRecordingRequest) -> Path:
    """Record finite XYZ telemetry samples into a portable CSV trajectory asset."""
    _validate_trajectory_request(request.samples, request.sample_interval_s)
    points = np.asarray(
        [
            _read_position(request.client, request.position_indices, request.sample_interval_s)
            for _ in range(request.samples)
        ]
    )
    request.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(request.output, points, delimiter=",")
    return request.output


def _validate_trajectory_request(samples: int, sample_interval_s: float) -> None:
    if samples < 2:
        raise ValueError("samples must be at least two")
    if sample_interval_s < 0.0:
        raise ValueError("sample_interval_s must not be negative")


def record_boundary(request: BoundaryRecordingRequest) -> Path:
    return _record_boundary(request)


def _record_boundary(request: BoundaryRecordingRequest) -> Path:
    if request.max_duration_s <= 0.0 or request.minimum_spacing_m <= 0.0:
        raise ValueError("max_duration_s and minimum_spacing_m must be positive")
    progress = _BoundaryProgress()
    _emit(request, "Waiting for an active run; restart the map if it is on the finish screen.")
    while monotonic() - progress.started_at < request.max_duration_s:
        sample = _boundary_sample(request, progress)
        if _process_sample(request, progress, sample) is _RecordingState.FINISHED:
            return _save_boundary(request.output, progress.points)
    state = "before a new run started" if not progress.active_run else "before the finish"
    raise RuntimeError(f"boundary recording exceeded {request.max_duration_s:.0f} seconds {state}")


def _boundary_sample(
    request: BoundaryRecordingRequest, progress: _BoundaryProgress
) -> _BoundarySample:
    frame = request.client.read()
    progress.sample_index += 1
    position = frame.values[list(request.position_indices)]
    if progress.initial_position is None:
        progress.initial_position = position.copy()
    moved_m = float(np.linalg.norm(position - progress.initial_position))
    return _BoundarySample(frame, position, bool(frame.values[request.finish_index]), moved_m)


def _process_sample(
    request: BoundaryRecordingRequest, progress: _BoundaryProgress, sample: _BoundarySample
) -> _RecordingState:
    if not progress.active_run:
        state = _start_recording(request, progress, sample)
        if state is _RecordingState.WAITING:
            return state
    _append_boundary_point(request, progress, sample.position)
    if len(progress.points) > 1 and sample.finished:
        _emit(
            request,
            f"Finish detected at race time {sample.frame.values[3]:.0f} ms "
            f"after {len(progress.points)} samples.",
        )
        return _RecordingState.FINISHED
    return _RecordingState.RECORDING


def _start_recording(
    request: BoundaryRecordingRequest, progress: _BoundaryProgress, sample: _BoundarySample
) -> _RecordingState:
    if progress.sample_index % 500 == 0:
        _emit(
            request,
            f"Still waiting: race time {sample.frame.values[3]:.0f} ms, "
            f"finished={sample.finished}, moved {sample.moved_m:.1f} m.",
        )
    if sample.finished or (sample.frame.values[3] <= 0.0 and sample.moved_m < 1.0):
        return _RecordingState.WAITING
    progress.active_run = True
    _emit(
        request,
        f"Recording started at race time {sample.frame.values[3]:.0f} ms "
        f"after moving {sample.moved_m:.1f} m.",
    )
    return _RecordingState.RECORDING


def _append_boundary_point(
    request: BoundaryRecordingRequest, progress: _BoundaryProgress, position: np.ndarray
) -> None:
    if progress.last_position is not None:
        distance = float(np.linalg.norm(position - progress.last_position))
        if distance < request.minimum_spacing_m:
            return
    progress.points.append(position.copy())
    progress.last_position = position.copy()


def _emit(request: BoundaryRecordingRequest, message: str) -> None:
    if request.status is not None:
        request.status(message)


def _save_boundary(output: Path, points: list[np.ndarray]) -> Path:
    points_array = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points_array).all():
        raise ValueError("boundary recording contains non-finite telemetry positions")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, points_array)
    return output


def _read_position(
    client: TelemetryReader,
    position_indices: tuple[int, int, int],
    sample_interval_s: float,
) -> np.ndarray:
    position = client.read().values[list(position_indices)]
    if sample_interval_s:
        sleep(sample_interval_s)
    return position
