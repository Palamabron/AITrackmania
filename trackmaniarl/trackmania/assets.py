"""Small asset-recording utilities for first-party TrackMania projects."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

import numpy as np

from trackmaniarl.trackmania.telemetry import DEFAULT_POSITION_INDICES, TelemetryFrame


class TelemetryReader(Protocol):
    """Supplies validated telemetry frames for asset recording."""

    def read(self) -> TelemetryFrame: ...


def record_trajectory(
    output: str | Path,
    client: TelemetryReader,
    *,
    samples: int = 2_000,
    sample_interval_s: float = 0.0,
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES,
) -> Path:
    """Record finite XYZ telemetry samples into a portable CSV trajectory asset."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    if sample_interval_s < 0.0:
        raise ValueError("sample_interval_s must not be negative")
    points = np.asarray(
        [_read_position(client, position_indices, sample_interval_s) for _ in range(samples)]
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, points, delimiter=",")
    return target


def record_boundary(
    output: str | Path,
    client: TelemetryReader,
    *,
    max_duration_s: float = 600.0,
    minimum_spacing_m: float = 0.25,
    finish_index: int = 2,
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES,
    status: Callable[[str], None] | None = None,
) -> Path:
    """Record one manually driven boundary until TrackMania reports a finish."""

    if max_duration_s <= 0.0 or minimum_spacing_m <= 0.0:
        raise ValueError("max_duration_s and minimum_spacing_m must be positive")
    points: list[np.ndarray] = []
    active_run = False
    initial_position: np.ndarray | None = None
    last_recorded_position: np.ndarray | None = None
    started_at = monotonic()
    if status is not None:
        status("Waiting for an active run; restart the map if it is on the finish screen.")
    sample_index = 0
    while monotonic() - started_at < max_duration_s:
        frame = client.read()
        sample_index += 1
        finished = bool(frame.values[finish_index])
        position = frame.values[list(position_indices)]
        if initial_position is None:
            initial_position = position.copy()
        moved_m = float(np.linalg.norm(position - initial_position))
        if not active_run:
            if status is not None and sample_index > 0 and sample_index % 500 == 0:
                status(
                    f"Still waiting: race time {frame.values[3]:.0f} ms, "
                    f"finished={finished}, moved {moved_m:.1f} m."
                )
            if finished or (frame.values[3] <= 0.0 and moved_m < 1.0):
                continue
            active_run = True
            if status is not None:
                status(
                    f"Recording started at race time {frame.values[3]:.0f} ms "
                    f"after moving {moved_m:.1f} m."
                )
        if (
            last_recorded_position is None
            or np.linalg.norm(position - last_recorded_position) >= minimum_spacing_m
        ):
            points.append(position.copy())
            last_recorded_position = position.copy()
        if len(points) > 1 and finished:
            if status is not None:
                status(
                    f"Finish detected at race time {frame.values[3]:.0f} ms "
                    f"after {len(points)} samples."
                )
            break
    else:
        state = "before a new run started" if not active_run else "before the finish"
        raise RuntimeError(f"boundary recording exceeded {max_duration_s:.0f} seconds {state}")
    points_array = np.asarray(points, dtype=np.float32)
    if not np.isfinite(points_array).all():
        raise ValueError("boundary recording contains non-finite telemetry positions")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, points_array)
    return target


def _read_position(
    client: TelemetryReader,
    position_indices: tuple[int, int, int],
    sample_interval_s: float,
) -> np.ndarray:
    position = client.read().values[list(position_indices)]
    if sample_interval_s:
        sleep(sample_interval_s)
    return position
