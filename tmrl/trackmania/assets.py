"""Small asset-recording utilities for first-party TrackMania projects."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from tmrl.trackmania.telemetry import DEFAULT_POSITION_INDICES, TelemetryFrame


class TelemetryReader(Protocol):
    """Supplies validated telemetry frames for asset recording."""

    def read(self) -> TelemetryFrame: ...


def record_trajectory(
    output: str | Path,
    client: TelemetryReader,
    *,
    samples: int = 2_000,
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES,
) -> Path:
    """Record finite XYZ telemetry samples into a portable CSV trajectory asset."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    points = np.asarray([client.read().values[list(position_indices)] for _ in range(samples)])
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(target, points, delimiter=",")
    return target


def record_boundary(
    output: str | Path,
    client: TelemetryReader,
    *,
    samples: int = 2_000,
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES,
) -> Path:
    """Record one manually driven left or right map boundary as ``.npy`` XYZ samples."""

    if samples < 2:
        raise ValueError("samples must be at least two")
    points = np.asarray(
        [client.read().values[list(position_indices)] for _ in range(samples)], dtype=np.float32
    )
    if not np.isfinite(points).all():
        raise ValueError("boundary recording contains non-finite telemetry positions")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, points)
    return target
