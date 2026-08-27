"""Run-length encoded control schedules for trajectory optimization."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

SCHEDULE_FORMAT = "trackmaniarl-trajectory-schedule-v1"


@dataclass(frozen=True, slots=True)
class SlowControlWindow:
    """A contiguous interval where the expert releases gas or applies brake."""

    first_segment: int
    stop_segment: int


@dataclass(frozen=True, slots=True)
class TrajectorySchedule:
    """Run-length encoded expert controls with optimizable switch boundaries."""

    boundaries: np.ndarray
    segment_controls: np.ndarray
    boundary_offsets: np.ndarray

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.boundaries, dtype=np.int64).copy()
        controls = np.asarray(self.segment_controls, dtype=np.float32).copy()
        offsets = np.asarray(self.boundary_offsets, dtype=np.int64).copy()
        if boundaries.ndim != 1 or len(boundaries) < 2:
            raise ValueError("trajectory schedule requires at least one control segment")
        segment_count = len(boundaries) - 1
        if controls.shape != (segment_count, 3):
            raise ValueError("trajectory schedule controls must have shape (segments, 3)")
        if offsets.shape != (max(0, segment_count - 1),):
            raise ValueError("trajectory schedule requires one offset per internal boundary")
        if boundaries[0] != 0 or np.any(np.diff(boundaries) <= 0):
            raise ValueError("trajectory schedule boundaries must start at zero and increase")
        if not np.isfinite(controls).all():
            raise ValueError("trajectory schedule controls must be finite")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "segment_controls", controls)
        object.__setattr__(self, "boundary_offsets", offsets)
        self.effective_boundaries()

    @classmethod
    def from_controls(cls, controls: np.ndarray) -> TrajectorySchedule:
        values = np.asarray(controls, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError("trajectory controls must have shape (steps, 3)")
        changes = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
        boundaries = np.concatenate(([0], changes, [len(values)])).astype(np.int64)
        return cls(
            boundaries,
            values[boundaries[:-1]],
            np.zeros(max(0, len(boundaries) - 2), dtype=np.int64),
        )

    @property
    def step_count(self) -> int:
        return int(self.boundaries[-1])

    def effective_boundaries(self) -> np.ndarray:
        effective = self.boundaries.copy()
        effective[1:-1] += self.boundary_offsets
        if np.any(np.diff(effective) <= 0):
            raise ValueError("trajectory boundary offsets collapse a control segment")
        return effective

    def materialize(self) -> np.ndarray:
        durations = np.diff(self.effective_boundaries())
        controls = np.repeat(self.segment_controls, durations, axis=0)
        if controls.shape != (self.step_count, 3):
            raise AssertionError("trajectory schedule changed its total duration")
        return controls

    def source_controls(self) -> np.ndarray:
        return np.repeat(self.segment_controls, np.diff(self.boundaries), axis=0)

    def slow_windows(self, *, minimum_ticks: int = 3) -> tuple[SlowControlWindow, ...]:
        if minimum_ticks < 1:
            raise ValueError("minimum_ticks must be positive")
        slow = (self.segment_controls[:, 0] < 0.5) | (self.segment_controls[:, 1] > 0.5)
        boundaries = self.effective_boundaries()
        windows: list[SlowControlWindow] = []
        first = 0
        while first < len(slow):
            if not slow[first]:
                first += 1
                continue
            stop = first + 1
            while stop < len(slow) and slow[stop]:
                stop += 1
            if int(boundaries[stop] - boundaries[first]) >= minimum_ticks:
                windows.append(SlowControlWindow(first, stop))
            first = stop
        return tuple(windows)

    def shorten(
        self,
        window: SlowControlWindow,
        side: Literal["start", "end"],
        ticks: int,
    ) -> TrajectorySchedule:
        if ticks < 1:
            raise ValueError("trajectory shortening must use a positive tick count")
        segment_count = len(self.segment_controls)
        if not 0 <= window.first_segment < window.stop_segment <= segment_count:
            raise ValueError("slow-control window lies outside the trajectory schedule")
        boundary = window.first_segment if side == "start" else window.stop_segment
        if boundary in {0, segment_count}:
            raise ValueError("the first or last control window cannot be shortened on this side")
        offsets = self.boundary_offsets.copy()
        offsets[boundary - 1] += ticks if side == "start" else -ticks
        return replace(self, boundary_offsets=offsets)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.tmp.npz")
        np.savez_compressed(
            temporary,
            format=np.asarray(SCHEDULE_FORMAT),
            boundaries=self.boundaries,
            segment_controls=self.segment_controls,
            boundary_offsets=self.boundary_offsets,
        )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrajectorySchedule:
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"].item()) != SCHEDULE_FORMAT:
                raise ValueError("unsupported trajectory schedule format")
            return cls(
                boundaries=np.asarray(data["boundaries"], dtype=np.int64),
                segment_controls=np.asarray(data["segment_controls"], dtype=np.float32),
                boundary_offsets=np.asarray(data["boundary_offsets"], dtype=np.int64),
            )
