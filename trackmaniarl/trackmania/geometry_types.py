"""Typed inputs for offline TrackMania geometry construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GeometryBuildRequest:
    output: Path
    left_recording: Path
    right_recording: Path
    map_uid: str
    map_path: Path
    spacing_m: float = 2.0
    smooth_window: int = 5
    lookahead_points: int = 60
