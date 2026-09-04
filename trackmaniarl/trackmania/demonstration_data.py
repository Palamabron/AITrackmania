"""Demonstration data contracts and archive I/O."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, TelemetryFrame

DEMONSTRATION_FORMAT = "trackmaniarl-trackmania-demo-v5"
CONTROL_INDICES = (31, 32, 30)


class TelemetryReader(Protocol):
    def read(self) -> TelemetryFrame: ...

    def read_next(self) -> TelemetryFrame: ...


@dataclass(frozen=True, slots=True)
class Demonstration:
    map_uid: str
    geometry_sha256: str
    action_repeat_frames: int
    frames: np.ndarray
    actions: np.ndarray
    controls: np.ndarray
    finish_time_s: float
    decision_interval_ms: float | None = None
    control_alignment: str = "frame_start"

    def __post_init__(self) -> None:
        _validate_identity(self)
        _validate_shapes(self)
        _validate_timing(self)
        _validate_actions(self)
        _validate_race(self)


def _validate_identity(demonstration: Demonstration) -> None:
    if not demonstration.map_uid or len(demonstration.geometry_sha256) != 64:
        raise ValueError("demonstration map identity metadata is invalid")


def _validate_shapes(demonstration: Demonstration) -> None:
    frames = demonstration.frames
    if frames.ndim != 2 or len(frames) < 2 or frames.shape[1] != DEFAULT_TELEMETRY_FIELD_COUNT:
        raise ValueError("demonstration frames must have shape (steps + 1, 33)")
    if demonstration.actions.shape != (len(frames) - 1,):
        raise ValueError("demonstration actions must contain one action per transition")
    if demonstration.controls.shape != (len(demonstration.actions), 3):
        raise ValueError("demonstration controls must have shape (transitions, 3)")
    if not np.isfinite(frames).all() or not np.isfinite(demonstration.controls).all():
        raise ValueError("demonstration contains non-finite values")


def _validate_timing(demonstration: Demonstration) -> None:
    if demonstration.action_repeat_frames < 1 or demonstration.finish_time_s <= 0.0:
        raise ValueError("demonstration timing metadata is invalid")
    interval = demonstration.decision_interval_ms
    if interval is not None and (not np.isfinite(interval) or interval <= 0.0):
        raise ValueError("demonstration decision interval must be finite and positive")
    if demonstration.control_alignment != "frame_start":
        raise ValueError("demonstration control alignment must be 'frame_start'")


def _validate_actions(demonstration: Demonstration) -> None:
    action_count, table = build_brake_tap_action_table()
    if np.any(demonstration.actions < 0) or np.any(demonstration.actions >= action_count):
        raise ValueError("demonstration contains an invalid discrete action")
    quantized = continuous_control_to_discrete_indices_batch(demonstration.controls, table)
    if not np.array_equal(demonstration.actions, quantized):
        raise ValueError("demonstration actions do not match the recorded controls")


def _validate_race(demonstration: Demonstration) -> None:
    race_times = demonstration.frames[:, 3]
    if np.any(np.diff(race_times) <= 0.0):
        raise ValueError("demonstration race time must increase without a restart")
    if np.any(demonstration.frames[:-1, 2]) or not bool(demonstration.frames[-1, 2]):
        raise ValueError("demonstration does not end with a finish frame")
    if abs(float(race_times[-1]) / 1_000.0 - demonstration.finish_time_s) > 0.05:
        raise ValueError("demonstration finish time does not match its final frame")


def save_demonstration(path: str | Path, demonstration: Demonstration) -> Path:
    target = _archive_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **_archive_arrays(demonstration))
    return target


def _archive_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix.lower() == ".npz" else Path(f"{target}.npz")


def _archive_arrays(demonstration: Demonstration) -> dict[str, Any]:
    return {
        "format": np.asarray(DEMONSTRATION_FORMAT),
        "map_uid": np.asarray(demonstration.map_uid),
        "geometry_sha256": np.asarray(demonstration.geometry_sha256),
        "action_repeat_frames": np.asarray(demonstration.action_repeat_frames, dtype=np.int32),
        "decision_interval_ms": np.asarray(demonstration.decision_interval_ms or 0.0),
        "control_alignment": np.asarray(demonstration.control_alignment),
        "frames": np.asarray(demonstration.frames, dtype=np.float32),
        "actions": np.asarray(demonstration.actions, dtype=np.int64),
        "controls": np.asarray(demonstration.controls, dtype=np.float32),
        "finish_time_s": np.asarray(demonstration.finish_time_s, dtype=np.float64),
    }


def resolve_demonstration_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand ``--demo`` arguments: directories load every ``*.npz``, files stay as-is."""

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        for candidate in _resolve_candidates(Path(raw)):
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)
    return tuple(resolved)


def _resolve_candidates(path: Path) -> list[Path]:
    if path.is_dir():
        matches = sorted(item.resolve() for item in path.rglob("*.npz") if item.is_file())
        if not matches:
            raise FileNotFoundError(f"demonstration directory has no .npz files: {path}")
        return matches
    if not path.is_file():
        raise FileNotFoundError(f"demonstration path does not exist: {path}")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"demonstration file must be a .npz archive: {path}")
    return [path.resolve()]


def load_demonstration(path: str | Path) -> Demonstration:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        _validate_archive(data)
        return _demonstration_from_archive(data)


def _validate_archive(data: Any) -> None:
    required = {
        "format",
        "map_uid",
        "geometry_sha256",
        "action_repeat_frames",
        "decision_interval_ms",
        "control_alignment",
        "frames",
        "actions",
        "controls",
        "finish_time_s",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"demonstration is missing keys: {sorted(missing)}")
    format_name = str(data["format"].item())
    if format_name != DEMONSTRATION_FORMAT:
        raise ValueError("unsupported TrackMania demonstration format")


def _demonstration_from_archive(data: Any) -> Demonstration:
    return Demonstration(
        map_uid=str(data["map_uid"].item()),
        geometry_sha256=str(data["geometry_sha256"].item()),
        action_repeat_frames=int(data["action_repeat_frames"].item()),
        frames=np.asarray(data["frames"], dtype=np.float32),
        actions=np.asarray(data["actions"], dtype=np.int64),
        controls=np.asarray(data["controls"], dtype=np.float32),
        finish_time_s=float(data["finish_time_s"].item()),
        decision_interval_ms=float(data["decision_interval_ms"].item()) or None,
        control_alignment=str(data["control_alignment"].item()),
    )


def _control(frame: TelemetryFrame) -> np.ndarray:
    values = frame.values[list(CONTROL_INDICES)]
    return np.asarray(
        [
            np.clip(values[0], 0.0, 1.0),
            np.clip(values[1], 0.0, 1.0),
            np.clip(values[2], -1.0, 1.0),
        ],
        dtype=np.float32,
    )
