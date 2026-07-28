"""Validated human-driving demonstrations for TrackMania replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

import numpy as np

from tmrl.core.contracts import FeaturePipeline
from tmrl.core.data import Transition
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
    continuous_control_to_discrete_indices_batch,
)
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig
from tmrl.trackmania.geometry import BoundaryGeometry, file_sha256
from tmrl.trackmania.reward import TrajectoryReward
from tmrl.trackmania.telemetry import TelemetryFrame

DEMONSTRATION_FORMAT = "tmrl-trackmania-demo-v1"
CONTROL_INDICES = (31, 32, 30)


class TelemetryReader(Protocol):
    def read(self) -> TelemetryFrame: ...


@dataclass(frozen=True, slots=True)
class Demonstration:
    map_uid: str
    geometry_sha256: str
    action_repeat_frames: int
    frames: np.ndarray
    actions: np.ndarray
    controls: np.ndarray
    finish_time_s: float

    def __post_init__(self) -> None:
        if not self.map_uid or len(self.geometry_sha256) != 64:
            raise ValueError("demonstration map identity metadata is invalid")
        if self.frames.ndim != 2 or len(self.frames) < 2 or self.frames.shape[1] < 33:
            raise ValueError("demonstration frames must have shape (steps + 1, fields >= 33)")
        if self.actions.shape != (len(self.frames) - 1,):
            raise ValueError("demonstration actions must contain one action per transition")
        if self.controls.shape != (len(self.actions), 3):
            raise ValueError("demonstration controls must have shape (transitions, 3)")
        if not np.isfinite(self.frames).all() or not np.isfinite(self.controls).all():
            raise ValueError("demonstration contains non-finite values")
        if self.action_repeat_frames < 1 or self.finish_time_s <= 0.0:
            raise ValueError("demonstration timing metadata is invalid")
        action_count, table = build_brake_tap_action_table()
        if np.any(self.actions < 0) or np.any(self.actions >= action_count):
            raise ValueError("demonstration contains an invalid discrete action")
        quantized = continuous_control_to_discrete_indices_batch(self.controls, table)
        if not np.array_equal(self.actions, quantized):
            raise ValueError("demonstration actions do not match the recorded controls")
        race_times = self.frames[:, 3]
        if np.any(np.diff(race_times) <= 0.0):
            raise ValueError("demonstration race time must increase without a restart")
        if np.any(self.frames[:-1, 2]) or not bool(self.frames[-1, 2]):
            raise ValueError("demonstration does not end with a finish frame")
        if abs(float(race_times[-1]) / 1_000.0 - self.finish_time_s) > 0.05:
            raise ValueError("demonstration finish time does not match its final frame")


def save_demonstration(path: str | Path, demonstration: Demonstration) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".npz":
        target = target.with_suffix(".npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format=np.asarray(DEMONSTRATION_FORMAT),
        map_uid=np.asarray(demonstration.map_uid),
        geometry_sha256=np.asarray(demonstration.geometry_sha256),
        action_repeat_frames=np.asarray(demonstration.action_repeat_frames, dtype=np.int32),
        frames=np.asarray(demonstration.frames, dtype=np.float32),
        actions=np.asarray(demonstration.actions, dtype=np.int64),
        controls=np.asarray(demonstration.controls, dtype=np.float32),
        finish_time_s=np.asarray(demonstration.finish_time_s, dtype=np.float64),
    )
    return target


def resolve_demonstration_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    """Expand ``--demo`` arguments: directories load every ``*.npz``, files stay as-is."""

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            matches = sorted(item.resolve() for item in path.glob("*.npz") if item.is_file())
            if not matches:
                raise FileNotFoundError(f"demonstration directory has no .npz files: {path}")
            candidates = matches
        elif path.is_file():
            if path.suffix.lower() != ".npz":
                raise ValueError(f"demonstration file must be a .npz archive: {path}")
            candidates = [path.resolve()]
        else:
            raise FileNotFoundError(f"demonstration path does not exist: {path}")
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)
    return tuple(resolved)


def load_demonstration(path: str | Path) -> Demonstration:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {
            "format",
            "map_uid",
            "geometry_sha256",
            "action_repeat_frames",
            "frames",
            "actions",
            "controls",
            "finish_time_s",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"demonstration is missing keys: {sorted(missing)}")
        if str(data["format"].item()) != DEMONSTRATION_FORMAT:
            raise ValueError("unsupported TrackMania demonstration format")
        return Demonstration(
            map_uid=str(data["map_uid"].item()),
            geometry_sha256=str(data["geometry_sha256"].item()),
            action_repeat_frames=int(data["action_repeat_frames"].item()),
            frames=np.asarray(data["frames"], dtype=np.float32),
            actions=np.asarray(data["actions"], dtype=np.int64),
            controls=np.asarray(data["controls"], dtype=np.float32),
            finish_time_s=float(data["finish_time_s"].item()),
        )


def _control(frame: TelemetryFrame) -> np.ndarray:
    values = frame.values[list(CONTROL_INDICES)]
    return np.asarray(
        [np.clip(values[0], 0.0, 1.0), np.clip(values[1], 0.0, 1.0), np.clip(values[2], -1.0, 1.0)],
        dtype=np.float32,
    )


def _wait_for_new_run(
    client: TelemetryReader, *, timeout_s: float, poll_s: float
) -> TelemetryFrame:
    previous_time = float(client.read().values[3])
    restart_observed = previous_time <= 0.0
    deadline = monotonic() + timeout_s
    while monotonic() < deadline:
        frame = client.read()
        race_time = float(frame.values[3])
        restart_observed = restart_observed or race_time < previous_time
        if restart_observed and race_time > 0.0:
            return frame
        previous_time = race_time
        if poll_s:
            sleep(poll_s)
    raise TimeoutError("no new TrackMania run was observed; restart the map and begin driving")


def record_demonstration(
    client: TelemetryReader,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    *,
    max_duration_s: float,
    status: Callable[[str], None] = print,
) -> Demonstration:
    if max_duration_s <= 0.0:
        raise ValueError("max_duration_s must be positive")
    status("Waiting for a fresh run. Restart the map, then drive the complete lap.")
    current = _wait_for_new_run(
        client, timeout_s=config.start_timeout_s, poll_s=config.start_poll_s
    )
    _, table = build_brake_tap_action_table()
    frames = [current.values.copy()]
    actions: list[int] = []
    controls: list[np.ndarray] = []
    deadline = monotonic() + max_duration_s
    while monotonic() < deadline:
        control = _control(current)
        actions.append(continuous_control_to_discrete_index(control, table))
        controls.append(control)
        for _ in range(config.action_repeat_frames):
            current = client.read()
        if float(current.values[3]) < float(frames[-1][3]):
            status("Restart detected; the partial lap was discarded. Recording the new run.")
            deadline = monotonic() + max_duration_s
            while float(current.values[3]) <= 0.0 and monotonic() < deadline:
                current = client.read()
            frames = [current.values.copy()]
            actions.clear()
            controls.clear()
            continue
        frames.append(current.values.copy())
        if bool(current.values[2]):
            finish_time_s = float(current.values[3]) / 1_000.0
            status(f"Finished demonstration in {finish_time_s:.3f}s.")
            return Demonstration(
                map_uid=geometry.map_uid,
                geometry_sha256=geometry.sha256,
                action_repeat_frames=config.action_repeat_frames,
                frames=np.asarray(frames, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                controls=np.asarray(controls, dtype=np.float32),
                finish_time_s=finish_time_s,
            )
    raise TimeoutError("demonstration did not reach the finish before max_duration_s")


def record_demonstration_session(
    client: TelemetryReader,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    *,
    count: int,
    max_duration_s: float,
    status: Callable[[str], None] = print,
) -> list[Demonstration]:
    """Record up to ``count`` finished laps, stopping early once a lap start times out."""

    if count < 1:
        raise ValueError("count must be positive")
    demonstrations: list[Demonstration] = []
    for lap in range(1, count + 1):
        status(f"Recording lap {lap} of {count}.")
        try:
            demonstrations.append(
                record_demonstration(
                    client, config, geometry, max_duration_s=max_duration_s, status=status
                )
            )
        except TimeoutError as error:
            if not demonstrations:
                raise
            status(f"Stopping the session after {len(demonstrations)} laps: {error}")
            break
    return demonstrations


def reject_outliers(
    demonstrations: Sequence[Demonstration], *, max_gap_s: float = 1.0
) -> list[Demonstration]:
    """Keep laps within ``max_gap_s`` of the best finish time, ranked fastest-first."""

    if max_gap_s < 0.0:
        raise ValueError("max_gap_s must be non-negative")
    if not demonstrations:
        return []
    best = min(demonstration.finish_time_s for demonstration in demonstrations)
    cutoff = best + max_gap_s
    return sorted(
        (
            demonstration
            for demonstration in demonstrations
            if demonstration.finish_time_s <= cutoff
        ),
        key=lambda demonstration: demonstration.finish_time_s,
    )


def validate_demonstration(
    demonstration: Demonstration,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> None:
    if demonstration.map_uid != geometry.map_uid:
        raise ValueError("demonstration map UID does not match the configured map")
    if demonstration.geometry_sha256 != geometry.sha256:
        raise ValueError("demonstration geometry hash does not match the configured geometry")
    if demonstration.action_repeat_frames != config.action_repeat_frames:
        raise ValueError("demonstration action repeat does not match the environment")
    if demonstration.frames.shape[1] != config.field_count:
        raise ValueError("demonstration telemetry schema does not match the environment")


def _reward(config: TrackmaniaEnvironmentConfig, geometry: BoundaryGeometry) -> TrajectoryReward:
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    return TrajectoryReward(reference, **config.reward_kwargs())


def demonstration_transitions(
    path: str | Path,
    pipeline: FeaturePipeline,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> list[Transition]:
    demo = load_demonstration(path)
    validate_demonstration(demo, config, geometry)
    reward = _reward(config, geometry)
    reset_pipeline = getattr(pipeline, "reset_episode", None)
    if callable(reset_pipeline):
        reset_pipeline()
    position = demo.frames[0, list(config.position_indices)]
    velocity = demo.frames[0, list(config.velocity_indices)]
    reward.reset(position, velocity=velocity, race_time_ms=float(demo.frames[0, 3]))
    prepared = pipeline.transform_observation(demo.frames[0])
    episode_id = f"demo-{file_sha256(path)[:16]}"
    transitions: list[Transition] = []
    _, table = build_brake_tap_action_table()
    for step, (action, next_frame) in enumerate(zip(demo.actions, demo.frames[1:], strict=True)):
        control = table[int(action)]
        result = reward.step(
            next_frame[list(config.position_indices)],
            finish_ui_active=bool(next_frame[2]),
            velocity=next_frame[list(config.velocity_indices)],
            race_time_ms=float(next_frame[3]),
            steering=float(control[2]),
        )
        if result.terminated and step != len(demo.actions) - 1:
            raise ValueError(f"demonstration reward terminated early: {result.reason}")
        next_prepared = pipeline.transform_observation(next_frame)
        transitions.append(
            Transition(
                observation=prepared,
                action=int(action),
                reward=result.reward,
                next_observation=next_prepared,
                terminated=result.terminated,
                truncated=False,
                info={
                    "source": "demo",
                    "is_demo": True,
                    "sampling/projected_lap_time_s": demo.finish_time_s,
                },
                episode_id=episode_id,
                step=step,
            )
        )
        prepared = next_prepared
    if not transitions[-1].terminated or result.reason != "finished":
        raise ValueError("demonstration does not satisfy the configured finish contract")
    return transitions
