"""Validated human-driving demonstrations for TrackMania replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from time import monotonic, sleep
from typing import Protocol

import numpy as np

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_DURATION_S,
    BRAKE_TAP_SENTINEL,
    BRAKE_TAP_TABLE_N_STEER,
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry, file_sha256
from trackmaniarl.trackmania.pace import ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.telemetry import TelemetryFrame

DEMONSTRATION_FORMAT = "trackmaniarl-trackmania-demo-v5"
LEGACY_DEMONSTRATION_FORMATS = {
    "trackmaniarl-trackmania-demo-v4",
    "trackmaniarl-trackmania-demo-v3",
    "trackmaniarl-trackmania-demo-v2",
    "trackmaniarl-trackmania-demo-v1",
    "tmrl-trackmania-demo-v1",
}
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
        if self.decision_interval_ms is not None and (
            not np.isfinite(self.decision_interval_ms) or self.decision_interval_ms <= 0.0
        ):
            raise ValueError("demonstration decision interval must be finite and positive")
        if self.control_alignment not in {"frame_start", "legacy_frame_start", "transition_end"}:
            raise ValueError("demonstration control alignment is invalid")
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
        target = Path(f"{target}.npz")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        format=np.asarray(DEMONSTRATION_FORMAT),
        map_uid=np.asarray(demonstration.map_uid),
        geometry_sha256=np.asarray(demonstration.geometry_sha256),
        action_repeat_frames=np.asarray(demonstration.action_repeat_frames, dtype=np.int32),
        decision_interval_ms=np.asarray(
            demonstration.decision_interval_ms or 0.0, dtype=np.float64
        ),
        control_alignment=np.asarray(demonstration.control_alignment),
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
            matches = sorted(item.resolve() for item in path.rglob("*.npz") if item.is_file())
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
        format_name = str(data["format"].item())
        if format_name not in {DEMONSTRATION_FORMAT, *LEGACY_DEMONSTRATION_FORMATS}:
            raise ValueError("unsupported TrackMania demonstration format")
        if format_name in {
            DEMONSTRATION_FORMAT,
            "trackmaniarl-trackmania-demo-v4",
            "trackmaniarl-trackmania-demo-v3",
        } and ("decision_interval_ms" not in data.files):
            raise ValueError("timed demonstration is missing its decision interval")
        if format_name in {DEMONSTRATION_FORMAT, "trackmaniarl-trackmania-demo-v4"}:
            required_timing = {"decision_interval_ms", "control_alignment"}
            missing_timing = required_timing - set(data.files)
            if missing_timing:
                raise ValueError(f"native demonstration is missing keys: {sorted(missing_timing)}")
        interval = (
            float(data["decision_interval_ms"].item())
            if "decision_interval_ms" in data.files
            else 0.0
        )
        controls = np.asarray(data["controls"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
        control_alignment = (
            str(data["control_alignment"].item())
            if "control_alignment" in data.files
            else "legacy_frame_start"
        )
        if format_name == "trackmaniarl-trackmania-demo-v4":
            controls = controls.copy()
            controls[:, 2] *= -1.0
            controls[1:] = controls[:-1].copy()
            _, table = build_brake_tap_action_table()
            actions = continuous_control_to_discrete_indices_batch(controls, table)
            control_alignment = "frame_start"
        return Demonstration(
            map_uid=str(data["map_uid"].item()),
            geometry_sha256=str(data["geometry_sha256"].item()),
            action_repeat_frames=int(data["action_repeat_frames"].item()),
            frames=np.asarray(data["frames"], dtype=np.float32),
            actions=actions,
            controls=controls,
            finish_time_s=float(data["finish_time_s"].item()),
            decision_interval_ms=interval or None,
            control_alignment=control_alignment,
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


def _wait_for_new_run(
    client: TelemetryReader,
    *,
    timeout_s: float,
    poll_s: float,
    previous_race_time_ms: float | None = None,
) -> TelemetryFrame:
    deadline = monotonic() + timeout_s
    previous_time = previous_race_time_ms
    restart_observed = previous_time is not None and previous_time <= 0.0
    stream_synchronized = previous_time is not None
    while monotonic() < deadline:
        try:
            frame = client.read_next() if stream_synchronized else client.read()
            stream_synchronized = True
        except TimeoutError:
            continue
        race_time = float(frame.values[3])
        if previous_time is None:
            previous_time = race_time
            restart_observed = race_time <= 0.0
            continue
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
    sampling_interval_ms: float | None = None,
    previous_race_time_ms: float | None = None,
    status: Callable[[str], None] = print,
) -> Demonstration:
    if max_duration_s <= 0.0 or (sampling_interval_ms is not None and sampling_interval_ms < 0.0):
        raise ValueError("recording duration and sampling interval are invalid")
    status("Waiting for a fresh run. Restart the map, then drive the complete lap.")
    current = _wait_for_new_run(
        client,
        timeout_s=config.start_timeout_s,
        poll_s=config.start_poll_s,
        previous_race_time_ms=previous_race_time_ms,
    )
    _, table = build_brake_tap_action_table()
    frames = [current.values.copy()]
    actions: list[int] = []
    controls: list[np.ndarray] = []
    control = _control(current)
    deadline = monotonic() + max_duration_s
    while monotonic() < deadline:
        following = _advance_demonstration_frame(
            client,
            current,
            config,
            sampling_interval_ms=sampling_interval_ms,
        )
        if float(following.values[3]) < float(frames[-1][3]):
            status("Restart detected; the partial lap was discarded. Recording the new run.")
            deadline = monotonic() + max_duration_s
            while float(following.values[3]) <= 0.0 and monotonic() < deadline:
                following = client.read_next()
            current = following
            frames = [current.values.copy()]
            actions.clear()
            controls.clear()
            control = _control(current)
            continue
        frames.append(following.values.copy())
        actions.append(continuous_control_to_discrete_index(control, table))
        controls.append(control)
        if bool(following.values[2]):
            native_sampling = sampling_interval_ms == 0.0
            finish_time_s = _estimated_finish_time_s(frames, native_sampling=native_sampling)
            status(f"Finished demonstration in {finish_time_s:.3f}s.")
            demonstration = Demonstration(
                map_uid=geometry.map_uid,
                geometry_sha256=geometry.sha256,
                action_repeat_frames=1
                if sampling_interval_ms is not None
                else config.action_repeat_frames,
                frames=np.asarray(frames, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                controls=np.asarray(controls, dtype=np.float32),
                finish_time_s=finish_time_s,
                decision_interval_ms=(
                    None
                    if native_sampling
                    else config.decision_interval_ms
                    if sampling_interval_ms is None
                    else sampling_interval_ms
                ),
                control_alignment="frame_start",
            )
            status(_recording_quality_message(demonstration))
            return demonstration
        current = following
        control = _control(current)
    raise TimeoutError("demonstration did not reach the finish before max_duration_s")


def _estimated_finish_time_s(frames: list[np.ndarray], *, native_sampling: bool) -> float:
    observed_ms = float(frames[-1][3])
    if not native_sampling or len(frames) < 2:
        return observed_ms / 1_000.0
    previous_ms = float(frames[-2][3])
    return (previous_ms + observed_ms) / 2_000.0


def _advance_demonstration_frame(
    client: TelemetryReader,
    current: TelemetryFrame,
    config: TrackmaniaEnvironmentConfig,
    *,
    sampling_interval_ms: float | None,
) -> TelemetryFrame:
    current_time_ms = float(current.values[3])
    if sampling_interval_ms is None:
        interval_ms = config.decision_interval_ms
        minimum_reads = config.action_repeat_frames
    else:
        interval_ms = sampling_interval_ms or None
        minimum_reads = 1
    target_time_ms = current_time_ms + interval_ms if interval_ms is not None else None
    for _ in range(64):
        current = client.read_next()
        minimum_reads -= 1
        observed_time_ms = float(current.values[3])
        if observed_time_ms < current_time_ms:
            return current
        time_advanced = observed_time_ms > current_time_ms
        interval_reached = target_time_ms is None or observed_time_ms >= target_time_ms
        if minimum_reads <= 0 and time_advanced and interval_reached:
            return current
    raise TimeoutError("TrackMania telemetry did not advance during demonstration recording")


def record_demonstration_session(
    client: TelemetryReader,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    *,
    count: int,
    max_duration_s: float,
    sampling_interval_ms: float | None = None,
    status: Callable[[str], None] = print,
) -> list[Demonstration]:
    """Record up to ``count`` finished laps, stopping early once a lap start times out."""

    if count < 1:
        raise ValueError("count must be positive")
    demonstrations: list[Demonstration] = []
    previous_race_time_ms: float | None = None
    for lap in range(1, count + 1):
        status(f"Recording lap {lap} of {count}.")
        try:
            demonstration = record_demonstration(
                client,
                config,
                geometry,
                max_duration_s=max_duration_s,
                sampling_interval_ms=sampling_interval_ms,
                previous_race_time_ms=previous_race_time_ms,
                status=status,
            )
            if sampling_interval_ms is not None:
                try:
                    validate_recording_quality(demonstration)
                except ValueError as error:
                    previous_race_time_ms = float(demonstration.frames[-1, 3])
                    status(f"Discarded lap {lap}: {error}")
                    continue
            demonstrations.append(demonstration)
            previous_race_time_ms = float(demonstration.frames[-1, 3])
        except TimeoutError as error:
            if not demonstrations:
                raise
            status(f"Stopping the session after {len(demonstrations)} laps: {error}")
            break
    return demonstrations


def demonstration_timing_summary(demonstration: Demonstration) -> dict[str, float]:
    """Summarize the physical telemetry cadence stored in a demonstration."""

    intervals = np.diff(demonstration.frames[:, 3]).astype(np.float64)
    switches = np.any(demonstration.controls[1:] != demonstration.controls[:-1], axis=1)
    return {
        "transitions": float(len(intervals)),
        "first_frame_ms": float(demonstration.frames[0, 3]),
        "interval_median_ms": float(np.median(intervals)),
        "interval_p95_ms": float(np.quantile(intervals, 0.95)),
        "interval_max_ms": float(np.max(intervals)),
        "control_switches": float(np.count_nonzero(switches)),
        "minimum_control_hold_ms": _minimum_control_hold_ms(demonstration),
    }


def _minimum_control_hold_ms(demonstration: Demonstration) -> float:
    controls = demonstration.controls
    if len(controls) < 2:
        return float(demonstration.frames[-1, 3] - demonstration.frames[0, 3])
    boundaries = np.flatnonzero(np.any(controls[1:] != controls[:-1], axis=1)) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    stops = np.concatenate((boundaries, np.asarray([len(controls)])))
    times = demonstration.frames[:, 3].astype(np.float64)
    durations = times[stops] - times[starts]
    return float(np.min(durations))


def validate_recording_quality(
    demonstration: Demonstration,
) -> None:
    """Reject recordings whose telemetry cadence cannot preserve short inputs."""

    summary = demonstration_timing_summary(demonstration)
    if summary["first_frame_ms"] > 50.0:
        raise ValueError("demonstration recording started more than 50 ms after race start")
    if summary["interval_p95_ms"] > 25.0 or summary["interval_max_ms"] > 50.0:
        raise ValueError(
            "demonstration telemetry cadence is too sparse; require p95 <=25 ms and max <=50 ms"
        )
    if demonstration.control_alignment != "frame_start":
        raise ValueError(
            "demonstration does not align each control with its transition start frame"
        )
    native = demonstration.action_repeat_frames == 1 and demonstration.decision_interval_ms is None
    if native and (
        summary["first_frame_ms"] > 15.0
        or not 8.0 <= summary["interval_median_ms"] <= 12.0
        or summary["interval_p95_ms"] > 12.0
        or summary["interval_max_ms"] > 20.0
    ):
        raise ValueError(
            "native demonstration missed its 100 Hz timing contract; require first <=15 ms, "
            "median 8-12 ms, p95 <=12 ms, and max <=20 ms"
        )


def _recording_quality_message(demonstration: Demonstration) -> str:
    summary = demonstration_timing_summary(demonstration)
    return (
        "Recording quality: "
        f"transitions={int(summary['transitions'])}, "
        f"first={summary['first_frame_ms']:.0f}ms, "
        f"interval(median={summary['interval_median_ms']:.1f}ms, "
        f"p95={summary['interval_p95_ms']:.1f}ms, max={summary['interval_max_ms']:.1f}ms), "
        f"control_switches={int(summary['control_switches'])}, "
        f"shortest_control={summary['minimum_control_hold_ms']:.1f}ms, "
        f"alignment={demonstration.control_alignment}"
    )


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
    if (
        config.decision_interval_ms is None
        and demonstration.action_repeat_frames != config.action_repeat_frames
    ):
        raise ValueError("demonstration action repeat does not match the environment")
    if (
        config.decision_interval_ms is not None
        and demonstration.decision_interval_ms is not None
        and not np.isclose(
            demonstration.decision_interval_ms,
            config.decision_interval_ms,
            rtol=0.0,
            atol=0.05,
        )
    ):
        raise ValueError("demonstration decision interval does not match the environment")
    if demonstration.frames.shape[1] != config.field_count:
        raise ValueError("demonstration telemetry schema does not match the environment")


def _reward(
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    *,
    demonstration_steps: int | None = None,
) -> TrajectoryReward:
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    kwargs = config.reward_kwargs()
    if demonstration_steps is not None:
        kwargs["no_progress_steps"] = demonstration_steps + 1
        kwargs["slow_progress_window_steps"] = demonstration_steps + 1
    pace_profile = (
        ReferencePaceProfile.from_demonstration(config.pace_reference_path, geometry, reference)
        if config.pace_reference_path is not None
        else None
    )
    return TrajectoryReward(reference, pace_profile=pace_profile, **kwargs)


def demonstration_transitions(
    path: str | Path,
    pipeline: FeaturePipeline,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> list[Transition]:
    demo = load_demonstration(path)
    validate_demonstration(demo, config, geometry)
    frames, actions = resample_demonstration(
        demo,
        config.decision_interval_ms,
        action_lead_ms=config.demonstration_action_lead_ms,
        aggregate_controls=config.demonstration_control_aggregation,
    )
    reward = _reward(config, geometry, demonstration_steps=len(actions))
    reset_pipeline = getattr(pipeline, "reset_episode", None)
    if callable(reset_pipeline):
        reset_pipeline()
    position = frames[0, list(config.position_indices)]
    velocity = frames[0, list(config.velocity_indices)]
    reward.reset(position, velocity=velocity, race_time_ms=float(frames[0, 3]))
    prepared = pipeline.transform_observation(frames[0])
    episode_id = f"demo-{file_sha256(path)[:16]}"
    transitions: list[Transition] = []
    _, table = build_brake_tap_action_table()
    action_ids = config.compact_action_ids
    compact_indices = (
        None if action_ids is None else {action: index for index, action in enumerate(action_ids)}
    )
    for step, (action, next_frame) in enumerate(zip(actions, frames[1:], strict=True)):
        source_action = int(action)
        if compact_indices is not None and source_action not in compact_indices:
            raise ValueError(f"demonstration action {source_action} is outside compact action IDs")
        control = table[source_action]
        result = reward.step(
            next_frame[list(config.position_indices)],
            finish_ui_active=bool(next_frame[2]),
            velocity=next_frame[list(config.velocity_indices)],
            race_time_ms=float(next_frame[3]),
            steering=float(control[2]),
        )
        if result.terminated and step != len(actions) - 1:
            raise ValueError(f"demonstration reward terminated early: {result.reason}")
        next_prepared = pipeline.transform_observation(next_frame)
        transitions.append(
            Transition(
                observation=prepared,
                action=source_action if compact_indices is None else compact_indices[source_action],
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


def resample_demonstration(
    demonstration: Demonstration,
    decision_interval_ms: float | None,
    *,
    action_lead_ms: float = 0.0,
    aggregate_controls: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select transitions on the physical decision-time grid used online."""

    if not np.isfinite(action_lead_ms) or action_lead_ms < 0.0:
        raise ValueError("demonstration action lead must be finite and non-negative")
    race_times = demonstration.frames[:, 3]
    indices = list(range(len(demonstration.frames)))
    if decision_interval_ms is not None:
        indices = [0]
        while indices[-1] < len(demonstration.actions):
            target = float(race_times[indices[-1]]) + decision_interval_ms
            candidate = int(np.searchsorted(race_times, target, side="left"))
            indices.append(min(max(candidate, indices[-1] + 1), len(demonstration.actions)))
    selected = np.asarray(indices, dtype=np.int64)
    if aggregate_controls and decision_interval_ms is not None:
        actions = _aggregate_demonstration_controls(
            demonstration,
            race_times[selected] + action_lead_ms,
        )
        return demonstration.frames[selected], actions
    action_times = race_times[selected[:-1]] + action_lead_ms
    action_indices = np.searchsorted(race_times, action_times, side="left")
    action_indices = np.clip(action_indices, 0, len(demonstration.actions) - 1)
    return demonstration.frames[selected], demonstration.actions[action_indices]


def _aggregate_demonstration_controls(
    demonstration: Demonstration, window_boundaries_ms: np.ndarray
) -> np.ndarray:
    if demonstration.control_alignment != "frame_start":
        raise ValueError("control aggregation requires frame_start demonstration controls")
    race_times = demonstration.frames[:, 3].astype(np.float64)
    actions = [
        _quantize_control_window(
            _integrate_control_window(
                demonstration.controls,
                race_times,
                float(start),
                float(stop),
            ),
            float(stop - start),
        )
        for start, stop in pairwise(window_boundaries_ms)
    ]
    return np.asarray(actions, dtype=np.int64)


def _integrate_control_window(
    controls: np.ndarray,
    race_times_ms: np.ndarray,
    start_ms: float,
    stop_ms: float,
) -> np.ndarray:
    duration_ms = stop_ms - start_ms
    if duration_ms <= 0.0:
        raise ValueError("demonstration control window must have positive duration")
    integral = np.zeros(3, dtype=np.float64)
    cursor = start_ms
    while cursor < stop_ms:
        index = int(np.searchsorted(race_times_ms, cursor, side="right") - 1)
        index = int(np.clip(index, 0, len(controls) - 1))
        boundary = race_times_ms[index + 1] if index + 1 < len(race_times_ms) else stop_ms
        segment_stop = min(stop_ms, max(cursor, float(boundary)))
        if segment_stop == cursor:
            segment_stop = stop_ms
        overlap_ms = segment_stop - cursor
        integral[[0, 2]] += controls[index, [0, 2]] * overlap_ms
        integral[1] += _brake_overlap_ms(
            float(controls[index, 1]),
            float(race_times_ms[index]),
            cursor,
            segment_stop,
        )
        cursor = segment_stop
    return (integral / duration_ms).astype(np.float32)


def _brake_overlap_ms(
    brake: float, start_ms: float, overlap_start: float, overlap_stop: float
) -> float:
    if brake != BRAKE_TAP_SENTINEL:
        return float(np.clip(brake, 0.0, 1.0)) * (overlap_stop - overlap_start)
    tap_stop = start_ms + BRAKE_TAP_DURATION_S * 1_000.0
    return max(0.0, min(overlap_stop, tap_stop) - overlap_start)


def _quantize_control_window(control: np.ndarray, duration_ms: float) -> int:
    _, table = build_brake_tap_action_table()
    steer_values = np.linspace(-1.0, 1.0, BRAKE_TAP_TABLE_N_STEER, dtype=np.float32)
    steer = float(steer_values[np.argmin(np.abs(steer_values - control[2]))])
    gas = float(control[0] >= 0.5)
    brake_duties = np.asarray(
        [0.0, BRAKE_TAP_DURATION_S * 1_000.0 / duration_ms, 1.0], dtype=np.float32
    )
    brake_values = (0.0, BRAKE_TAP_SENTINEL, 1.0)
    brake = brake_values[int(np.argmin(np.abs(brake_duties - control[1])))]
    return continuous_control_to_discrete_index(
        np.asarray([gas, brake, steer], dtype=np.float32), table
    )
