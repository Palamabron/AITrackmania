"""Live TrackMania demonstration recording."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic, sleep

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstration_data import Demonstration, TelemetryReader, _control
from trackmaniarl.trackmania.demonstration_processing import (
    _recording_quality_message,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.telemetry import TelemetryFrame

StatusReporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class DemonstrationRecordingConfig:
    max_duration_s: float
    sampling_interval_ms: float | None = None
    previous_race_time_ms: float | None = None
    status: StatusReporter = print


@dataclass(frozen=True, slots=True)
class DemonstrationRecordingRequest:
    client: TelemetryReader
    environment: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    config: DemonstrationRecordingConfig


@dataclass(frozen=True, slots=True)
class DemonstrationSessionConfig:
    count: int
    max_duration_s: float
    sampling_interval_ms: float | None = None
    status: StatusReporter = print


@dataclass(frozen=True, slots=True)
class DemonstrationSessionRequest:
    client: TelemetryReader
    environment: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    config: DemonstrationSessionConfig


@dataclass(slots=True)
class _RunWaitState:
    deadline: float
    poll_s: float
    previous_time: float | None
    restart_observed: bool
    stream_synchronized: bool


@dataclass(frozen=True, slots=True)
class _WaitSettings:
    timeout_s: float
    poll_s: float
    previous_race_time_ms: float | None


@dataclass(frozen=True, slots=True)
class _RecordingContext:
    client: TelemetryReader
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    settings: DemonstrationRecordingConfig
    action_table: list[np.ndarray]


@dataclass(slots=True)
class _RecordingState:
    current: TelemetryFrame
    frames: list[np.ndarray]
    actions: list[int]
    controls: list[np.ndarray]
    control: np.ndarray
    deadline: float


@dataclass(frozen=True, slots=True)
class _SamplingPlan:
    interval_ms: float | None
    minimum_reads: int


@dataclass(slots=True)
class _SamplingProgress:
    current_time_ms: float
    target_time_ms: float | None
    remaining_reads: int


@dataclass(frozen=True, slots=True)
class _SessionContext:
    client: TelemetryReader
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    settings: DemonstrationSessionConfig


def _wait_for_new_run(client: TelemetryReader, settings: _WaitSettings) -> TelemetryFrame:
    state = _run_wait_state(settings)
    while monotonic() < state.deadline:
        try:
            frame = client.read_next() if state.stream_synchronized else client.read()
            state.stream_synchronized = True
        except TimeoutError:
            continue
        if _is_new_run(state, frame):
            return frame
        if state.poll_s:
            sleep(state.poll_s)
    raise TimeoutError("no new TrackMania run was observed; restart the map and begin driving")


def _run_wait_state(settings: _WaitSettings) -> _RunWaitState:
    previous = settings.previous_race_time_ms
    return _RunWaitState(
        monotonic() + settings.timeout_s,
        settings.poll_s,
        previous,
        previous is not None and previous <= 0.0,
        previous is not None,
    )


def _is_new_run(state: _RunWaitState, frame: TelemetryFrame) -> bool:
    race_time = float(frame.values[3])
    if state.previous_time is None:
        state.previous_time = race_time
        state.restart_observed = race_time <= 0.0
        return False
    state.restart_observed = state.restart_observed or race_time < state.previous_time
    state.previous_time = race_time
    return state.restart_observed and race_time > 0.0


def record_demonstration(request: DemonstrationRecordingRequest) -> Demonstration:
    client, config, geometry = request.client, request.environment, request.geometry
    settings = request.config
    _validate_record_settings(settings)
    settings.status("Waiting for a fresh run. Restart the map, then drive the complete lap.")
    wait = _WaitSettings(
        config.start_timeout_s, config.start_poll_s, settings.previous_race_time_ms
    )
    current = _wait_for_new_run(client, wait)
    _, table = build_brake_tap_action_table()
    context = _RecordingContext(client, config, geometry, settings, table)
    return _record(context, _initial_recording_state(current, settings.max_duration_s))


def _validate_record_settings(settings: DemonstrationRecordingConfig) -> None:
    interval = settings.sampling_interval_ms
    if settings.max_duration_s <= 0.0 or (interval is not None and interval < 0.0):
        raise ValueError("recording duration and sampling interval are invalid")


def _initial_recording_state(current: TelemetryFrame, duration_s: float) -> _RecordingState:
    return _RecordingState(
        current, [current.values.copy()], [], [], _control(current), monotonic() + duration_s
    )


def _record(context: _RecordingContext, state: _RecordingState) -> Demonstration:
    while monotonic() < state.deadline:
        following = _next_recording_frame(context, state)
        if float(following.values[3]) < float(state.frames[-1][3]):
            _restart_recording(context, state, following)
            continue
        _append_recording_frame(context, state, following)
        if bool(following.values[2]):
            return _finish_demonstration(context, state)
        state.current = following
        state.control = _control(following)
    raise TimeoutError("demonstration did not reach the finish before max_duration_s")


def _next_recording_frame(context: _RecordingContext, state: _RecordingState) -> TelemetryFrame:
    plan = _sampling_plan(context.config, context.settings.sampling_interval_ms)
    return _advance_demonstration_frame(context.client, state.current, plan)


def _restart_recording(
    context: _RecordingContext, state: _RecordingState, following: TelemetryFrame
) -> None:
    context.settings.status(
        "Restart detected; the partial lap was discarded. Recording the new run."
    )
    state.deadline = monotonic() + context.settings.max_duration_s
    while float(following.values[3]) <= 0.0 and monotonic() < state.deadline:
        following = context.client.read_next()
    state.current = following
    state.frames = [following.values.copy()]
    state.actions.clear()
    state.controls.clear()
    state.control = _control(following)


def _append_recording_frame(
    context: _RecordingContext, state: _RecordingState, following: TelemetryFrame
) -> None:
    state.frames.append(following.values.copy())
    state.actions.append(continuous_control_to_discrete_index(state.control, context.action_table))
    state.controls.append(state.control)


def _finish_demonstration(context: _RecordingContext, state: _RecordingState) -> Demonstration:
    finish_time_s = _estimated_finish_time_s(state.frames, context.settings.sampling_interval_ms)
    context.settings.status(f"Finished demonstration in {finish_time_s:.3f}s.")
    demonstration = _completed_demonstration(context, state, finish_time_s)
    context.settings.status(_recording_quality_message(demonstration))
    return demonstration


def _completed_demonstration(
    context: _RecordingContext, state: _RecordingState, finish_time_s: float
) -> Demonstration:
    interval = context.settings.sampling_interval_ms
    return Demonstration(
        map_uid=context.geometry.map_uid,
        geometry_sha256=context.geometry.sha256,
        action_repeat_frames=1 if interval is not None else context.config.action_repeat_frames,
        frames=np.asarray(state.frames, dtype=np.float32),
        actions=np.asarray(state.actions, dtype=np.int64),
        controls=np.asarray(state.controls, dtype=np.float32),
        finish_time_s=finish_time_s,
        decision_interval_ms=_recorded_interval(context.config, interval),
        control_alignment="frame_start",
    )


def _recorded_interval(
    config: TrackmaniaEnvironmentConfig, sampling_interval_ms: float | None
) -> float | None:
    if sampling_interval_ms == 0.0:
        return None
    return config.decision_interval_ms if sampling_interval_ms is None else sampling_interval_ms


def _estimated_finish_time_s(frames: list[np.ndarray], sampling_interval_ms: float | None) -> float:
    observed_ms = float(frames[-1][3])
    if sampling_interval_ms != 0.0 or len(frames) < 2:
        return observed_ms / 1_000.0
    previous_ms = float(frames[-2][3])
    return (previous_ms + observed_ms) / 2_000.0


def _advance_demonstration_frame(
    client: TelemetryReader, current: TelemetryFrame, plan: _SamplingPlan
) -> TelemetryFrame:
    progress = _sampling_progress(float(current.values[3]), plan)
    for _ in range(64):
        current = client.read_next()
        progress.remaining_reads -= 1
        observed_time_ms = float(current.values[3])
        if observed_time_ms < progress.current_time_ms:
            return current
        if _sampling_target_reached(progress, observed_time_ms):
            return current
    raise TimeoutError("TrackMania telemetry did not advance during demonstration recording")


def _sampling_plan(
    config: TrackmaniaEnvironmentConfig, sampling_interval_ms: float | None
) -> _SamplingPlan:
    if sampling_interval_ms is None:
        return _SamplingPlan(config.decision_interval_ms, config.action_repeat_frames)
    return _SamplingPlan(sampling_interval_ms or None, 1)


def _sampling_progress(current_time_ms: float, plan: _SamplingPlan) -> _SamplingProgress:
    target = current_time_ms + plan.interval_ms if plan.interval_ms is not None else None
    return _SamplingProgress(current_time_ms, target, plan.minimum_reads)


def _sampling_target_reached(progress: _SamplingProgress, observed_ms: float) -> bool:
    time_advanced = observed_ms > progress.current_time_ms
    interval_reached = progress.target_time_ms is None or observed_ms >= progress.target_time_ms
    return progress.remaining_reads <= 0 and time_advanced and interval_reached


def record_demonstration_session(
    request: DemonstrationSessionRequest,
) -> list[Demonstration]:
    """Record up to ``count`` finished laps, stopping early once a lap start times out."""

    settings = request.config
    if settings.count < 1:
        raise ValueError("count must be positive")
    context = _SessionContext(request.client, request.environment, request.geometry, settings)
    return _record_session(context)


def _record_session(context: _SessionContext) -> list[Demonstration]:
    demonstrations: list[Demonstration] = []
    previous_race_time_ms: float | None = None
    for lap in range(1, context.settings.count + 1):
        context.settings.status(f"Recording lap {lap} of {context.settings.count}.")
        try:
            demonstration = _record_session_lap(context, previous_race_time_ms)
        except TimeoutError as error:
            _report_session_timeout(context, demonstrations, error)
            break
        previous_race_time_ms = float(demonstration.frames[-1, 3])
        if _accept_session_lap(demonstration, context.settings, lap):
            demonstrations.append(demonstration)
    return demonstrations


def _record_session_lap(
    context: _SessionContext, previous_race_time_ms: float | None
) -> Demonstration:
    return record_demonstration(
        DemonstrationRecordingRequest(
            context.client,
            context.config,
            context.geometry,
            DemonstrationRecordingConfig(
                context.settings.max_duration_s,
                context.settings.sampling_interval_ms,
                previous_race_time_ms,
                context.settings.status,
            ),
        )
    )


def _report_session_timeout(
    context: _SessionContext, demonstrations: list[Demonstration], error: TimeoutError
) -> None:
    if not demonstrations:
        raise error
    context.settings.status(f"Stopping the session after {len(demonstrations)} laps: {error}")


def _accept_session_lap(
    demonstration: Demonstration, settings: DemonstrationSessionConfig, lap: int
) -> bool:
    if settings.sampling_interval_ms is None:
        return True
    try:
        validate_recording_quality(demonstration)
    except ValueError as error:
        settings.status(f"Discarded lap {lap}: {error}")
        return False
    return True
