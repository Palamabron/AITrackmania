from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    demonstration_timing_summary,
    demonstration_transitions,
    load_demonstration,
    record_demonstration,
    record_demonstration_session,
    reject_outliers,
    resample_demonstration,
    resolve_demonstration_paths,
    save_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.telemetry import TelemetryFrame


class _IdentityPipeline:
    def reset_episode(self) -> None:
        return

    def transform_observation(self, observation: Any) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32).copy()

    def collate(self, transitions: list[Transition]) -> list[Transition]:
        return transitions


class _TelemetryClient:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = list(frames)
        self.cursor = 0

    def read(self) -> TelemetryFrame:
        frame = self.frames[min(self.cursor, len(self.frames) - 1)]
        self.cursor += 1
        return TelemetryFrame(frame)

    def read_next(self) -> TelemetryFrame:
        return self.read()


class _EventTelemetryClient:
    def __init__(self, events: list[np.ndarray | TimeoutError]) -> None:
        self.events = list(events)
        self.cursor = 0

    def read(self) -> TelemetryFrame:
        event = self.events[min(self.cursor, len(self.events) - 1)]
        self.cursor += 1
        if isinstance(event, TimeoutError):
            raise event
        return TelemetryFrame(event)

    def read_next(self) -> TelemetryFrame:
        return self.read()


def _geometry(tmp_path: Path) -> BoundaryGeometry:
    left = np.asarray([[float(x), 0.0, -5.0] for x in range(101)], dtype=np.float32)
    right = left + np.asarray([0.0, 0.0, 10.0], dtype=np.float32)
    np.save(tmp_path / "left.npy", left)
    np.save(tmp_path / "right.npy", right)
    map_path = tmp_path / "map.Map.Gbx"
    map_path.write_bytes(b"map")
    asset = build_geometry_asset(
        tmp_path / "geometry.npz",
        tmp_path / "left.npy",
        tmp_path / "right.npy",
        map_uid="test-map",
        map_path=map_path,
        spacing_m=1.0,
        lookahead_points=0,
    )
    return BoundaryGeometry(asset, expected_map_uid="test-map")


def _config(
    geometry: BoundaryGeometry,
    *,
    action_repeat_frames: int = 2,
    decision_interval_ms: float | None = None,
    start_timeout_s: float = 15.0,
) -> TrackmaniaEnvironmentConfig:
    return TrackmaniaEnvironmentConfig(
        geometry_path=geometry.path,
        expected_map_uid=geometry.map_uid,
        action_repeat_frames=action_repeat_frames,
        decision_interval_ms=decision_interval_ms,
        start_timeout_s=start_timeout_s,
        start_poll_s=0.0,
        velocity_to_mps_scale=1.0,
        minimum_finish_steps=50,
        no_progress_steps=100,
        slow_progress_window_steps=80,
        minimum_progress_per_window_m=1.0,
    )


def _frames(count: int = 61) -> np.ndarray:
    frames = np.zeros((count, 33), dtype=np.float32)
    frames[:, 3] = np.linspace(0.0, 36_000.0, count)
    frames[:, 4] = np.linspace(0.0, 100.0, count)
    frames[:, 7] = 40.0
    frames[:, 10] = 1.0
    frames[:, 31] = 1.0
    frames[-1, 2] = 1.0
    return frames


def _demonstration(geometry: BoundaryGeometry) -> Demonstration:
    frames = _frames()
    _, table = build_brake_tap_action_table()
    action = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), table
    )
    return Demonstration(
        map_uid=geometry.map_uid,
        geometry_sha256=geometry.sha256,
        action_repeat_frames=2,
        frames=frames,
        actions=np.full(len(frames) - 1, action, dtype=np.int64),
        controls=np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (len(frames) - 1, 1)),
        finish_time_s=36.0,
    )


def test_demonstration_round_trip_and_transition_conversion(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))

    loaded = load_demonstration(path)
    transitions = demonstration_transitions(path, _IdentityPipeline(), _config(geometry), geometry)

    assert loaded.finish_time_s == 36.0
    assert loaded.decision_interval_ms is None
    assert len(transitions) == 60
    assert transitions[-1].terminated
    assert transitions[-1].info["is_demo"] is True
    assert transitions[-1].info["sampling/projected_lap_time_s"] == 36.0


def test_save_preserves_decimal_finish_time_in_filename(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)

    path = save_demonstration(tmp_path / "demo-01-37.785s", _demonstration(geometry))

    assert path.name == "demo-01-37.785s.npz"


def test_resample_demonstration_uses_the_online_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames.copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    demonstration = replace(demonstration, frames=frames, finish_time_s=0.6)

    selected_frames, selected_actions = resample_demonstration(demonstration, 20.0)

    assert np.array_equal(selected_frames[:, 3], frames[::2, 3])
    assert len(selected_actions) == len(selected_frames) - 1


def test_legacy_demonstration_preserves_openplanet_gamepad_steering_convention(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    controls = demonstration.controls.copy()
    controls[:, 2] = 1.0
    _, table = build_brake_tap_action_table()
    legacy_action = continuous_control_to_discrete_index(controls[0], table)
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        format=np.asarray("trackmaniarl-trackmania-demo-v1"),
        map_uid=np.asarray(demonstration.map_uid),
        geometry_sha256=np.asarray(demonstration.geometry_sha256),
        action_repeat_frames=np.asarray(demonstration.action_repeat_frames, dtype=np.int32),
        frames=demonstration.frames,
        actions=np.full(len(controls), legacy_action, dtype=np.int64),
        controls=controls,
        finish_time_s=np.asarray(demonstration.finish_time_s, dtype=np.float64),
    )

    loaded = load_demonstration(path)

    assert np.all(loaded.controls[:, 2] == 1.0)
    assert np.all(loaded.actions == legacy_action)


def test_v4_demonstration_migrates_openplanet_steering_to_controller_sign(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    controls = demonstration.controls.copy()
    controls[:, 2] = 1.0
    _, table = build_brake_tap_action_table()
    stored_action = continuous_control_to_discrete_index(controls[0], table)
    path = tmp_path / "native-v4.npz"
    np.savez_compressed(
        path,
        format=np.asarray("trackmaniarl-trackmania-demo-v4"),
        map_uid=np.asarray(demonstration.map_uid),
        geometry_sha256=np.asarray(demonstration.geometry_sha256),
        action_repeat_frames=np.asarray(1, dtype=np.int32),
        decision_interval_ms=np.asarray(0.0, dtype=np.float64),
        control_alignment=np.asarray("transition_end"),
        frames=demonstration.frames,
        actions=np.full(len(controls), stored_action, dtype=np.int64),
        controls=controls,
        finish_time_s=np.asarray(demonstration.finish_time_s, dtype=np.float64),
    )

    loaded = load_demonstration(path)

    expected_action = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, -1.0], dtype=np.float32), table
    )
    assert np.all(loaded.controls[:, 2] == -1.0)
    assert np.all(loaded.actions == expected_action)


def test_resolve_demonstration_paths_expands_directories(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    folder = tmp_path / "demos"
    first = save_demonstration(folder / "demo-01-36.000s", _demonstration(geometry))
    second = save_demonstration(folder / "demo-02-36.500s", _demonstration(geometry))
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")

    resolved = resolve_demonstration_paths([folder, first])

    assert resolved == (first.resolve(), second.resolve())


def test_resolve_demonstration_paths_expands_nested_directories(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    folder = tmp_path / "demos"
    first = save_demonstration(folder / "elite" / "demo-01-36.000s", _demonstration(geometry))
    second = save_demonstration(folder / "recovery" / "demo-02-36.500s", _demonstration(geometry))

    resolved = resolve_demonstration_paths([folder])

    assert resolved == (first.resolve(), second.resolve())


def test_resolve_demonstration_paths_rejects_empty_directory(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match=r"no \.npz files"):
        resolve_demonstration_paths([empty])


def test_demonstration_rejects_action_repeat_mismatch(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap.npz", _demonstration(geometry))

    with pytest.raises(ValueError, match="action repeat"):
        demonstration_transitions(
            path,
            _IdentityPipeline(),
            _config(geometry, action_repeat_frames=4),
            geometry,
        )


def test_legacy_demonstration_is_resampled_to_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    frames = _frames(121)
    frames[:, 3] = np.arange(121, dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    demonstration = _demonstration(geometry)
    demonstration = replace(
        demonstration,
        frames=frames,
        actions=np.full(120, demonstration.actions[0], dtype=np.int64),
        controls=np.tile(demonstration.controls[0], (120, 1)),
        finish_time_s=1.2,
    )
    path = save_demonstration(tmp_path / "legacy-timing.npz", demonstration)
    config = _config(geometry, action_repeat_frames=1, decision_interval_ms=20.0)

    transitions = demonstration_transitions(path, _IdentityPipeline(), config, geometry)

    assert len(transitions) == 60
    assert [transition.step for transition in transitions[:3]] == [0, 1, 2]
    assert transitions[-1].terminated


def test_demonstration_rejects_explicit_decision_interval_mismatch(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = replace(_demonstration(geometry), decision_interval_ms=10.0)
    path = save_demonstration(tmp_path / "timed.npz", demonstration)
    config = _config(geometry, action_repeat_frames=1, decision_interval_ms=20.0)

    with pytest.raises(ValueError, match="decision interval"):
        demonstration_transitions(path, _IdentityPipeline(), config, geometry)


def test_demonstration_import_preserves_completed_recovery_lap(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames.copy()
    frames[:21, 4] = 0.0
    frames[21:, 4] = np.linspace(0.0, 100.0, len(frames) - 21)
    recovery = Demonstration(
        map_uid=demonstration.map_uid,
        geometry_sha256=demonstration.geometry_sha256,
        action_repeat_frames=demonstration.action_repeat_frames,
        frames=frames,
        actions=demonstration.actions,
        controls=demonstration.controls,
        finish_time_s=demonstration.finish_time_s,
    )
    path = save_demonstration(tmp_path / "recovery.npz", recovery)
    config = _config(geometry).model_copy(update={"no_progress_steps": 10})

    transitions = demonstration_transitions(path, _IdentityPipeline(), config, geometry)

    assert len(transitions) == len(recovery.actions)
    assert transitions[-1].terminated


def test_recorder_waits_for_restart_and_quantizes_human_control(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    baseline = _frames(1)[0]
    baseline[2] = 0.0
    baseline[3] = 1_000.0
    reset = baseline.copy()
    reset[3] = 0.0
    start = baseline.copy()
    start[3] = 16.0
    start[31] = 1.0
    repeated = start.copy()
    repeated[3] = 32.0
    finish = repeated.copy()
    finish[2] = 1.0
    finish[3] = 48.0

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, repeated, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert demo.frames.shape == (2, 33)
    _, table = build_brake_tap_action_table()
    expected = continuous_control_to_discrete_index(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32), table
    )
    assert demo.actions.tolist() == [expected]
    assert demo.finish_time_s == pytest.approx(0.048)


def test_recorder_converts_openplanet_steering_to_controller_sign(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    baseline, reset, start, repeated, finish = _lap_frames(16.0, 32.0, 48.0)
    start[30] = repeated[30] = finish[30] = 1.0

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, repeated, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert np.all(demo.controls[:, 2] == -1.0)


def test_recorder_holds_action_until_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    baseline, reset, start, middle, finish = _lap_frames(10.0, 20.0, 30.0)

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, middle, finish]),
        _config(geometry, action_repeat_frames=1, decision_interval_ms=20.0),
        geometry,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert demo.frames[:, 3].tolist() == [10.0, 30.0]
    assert demo.decision_interval_ms == 20.0


def test_recorder_native_sampling_keeps_every_new_race_time(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    frames = _lap_frames(10.0, 20.0, 30.0)

    demo = record_demonstration(
        _TelemetryClient(frames),
        _config(geometry, action_repeat_frames=2),
        geometry,
        max_duration_s=1.0,
        sampling_interval_ms=0.0,
        status=lambda _message: None,
    )
    summary = demonstration_timing_summary(demo)
    validate_recording_quality(demo)

    assert demo.frames[:, 3].tolist() == [10.0, 20.0, 30.0]
    assert demo.action_repeat_frames == 1
    assert demo.decision_interval_ms is None
    assert demo.control_alignment == "frame_start"
    assert summary["interval_median_ms"] == 10.0


def test_recorder_aligns_each_control_with_the_transition_start_frame(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    baseline, reset, start, middle, finish = _lap_frames(10.0, 20.0, 30.0)
    start[30] = 0.0
    middle[30] = 1.0
    finish[30] = -1.0

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, middle, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        sampling_interval_ms=0.0,
        status=lambda _message: None,
    )

    assert demo.controls[:, 2].tolist() == [-0.0, -1.0]


def test_recorder_does_not_store_inputs_cleared_by_the_finish_frame(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    baseline, reset, start, middle, finish = _lap_frames(10.0, 20.0, 30.0)
    middle[30] = 1.0
    finish[30] = 0.0
    finish[31] = 0.0

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, start, middle, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        sampling_interval_ms=0.0,
        status=lambda _message: None,
    )

    assert demo.controls[-1].tolist() == [1.0, 0.0, -1.0]


def test_recorder_estimates_native_finish_between_telemetry_frames(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    baseline, reset, before, finish = _lap_frames(37_780.0, 37_790.0)

    demo = record_demonstration(
        _TelemetryClient([baseline, reset, before, finish]),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        sampling_interval_ms=0.0,
        status=lambda _message: None,
    )

    assert demo.finish_time_s == pytest.approx(37.785)


def _lap_frames(*race_times_ms: float) -> list[np.ndarray]:
    """Frames for one full-throttle run: an idle frame, a reset, then the driven lap."""

    baseline = _frames(1)[0]
    baseline[2] = 0.0
    baseline[31] = 1.0
    idle = baseline.copy()
    idle[3] = 1_000.0
    reset = baseline.copy()
    reset[3] = 0.0
    driven = []
    for race_time_ms in race_times_ms:
        frame = baseline.copy()
        frame[3] = race_time_ms
        driven.append(frame)
    driven[-1][2] = 1.0
    return [idle, reset, *driven]


def test_recorder_discards_partial_lap_after_mid_lap_restart(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    frames = _lap_frames(16.0, 32.0)
    restarted = frames[2].copy()
    restarted[3] = 5.0
    restarted[2] = 0.0
    continued = restarted.copy()
    continued[3] = 15.0
    finish = continued.copy()
    finish[2] = 1.0
    finish[3] = 25.0
    frames = [*frames[:4], restarted, continued, finish]
    frames[3][2] = 0.0

    demo = record_demonstration(
        _TelemetryClient(frames),
        _config(geometry),
        geometry,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert demo.frames[:, 3].tolist() == [5.0, 25.0]
    assert demo.finish_time_s == pytest.approx(0.025)


def test_session_records_each_lap_and_reject_outliers_drops_slow_laps(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    frames = (
        _lap_frames(16.0, 32.0, 36_000.0)
        + _lap_frames(10.0, 20.0, 36_500.0)
        + _lap_frames(10.0, 20.0, 38_000.0)
    )

    demos = record_demonstration_session(
        _TelemetryClient(frames),
        _config(geometry),
        geometry,
        count=3,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert [demo.finish_time_s for demo in demos] == pytest.approx([36.0, 36.5, 38.0])
    kept = reject_outliers(demos, max_gap_s=1.0)
    assert kept == [demos[0], demos[1]]
    assert reject_outliers(demos, max_gap_s=0.0) == [demos[0]]
    with pytest.raises(ValueError, match="max_gap_s must be non-negative"):
        reject_outliers(demos, max_gap_s=-0.1)


def test_session_returns_completed_laps_when_the_next_start_never_happens(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    frames = _lap_frames(16.0, 32.0, 48.0)

    demos = record_demonstration_session(
        _TelemetryClient(frames),
        _config(geometry, start_timeout_s=0.05),
        geometry,
        count=3,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert len(demos) == 1
    assert demos[0].finish_time_s == pytest.approx(0.048)


def test_session_retries_telemetry_timeout_while_waiting_for_next_lap(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    first = _lap_frames(10.0, 20.0, 30.0)
    second = _lap_frames(10.0, 20.0, 40.0)[2:]
    events: list[np.ndarray | TimeoutError] = [
        *first,
        TimeoutError("temporary telemetry silence"),
        *second,
    ]

    demos = record_demonstration_session(
        _EventTelemetryClient(events),
        _config(geometry, start_timeout_s=0.1),
        geometry,
        count=2,
        max_duration_s=1.0,
        status=lambda _message: None,
    )

    assert [demo.finish_time_s for demo in demos] == pytest.approx([0.03, 0.04])


def test_session_discards_a_late_native_start_and_continues(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    late = _lap_frames(20.0, 30.0, 40.0)
    valid = _lap_frames(10.0, 20.0, 30.0)[2:]
    messages: list[str] = []

    demos = record_demonstration_session(
        _TelemetryClient([*late, *valid]),
        _config(geometry),
        geometry,
        count=2,
        max_duration_s=1.0,
        sampling_interval_ms=0.0,
        status=messages.append,
    )

    assert len(demos) == 1
    assert demos[0].frames[0, 3] == 10.0
    assert any(message.startswith("Discarded lap 1:") for message in messages)
