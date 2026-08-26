from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_SENTINEL,
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    _control,
    demonstration_transitions,
    load_demonstration,
    record_demonstration,
    record_demonstration_session,
    reject_outliers,
    resample_demonstration,
    resolve_demonstration_paths,
    save_demonstration,
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


def test_recorded_control_preserves_openplanet_steering_direction() -> None:
    values = np.zeros(33, dtype=np.float32)
    values[30] = -1.0
    values[31] = 1.0

    control = _control(TelemetryFrame(values))

    np.testing.assert_array_equal(control, np.asarray([1.0, 0.0, -1.0], dtype=np.float32))


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"decision_interval_ms": None},
        {"decision_interval_ms": 50.0, "control_backend": "keyboard"},
    ],
)
def test_control_aggregation_requires_an_analog_timed_environment(
    tmp_path: Path, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="demonstration control aggregation requires"):
        TrackmaniaEnvironmentConfig(
            trajectory_path=tmp_path / "track.npy",
            action_repeat_frames=1,
            demonstration_control_aggregation=True,
            **overrides,
        )


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


def test_resample_demonstration_uses_the_online_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames.copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    demonstration = replace(demonstration, frames=frames, finish_time_s=0.6)

    selected_frames, selected_actions = resample_demonstration(demonstration, 20.0)

    assert np.array_equal(selected_frames[:, 3], frames[::2, 3])
    assert len(selected_actions) == len(selected_frames) - 1


def test_resample_demonstration_leads_actions_by_race_time(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames[:6].copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    actions = np.asarray([0, 1, 3, 39, 75], dtype=np.int64)
    _, table = build_brake_tap_action_table()
    demonstration = replace(
        demonstration,
        frames=frames,
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.05,
    )

    selected_frames, selected_actions = resample_demonstration(
        demonstration,
        20.0,
        action_lead_ms=20.0,
    )

    assert selected_frames[:, 3].tolist() == [0.0, 20.0, 40.0, 50.0]
    assert selected_actions.tolist() == [3, 75, 75]


@pytest.mark.parametrize("action_lead_ms", [float("nan"), float("inf")])
def test_resample_demonstration_rejects_non_finite_action_lead(
    tmp_path: Path, action_lead_ms: float
) -> None:
    demonstration = _demonstration(_geometry(tmp_path))

    with pytest.raises(ValueError, match="finite and non-negative"):
        resample_demonstration(demonstration, None, action_lead_ms=action_lead_ms)


def test_resample_demonstration_aggregates_keyboard_pwm_into_analog_action(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames[:6].copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    controls = np.asarray(
        [
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    _, table = build_brake_tap_action_table()
    demonstration = replace(
        demonstration,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
        control_alignment="frame_start",
    )

    _, default_actions = resample_demonstration(demonstration, 50.0)
    _, aggregated_actions = resample_demonstration(
        demonstration,
        50.0,
        aggregate_controls=True,
    )

    assert int(default_actions[0]) == int(demonstration.actions[0])
    gas, brake, steer = table[int(aggregated_actions[0])]
    assert gas == pytest.approx(1.0)
    assert brake == pytest.approx(BRAKE_TAP_SENTINEL)
    assert steer == pytest.approx(1.0 / 6.0)


def test_control_aggregation_shifts_the_entire_window_by_action_lead(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames[:6].copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    controls = np.asarray(
        [[1.0, 0.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 3,
        dtype=np.float32,
    )
    _, table = build_brake_tap_action_table()
    demonstration = replace(
        demonstration,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
        control_alignment="frame_start",
    )

    _, actions = resample_demonstration(
        demonstration,
        20.0,
        action_lead_ms=20.0,
        aggregate_controls=True,
    )

    assert [float(table[int(action)][2]) for action in actions] == [1.0, 1.0, 1.0]


def test_resolve_demonstration_paths_expands_directories(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    folder = tmp_path / "demos"
    first = save_demonstration(folder / "demo-01-36.000s", _demonstration(geometry))
    second = save_demonstration(folder / "demo-02-36.500s", _demonstration(geometry))
    (folder / "notes.txt").write_text("ignore", encoding="utf-8")

    resolved = resolve_demonstration_paths([folder, first])

    assert resolved == (first.resolve(), second.resolve())


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

    assert demo.controls[:, 2].tolist() == [0.0, 1.0]


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
