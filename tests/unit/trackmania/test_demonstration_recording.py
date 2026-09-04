from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.unit.trackmania._demonstration_fixtures import (
    _config,
    _demonstration,
    _frames,
    _geometry,
    _IdentityPipeline,
    _lap_frames,
    _TelemetryClient,
)
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationRecordingConfig,
    DemonstrationRecordingRequest,
    DemonstrationSessionConfig,
    DemonstrationSessionRequest,
    DemonstrationTransitionContext,
    demonstration_transitions,
    record_demonstration,
    record_demonstration_session,
    reject_outliers,
    save_demonstration,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry


def _record(
    client: _TelemetryClient,
    geometry: BoundaryGeometry,
    sampling_interval_ms: float | None = None,
) -> Demonstration:
    request = DemonstrationRecordingRequest(
        client,
        _config(geometry),
        geometry,
        DemonstrationRecordingConfig(1.0, sampling_interval_ms, status=lambda _message: None),
    )
    return record_demonstration(request)


def _recovery(demonstration: Demonstration) -> Demonstration:
    frames = demonstration.frames.copy()
    frames[:21, 4] = 0.0
    frames[21:, 4] = np.linspace(0.0, 100.0, len(frames) - 21)
    return Demonstration(
        demonstration.map_uid,
        demonstration.geometry_sha256,
        demonstration.action_repeat_frames,
        frames,
        demonstration.actions,
        demonstration.controls,
        demonstration.finish_time_s,
    )


def _fresh_run_frames() -> list[np.ndarray]:
    baseline = _frames(1)[0]
    baseline[2], baseline[3] = 0.0, 1_000.0
    reset, start = baseline.copy(), baseline.copy()
    reset[3], start[3], start[31] = 0.0, 16.0, 1.0
    repeated = start.copy()
    repeated[3] = 32.0
    finish = repeated.copy()
    finish[2], finish[3] = 1.0, 48.0
    return [baseline, reset, start, repeated, finish]


def _record_session_laps(geometry: BoundaryGeometry) -> list[Demonstration]:
    frames = (
        _lap_frames(16.0, 32.0, 36_000.0)
        + _lap_frames(10.0, 20.0, 36_500.0)
        + _lap_frames(10.0, 20.0, 38_000.0)
    )
    request = DemonstrationSessionRequest(
        _TelemetryClient(frames),
        _config(geometry),
        geometry,
        DemonstrationSessionConfig(3, 1.0, status=lambda _message: None),
    )
    return record_demonstration_session(request)


def test_demonstration_rejects_action_repeat_mismatch(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap.npz", _demonstration(geometry))

    context = DemonstrationTransitionContext(_config(geometry, action_repeat_frames=4), geometry)
    with pytest.raises(ValueError, match="action repeat"):
        demonstration_transitions(path, _IdentityPipeline(), context)


def test_demonstration_import_preserves_completed_recovery_lap(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    recovery = _recovery(demonstration)
    path = save_demonstration(tmp_path / "recovery.npz", recovery)
    config = _config(geometry).model_copy(update={"no_progress_steps": 10})

    context = DemonstrationTransitionContext(config, geometry)
    transitions = demonstration_transitions(path, _IdentityPipeline(), context)

    assert len(transitions) == len(recovery.actions)
    assert transitions[-1].terminated


def test_recorder_waits_for_restart_and_quantizes_human_control(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demo = _record(_TelemetryClient(_fresh_run_frames()), geometry)

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

    demo = _record(_TelemetryClient([baseline, reset, start, middle, finish]), geometry, 0.0)

    assert demo.controls[:, 2].tolist() == [0.0, 1.0]


def test_session_records_each_lap_and_reject_outliers_drops_slow_laps(
    tmp_path: Path,
) -> None:
    geometry = _geometry(tmp_path)
    demos = _record_session_laps(geometry)

    assert [demo.finish_time_s for demo in demos] == pytest.approx([36.0, 36.5, 38.0])
    kept = reject_outliers(demos, max_gap_s=1.0)
    assert kept == [demos[0], demos[1]]
    assert reject_outliers(demos, max_gap_s=0.0) == [demos[0]]
    with pytest.raises(ValueError, match="max_gap_s must be non-negative"):
        reject_outliers(demos, max_gap_s=-0.1)
