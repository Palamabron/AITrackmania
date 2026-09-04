"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.unit.learning._imitation_fixtures import (
    _RecoveryPipeline,
)
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.imitation_learning import (
    LapLoadRequest,
    load_behavior_cloning_laps,
)


def _frames(timestamps: list[float]) -> np.ndarray:
    frames = np.zeros((len(timestamps), 33), dtype=np.float32)
    frames[:, 3] = timestamps
    frames[-1, 2] = 1.0
    return frames


def _constant_control_demo(timestamps: list[float], interval_ms: float) -> Demonstration:
    transition_count = len(timestamps) - 1
    controls = np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (transition_count, 1))
    return Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=interval_ms,
        frames=_frames(timestamps),
        actions=np.full(transition_count, 39, dtype=np.int64),
        controls=controls,
        finish_time_s=timestamps[-1] / 1000.0,
    )


def _save_copies(tmp_path: Path, prefix: str, demonstration: Demonstration) -> list[Path]:
    return [save_demonstration(tmp_path / f"{prefix}-{index}", demonstration) for index in range(3)]


def _lead_demo() -> Demonstration:
    actions = np.asarray([0, 3, 39, 72, 75], dtype=np.int64)
    _, table = build_brake_tap_action_table()
    return Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=10.0,
        frames=_frames([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.05,
    )


def _aggregated_demo() -> Demonstration:
    controls = np.asarray(
        [[1.0, 0.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 3,
        dtype=np.float32,
    )
    _, table = build_brake_tap_action_table()
    return Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        frames=_frames([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
    )


def _assert_rejects_explicit_decision_interval_mismatch(
    tmp_path: Path,
) -> None:
    demonstration = _constant_control_demo([20.0, 40.0], 20.0)
    paths = _save_copies(tmp_path, "demo", demonstration)

    with pytest.raises(ValueError, match="decision interval 20ms"):
        load_behavior_cloning_laps(
            LapLoadRequest(
                paths,
                _RecoveryPipeline(),
                (0, 1, 3, 39, 72, 73, 75),
                expected_action_repeat_frames=1,
                expected_decision_interval_ms=10.0,
            )
        )


def _assert_rejects_sparse_recording_during_ingestion(tmp_path: Path) -> None:
    demonstration = _constant_control_demo([0.0, 10.0, 70.0], 10.0)
    paths = _save_copies(tmp_path, "sparse", demonstration)

    with pytest.raises(ValueError, match="telemetry cadence is too sparse"):
        load_behavior_cloning_laps(
            LapLoadRequest(
                paths,
                _RecoveryPipeline(),
                (0, 1, 3, 39, 72, 73, 75),
                expected_decision_interval_ms=10.0,
            )
        )


def test_behavior_cloning_rejects_invalid_recording_cadence(tmp_path: Path) -> None:
    _assert_rejects_explicit_decision_interval_mismatch(tmp_path)
    _assert_rejects_sparse_recording_during_ingestion(tmp_path)


def _assert_can_lead_labels_for_delayed_observations(tmp_path: Path) -> None:
    paths = _save_copies(tmp_path, "lead", _lead_demo())

    laps = load_behavior_cloning_laps(
        LapLoadRequest(
            paths,
            _RecoveryPipeline(),
            (0, 1, 3, 39, 72, 73, 75),
            expected_decision_interval_ms=10.0,
            action_lead_ms=20.0,
        )
    )

    assert laps[0].labels.tolist() == [3, 4, 6, 6, 6]


def _assert_can_ingest_aggregated_control_windows(tmp_path: Path) -> None:
    paths = _save_copies(tmp_path, "aggregate", _aggregated_demo())

    laps = load_behavior_cloning_laps(
        LapLoadRequest(
            paths,
            _RecoveryPipeline(),
            tuple(range(78)),
            expected_decision_interval_ms=50.0,
            aggregate_controls=True,
        )
    )

    assert laps[0].labels.tolist() == [45]


def test_behavior_cloning_label_alignment_modes(tmp_path: Path) -> None:
    _assert_can_lead_labels_for_delayed_observations(tmp_path)
    _assert_can_ingest_aggregated_control_windows(tmp_path)
