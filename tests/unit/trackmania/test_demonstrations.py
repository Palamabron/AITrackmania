from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tests.unit.trackmania._demonstration_fixtures import (
    _config,
    _demonstration,
    _geometry,
    _IdentityPipeline,
)
from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_SENTINEL,
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    DemonstrationTransitionContext,
    _control,
    demonstration_transitions,
    load_demonstration,
    resample_demonstration,
    save_demonstration,
)
from trackmaniarl.trackmania.telemetry import TelemetryFrame

DEFAULT_RESAMPLING_CONFIG = DemonstrationResamplingConfig()


def _resample(
    demonstration: Demonstration,
    decision_interval_ms: float | None,
    config: DemonstrationResamplingConfig = DEFAULT_RESAMPLING_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    return resample_demonstration(
        DemonstrationResamplingRequest(demonstration, decision_interval_ms, config)
    )


def _short_demonstration(tmp_path: Path) -> tuple[Demonstration, np.ndarray]:
    demonstration = _demonstration(_geometry(tmp_path))
    frames = demonstration.frames[:6].copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    return demonstration, frames


def _lead_demonstration(tmp_path: Path) -> tuple[Demonstration, np.ndarray]:
    demonstration, frames = _short_demonstration(tmp_path)
    actions = np.asarray([0, 1, 3, 39, 75], dtype=np.int64)
    _, table = build_brake_tap_action_table()
    updated = replace(
        demonstration,
        frames=frames,
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.05,
    )
    return updated, frames


def _controlled_demonstration(
    tmp_path: Path, controls: np.ndarray
) -> tuple[Demonstration, list[np.ndarray]]:
    demonstration, frames = _short_demonstration(tmp_path)
    _, table = build_brake_tap_action_table()
    updated = replace(
        demonstration,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
        control_alignment="frame_start",
    )
    return updated, table


def _pwm_controls() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_recorded_control_preserves_openplanet_steering_direction() -> None:
    values = np.zeros(33, dtype=np.float32)
    values[30] = -1.0
    values[31] = 1.0

    control = _control(TelemetryFrame(values))

    np.testing.assert_array_equal(control, np.asarray([1.0, 0.0, -1.0], dtype=np.float32))


def test_demonstration_round_trip_and_transition_conversion(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))

    loaded = load_demonstration(path)
    context = DemonstrationTransitionContext(_config(geometry), geometry)
    transitions = demonstration_transitions(path, _IdentityPipeline(), context)

    assert loaded.finish_time_s == 36.0
    assert loaded.decision_interval_ms is None
    assert len(transitions) == 60
    assert transitions[-1].terminated
    assert transitions[-1].info["is_demo"] is True
    assert transitions[-1].info["sampling/projected_lap_time_s"] == 36.0


def _archive_values(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


def test_demonstration_loader_rejects_an_outdated_archive(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    path = save_demonstration(tmp_path / "lap", _demonstration(geometry))
    values = _archive_values(path)
    values["format"] = np.asarray("trackmaniarl-trackmania-demo-v4")
    np.savez_compressed(path, **values)

    with pytest.raises(ValueError, match="unsupported"):
        load_demonstration(path)


def test_resample_demonstration_uses_the_online_decision_interval(tmp_path: Path) -> None:
    geometry = _geometry(tmp_path)
    demonstration = _demonstration(geometry)
    frames = demonstration.frames.copy()
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    demonstration = replace(demonstration, frames=frames, finish_time_s=0.6)

    selected_frames, selected_actions = _resample(demonstration, 20.0)

    assert np.array_equal(selected_frames[:, 3], frames[::2, 3])
    assert len(selected_actions) == len(selected_frames) - 1


def test_resample_demonstration_leads_actions_by_race_time(tmp_path: Path) -> None:
    demonstration, _ = _lead_demonstration(tmp_path)

    selected_frames, selected_actions = _resample(
        demonstration, 20.0, DemonstrationResamplingConfig(action_lead_ms=20.0)
    )

    assert selected_frames[:, 3].tolist() == [0.0, 20.0, 40.0, 50.0]
    assert selected_actions.tolist() == [3, 75, 75]


def test_resample_demonstration_aggregates_keyboard_pwm_into_analog_action(
    tmp_path: Path,
) -> None:
    demonstration, table = _controlled_demonstration(tmp_path, _pwm_controls())

    _, default_actions = _resample(demonstration, 50.0)
    _, aggregated_actions = _resample(
        demonstration, 50.0, DemonstrationResamplingConfig(aggregate_controls=True)
    )

    assert int(default_actions[0]) == int(demonstration.actions[0])
    gas, brake, steer = table[int(aggregated_actions[0])]
    assert gas == pytest.approx(1.0)
    assert brake == pytest.approx(BRAKE_TAP_SENTINEL)
    assert steer == pytest.approx(1.0 / 6.0)
