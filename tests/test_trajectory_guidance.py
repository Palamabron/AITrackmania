from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from trackmaniarl.trackmania.actions import build_brake_tap_action_table
from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.guidance import (
    DemonstrationReplayPolicy,
    PhaseLockedDemonstrationPolicy,
    TrajectoryTrackingDemonstrationPolicy,
    _TrajectoryGuidedPolicy,
)


class _FallbackPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        del observation, deterministic
        self.calls += 1
        return 7

    def export_state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: Any) -> None:
        del state

    def set_exploration_epsilon(self, epsilon: float) -> None:
        del epsilon

    def reset_episode(self) -> None:
        pass


def _observation(features: list[float]) -> dict[str, torch.Tensor]:
    telemetry = torch.zeros(26)
    telemetry[-6:] = torch.tensor(features)
    return {"telemetry": telemetry}


def _raw_frame(time_ms: float, position_z: float) -> np.ndarray:
    frame = np.zeros(33, dtype=np.float32)
    frame[3] = time_ms
    frame[6] = position_z
    frame[9] = 10.0
    frame[12] = 1.0
    return frame


def test_trajectory_tracking_reproduces_nominal_controls_without_feedback() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0)])
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    policy = TrajectoryTrackingDemonstrationPolicy(frames, controls, action_lead_steps=0)

    assert np.array_equal(policy.act(frames[0]), controls[0])
    assert np.array_equal(policy.act(frames[1]), controls[1])


def test_trajectory_tracking_does_not_anticipate_a_curve_through_feedback() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0)])
    frames[1, 10:13] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    policy = TrajectoryTrackingDemonstrationPolicy(frames, controls, action_lead_steps=0)

    assert np.array_equal(policy.act(frames[0]), controls[0])


def test_trajectory_tracking_holds_then_releases_a_correction() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        action_lead_steps=0,
        minimum_correction_steps=3,
    )
    displaced = frame.copy()
    displaced[4] = 1.0

    steering = [float(policy.act(displaced)[2])]
    steering.extend(float(policy.act(frame)[2]) for _ in range(3))

    assert steering == [-1.0, -1.0, -1.0, 0.0]


def test_trajectory_tracking_inserts_neutral_ticks_before_reversing() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        action_lead_steps=0,
        minimum_correction_steps=1,
        reversal_neutral_steps=2,
    )
    right = frame.copy()
    right[4] = 1.0
    left = frame.copy()
    left[4] = -1.0

    steering = [float(policy.act(right)[2])]
    steering.extend(float(policy.act(left)[2]) for _ in range(3))

    assert steering == [-1.0, 0.0, 0.0, 1.0]
    assert policy.opposing_switch_count == 1


def test_trajectory_tracking_uses_native_tick_lookahead() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0)])
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    policy = TrajectoryTrackingDemonstrationPolicy(frames, controls, action_lead_steps=1)

    assert np.array_equal(policy.act(frames[0]), controls[1])


def test_trajectory_tracking_uses_time_lookahead_for_irregular_frames() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(30.0, 1.0), _raw_frame(40.0, 2.0)])
    controls = np.asarray(
        [[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    policy = TrajectoryTrackingDemonstrationPolicy(
        frames,
        controls,
        action_lead_steps=0,
        action_lead_ms=20.0,
    )

    assert np.array_equal(policy.act(frames[0]), controls[1])


def test_trajectory_tracking_rejects_invalid_time_lookahead() -> None:
    frame = _raw_frame(10.0, 0.0)
    with pytest.raises(ValueError, match="milliseconds"):
        TrajectoryTrackingDemonstrationPolicy(
            frame[None, :],
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            action_lead_ms=float("nan"),
        )


def test_trajectory_tracking_corrects_rightward_displacement_to_the_left() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        action_lead_steps=0,
    )
    displaced = frame.copy()
    displaced[4] = 1.0

    action = policy.act(displaced)

    assert np.array_equal(action, np.asarray([1.0, 0.0, -1.0], dtype=np.float32))


def test_trajectory_tracking_releases_conflicting_expert_steering_before_countersteer() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32),
        action_lead_steps=0,
    )
    displaced = frame.copy()
    displaced[4] = 1.0

    action = policy.act(displaced)

    assert np.array_equal(action, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_trajectory_tracking_reference_index_never_moves_backwards() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0), _raw_frame(30.0, 2.0)])
    controls = np.asarray(
        [[1.0, 0.0, -1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    policy = TrajectoryTrackingDemonstrationPolicy(frames, controls, action_lead_steps=0)

    policy.act(frames[2])
    action = policy.act(frames[0])

    assert policy.reference_index == 2
    assert np.array_equal(action[:2], controls[2, :2])


def test_demonstration_replay_selects_the_action_for_the_current_race_time() -> None:
    policy = DemonstrationReplayPolicy(
        np.asarray([20.0, 40.0, 60.0], dtype=np.float32),
        np.asarray([3, 5, 7], dtype=np.int64),
    )
    observation = _observation([0.0] * 6)
    observation["telemetry"][3] = 45.0 / 60_000.0

    assert policy.act(observation, deterministic=True) == 5


def test_demonstration_replay_accepts_raw_identity_telemetry() -> None:
    policy = DemonstrationReplayPolicy(
        np.asarray([20.0, 40.0, 60.0], dtype=np.float32),
        np.asarray([3, 5, 7], dtype=np.int64),
    )

    assert policy.act(_raw_frame(45.0, 0.0), deterministic=True) == 5


def test_demonstration_replay_applies_a_signed_action_timestamp_offset() -> None:
    policy = DemonstrationReplayPolicy(
        np.asarray([20.0, 40.0, 60.0], dtype=np.float32),
        np.asarray([3, 5, 7], dtype=np.int64),
        action_offset_ms=10.0,
    )

    assert policy.act(_raw_frame(45.0, 0.0), deterministic=True) == 3
    assert policy.act(_raw_frame(50.0, 0.0), deterministic=True) == 5


def test_frame_start_replay_emits_transition_labels_from_the_start_state(tmp_path: Path) -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0), _raw_frame(30.0, 2.0)])
    frames[-1, 2] = 1.0
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    actions = np.asarray([39, 75], dtype=np.int64)

    demonstration = Demonstration(
        map_uid="map",
        geometry_sha256="0" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=actions,
        controls=controls,
        finish_time_s=0.03,
        control_alignment="frame_start",
    )
    path = save_demonstration(tmp_path / "demo", demonstration)

    policy = DemonstrationReplayPolicy.from_path(path)

    assert policy.race_times_ms.tolist() == [10.0, 20.0]


def test_frame_start_trajectory_tracker_emits_controls_from_the_start_state(
    tmp_path: Path,
) -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0), _raw_frame(30.0, 2.0)])
    frames[-1, 2] = 1.0
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    demonstration = Demonstration(
        map_uid="map",
        geometry_sha256="0" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=np.asarray([39, 75], dtype=np.int64),
        controls=controls,
        finish_time_s=0.03,
        control_alignment="frame_start",
    )
    path = save_demonstration(tmp_path / "demo", demonstration)

    policy = TrajectoryTrackingDemonstrationPolicy.from_path(path, action_lead_steps=0)

    assert policy.reference_times_ms.tolist() == [10.0, 20.0]
    assert np.array_equal(policy.act(frames[0]), controls[0])


def test_demonstration_replay_remaps_canonical_actions_to_compact_indices() -> None:
    policy = DemonstrationReplayPolicy(
        np.asarray([20.0, 40.0, 60.0], dtype=np.float32),
        np.asarray([3, 39, 75], dtype=np.int64),
        (0, 1, 3, 39, 72, 73, 75),
    )
    observation = _observation([0.0] * 6)
    observation["telemetry"][3] = 45.0 / 60_000.0

    assert policy.action_count == 7
    assert policy.act(observation, deterministic=True) == 3


def test_phase_locked_replay_uses_vehicle_state_and_returns_compact_action() -> None:
    policy = PhaseLockedDemonstrationPolicy(
        np.asarray(
            [
                [1.0, 0.1, 0.0, 0.0, 1.0, 0.4, 0.0],
                [2.0, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0],
                [3.0, 0.3, 0.0, 0.0, 1.0, 0.7, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0, 39, 75], dtype=np.int64),
        (0, 1, 3, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    observation = _observation([0.2, 0.0, 0.0, 1.0, 0.6, 0.0])
    observation["telemetry"][3] = 10.0 / 60.0

    assert policy.act(observation, deterministic=True) == 3


def test_phase_locked_replay_corrects_lateral_error_without_changing_throttle() -> None:
    policy = PhaseLockedDemonstrationPolicy(
        np.asarray([[1.0, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0]], dtype=np.float32),
        np.asarray([39], dtype=np.int64),
        (0, 1, 3, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    observation = _observation([0.2, 0.3, 0.0, 1.0, 0.6, 0.0])

    assert policy.act(observation, deterministic=True) == 2


def test_phase_locked_replay_never_selects_an_action_from_future_race_time() -> None:
    policy = PhaseLockedDemonstrationPolicy(
        np.asarray(
            [
                [1.0, 0.1, 0.0, 0.0, 1.0, 0.4, 0.0],
                [2.0, 0.2, 0.0, 0.0, 1.0, 0.6, 0.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([39, 75], dtype=np.int64),
        (0, 1, 3, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    observation = _observation([0.2, 0.0, 0.0, 1.0, 0.6, 0.0])
    observation["telemetry"][3] = 1.5 / 60.0

    assert policy.act(observation, deterministic=True) == 3


def test_guidance_uses_nearest_expert_action_for_a_close_state() -> None:
    fallback = _FallbackPolicy()
    policy = _TrajectoryGuidedPolicy(
        fallback,
        np.asarray(
            [[1.0, 0.1, 0, 0, 1, 0.5, 0], [2.0, 0.2, 0, 0, 1, 0.6, 0]],
            dtype=np.float32,
        ),
        np.asarray([[1.0, 0.0, -0.5], [1.0, 0.0, 0.5]], dtype=np.float32),
        0.35,
        0.0,
    )

    observation = _observation([0.2, 0, 0, 1, 0.6, 0])
    observation["telemetry"][3] = 2.0 / 60.0
    action = policy.act(observation, deterministic=True)

    _, table = build_brake_tap_action_table()
    assert np.array_equal(action, np.asarray([1.0, 0.0, table[7][2]], dtype=np.float32))
    assert fallback.calls == 1


def test_guidance_preserves_policy_steering_and_uses_expert_longitudinal_control() -> None:
    fallback = _FallbackPolicy()
    policy = _TrajectoryGuidedPolicy(
        fallback,
        np.asarray([[2.0, 0.2, 0, 0, 1, 0.6, 0]], dtype=np.float32),
        np.asarray([[0.0, 1.0, 1.0]], dtype=np.float32),
        0.35,
        0.0,
    )
    observation = _observation([0.2, 0, 0, 1, 0.6, 0])
    observation["telemetry"][3] = 2.0 / 60.0

    action = policy.act(observation, deterministic=True)

    _, table = build_brake_tap_action_table()
    assert np.array_equal(action, np.asarray([0.0, 1.0, table[7][2]], dtype=np.float32))


def test_guidance_falls_back_when_the_car_leaves_the_expert_trajectory() -> None:
    fallback = _FallbackPolicy()
    policy = _TrajectoryGuidedPolicy(
        fallback,
        np.asarray([[2.0, 0.2, 0, 0, 1, 0.6, 0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32),
        0.35,
        0.0,
    )

    observation = _observation([0.2, 1, 1, 0, 0, 1])
    observation["telemetry"][3] = 2.0 / 60.0
    action = policy.act(observation, deterministic=True)

    assert action == 7


def test_guidance_does_not_override_exploratory_rollouts() -> None:
    fallback = _FallbackPolicy()
    policy = _TrajectoryGuidedPolicy(
        fallback,
        np.asarray([[2.0, 0.2, 0, 0, 1, 0.6, 0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32),
        0.35,
        0.0,
    )

    assert policy.act(_observation([0.2, 0, 0, 1, 0.6, 0])) == 7


def test_guidance_preserves_fallback_before_the_configured_lap_segment() -> None:
    fallback = _FallbackPolicy()
    policy = _TrajectoryGuidedPolicy(
        fallback,
        np.asarray([[2.0, 0.2, 0, 0, 1, 0.6, 0]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.5]], dtype=np.float32),
        0.35,
        0.8,
    )

    observation = _observation([0.2, 0, 0, 1, 0.6, 0])
    observation["telemetry"][3] = 2.0 / 60.0

    assert policy.act(observation, deterministic=True) == 7
