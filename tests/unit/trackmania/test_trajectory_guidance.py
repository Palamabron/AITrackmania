from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

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


def test_trajectory_tracking_inserts_neutral_ticks_before_reversing() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        action_lead_steps=0,
        minimum_correction_steps=1,
        reversal_neutral_steps=2,
    )
    left = frame.copy()
    left[4] = 1.0
    right = frame.copy()
    right[4] = -1.0

    steering = [float(policy.act(left)[2])]
    steering.extend(float(policy.act(right)[2]) for _ in range(3))

    assert steering == [1.0, 0.0, 0.0, -1.0]
    assert policy.opposing_switch_count == 1


def test_trajectory_tracking_releases_feedback_after_returning_to_reference() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = TrajectoryTrackingDemonstrationPolicy(
        frame[None, :],
        np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        action_lead_steps=0,
        minimum_correction_steps=4,
    )
    left = frame.copy()
    left[4] = 1.0

    steering = [float(policy.act(left)[2])]
    steering.extend(float(policy.act(frame)[2]) for _ in range(4))

    assert steering == [1.0, 1.0, 1.0, 1.0, 0.0]
    assert policy.correction_count == 1


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


def test_replay_can_aggregate_native_controls_onto_the_online_decision_grid(
    tmp_path: Path,
) -> None:
    frames = np.stack([_raw_frame(float(time_ms), 0.0) for time_ms in range(0, 60, 10)])
    frames[-1, 2] = 1.0
    controls = np.asarray(
        [[1.0, 0.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 3,
        dtype=np.float32,
    )
    demonstration = Demonstration(
        map_uid="map",
        geometry_sha256="0" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=np.asarray([3, 3, 75, 75, 75], dtype=np.int64),
        controls=controls,
        finish_time_s=0.05,
        control_alignment="frame_start",
    )
    path = save_demonstration(tmp_path / "demo", demonstration)

    policy = DemonstrationReplayPolicy.from_path(
        path,
        decision_interval_ms=50.0,
        aggregate_controls=True,
    )

    assert policy.race_times_ms.tolist() == [0.0]
    assert policy.actions.tolist() == [45]


def test_phase_locked_replay_uses_the_configured_action_lead(tmp_path: Path) -> None:
    class Pipeline:
        include_control_inputs = False

        def reset_episode(self) -> None:
            return None

        def transform_observation(self, frame: np.ndarray) -> dict[str, torch.Tensor]:
            telemetry = torch.zeros(23)
            telemetry[3] = float(frame[3]) / 60_000.0
            telemetry[17] = float(frame[6])
            telemetry[20] = 1.0
            return {"telemetry": telemetry}

    frames = np.stack(
        [_raw_frame(float(time_ms), float(index)) for index, time_ms in enumerate((10, 20, 30, 40))]
    )
    frames[-1, 2] = 1.0
    demonstration = Demonstration(
        map_uid="map",
        geometry_sha256="0" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=np.asarray([39, 39, 75], dtype=np.int64),
        controls=np.asarray(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        finish_time_s=0.04,
        control_alignment="frame_start",
    )
    path = save_demonstration(tmp_path / "demo", demonstration)
    pipeline: Any = Pipeline()

    policy = PhaseLockedDemonstrationPolicy.from_path(
        path,
        pipeline,
        (39, 75),
        20.0,
        action_lead_ms=20.0,
    )

    assert policy.reference_actions.tolist() == [1, 1]


def test_phase_locked_replay_cannot_freeze_on_a_stale_matching_state() -> None:
    reference = np.zeros((30, 7), dtype=np.float32)
    reference[:, 0] = np.arange(1, 31, dtype=np.float32) / 100.0
    reference[:, 1] = np.arange(30, dtype=np.float32) / 100.0
    reference[1:, 2] = 1.0
    reference[:, 4] = 1.0
    actions = np.full(30, 75, dtype=np.int64)
    actions[0] = 3
    policy = PhaseLockedDemonstrationPolicy(
        reference,
        actions,
        (0, 1, 3, 36, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    observation = _observation([0.2, 0.0, 0.0, 1.0, 0.0, 0.0])
    observation["telemetry"][3] = 0.21 / 60.0

    assert policy.act(observation, deterministic=True) == 7


def test_phase_locked_replay_never_moves_behind_its_previous_phase() -> None:
    reference = np.zeros((3, 7), dtype=np.float32)
    reference[:, 0] = [1.0, 2.0, 3.0]
    reference[:, 1] = [0.1, 0.2, 0.3]
    reference[:, 4] = 1.0
    policy = PhaseLockedDemonstrationPolicy(
        reference,
        np.asarray([3, 39, 75], dtype=np.int64),
        (0, 1, 3, 36, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    middle = _observation([0.2, 0.0, 0.0, 1.0, 0.0, 0.0])
    middle["telemetry"][3] = 2.0 / 60.0
    stale_match = _observation([0.11, 0.0, 0.0, 1.0, 0.0, 0.0])
    stale_match["telemetry"][3] = 2.1 / 60.0

    assert policy.act(middle, deterministic=True) == 4
    assert policy.act(stale_match, deterministic=True) == 4


def test_phase_locked_replay_advances_after_state_progress_stalls() -> None:
    reference = np.zeros((280, 7), dtype=np.float32)
    reference[:, 0] = 0.01 + 0.02 * np.arange(280, dtype=np.float32)
    reference[:, 1] = 0.003 * np.arange(280, dtype=np.float32)
    reference[:, 4] = 1.0
    actions = np.full(280, 39, dtype=np.int64)
    actions[262:] = 75
    policy = PhaseLockedDemonstrationPolicy(
        reference,
        actions,
        (0, 1, 3, 36, 39, 72, 73, 75),
    )
    policy.track_relative_start = 20
    stalled = _observation([float(reference[261, 1]), 0.0, 0.0, 1.0, 0.0, 0.0])
    stalled["telemetry"][3] = float(reference[261, 0]) / 60.0

    assert policy.act(stalled, deterministic=True) == 4

    stalled["telemetry"][3] = float(reference[278, 0]) / 60.0

    assert policy.act(stalled, deterministic=True) == 7


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
