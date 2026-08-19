from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.guidance import (
    DemonstrationReplayPolicy,
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
    right = frame.copy()
    right[4] = 1.0
    left = frame.copy()
    left[4] = -1.0

    steering = [float(policy.act(right)[2])]
    steering.extend(float(policy.act(left)[2]) for _ in range(3))

    assert steering == [-1.0, 0.0, 0.0, 1.0]
    assert policy.opposing_switch_count == 1


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
