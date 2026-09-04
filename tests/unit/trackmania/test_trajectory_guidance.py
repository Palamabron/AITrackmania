from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.guidance import (
    PhaseLockedDemonstrationPolicy,
    TrajectoryTrackingConfig,
    TrajectoryTrackingDemonstrationPolicy,
    TrajectoryTrackingReference,
)
from trackmaniarl.trackmania.guidance_phase import PhaseLockedPathRequest


class _PhasePipeline:
    include_control_inputs = False

    def reset_episode(self) -> None:
        return None

    def transform_observation(self, frame: np.ndarray) -> dict[str, torch.Tensor]:
        telemetry = torch.zeros(23)
        telemetry[3] = float(frame[3]) / 60_000.0
        telemetry[17] = float(frame[6])
        telemetry[20] = 1.0
        return {"telemetry": telemetry}


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


def _tracking_policy(
    frames: np.ndarray, controls: np.ndarray, config: TrajectoryTrackingConfig
) -> TrajectoryTrackingDemonstrationPolicy:
    return TrajectoryTrackingDemonstrationPolicy(
        TrajectoryTrackingReference(frames, controls), config
    )


def _replay_demonstration(
    frames: np.ndarray, controls: np.ndarray, actions: np.ndarray
) -> Demonstration:
    return Demonstration(
        map_uid="map",
        geometry_sha256="0" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=actions,
        controls=controls,
        finish_time_s=float(frames[-1, 3]) / 1_000.0,
        control_alignment="frame_start",
    )


def _phase_demo_path(tmp_path: Path) -> Path:
    frames = np.stack(
        [_raw_frame(float(time_ms), float(index)) for index, time_ms in enumerate((10, 20, 30, 40))]
    )
    frames[-1, 2] = 1.0
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    demonstration = _replay_demonstration(
        frames, controls, np.asarray([39, 39, 75], dtype=np.int64)
    )
    return save_demonstration(tmp_path / "demo", demonstration)


def _phase_policy(reference: np.ndarray, actions: np.ndarray) -> PhaseLockedDemonstrationPolicy:
    policy = PhaseLockedDemonstrationPolicy(reference, actions, (0, 1, 3, 36, 39, 72, 73, 75))
    policy.track_relative_start = 20
    return policy


def _reversal_policy(frame: np.ndarray) -> TrajectoryTrackingDemonstrationPolicy:
    controls = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    config = TrajectoryTrackingConfig(
        action_lead_steps=0,
        minimum_correction_steps=1,
        reversal_neutral_steps=2,
    )
    return _tracking_policy(frame[None, :], controls, config)


def _reversal_steering(
    policy: TrajectoryTrackingDemonstrationPolicy, frame: np.ndarray
) -> list[float]:
    left, right = frame.copy(), frame.copy()
    left[4], right[4] = 1.0, -1.0
    steering = [float(policy.act(left)[2])]
    steering.extend(float(policy.act(right)[2]) for _ in range(3))
    return steering


def test_trajectory_tracking_reproduces_nominal_controls_without_feedback() -> None:
    frames = np.stack([_raw_frame(10.0, 0.0), _raw_frame(20.0, 1.0)])
    controls = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float32)
    policy = _tracking_policy(frames, controls, TrajectoryTrackingConfig(action_lead_steps=0))

    assert np.array_equal(policy.act(frames[0]), controls[0])
    assert np.array_equal(policy.act(frames[1]), controls[1])


def test_trajectory_tracking_inserts_neutral_ticks_before_reversing() -> None:
    frame = _raw_frame(10.0, 0.0)
    policy = _reversal_policy(frame)
    steering = _reversal_steering(policy, frame)

    assert steering == [1.0, 0.0, 0.0, -1.0]
    assert policy.opposing_switch_count == 1


def test_phase_locked_replay_uses_the_configured_action_lead(tmp_path: Path) -> None:
    path = _phase_demo_path(tmp_path)
    pipeline: Any = _PhasePipeline()

    policy = PhaseLockedDemonstrationPolicy.from_path(
        PhaseLockedPathRequest(path, pipeline, (39, 75), 20.0, 20.0)
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
    policy = _phase_policy(reference, actions)
    observation = _observation([0.2, 0.0, 0.0, 1.0, 0.0, 0.0])
    observation["telemetry"][3] = 0.21 / 60.0

    assert policy.act(observation, PolicyMode.EVALUATION) == 7
