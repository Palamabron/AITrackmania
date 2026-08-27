"""Track-relative phase-locked demonstration replay."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    load_demonstration,
    resample_demonstration,
)
from trackmaniarl.trackmania.features import LidarFeaturePipeline


@dataclass(frozen=True, slots=True)
class PhaseLockedPathRequest:
    path: str | Path
    pipeline: LidarFeaturePipeline
    action_ids: tuple[int, ...]
    decision_interval_ms: float | None
    action_lead_ms: float = 0.0


class PhaseLockedDemonstrationPolicy:
    """Follow the expert action indexed by current track-relative vehicle state."""

    def __init__(
        self,
        reference_features: np.ndarray,
        reference_actions: np.ndarray,
        action_ids: tuple[int, ...],
    ) -> None:
        if reference_features.ndim != 2 or reference_features.shape[1] != 7:
            raise ValueError("phase-locked reference features must have shape (steps, 7)")
        if reference_actions.shape != (len(reference_features),):
            raise ValueError("phase-locked reference requires one action per feature")
        self.reference_features = reference_features.astype(np.float32, copy=True)
        self.reference_actions = self._mapped_actions(reference_actions, action_ids)
        self.action_count = len(action_ids)
        _, action_table = select_brake_tap_actions(action_ids)
        self.action_table = np.asarray(action_table, dtype=np.float32)
        self.track_relative_start = 17
        self.reset_episode()

    @staticmethod
    def _mapped_actions(reference_actions: np.ndarray, action_ids: tuple[int, ...]) -> np.ndarray:
        mapping = {action: index for index, action in enumerate(action_ids)}
        missing = sorted({int(action) for action in reference_actions} - mapping.keys())
        if missing:
            raise ValueError(f"demonstration actions are outside compact action IDs: {missing}")
        return np.asarray([mapping[int(action)] for action in reference_actions], dtype=np.int64)

    @classmethod
    def from_path(cls, request: PhaseLockedPathRequest) -> PhaseLockedDemonstrationPolicy:
        demonstration = load_demonstration(request.path)
        frames, actions = cls._resampled(
            demonstration, request.decision_interval_ms, request.action_lead_ms
        )
        start = 17 + 3 * int(request.pipeline.include_control_inputs)
        features = cls._reference_features(request.pipeline, frames, start)
        policy = cls(np.asarray(features, dtype=np.float32), actions, request.action_ids)
        policy.track_relative_start = start
        request.pipeline.reset_episode()
        return policy

    @staticmethod
    def _resampled(
        demonstration: Demonstration, decision_interval_ms: float | None, action_lead_ms: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return resample_demonstration(
            DemonstrationResamplingRequest(
                demonstration,
                decision_interval_ms,
                DemonstrationResamplingConfig(action_lead_ms),
            )
        )

    @classmethod
    def _reference_features(
        cls, pipeline: LidarFeaturePipeline, frames: np.ndarray, start: int
    ) -> list[np.ndarray]:
        pipeline.reset_episode()
        return [
            cls._features(pipeline.transform_observation(frame), start) for frame in frames[:-1]
        ]

    def reset_episode(self) -> None:
        self._reference_index = 0
        self.last_state_error = 0.0

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del mode
        current = self._features(observation, self.track_relative_start)
        index = self._nearest_reference(current)
        self.last_state_error = self._state_error(current, self.reference_features[index])
        self._reference_index = max(self._reference_index, index)
        return self._recovery_action(
            int(self.reference_actions[index]), current - self.reference_features[index]
        )

    def _nearest_reference(self, current: np.ndarray) -> int:
        state_index = self._state_reference_index(current)
        time_index = self._time_reference_index(current)
        time_floor = max(0, time_index - 16)
        monotonic_floor = min(self._reference_index, time_index)
        return max(min(state_index, time_index), time_floor, monotonic_floor)

    def _state_reference_index(self, current: np.ndarray) -> int:
        progress = self.reference_features[:, 1]
        center = int(
            np.clip(
                np.searchsorted(progress, current[1], side="left"),
                0,
                len(progress) - 1,
            )
        )
        start = max(self._reference_index - 16, center - 96, 0)
        stop = min(len(progress), max(self._reference_index + 97, center + 97))
        candidates = self.reference_features[start:stop]
        delta = candidates - current
        weights = np.asarray([0.15, 12.0, 4.0, 4.0, 2.0, 3.0, 3.0], dtype=np.float32)
        errors = np.mean(np.square(delta * weights), axis=1)
        return start + int(np.argmin(errors))

    def _time_reference_index(self, current: np.ndarray) -> int:
        return int(
            np.clip(
                np.searchsorted(self.reference_features[:, 0], current[0], side="right") - 1,
                0,
                len(self.reference_features) - 1,
            )
        )

    @staticmethod
    def _state_error(current: np.ndarray, reference: np.ndarray) -> float:
        weights = np.asarray([0.0, 12.0, 4.0, 4.0, 2.0, 3.0, 3.0], dtype=np.float32)
        return float(np.sqrt(np.mean(np.square((current - reference) * weights))))

    def _recovery_action(self, expert_action: int, current: np.ndarray) -> int:
        if self.last_state_error < 0.4:
            return expert_action
        lateral_error = float(current[2])
        heading_error = float(current[3])
        lateral_velocity = float(current[6])
        correction = -2.0 * lateral_error - heading_error - 0.5 * lateral_velocity
        if abs(correction) < 0.2:
            return expert_action
        desired_steering = float(np.sign(correction))
        expert_control = self.action_table[expert_action]
        candidates = np.flatnonzero(
            np.isclose(self.action_table[:, 0], expert_control[0])
            & np.isclose(self.action_table[:, 1], expert_control[1])
            & np.isclose(self.action_table[:, 2], desired_steering)
        )
        return expert_action if not len(candidates) else int(candidates[0])

    @staticmethod
    def _features(observation: Any, track_relative_start: int) -> np.ndarray:
        if not isinstance(observation, Mapping) or "telemetry" not in observation:
            raise TypeError("phase-locked demonstration requires lidar telemetry")
        telemetry = observation["telemetry"]
        if isinstance(telemetry, torch.Tensor):
            telemetry = telemetry.detach().float().cpu().numpy()
        values = np.asarray(telemetry, dtype=np.float32)
        current = values.reshape(-1, values.shape[-1])[-1]
        stop = track_relative_start + 6
        if current.shape[0] < stop:
            raise ValueError("phase-locked demonstration requires track-relative telemetry")
        return np.concatenate(
            (
                np.asarray([current[3] * 60.0], dtype=np.float32),
                current[track_relative_start:stop],
            )
        )
