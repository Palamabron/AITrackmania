"""Trajectory-guided TrackMania IQN policy with a learned recovery fallback."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

from trackmaniarl.algorithms.implicit_quantile_q_learning import ImplicitQuantileQLearning
from trackmaniarl.core.contracts import ExploratoryPolicy, Policy, ReplicablePolicy
from trackmaniarl.trackmania.actions import build_brake_tap_action_table, select_brake_tap_actions
from trackmaniarl.trackmania.demonstrations import load_demonstration, resample_demonstration
from trackmaniarl.trackmania.features import LidarFeaturePipeline


class _GuidanceFallback(ReplicablePolicy, ExploratoryPolicy, Protocol):
    def reset_episode(self) -> None: ...


class DemonstrationReplayPolicy:
    """Replays the recorded expert command selected by the current race timer."""

    def __init__(
        self,
        race_times_ms: np.ndarray,
        actions: np.ndarray,
        action_ids: tuple[int, ...] | None = None,
        *,
        action_offset_ms: float = 0.0,
    ) -> None:
        if race_times_ms.shape != actions.shape or race_times_ms.ndim != 1:
            raise ValueError("demonstration replay requires one time stamp per action")
        if len(actions) < 1 or np.any(np.diff(race_times_ms) <= 0.0):
            raise ValueError("demonstration replay time stamps must increase")
        if not np.isfinite(action_offset_ms):
            raise ValueError("demonstration action offset must be finite")
        selected_ids = tuple(range(78)) if action_ids is None else action_ids
        mapping = {action: index for index, action in enumerate(selected_ids)}
        missing = sorted({int(action) for action in actions} - mapping.keys())
        if missing:
            raise ValueError(f"demonstration actions are outside selected action IDs: {missing}")
        self.race_times_ms = race_times_ms.astype(np.float32, copy=True) + action_offset_ms
        self.actions = np.asarray([mapping[int(action)] for action in actions], dtype=np.int64)
        self.action_count = len(selected_ids)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        action_ids: tuple[int, ...] | None = None,
        *,
        action_offset_ms: float = 0.0,
    ) -> DemonstrationReplayPolicy:
        demonstration = load_demonstration(path)
        return cls(
            demonstration.frames[:-1, 3],
            demonstration.actions,
            action_ids,
            action_offset_ms=action_offset_ms,
        )

    def reset_episode(self) -> None:
        return None

    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        del deterministic
        race_time_ms = self._race_time_ms(observation)
        index = int(np.searchsorted(self.race_times_ms, race_time_ms, side="right") - 1)
        return int(self.actions[np.clip(index, 0, len(self.actions) - 1)])

    @staticmethod
    def _race_time_ms(observation: Any) -> float:
        if isinstance(observation, Mapping):
            if "raw_telemetry" in observation:
                return DemonstrationReplayPolicy._raw_race_time_ms(observation["raw_telemetry"])
            if "telemetry" not in observation:
                raise TypeError("demonstration replay requires telemetry")
            telemetry = observation["telemetry"]
            if isinstance(telemetry, torch.Tensor):
                telemetry = telemetry.detach().float().cpu().numpy()
            values = np.asarray(telemetry, dtype=np.float32)
            if values.shape[-1] < 4:
                raise ValueError("demonstration replay requires race-time telemetry")
            return float(values.reshape(-1, values.shape[-1])[-1, 3] * 60_000.0)

        return DemonstrationReplayPolicy._raw_race_time_ms(observation)

    @staticmethod
    def _raw_race_time_ms(telemetry: Any) -> float:
        if isinstance(telemetry, torch.Tensor):
            telemetry = telemetry.detach().float().cpu().numpy()
        values = np.asarray(telemetry, dtype=np.float32).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("raw demonstration replay requires one finite 33-field frame")
        return float(values[3])


class TrajectoryTrackingDemonstrationPolicy:
    """Track a recorded world-space trajectory with expert feed-forward controls."""

    def __init__(
        self,
        reference_frames: np.ndarray,
        reference_controls: np.ndarray,
        *,
        action_lead_steps: int = 1,
        action_lead_ms: float | None = None,
        lateral_gain: float = 0.8,
        heading_gain: float = 4.0,
        lateral_velocity_gain: float = 0.03,
        steering_threshold: float = 0.35,
        steering_release_threshold: float = 0.15,
        preview_ms: float = 0.0,
        minimum_correction_steps: int = 4,
        reversal_neutral_steps: int = 2,
    ) -> None:
        if reference_frames.ndim != 2 or reference_frames.shape[1] < 33:
            raise ValueError("trajectory tracking requires 33-field reference frames")
        if reference_controls.shape != (len(reference_frames), 3):
            raise ValueError("trajectory tracking requires one control per reference frame")
        if action_lead_steps < 0:
            raise ValueError("trajectory action lead must be non-negative")
        if action_lead_ms is not None and (not np.isfinite(action_lead_ms) or action_lead_ms < 0.0):
            raise ValueError("trajectory action lead milliseconds must be finite and non-negative")
        if min(lateral_gain, heading_gain, lateral_velocity_gain) < 0.0:
            raise ValueError("trajectory tracking gains must be non-negative")
        if not 0.0 < steering_threshold <= 1.0:
            raise ValueError("trajectory steering threshold must be in (0, 1]")
        if not 0.0 <= steering_release_threshold < steering_threshold:
            raise ValueError("trajectory release threshold must be below its engage threshold")
        if not np.isfinite(preview_ms) or preview_ms < 0.0:
            raise ValueError("trajectory preview milliseconds must be finite and non-negative")
        if minimum_correction_steps < 1 or reversal_neutral_steps < 0:
            raise ValueError("trajectory correction timing is invalid")
        self.reference_frames = reference_frames.astype(np.float32, copy=True)
        self.reference_controls = reference_controls.astype(np.float32, copy=True)
        self.reference_headings = self._horizontal_units(self.reference_frames[:, 10:13])
        self.reference_times_ms = self.reference_frames[:, 3].astype(np.float64)
        self.expert_steering_switch_count = int(
            np.count_nonzero(np.diff(self.reference_controls[:, 2]))
        )
        self.action_lead_steps = action_lead_steps
        self.action_lead_ms = action_lead_ms
        self.lateral_gain = lateral_gain
        self.heading_gain = heading_gain
        self.lateral_velocity_gain = lateral_velocity_gain
        self.steering_threshold = steering_threshold
        self.steering_release_threshold = steering_release_threshold
        self.preview_ms = preview_ms
        self.minimum_correction_steps = minimum_correction_steps
        self.reversal_neutral_steps = reversal_neutral_steps
        self.action_count = 78
        self.reference_index = 0
        self.last_position_error_m = 0.0
        self.last_lateral_error_m = 0.0
        self.last_heading_error = 0.0
        self.last_lateral_velocity_error_mps = 0.0
        self.correction_count = 0
        self.output_switch_count = 0
        self.opposing_switch_count = 0
        self.correction_step_count = 0
        self.neutralized_expert_step_count = 0
        self.max_position_error_m = 0.0
        self.max_abs_lateral_error_m = 0.0
        self.max_abs_heading_error = 0.0
        self.max_abs_lateral_velocity_error_mps = 0.0
        self._correction_direction = 0.0
        self._correction_hold_steps = 0
        self._neutral_steps = 0
        self._pending_direction = 0.0
        self._last_output_steering = 0.0
        self._last_current_heading = self.reference_headings[0].copy()

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        action_lead_steps: int = 1,
        action_lead_ms: float | None = None,
        lateral_gain: float = 0.8,
        heading_gain: float = 4.0,
        lateral_velocity_gain: float = 0.03,
        steering_threshold: float = 0.35,
        steering_release_threshold: float = 0.15,
        preview_ms: float = 0.0,
        minimum_correction_steps: int = 4,
        reversal_neutral_steps: int = 2,
    ) -> TrajectoryTrackingDemonstrationPolicy:
        demonstration = load_demonstration(path)
        return cls(
            demonstration.frames[:-1],
            demonstration.controls,
            action_lead_steps=action_lead_steps,
            action_lead_ms=action_lead_ms,
            lateral_gain=lateral_gain,
            heading_gain=heading_gain,
            lateral_velocity_gain=lateral_velocity_gain,
            steering_threshold=steering_threshold,
            steering_release_threshold=steering_release_threshold,
            preview_ms=preview_ms,
            minimum_correction_steps=minimum_correction_steps,
            reversal_neutral_steps=reversal_neutral_steps,
        )

    def reset_episode(self) -> None:
        self.reference_index = 0
        self.last_position_error_m = 0.0
        self.last_lateral_error_m = 0.0
        self.last_heading_error = 0.0
        self.last_lateral_velocity_error_mps = 0.0
        self.correction_count = 0
        self.output_switch_count = 0
        self.opposing_switch_count = 0
        self.correction_step_count = 0
        self.neutralized_expert_step_count = 0
        self.max_position_error_m = 0.0
        self.max_abs_lateral_error_m = 0.0
        self.max_abs_heading_error = 0.0
        self.max_abs_lateral_velocity_error_mps = 0.0
        self._correction_direction = 0.0
        self._correction_hold_steps = 0
        self._neutral_steps = 0
        self._pending_direction = 0.0
        self._last_output_steering = 0.0
        self._last_current_heading = self.reference_headings[0].copy()

    def act(self, observation: Any, *, deterministic: bool = False) -> np.ndarray:
        del deterministic
        current = self._raw_telemetry(observation)
        self.reference_index = self._nearest_reference(current)
        command_index = self._command_index()
        control = self.reference_controls[command_index].copy()
        control[2] = self._steering(control[2], current, self.reference_index)
        self._record_output(float(control[2]))
        return np.asarray(control, dtype=np.float32)

    def _command_index(self) -> int:
        if self.action_lead_ms is None:
            return min(
                self.reference_index + self.action_lead_steps,
                len(self.reference_controls) - 1,
            )
        target_ms = self.reference_times_ms[self.reference_index] + self.action_lead_ms
        index = int(np.searchsorted(self.reference_times_ms, target_ms, side="left"))
        return min(index, len(self.reference_controls) - 1)

    def _nearest_reference(self, current: np.ndarray) -> int:
        start = self.reference_index
        temporal_stop = int(
            np.searchsorted(self.reference_times_ms, float(current[3]) + 250.0, side="right")
        )
        stop = min(len(self.reference_frames), max(start + 1, temporal_stop))
        candidates = self.reference_frames[start:stop]
        position_delta = candidates[:, 4:7] - current[4:7]
        velocity_delta = candidates[:, 7:10] - current[7:10]
        current_heading = self._current_heading(current[10:13])
        headings = self.reference_headings[start:stop]
        heading_cost = 2.0 * (1.0 - np.clip(headings @ current_heading, -1.0, 1.0))
        time_delta_s = (candidates[:, 3] - current[3]) / 1_000.0
        costs = (
            np.square(position_delta[:, 0]).astype(np.float64)
            + 0.25 * np.square(position_delta[:, 1])
            + np.square(position_delta[:, 2])
            + 0.0025 * np.sum(np.square(velocity_delta), axis=1)
            + heading_cost
            + 0.01 * np.square(time_delta_s)
        )
        index = start + int(np.argmin(costs))
        self.last_position_error_m = float(np.linalg.norm(position_delta[index - start]))
        self.max_position_error_m = max(self.max_position_error_m, self.last_position_error_m)
        return index

    def _steering(self, expert_steering: float, current: np.ndarray, index: int) -> float:
        reference = self.reference_frames[index]
        heading = self.reference_headings[index]
        right = np.asarray([heading[2], 0.0, -heading[0]], dtype=np.float32)
        current_heading = self._current_heading(current[10:13])
        preview_index = self._preview_index(index)
        preview_heading = self.reference_headings[preview_index]
        preview_right = np.asarray([preview_heading[2], 0.0, -preview_heading[0]], dtype=np.float32)
        self.last_lateral_error_m = float(np.dot(current[4:7] - reference[4:7], right))
        self.last_heading_error = float(np.dot(current_heading, preview_right))
        self.last_lateral_velocity_error_mps = float(np.dot(current[7:10] - reference[7:10], right))
        self._record_errors()
        correction = -(
            self.lateral_gain * self.last_lateral_error_m
            + self.heading_gain * self.last_heading_error
            + self.lateral_velocity_gain * self.last_lateral_velocity_error_mps
        )
        direction = self._update_correction(correction)
        if not direction:
            return float(expert_steering)
        self.correction_step_count += 1
        if not expert_steering or expert_steering == direction:
            return direction
        self.neutralized_expert_step_count += 1
        return 0.0

    def _preview_index(self, index: int) -> int:
        target_ms = self.reference_times_ms[index] + self.preview_ms
        preview = int(np.searchsorted(self.reference_times_ms, target_ms, side="left"))
        return min(preview, len(self.reference_frames) - 1)

    def _update_correction(self, correction: float) -> float:
        desired = float(np.sign(correction)) if abs(correction) >= self.steering_threshold else 0.0
        if self._neutral_steps:
            self._neutral_steps -= 1
            if not self._neutral_steps and self._pending_direction:
                self._engage(self._pending_direction)
                self._pending_direction = 0.0
            return 0.0
        if not self._correction_direction:
            if desired:
                self._engage(desired)
            return self._correction_direction
        if self._correction_hold_steps:
            self._correction_hold_steps -= 1
            return self._correction_direction
        if desired and desired != self._correction_direction:
            self._pending_direction = desired
            self._correction_direction = 0.0
            self._neutral_steps = max(0, self.reversal_neutral_steps - 1)
            self.opposing_switch_count += 1
            if not self._neutral_steps:
                self._engage(desired)
                self._pending_direction = 0.0
            return self._correction_direction
        if abs(correction) <= self.steering_release_threshold:
            self._correction_direction = 0.0
        return self._correction_direction

    def _engage(self, direction: float) -> None:
        self._correction_direction = direction
        self._correction_hold_steps = self.minimum_correction_steps - 1
        self.correction_count += 1

    def _record_errors(self) -> None:
        self.max_abs_lateral_error_m = max(
            self.max_abs_lateral_error_m, abs(self.last_lateral_error_m)
        )
        self.max_abs_heading_error = max(self.max_abs_heading_error, abs(self.last_heading_error))
        self.max_abs_lateral_velocity_error_mps = max(
            self.max_abs_lateral_velocity_error_mps,
            abs(self.last_lateral_velocity_error_mps),
        )

    def _record_output(self, steering: float) -> None:
        if steering != self._last_output_steering:
            self.output_switch_count += 1
        self._last_output_steering = steering

    @staticmethod
    def _raw_telemetry(observation: Any) -> np.ndarray:
        if isinstance(observation, Mapping):
            observation = observation.get("raw_telemetry", observation.get("telemetry"))
        if isinstance(observation, torch.Tensor):
            observation = observation.detach().float().cpu().numpy()
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("trajectory tracking requires one finite 33-field telemetry frame")
        return values

    def _current_heading(self, vector: np.ndarray) -> np.ndarray:
        horizontal = np.asarray(vector, dtype=np.float32).copy()
        horizontal[1] = 0.0
        norm = float(np.linalg.norm(horizontal))
        if norm > 1e-5:
            self._last_current_heading = horizontal / norm
        return np.asarray(self._last_current_heading, dtype=np.float32)

    @staticmethod
    def _horizontal_units(vectors: np.ndarray) -> np.ndarray:
        horizontal = np.asarray(vectors, dtype=np.float32).copy()
        horizontal[:, 1] = 0.0
        norms = np.linalg.norm(horizontal, axis=1)
        result = np.empty_like(horizontal)
        previous = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        for index, norm in enumerate(norms):
            if norm > 1e-5:
                previous = horizontal[index] / norm
            result[index] = previous
        return result


def digital_recovery_steering(expert_steering: float, correction: float, threshold: float) -> float:
    """Apply a digital correction without commanding an immediate countersteer."""

    desired = 1.0 if correction > threshold else -1.0 if correction < -threshold else 0.0
    if not desired or desired == expert_steering:
        return float(expert_steering)
    return desired if expert_steering == 0.0 else 0.0


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
        mapping = {action: index for index, action in enumerate(action_ids)}
        missing = sorted({int(action) for action in reference_actions} - mapping.keys())
        if missing:
            raise ValueError(f"demonstration actions are outside compact action IDs: {missing}")
        self.reference_features = reference_features.astype(np.float32, copy=True)
        self.reference_actions = np.asarray(
            [mapping[int(action)] for action in reference_actions], dtype=np.int64
        )
        self.action_count = len(action_ids)
        _, action_table = select_brake_tap_actions(action_ids)
        self.action_table = np.asarray(action_table, dtype=np.float32)
        self.track_relative_start = 17
        self._reference_index = 0
        self.last_state_error = 0.0

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        pipeline: LidarFeaturePipeline,
        action_ids: tuple[int, ...],
        decision_interval_ms: float | None,
    ) -> PhaseLockedDemonstrationPolicy:
        demonstration = load_demonstration(path)
        frames, actions = resample_demonstration(demonstration, decision_interval_ms)
        pipeline.reset_episode()
        start = 17 + 3 * int(pipeline.include_control_inputs)
        features = [
            cls._features(pipeline.transform_observation(frame), start) for frame in frames[:-1]
        ]
        policy = cls(np.asarray(features, dtype=np.float32), actions, action_ids)
        policy.track_relative_start = start
        pipeline.reset_episode()
        return policy

    def reset_episode(self) -> None:
        self._reference_index = 0
        self.last_state_error = 0.0

    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        del deterministic
        current = self._features(observation, self.track_relative_start)
        index = self._nearest_reference(current)
        self.last_state_error = self._state_error(current, self.reference_features[index])
        self._reference_index = max(self._reference_index, index)
        return self._recovery_action(
            int(self.reference_actions[index]), current - self.reference_features[index]
        )

    def _nearest_reference(self, current: np.ndarray) -> int:
        progress = self.reference_features[:, 1]
        center = int(np.searchsorted(progress, current[1], side="left"))
        start = max(self._reference_index - 16, center - 96, 0)
        stop = min(len(progress), max(self._reference_index + 97, center + 97))
        candidates = self.reference_features[start:stop]
        delta = candidates - current
        weights = np.asarray([0.15, 12.0, 4.0, 4.0, 2.0, 3.0, 3.0], dtype=np.float32)
        errors = np.mean(np.square(delta * weights), axis=1)
        state_index = start + int(np.argmin(errors))
        time_index = int(
            np.clip(
                np.searchsorted(self.reference_features[:, 0], current[0], side="right") - 1,
                0,
                len(self.reference_features) - 1,
            )
        )
        return min(state_index, time_index)

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


class _TrajectoryGuidedPolicy:
    def __init__(
        self,
        fallback: _GuidanceFallback,
        reference_features: np.ndarray,
        reference_controls: np.ndarray,
        max_state_error: float,
        min_progress: float,
    ) -> None:
        self.fallback = fallback
        self.reference_features = reference_features
        self.reference_controls = reference_controls
        self.max_state_error = max_state_error
        self.min_progress = min_progress
        _, self.action_table = build_brake_tap_action_table()

    def act(self, observation: Any, *, deterministic: bool = False) -> Any:
        fallback_action = self.fallback.act(observation, deterministic=deterministic)
        if not deterministic:
            return fallback_action
        current = self._track_relative_features(observation)
        if current[1] < self.min_progress:
            return fallback_action
        candidate = self._nearest_reference(current)
        error = self._state_error(current, self.reference_features[candidate])
        if error > self.max_state_error:
            return fallback_action
        fallback_control = self._control(fallback_action)
        expert_control = self.reference_controls[candidate]
        return np.asarray(
            [expert_control[0], expert_control[1], fallback_control[2]], dtype=np.float32
        )

    def export_state(self) -> Mapping[str, Any]:
        return self.fallback.export_state()

    def load_state(self, state: Any) -> None:
        self.fallback.load_state(state)

    def set_exploration_epsilon(self, epsilon: float) -> None:
        self.fallback.set_exploration_epsilon(epsilon)

    def reset_episode(self) -> None:
        reset = getattr(self.fallback, "reset_episode", None)
        if callable(reset):
            reset()

    def _control(self, action: Any) -> np.ndarray:
        if isinstance(action, (int, np.integer)):
            index = int(action)
            if not 0 <= index < len(self.action_table):
                raise ValueError("fallback action is outside the canonical action table")
            return self.action_table[index]
        control = np.asarray(action, dtype=np.float32).reshape(-1)
        if control.shape != (3,):
            raise ValueError("fallback action must be an index or [gas, brake, steer]")
        return control

    def _nearest_reference(self, current: np.ndarray) -> int:
        progress = self.reference_features[:, 1]
        center = int(np.searchsorted(progress, current[1], side="left"))
        start = max(0, center - 64)
        stop = min(len(progress), center + 65)
        candidates = self.reference_features[start:stop]
        errors = np.asarray(
            [self._state_error(current, candidate) for candidate in candidates],
            dtype=np.float32,
        )
        return start + int(np.argmin(errors))

    @staticmethod
    def _state_error(current: np.ndarray, reference: np.ndarray) -> float:
        weights = np.asarray([0.5, 8.0, 3.0, 3.0, 1.0, 2.0, 2.0], dtype=np.float32)
        return float(np.sqrt(np.mean(np.square((current - reference) * weights))))

    @staticmethod
    def _track_relative_features(observation: Any) -> np.ndarray:
        if not isinstance(observation, dict) or "telemetry" not in observation:
            raise TypeError("trajectory guidance requires a lidar telemetry observation")
        telemetry = observation["telemetry"]
        if isinstance(telemetry, torch.Tensor):
            telemetry = telemetry.detach().float().cpu().numpy()
        values = np.asarray(telemetry, dtype=np.float32)
        if values.shape[-1] < 26:
            raise ValueError("trajectory guidance requires track-relative telemetry")
        current = values.reshape(-1, values.shape[-1])[-1]
        race_time_s = current[3] * 60.0
        return np.concatenate((np.asarray([race_time_s], dtype=np.float32), current[-6:]))


class DemonstrationGuidedImplicitQuantileQLearning(ImplicitQuantileQLearning):
    def __init__(
        self,
        *args: Any,
        guidance_demo_path: str | Path,
        guidance_geometry_path: str | Path,
        guidance_max_state_error: float = 0.35,
        guidance_min_progress: float = 0.8,
        base_dir: str | Path = ".",
        **kwargs: Any,
    ) -> None:
        if guidance_max_state_error <= 0.0 or not 0.0 <= guidance_min_progress <= 1.0:
            raise ValueError("guidance thresholds are invalid")
        super().__init__(*args, base_dir=base_dir, **kwargs)
        root = Path(base_dir)
        demo_path = self._resolved(root, guidance_demo_path)
        geometry_path = self._resolved(root, guidance_geometry_path)
        self.guidance_features, self.guidance_controls = self._load_reference(
            demo_path, geometry_path
        )
        self.guidance_max_state_error = guidance_max_state_error
        self.guidance_min_progress = guidance_min_progress

    def policy(self) -> Policy:
        return _TrajectoryGuidedPolicy(
            cast(_GuidanceFallback, super().policy()),
            self.guidance_features,
            self.guidance_controls,
            self.guidance_max_state_error,
            self.guidance_min_progress,
        )

    @staticmethod
    def _resolved(root: Path, path: str | Path) -> Path:
        value = Path(path)
        return value.resolve() if value.is_absolute() else (root / value).resolve()

    @staticmethod
    def _load_reference(demo_path: Path, geometry_path: Path) -> tuple[np.ndarray, np.ndarray]:
        demonstration = load_demonstration(demo_path)
        pipeline = LidarFeaturePipeline(
            geometry_path,
            expected_map_uid=demonstration.map_uid,
            samples_per_side=90,
            max_distance_m=180.0,
            include_track_relative=True,
            use_racing_line=True,
            max_speed_mps=80.0,
            velocity_to_mps_scale=1.0,
            nearest_forward_points=100,
            nearest_backward_points=10,
        )
        features = []
        for frame in demonstration.frames[:-1]:
            observation = pipeline.transform_observation(frame)
            telemetry = observation["telemetry"].numpy()
            features.append(
                np.concatenate(
                    (
                        np.asarray([telemetry[3] * 60.0], dtype=np.float32),
                        telemetry[-6:],
                    )
                )
            )
        return np.asarray(features, dtype=np.float32), demonstration.controls.copy()
