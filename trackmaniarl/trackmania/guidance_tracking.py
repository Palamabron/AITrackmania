"""World-space demonstration trajectory tracking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.trackmania.demonstrations import load_demonstration


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingConfig:
    action_lead_steps: int = 1
    action_lead_ms: float | None = None
    lateral_gain: float = 0.8
    heading_gain: float = 4.0
    lateral_velocity_gain: float = 0.03
    steering_threshold: float = 0.35
    steering_release_threshold: float = 0.15
    preview_ms: float = 0.0
    minimum_correction_steps: int = 4
    reversal_neutral_steps: int = 2


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingReference:
    frames: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True, slots=True)
class TrajectoryTrackingPathRequest:
    path: str | Path
    config: TrajectoryTrackingConfig = TrajectoryTrackingConfig()


class TrajectoryTrackingDemonstrationPolicy:
    """Track a recorded world-space trajectory with expert feed-forward controls."""

    requires_raw_observation = True

    def __init__(
        self,
        reference: TrajectoryTrackingReference,
        config: TrajectoryTrackingConfig | None = None,
    ) -> None:
        config = config or TrajectoryTrackingConfig()
        self._validate_reference(reference.frames, reference.controls)
        self._validate_config(config)
        self._initialize_reference(reference.frames, reference.controls)
        self._initialize_config(config)
        self.action_count = 78
        self.reset_episode()

    @staticmethod
    def _validate_reference(frames: np.ndarray, controls: np.ndarray) -> None:
        if frames.ndim != 2 or frames.shape[1] < 33:
            raise ValueError("trajectory tracking requires 33-field reference frames")
        if controls.shape != (len(frames), 3):
            raise ValueError("trajectory tracking requires one control per reference frame")

    @staticmethod
    def _validate_config(config: TrajectoryTrackingConfig) -> None:
        if config.action_lead_steps < 0:
            raise ValueError("trajectory action lead must be non-negative")
        lead_ms = config.action_lead_ms
        if lead_ms is not None and (not np.isfinite(lead_ms) or lead_ms < 0.0):
            raise ValueError("trajectory action lead milliseconds must be finite and non-negative")
        if min(config.lateral_gain, config.heading_gain, config.lateral_velocity_gain) < 0.0:
            raise ValueError("trajectory tracking gains must be non-negative")
        TrajectoryTrackingDemonstrationPolicy._validate_thresholds(config)
        if not np.isfinite(config.preview_ms) or config.preview_ms < 0.0:
            raise ValueError("trajectory preview milliseconds must be finite and non-negative")
        if config.minimum_correction_steps < 1 or config.reversal_neutral_steps < 0:
            raise ValueError("trajectory correction timing is invalid")

    @staticmethod
    def _validate_thresholds(config: TrajectoryTrackingConfig) -> None:
        if not 0.0 < config.steering_threshold <= 1.0:
            raise ValueError("trajectory steering threshold must be in (0, 1]")
        if not 0.0 <= config.steering_release_threshold < config.steering_threshold:
            raise ValueError("trajectory release threshold must be below its engage threshold")

    def _initialize_reference(self, frames: np.ndarray, controls: np.ndarray) -> None:
        reference_frames = frames.astype(np.float32, copy=True)
        reference_controls = controls.astype(np.float32, copy=True)
        self.reference_frames = reference_frames
        self.reference_controls = reference_controls
        self.reference_headings = self._horizontal_units(self.reference_frames[:, 10:13])
        self.reference_times_ms = self.reference_frames[:, 3].astype(np.float64)
        self.expert_steering_switch_count = int(
            np.count_nonzero(np.diff(self.reference_controls[:, 2]))
        )

    def _initialize_config(self, config: TrajectoryTrackingConfig) -> None:
        self.action_lead_steps = config.action_lead_steps
        self.action_lead_ms = config.action_lead_ms
        self.lateral_gain = config.lateral_gain
        self.heading_gain = config.heading_gain
        self.lateral_velocity_gain = config.lateral_velocity_gain
        self.steering_threshold = config.steering_threshold
        self.steering_release_threshold = config.steering_release_threshold
        self.preview_ms = config.preview_ms
        self.minimum_correction_steps = config.minimum_correction_steps
        self.reversal_neutral_steps = config.reversal_neutral_steps

    def reset_episode(self) -> None:
        self._reset_error_metrics()
        self._reset_correction_state()

    def _reset_error_metrics(self) -> None:
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

    def _reset_correction_state(self) -> None:
        self._correction_direction = 0.0
        self._correction_hold_steps = 0
        self._neutral_steps = 0
        self._pending_direction = 0.0
        self._last_output_steering = 0.0
        self._last_current_heading = self.reference_headings[0].copy()

    @classmethod
    def from_path(
        cls, request: TrajectoryTrackingPathRequest
    ) -> TrajectoryTrackingDemonstrationPolicy:
        demonstration = load_demonstration(request.path)
        reference = TrajectoryTrackingReference(demonstration.frames[:-1], demonstration.controls)
        return cls(reference, request.config)

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> np.ndarray:
        del mode
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
        start, stop = self._reference_window(current)
        candidates = self.reference_frames[start:stop]
        position_delta = candidates[:, 4:7] - current[4:7]
        costs = self._reference_costs(current, candidates, start)
        index = start + int(np.argmin(costs))
        self._record_position_error(position_delta[index - start])
        return index

    def _reference_window(self, current: np.ndarray) -> tuple[int, int]:
        start = self.reference_index
        temporal_stop = int(
            np.searchsorted(self.reference_times_ms, float(current[3]) + 250.0, side="right")
        )
        stop = min(len(self.reference_frames), max(start + 1, temporal_stop))
        return start, stop

    def _reference_costs(
        self, current: np.ndarray, candidates: np.ndarray, start: int
    ) -> np.ndarray:
        position_delta = candidates[:, 4:7] - current[4:7]
        velocity_delta = candidates[:, 7:10] - current[7:10]
        current_heading = self._current_heading(current[10:13])
        headings = self.reference_headings[start : start + len(candidates)]
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
        return np.asarray(costs, dtype=np.float64)

    def _record_position_error(self, delta: np.ndarray) -> None:
        self.last_position_error_m = float(np.linalg.norm(delta))
        self.max_position_error_m = max(self.max_position_error_m, self.last_position_error_m)

    def _steering(self, expert_steering: float, current: np.ndarray, index: int) -> float:
        correction = self._steering_correction(current, index)
        direction = self._update_correction(correction)
        if not direction:
            return float(expert_steering)
        return self._corrected_steering(expert_steering, direction)

    def _steering_correction(self, current: np.ndarray, index: int) -> float:
        reference = self.reference_frames[index]
        heading = self.reference_headings[index]
        right = np.asarray([-heading[2], 0.0, heading[0]], dtype=np.float32)
        current_heading = self._current_heading(current[10:13])
        preview_index = self._preview_index(index)
        preview_heading = self.reference_headings[preview_index]
        preview_right = np.asarray([-preview_heading[2], 0.0, preview_heading[0]], dtype=np.float32)
        self.last_lateral_error_m = float(np.dot(current[4:7] - reference[4:7], right))
        self.last_heading_error = float(np.dot(current_heading, preview_right))
        self.last_lateral_velocity_error_mps = float(np.dot(current[7:10] - reference[7:10], right))
        self._record_errors()
        return -(
            self.lateral_gain * self.last_lateral_error_m
            + self.heading_gain * self.last_heading_error
            + self.lateral_velocity_gain * self.last_lateral_velocity_error_mps
        )

    def _corrected_steering(self, expert_steering: float, direction: float) -> float:
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
            return self._advance_neutral_period()
        if not self._correction_direction:
            if desired:
                self._engage(desired)
            return self._correction_direction
        if self._correction_hold_steps:
            self._correction_hold_steps -= 1
            return self._correction_direction
        if desired and desired != self._correction_direction:
            return self._reverse_correction(desired)
        if abs(correction) <= self.steering_release_threshold:
            self._correction_direction = 0.0
        return self._correction_direction

    def _advance_neutral_period(self) -> float:
        self._neutral_steps -= 1
        if not self._neutral_steps and self._pending_direction:
            self._engage(self._pending_direction)
            self._pending_direction = 0.0
        return 0.0

    def _reverse_correction(self, desired: float) -> float:
        self._pending_direction = desired
        self._correction_direction = 0.0
        self._neutral_steps = max(0, self.reversal_neutral_steps - 1)
        self.opposing_switch_count += 1
        if not self._neutral_steps:
            self._engage(desired)
            self._pending_direction = 0.0
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
