"""Deterministic counterfactual recovery data around an expert trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    select_brake_tap_actions,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    load_demonstration,
    resample_demonstration,
)
from trackmaniarl.trackmania.guidance import digital_recovery_steering
from trackmaniarl.trackmania.imitation_learning import (
    RecoveryContract,
    RecoveryProvenance,
    save_behavior_cloning_recovery,
)


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryConfig:
    sample_stride: int = 4
    minimum_speed_mps: float = 10.0
    lateral_offset_m: float = 0.55
    heading_offset_rad: float = 0.1
    lateral_velocity_offset_mps: float = 3.0
    action_lead_ms: float = 10.0
    nominal_sample_weight: float = 0.5
    maximum_recovery_weight: float = 3.0
    lateral_gain: float = 0.8
    heading_gain: float = 4.0
    lateral_velocity_gain: float = 0.03
    steering_threshold: float = 0.35

    def __post_init__(self) -> None:
        positive = (
            self.minimum_speed_mps,
            self.lateral_offset_m,
            self.heading_offset_rad,
            self.lateral_velocity_offset_mps,
            self.maximum_recovery_weight,
            self.steering_threshold,
        )
        non_negative = (
            self.action_lead_ms,
            self.lateral_gain,
            self.heading_gain,
            self.lateral_velocity_gain,
        )
        if self.sample_stride < 1 or not all(np.isfinite(value) for value in positive):
            raise ValueError("synthetic recovery configuration must be finite and positive")
        if any(value <= 0.0 for value in positive):
            raise ValueError("synthetic recovery configuration must be finite and positive")
        if not all(np.isfinite(value) and value >= 0.0 for value in non_negative):
            raise ValueError("synthetic recovery gains and action lead must be non-negative")
        if not 0.0 < self.nominal_sample_weight <= self.maximum_recovery_weight:
            raise ValueError("synthetic recovery sample weights are invalid")
        if self.maximum_recovery_weight < 1.0:
            raise ValueError("maximum synthetic recovery weight must be at least one")
        if self.steering_threshold > 1.0:
            raise ValueError("synthetic recovery steering threshold must not exceed one")


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryDataset:
    frames: np.ndarray
    labels: np.ndarray
    episode_starts: np.ndarray
    sample_weights: np.ndarray
    interventions: np.ndarray
    state_errors: np.ndarray
    action_ids: tuple[int, ...]
    provenance: RecoveryProvenance

    def __post_init__(self) -> None:
        sample_count = len(self.frames)
        if self.frames.shape != (sample_count, 33) or sample_count < 1:
            raise ValueError("synthetic recovery frames must have shape (samples, 33)")
        expected = (sample_count,)
        arrays = (
            self.labels,
            self.episode_starts,
            self.sample_weights,
            self.interventions,
            self.state_errors,
        )
        if any(values.shape != expected for values in arrays):
            raise ValueError("synthetic recovery metadata must match frames")
        if (
            not self.action_ids
            or np.any(self.labels < 0)
            or np.any(self.labels >= len(self.action_ids))
        ):
            raise ValueError("synthetic recovery labels violate the compact action contract")
        if not np.isfinite(self.frames).all() or not np.isfinite(self.sample_weights).all():
            raise ValueError("synthetic recovery data must be finite")
        if np.any(self.sample_weights <= 0.0) or not np.isfinite(self.state_errors).all():
            raise ValueError("synthetic recovery weights and errors are invalid")
        if not bool(self.episode_starts[0]):
            raise ValueError("synthetic recovery data must begin with an episode")

    def save(self, path: str | Path) -> Path:
        return save_behavior_cloning_recovery(
            path,
            self.frames,
            self.labels,
            self.episode_starts,
            self.action_ids,
            provenance=self.provenance,
            sample_weights=self.sample_weights,
            interventions=self.interventions,
            state_errors=self.state_errors,
        )


@dataclass(frozen=True, slots=True)
class _Perturbation:
    lateral_m: float
    heading_rad: float
    lateral_velocity_mps: float

    @property
    def is_nominal(self) -> bool:
        return self.lateral_m == self.heading_rad == self.lateral_velocity_mps == 0.0


def generate_synthetic_recovery(
    demonstration: Demonstration,
    action_ids: tuple[int, ...],
    config: SyntheticRecoveryConfig | None = None,
    *,
    provenance: RecoveryProvenance | None = None,
) -> SyntheticRecoveryDataset:
    """Build monotonic counterfactual trajectories around one demonstration."""

    selected = config or SyntheticRecoveryConfig()
    _, action_table = select_brake_tap_actions(action_ids)
    compact_table = np.asarray(action_table, dtype=np.float32)
    reference_frames = demonstration.frames[:-1]
    if reference_frames.shape[1] != 33:
        raise ValueError("synthetic recovery requires the raw 33-field telemetry schema")
    missing = sorted({int(action) for action in demonstration.actions} - set(action_ids))
    if missing:
        raise ValueError(f"demonstration actions are outside compact action IDs: {missing}")
    eligible = _eligible_indices(reference_frames, selected)
    perturbations = _perturbations(selected)
    frames: list[np.ndarray] = []
    labels: list[int] = []
    weights: list[float] = []
    interventions: list[bool] = []
    errors: list[float] = []
    episode_starts: list[bool] = []
    for perturbation in perturbations:
        for index in eligible:
            expert = demonstration.controls[_command_index(reference_frames, index, selected)]
            frame = _perturb_frame(reference_frames[index], perturbation)
            control, error = _recovery_control(expert, perturbation, selected)
            frames.append(frame)
            labels.append(_compact_label(control, expert, compact_table))
            weights.append(_sample_weight(perturbation, selected))
            interventions.append(not perturbation.is_nominal)
            errors.append(error)
            episode_starts.append(True)
    if not frames:
        raise ValueError("demonstration has no eligible synthetic recovery frames")
    return SyntheticRecoveryDataset(
        frames=np.asarray(frames, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        episode_starts=np.asarray(episode_starts, dtype=np.bool_),
        sample_weights=np.asarray(weights, dtype=np.float32),
        interventions=np.asarray(interventions, dtype=np.bool_),
        state_errors=np.asarray(errors, dtype=np.float32),
        action_ids=action_ids,
        provenance=provenance or RecoveryProvenance.from_demonstration(demonstration),
    )


def generate_synthetic_recovery_from_path(
    demonstration_path: str | Path,
    action_ids: tuple[int, ...],
    config: SyntheticRecoveryConfig | None = None,
    *,
    contract: RecoveryContract | None = None,
    aggregate_controls: bool = False,
) -> SyntheticRecoveryDataset:
    source = load_demonstration(demonstration_path)
    selected_contract = contract or RecoveryContract.from_demonstration(source)
    demonstration = _align_demonstration(source, selected_contract, aggregate_controls)
    return generate_synthetic_recovery(
        demonstration,
        action_ids,
        config,
        provenance=RecoveryProvenance.from_demonstration(
            source,
            contract=selected_contract,
        ),
    )


def _align_demonstration(
    demonstration: Demonstration,
    contract: RecoveryContract,
    aggregate_controls: bool,
) -> Demonstration:
    if demonstration.map_uid != contract.map_uid:
        raise ValueError("synthetic recovery map UID does not match its target contract")
    if demonstration.geometry_sha256 != contract.geometry_sha256:
        raise ValueError("synthetic recovery geometry does not match its target contract")
    if contract.decision_interval_ms is None:
        if (
            demonstration.decision_interval_ms is not None
            or demonstration.action_repeat_frames != contract.action_repeat_frames
        ):
            raise ValueError("synthetic recovery action repeat does not match its target contract")
        return demonstration
    frames, actions = resample_demonstration(
        demonstration,
        contract.decision_interval_ms,
        aggregate_controls=aggregate_controls,
    )
    _, action_table = build_brake_tap_action_table()
    controls = np.asarray(action_table, dtype=np.float32)[actions]
    return Demonstration(
        map_uid=demonstration.map_uid,
        geometry_sha256=demonstration.geometry_sha256,
        action_repeat_frames=contract.action_repeat_frames,
        decision_interval_ms=contract.decision_interval_ms,
        frames=frames,
        actions=actions,
        controls=controls,
        finish_time_s=demonstration.finish_time_s,
        control_alignment=contract.control_alignment,
    )


def _eligible_indices(frames: np.ndarray, config: SyntheticRecoveryConfig) -> np.ndarray:
    speed = frames[:, 16]
    flying_duration_ms = frames[:, 28]
    adherence = frames[:, 29]
    eligible = np.flatnonzero(
        (speed >= config.minimum_speed_mps) & (flying_duration_ms <= 50.0) & (adherence > 0.1)
    )
    return eligible[:: config.sample_stride]


def _perturbations(config: SyntheticRecoveryConfig) -> tuple[_Perturbation, ...]:
    result = [_Perturbation(0.0, 0.0, 0.0)]
    for sign in (-1.0, 1.0):
        lateral = sign * config.lateral_offset_m
        heading = sign * config.heading_offset_rad
        velocity = sign * config.lateral_velocity_offset_mps
        result.extend(
            (
                _Perturbation(lateral, 0.0, 0.0),
                _Perturbation(0.0, heading, 0.0),
                _Perturbation(0.0, 0.0, velocity),
                _Perturbation(lateral, heading, velocity),
                _Perturbation(lateral, -heading, -velocity),
            )
        )
    return tuple(result)


def _command_index(frames: np.ndarray, index: int, config: SyntheticRecoveryConfig) -> int:
    target_ms = float(frames[index, 3]) + config.action_lead_ms
    command = int(np.searchsorted(frames[:, 3], target_ms, side="left"))
    return min(command, len(frames) - 1)


def _perturb_frame(reference: np.ndarray, perturbation: _Perturbation) -> np.ndarray:
    frame = np.asarray(reference, dtype=np.float32).copy()
    forward = _horizontal_unit(frame[10:13])
    right = np.asarray([-forward[2], 0.0, forward[0]], dtype=np.float32)
    frame[4:7] += perturbation.lateral_m * right
    _rotate_yaw(frame[10:13], -perturbation.heading_rad)
    _rotate_yaw(frame[13:16], -perturbation.heading_rad)
    frame[7:10] += perturbation.lateral_velocity_mps * right
    frame[16] = float(np.linalg.norm(frame[7:10]))
    return frame


def _rotate_yaw(vector: np.ndarray, angle: float) -> None:
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    x, z = float(vector[0]), float(vector[2])
    vector[0] = x * cosine + z * sine
    vector[2] = z * cosine - x * sine


def _horizontal_unit(vector: np.ndarray) -> np.ndarray:
    horizontal = np.asarray([vector[0], 0.0, vector[2]], dtype=np.float32)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-5:
        raise ValueError("synthetic recovery requires a horizontal reference heading")
    return horizontal / norm


def _recovery_control(
    expert: np.ndarray,
    perturbation: _Perturbation,
    config: SyntheticRecoveryConfig,
) -> tuple[np.ndarray, float]:
    heading_error = float(np.sin(perturbation.heading_rad))
    components = np.asarray(
        (
            config.lateral_gain * perturbation.lateral_m,
            config.heading_gain * heading_error,
            config.lateral_velocity_gain * perturbation.lateral_velocity_mps,
        ),
        dtype=np.float64,
    )
    control = np.asarray(expert, dtype=np.float32).copy()
    requires_correction = (
        abs(perturbation.lateral_m) > 0.30
        or abs(heading_error) > 0.05
        or abs(perturbation.lateral_velocity_mps) > 1.5
    )
    if requires_correction:
        control[2] = digital_recovery_steering(
            float(expert[2]), -float(components.sum()), config.steering_threshold
        )
    return control, float(np.linalg.norm(components))


def _compact_label(control: np.ndarray, expert: np.ndarray, table: np.ndarray) -> int:
    compatible = np.flatnonzero(
        np.isclose(table[:, 0], control[0]) & np.isclose(table[:, 1], control[1])
    )
    if not len(compatible):
        distances = np.sum(np.square(table - control), axis=1)
        return int(np.argmin(distances))
    steering_distance = np.abs(table[compatible, 2] - control[2])
    closest = compatible[np.isclose(steering_distance, steering_distance.min())]
    expert_match = closest[np.isclose(table[closest, 2], expert[2])]
    return int(expert_match[0] if len(expert_match) else closest[0])


def _sample_weight(
    perturbation: _Perturbation,
    config: SyntheticRecoveryConfig,
) -> float:
    if perturbation.is_nominal:
        return config.nominal_sample_weight
    severity = np.linalg.norm(
        (
            perturbation.lateral_m / config.lateral_offset_m,
            perturbation.heading_rad / config.heading_offset_rad,
            perturbation.lateral_velocity_mps / config.lateral_velocity_offset_mps,
        )
    )
    fraction = min(1.0, float(severity) / (3.0**0.5))
    return float(1.0 + fraction * (config.maximum_recovery_weight - 1.0))
