"""Deterministic counterfactual recovery data around an expert trajectory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    select_brake_tap_actions,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    load_demonstration,
    resample_demonstration,
)
from trackmaniarl.trackmania.guidance import digital_recovery_steering
from trackmaniarl.trackmania.imitation_learning import (
    RecoveryContract,
    RecoveryProvenance,
)
from trackmaniarl.trackmania.synthetic_recovery_types import (
    SyntheticRecoveryConfig,
    SyntheticRecoveryDataset,
)


@dataclass(frozen=True, slots=True)
class _Perturbation:
    lateral_m: float
    heading_rad: float
    lateral_velocity_mps: float

    @property
    def is_nominal(self) -> bool:
        return self.lateral_m == self.heading_rad == self.lateral_velocity_mps == 0.0


@dataclass(frozen=True, slots=True)
class _GenerationContext:
    demonstration: Demonstration
    config: SyntheticRecoveryConfig
    compact_table: np.ndarray
    reference_frames: np.ndarray
    eligible_indices: np.ndarray


@dataclass(slots=True)
class _RecoverySamples:
    frames: list[np.ndarray] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    interventions: list[bool] = field(default_factory=list)
    errors: list[float] = field(default_factory=list)
    episode_starts: list[bool] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _AlignmentRequest:
    demonstration: Demonstration
    contract: RecoveryContract
    aggregate_controls: bool


@dataclass(frozen=True, slots=True)
class _AlignedData:
    frames: np.ndarray
    actions: np.ndarray
    controls: np.ndarray


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryRequest:
    demonstration: Demonstration
    action_ids: tuple[int, ...]
    config: SyntheticRecoveryConfig = field(default_factory=SyntheticRecoveryConfig)
    provenance: RecoveryProvenance | None = None


@dataclass(frozen=True, slots=True)
class SyntheticRecoveryPathRequest:
    demonstration_path: str | Path
    action_ids: tuple[int, ...]
    config: SyntheticRecoveryConfig = field(default_factory=SyntheticRecoveryConfig)
    contract: RecoveryContract | None = None
    aggregate_controls: bool = False


def _validate_demonstration(
    demonstration: Demonstration, action_ids: tuple[int, ...], frames: np.ndarray
) -> None:
    if frames.shape[1] != 33:
        raise ValueError("synthetic recovery requires the raw 33-field telemetry schema")
    missing = sorted({int(action) for action in demonstration.actions} - set(action_ids))
    if missing:
        raise ValueError(f"demonstration actions are outside compact action IDs: {missing}")


def _collect_samples(context: _GenerationContext) -> _RecoverySamples:
    samples = _RecoverySamples()
    for perturbation in _perturbations(context.config):
        _collect_perturbation(samples, context, perturbation)
    return samples


def _collect_perturbation(
    samples: _RecoverySamples, context: _GenerationContext, perturbation: _Perturbation
) -> None:
    for index in context.eligible_indices:
        command = _command_index(context.reference_frames, index, context.config)
        expert = context.demonstration.controls[command]
        control, error = _recovery_control(expert, perturbation, context.config)
        samples.frames.append(_perturb_frame(context.reference_frames[index], perturbation))
        samples.labels.append(_compact_label(control, expert, context.compact_table))
        samples.weights.append(_sample_weight(perturbation, context.config))
        samples.interventions.append(not perturbation.is_nominal)
        samples.errors.append(error)
        samples.episode_starts.append(True)


def _build_dataset(
    samples: _RecoverySamples, action_ids: tuple[int, ...], provenance: RecoveryProvenance
) -> SyntheticRecoveryDataset:
    return SyntheticRecoveryDataset(
        frames=np.asarray(samples.frames, dtype=np.float32),
        labels=np.asarray(samples.labels, dtype=np.int64),
        episode_starts=np.asarray(samples.episode_starts, dtype=np.bool_),
        sample_weights=np.asarray(samples.weights, dtype=np.float32),
        interventions=np.asarray(samples.interventions, dtype=np.bool_),
        state_errors=np.asarray(samples.errors, dtype=np.float32),
        action_ids=action_ids,
        provenance=provenance,
    )


def generate_synthetic_recovery(request: SyntheticRecoveryRequest) -> SyntheticRecoveryDataset:
    """Build monotonic counterfactual trajectories around one demonstration."""

    context = _generation_context(request.demonstration, request.action_ids, request.config)
    samples = _collect_samples(context)
    if not samples.frames:
        raise ValueError("demonstration has no eligible synthetic recovery frames")
    source = request.provenance or RecoveryProvenance.from_demonstration(request.demonstration)
    return _build_dataset(samples, request.action_ids, source)


def _generation_context(
    demonstration: Demonstration,
    action_ids: tuple[int, ...],
    config: SyntheticRecoveryConfig,
) -> _GenerationContext:
    _, action_table = select_brake_tap_actions(action_ids)
    reference_frames = demonstration.frames[:-1]
    _validate_demonstration(demonstration, action_ids, reference_frames)
    return _GenerationContext(
        demonstration,
        config,
        np.asarray(action_table, dtype=np.float32),
        reference_frames,
        _eligible_indices(reference_frames, config),
    )


def generate_synthetic_recovery_from_path(
    request: SyntheticRecoveryPathRequest,
) -> SyntheticRecoveryDataset:
    source = load_demonstration(request.demonstration_path)
    selected_contract = request.contract or RecoveryContract.from_demonstration(source)
    alignment = _AlignmentRequest(source, selected_contract, request.aggregate_controls)
    demonstration = _align_demonstration(alignment)
    provenance = _recovery_provenance(source, selected_contract)
    generation = SyntheticRecoveryRequest(
        demonstration,
        request.action_ids,
        request.config,
        provenance,
    )
    return generate_synthetic_recovery(generation)


def _recovery_provenance(
    demonstration: Demonstration, contract: RecoveryContract
) -> RecoveryProvenance:
    return RecoveryProvenance.from_demonstration(demonstration, contract=contract)


def _align_demonstration(
    request: _AlignmentRequest,
) -> Demonstration:
    demonstration, contract = request.demonstration, request.contract
    _validate_recovery_contract(demonstration, contract)
    if contract.decision_interval_ms is None:
        return _unaligned_demonstration(demonstration, contract)
    frames, actions = resample_demonstration(
        DemonstrationResamplingRequest(
            demonstration,
            contract.decision_interval_ms,
            DemonstrationResamplingConfig(aggregate_controls=request.aggregate_controls),
        )
    )
    _, action_table = build_brake_tap_action_table()
    controls = np.asarray(action_table, dtype=np.float32)[actions]
    return _aligned_demonstration(demonstration, contract, _AlignedData(frames, actions, controls))


def _validate_recovery_contract(demonstration: Demonstration, contract: RecoveryContract) -> None:
    if demonstration.map_uid != contract.map_uid:
        raise ValueError("synthetic recovery map UID does not match its target contract")
    if demonstration.geometry_sha256 != contract.geometry_sha256:
        raise ValueError("synthetic recovery geometry does not match its target contract")


def _unaligned_demonstration(
    demonstration: Demonstration, contract: RecoveryContract
) -> Demonstration:
    if (
        demonstration.decision_interval_ms is not None
        or demonstration.action_repeat_frames != contract.action_repeat_frames
    ):
        raise ValueError("synthetic recovery action repeat does not match its target contract")
    return demonstration


def _aligned_demonstration(
    demonstration: Demonstration, contract: RecoveryContract, data: _AlignedData
) -> Demonstration:
    return Demonstration(
        map_uid=demonstration.map_uid,
        geometry_sha256=demonstration.geometry_sha256,
        action_repeat_frames=contract.action_repeat_frames,
        decision_interval_ms=contract.decision_interval_ms,
        frames=data.frames,
        actions=data.actions,
        controls=data.controls,
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
    components = _recovery_components(perturbation, config, heading_error)
    control = np.asarray(expert, dtype=np.float32).copy()
    if _requires_correction(perturbation, heading_error):
        control[2] = digital_recovery_steering(
            float(expert[2]), -float(components.sum()), config.steering_threshold
        )
    return control, float(np.linalg.norm(components))


def _recovery_components(
    perturbation: _Perturbation, config: SyntheticRecoveryConfig, heading_error: float
) -> np.ndarray:
    return np.asarray(
        (
            config.lateral_gain * perturbation.lateral_m,
            config.heading_gain * heading_error,
            config.lateral_velocity_gain * perturbation.lateral_velocity_mps,
        ),
        dtype=np.float64,
    )


def _requires_correction(perturbation: _Perturbation, heading_error: float) -> bool:
    return (
        abs(perturbation.lateral_m) > 0.30
        or abs(heading_error) > 0.05
        or abs(perturbation.lateral_velocity_mps) > 1.5
    )


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
