"""Configuration and validated dataset types for synthetic recovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.imitation_learning import (
    RecoveryArrays,
    RecoveryMetadata,
    RecoveryProvenance,
    RecoverySaveRequest,
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
        _validate_recovery_config(self)


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
        _validate_recovery_dataset(self)

    def save(self, path: str | Path) -> Path:
        arrays = RecoveryArrays(
            self.frames,
            self.labels,
            self.episode_starts,
            self.action_ids,
        )
        metadata = RecoveryMetadata(
            sample_weights=self.sample_weights,
            interventions=self.interventions,
            state_errors=self.state_errors,
        )
        request = RecoverySaveRequest(
            path,
            arrays,
            self.provenance,
            metadata,
        )
        return save_behavior_cloning_recovery(request)


def _validate_recovery_config(config: SyntheticRecoveryConfig) -> None:
    _validate_positive_config(config)
    non_negative = (
        config.action_lead_ms,
        config.lateral_gain,
        config.heading_gain,
        config.lateral_velocity_gain,
    )
    if not all(np.isfinite(value) and value >= 0.0 for value in non_negative):
        raise ValueError("synthetic recovery gains and action lead must be non-negative")
    _validate_recovery_weights(config)


def _validate_positive_config(config: SyntheticRecoveryConfig) -> None:
    positive = (
        config.minimum_speed_mps,
        config.lateral_offset_m,
        config.heading_offset_rad,
        config.lateral_velocity_offset_mps,
        config.maximum_recovery_weight,
        config.steering_threshold,
    )
    valid = config.sample_stride >= 1 and all(
        np.isfinite(value) and value > 0.0 for value in positive
    )
    if not valid:
        raise ValueError("synthetic recovery configuration must be finite and positive")


def _validate_recovery_weights(config: SyntheticRecoveryConfig) -> None:
    if not 0.0 < config.nominal_sample_weight <= config.maximum_recovery_weight:
        raise ValueError("synthetic recovery sample weights are invalid")
    if config.maximum_recovery_weight < 1.0:
        raise ValueError("maximum synthetic recovery weight must be at least one")
    if config.steering_threshold > 1.0:
        raise ValueError("synthetic recovery steering threshold must not exceed one")


def _validate_recovery_dataset(dataset: SyntheticRecoveryDataset) -> None:
    sample_count = len(dataset.frames)
    if dataset.frames.shape != (sample_count, 33) or sample_count < 1:
        raise ValueError("synthetic recovery frames must have shape (samples, 33)")
    _validate_metadata_shapes(dataset, sample_count)
    _validate_dataset_labels(dataset)
    _validate_dataset_values(dataset)
    if not bool(dataset.episode_starts[0]):
        raise ValueError("synthetic recovery data must begin with an episode")


def _validate_metadata_shapes(dataset: SyntheticRecoveryDataset, sample_count: int) -> None:
    arrays = (
        dataset.labels,
        dataset.episode_starts,
        dataset.sample_weights,
        dataset.interventions,
        dataset.state_errors,
    )
    if any(values.shape != (sample_count,) for values in arrays):
        raise ValueError("synthetic recovery metadata must match frames")


def _validate_dataset_labels(dataset: SyntheticRecoveryDataset) -> None:
    labels_valid = bool(dataset.action_ids) and bool(np.all(dataset.labels >= 0))
    labels_valid = labels_valid and bool(np.all(dataset.labels < len(dataset.action_ids)))
    if not labels_valid:
        raise ValueError("synthetic recovery labels violate the compact action contract")


def _validate_dataset_values(dataset: SyntheticRecoveryDataset) -> None:
    if not np.isfinite(dataset.frames).all() or not np.isfinite(dataset.sample_weights).all():
        raise ValueError("synthetic recovery data must be finite")
    weights_valid = np.all(dataset.sample_weights > 0.0)
    if not weights_valid or not np.isfinite(dataset.state_errors).all():
        raise ValueError("synthetic recovery weights and errors are invalid")
