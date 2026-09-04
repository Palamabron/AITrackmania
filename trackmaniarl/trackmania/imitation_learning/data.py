"""Imitation-learning datasets, recovery archives, augmentation and batching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.imitation_learning._data_types import (
    INTERVENTION_KEY,
    RECOVERY_DATASET_FORMAT,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    BehaviorCloningLap,
    RecoveryContract,
    RecoveryProvenance,
)
from trackmaniarl.trackmania.imitation_learning._lap_loading import (
    LapLoadRequest,
    load_behavior_cloning_laps,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoveryArrays,
    RecoveryMetadata,
    RecoverySaveRequest,
)
from trackmaniarl.trackmania.imitation_learning.recovery import (
    RecoveryLoadRequest,
    load_behavior_cloning_recovery,
    save_behavior_cloning_recovery,
)


def split_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], seed: int
) -> tuple[list[BehaviorCloningLap], list[BehaviorCloningLap]]:
    generator = torch.Generator().manual_seed(seed)
    if len(laps) < 2:
        raise ValueError("behavior-cloning split requires at least two complete episodes")
    order = torch.randperm(len(laps), generator=generator).tolist()
    validation_count = max(1, round(len(laps) * 0.2))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    elite_index = max(range(len(laps)), key=lambda index: laps[index].quality_weight)
    if elite_index in validation_indices:
        replacement = training_indices[0]
        validation_indices[validation_indices.index(elite_index)] = replacement
        training_indices[0] = elite_index
    validation = [laps[index] for index in validation_indices]
    training = [laps[index] for index in training_indices]
    return training, validation


def augment_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], action_ids: tuple[int, ...]
) -> list[BehaviorCloningLap]:
    """Add a reflected copy of each local-frame demonstration lap."""

    mapping = _horizontal_flip_action_indices(action_ids)
    reflected = [
        BehaviorCloningLap(
            tuple(
                _horizontal_flip_conditioned_observation(observation, mapping)
                for observation in lap.observations
            ),
            mapping[lap.labels],
            quality_weight=lap.quality_weight,
            source_id=f"{lap.source_id}#horizontal-reflection",
        )
        for lap in laps
    ]
    return [*laps, *reflected]


def _horizontal_flip_conditioned_observation(
    observation: Mapping[str, torch.Tensor], mapping: torch.Tensor
) -> dict[str, torch.Tensor]:
    reflected = horizontal_flip_observation(observation)
    for key in ("expert_previous_action", "previous_action", STUDENT_ACTION_KEY):
        previous_action = observation.get(key)
        if previous_action is not None:
            index = int(previous_action)
            reflected[key] = (
                mapping[index].clone()
                if index < len(mapping)
                else torch.tensor(len(mapping), dtype=torch.long)
            )
    for key in (SAMPLE_WEIGHT_KEY, INTERVENTION_KEY, STATE_ERROR_KEY):
        value = observation.get(key)
        if value is not None:
            reflected[key] = value.clone()
    return reflected


def horizontal_flip_observation(
    observation: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reflect the v59 local TrackMania observation across the car forward axis."""

    lidar = observation["lidar"]
    telemetry = observation["telemetry"]
    if lidar.shape[-2] != 8 or telemetry.shape[-1] not in {46, 49}:
        raise ValueError("horizontal flip requires the 8-channel, 46- or 49-feature observation")
    reflected = {key: value.clone() for key, value in observation.items()}
    reflected["lidar"] = _horizontal_flip_lidar(lidar)
    reflected["telemetry"] = _horizontal_flip_telemetry(telemetry)
    return reflected


def _horizontal_flip_lidar(lidar: torch.Tensor) -> torch.Tensor:
    reflected_lidar = lidar.clone()
    reflected_lidar[..., 0, :] = -lidar[..., 2, :]
    reflected_lidar[..., 1, :] = lidar[..., 3, :]
    reflected_lidar[..., 2, :] = -lidar[..., 0, :]
    reflected_lidar[..., 3, :] = lidar[..., 1, :]
    reflected_lidar[..., 4, :] = -lidar[..., 4, :]
    reflected_lidar[..., 5, :] = lidar[..., 5, :]
    return reflected_lidar


def _horizontal_flip_telemetry(telemetry: torch.Tensor) -> torch.Tensor:
    reflected_telemetry = _flip_primary_telemetry(telemetry)
    offset = 3 if telemetry.shape[-1] == 49 else 0
    if offset:
        reflected_telemetry[..., 17] = -telemetry[..., 17]
    lateral_indices = (18, 19, 22, 29, 31, 32, 39, 41)
    for index in lateral_indices:
        reflected_telemetry[..., index + offset] = -telemetry[..., index + offset]
    return _flip_telemetry_vectors(reflected_telemetry, telemetry, offset)


def _flip_primary_telemetry(telemetry: torch.Tensor) -> torch.Tensor:
    reflected_telemetry = telemetry.clone()
    reflected_telemetry[..., 6] = -telemetry[..., 6]
    reflected_telemetry[..., 10] = telemetry[..., 11]
    reflected_telemetry[..., 11] = telemetry[..., 10]
    reflected_telemetry[..., 12] = telemetry[..., 13]
    reflected_telemetry[..., 13] = telemetry[..., 12]
    return reflected_telemetry


def _flip_telemetry_vectors(
    reflected: torch.Tensor, telemetry: torch.Tensor, offset: int
) -> torch.Tensor:
    reflected[..., 34 + offset] = -telemetry[..., 36 + offset]
    reflected[..., 35 + offset] = telemetry[..., 37 + offset]
    reflected[..., 36 + offset] = -telemetry[..., 34 + offset]
    reflected[..., 37 + offset] = telemetry[..., 35 + offset]
    return reflected


def _horizontal_flip_action_indices(action_ids: tuple[int, ...]) -> torch.Tensor:
    _, table = select_brake_tap_actions(action_ids)
    mirrored: list[int] = []
    for control in table:
        match = next(
            (
                index
                for index, candidate in enumerate(table)
                if np.array_equal(candidate[:2], control[:2])
                and np.isclose(candidate[2], -control[2])
            ),
            None,
        )
        if match is None:
            raise ValueError("horizontal flip requires left-right paired compact actions")
        mirrored.append(match)
    return torch.tensor(mirrored, dtype=torch.long)


def flatten_behavior_cloning_laps(
    laps: Sequence[BehaviorCloningLap], indices: torch.Tensor | None = None
) -> tuple[list[Mapping[str, torch.Tensor]], torch.Tensor]:
    observations = [observation for lap in laps for observation in lap.observations]
    labels = torch.cat([lap.labels for lap in laps])
    if indices is None:
        return observations, labels
    return [observations[int(index)] for index in indices], labels[indices]


def class_weights(labels: torch.Tensor, action_count: int, *, power: float = 0.5) -> torch.Tensor:
    if not 0.0 <= power <= 1.0:
        raise ValueError("class weight power must be in [0, 1]")
    counts = torch.bincount(labels, minlength=action_count).float()
    observed = counts > 0
    if not bool(observed.any()):
        raise ValueError("behavior cloning labels must not be empty")
    weights = torch.ones_like(counts)
    weights[observed] = counts[observed].pow(-power)
    weights[observed] /= weights[observed].mean()
    return weights.clamp(0.5, 3.0)


def collate_behavior_cloning(
    observations: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not observations:
        raise ValueError("behavior cloning batch must not be empty")
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy tensor state before the next optimizer update mutates it."""

    return deepcopy(dict(state))


__all__ = [
    "INTERVENTION_KEY",
    "RECOVERY_DATASET_FORMAT",
    "SAMPLE_WEIGHT_KEY",
    "STATE_ERROR_KEY",
    "STUDENT_ACTION_KEY",
    "BehaviorCloningLap",
    "LapLoadRequest",
    "RecoveryArrays",
    "RecoveryContract",
    "RecoveryLoadRequest",
    "RecoveryMetadata",
    "RecoveryProvenance",
    "RecoverySaveRequest",
    "augment_behavior_cloning_laps",
    "class_weights",
    "clone_state",
    "collate_behavior_cloning",
    "flatten_behavior_cloning_laps",
    "horizontal_flip_observation",
    "load_behavior_cloning_laps",
    "load_behavior_cloning_recovery",
    "save_behavior_cloning_recovery",
    "split_behavior_cloning_laps",
]
