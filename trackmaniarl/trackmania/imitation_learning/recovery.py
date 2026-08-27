"""DAgger recovery archive persistence and feature reconstruction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.trackmania.imitation_learning._data_types import (
    INTERVENTION_KEY,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    BehaviorCloningLap,
    RecoveryContract,
    RecoveryProvenance,
    is_sha256,
)
from trackmaniarl.trackmania.imitation_learning._recovery_archive import (
    read_recovery_archive,
    save_behavior_cloning_recovery,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoveryArchive,
    RecoveryReadRequest,
)


@dataclass(frozen=True, slots=True)
class RecoveryLoadRequest:
    paths: Sequence[Path]
    pipeline: FeaturePipeline
    action_ids: tuple[int, ...]
    expected_contract: RecoveryContract
    expected_source_demonstration_sha256: frozenset[str]
    previous_action_conditioning: bool = False


@dataclass(frozen=True, slots=True)
class _RecoveryLoadSettings:
    expected_contract: RecoveryContract
    expected_source_hashes: frozenset[str]
    previous_action_conditioning: bool = False


@dataclass(frozen=True, slots=True)
class _RecoveryBuildContext:
    pipeline: FeaturePipeline
    action_count: int
    previous_action_conditioning: bool


@dataclass(frozen=True, slots=True)
class _RecoverySlice:
    archive: RecoveryArchive
    episode: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _RecoverySample:
    archive: RecoveryArchive
    index: int
    previous_action: int


def load_behavior_cloning_recovery(request: RecoveryLoadRequest) -> list[BehaviorCloningLap]:
    """Rebuild feature histories for DAgger states and keep episodes separate."""

    settings = _RecoveryLoadSettings(
        request.expected_contract,
        request.expected_source_demonstration_sha256,
        request.previous_action_conditioning,
    )
    resolved_paths = _validated_recovery_paths(request.paths, settings.expected_source_hashes)
    context = _RecoveryBuildContext(
        request.pipeline, len(request.action_ids), settings.previous_action_conditioning
    )
    archives = _load_recovery_archives(resolved_paths, request.action_ids, settings)
    return [lap for archive in archives for lap in _rebuild_recovery_laps(archive, context)]


def _load_recovery_archives(
    paths: tuple[Path, ...], action_ids: tuple[int, ...], settings: _RecoveryLoadSettings
) -> list[RecoveryArchive]:
    archives: list[RecoveryArchive] = []
    for path in paths:
        request = RecoveryReadRequest(
            path,
            action_ids,
            settings.expected_contract,
            settings.expected_source_hashes,
        )
        archives.append(read_recovery_archive(request))
    return archives


def _validated_recovery_paths(
    paths: Sequence[Path], source_hashes: frozenset[str]
) -> tuple[Path, ...]:
    if not source_hashes or any(not is_sha256(value) for value in source_hashes):
        raise ValueError("expected recovery source hashes must be non-empty SHA-256 digests")
    resolved = tuple(path.resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ValueError("behavior-cloning recovery paths must be unique")
    return resolved


def _rebuild_recovery_laps(
    archive: RecoveryArchive, context: _RecoveryBuildContext
) -> list[BehaviorCloningLap]:
    boundaries = np.flatnonzero(archive.episode_starts)
    stops = [*boundaries[1:], len(archive.frames)]
    return [
        _rebuild_recovery_lap(context, _RecoverySlice(archive, episode, int(start), int(stop)))
        for episode, (start, stop) in enumerate(zip(boundaries, stops, strict=True))
    ]


def _rebuild_recovery_lap(
    context: _RecoveryBuildContext, recovery_slice: _RecoverySlice
) -> BehaviorCloningLap:
    reset = getattr(context.pipeline, "reset_episode", None)
    if callable(reset):
        reset()
    observations = _rebuild_observations(context, recovery_slice)
    archive = recovery_slice.archive
    start, stop = recovery_slice.start, recovery_slice.stop
    return BehaviorCloningLap(
        tuple(observations),
        torch.from_numpy(archive.labels[start:stop].copy()),
        source_id=f"{archive.path.resolve()}#episode-{recovery_slice.episode}",
    )


def _rebuild_observations(
    context: _RecoveryBuildContext, recovery_slice: _RecoverySlice
) -> list[Mapping[str, torch.Tensor]]:
    observations: list[Mapping[str, torch.Tensor]] = []
    previous_action = context.action_count
    archive = recovery_slice.archive
    for index in range(recovery_slice.start, recovery_slice.stop):
        sample = _RecoverySample(archive, index, previous_action)
        observations.append(_rebuild_observation(context, sample))
        label = archive.labels[index]
        previous_action = int(label)
    return observations


def _rebuild_observation(
    context: _RecoveryBuildContext, sample: _RecoverySample
) -> Mapping[str, torch.Tensor]:
    frame = sample.archive.frames[sample.index]
    transformed = context.pipeline.transform_observation(frame)
    observation = dict(_clone_mapping_observation(transformed))
    observation["expert_previous_action"] = torch.tensor(sample.previous_action, dtype=torch.long)
    if context.previous_action_conditioning:
        observation["previous_action"] = torch.tensor(sample.previous_action, dtype=torch.long)
    _attach_recovery_metadata(observation, sample.archive.metadata, sample.index)
    return observation


def _clone_mapping_observation(observation: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(observation, Mapping):
        raise TypeError("behavior cloning recovery requires mapping lidar observations")
    return {key: value.detach().clone() for key, value in observation.items()}


def _attach_recovery_metadata(
    observation: dict[str, torch.Tensor], metadata: Mapping[str, np.ndarray], index: int
) -> None:
    observation[SAMPLE_WEIGHT_KEY] = torch.tensor(metadata[SAMPLE_WEIGHT_KEY][index])
    observation[STUDENT_ACTION_KEY] = torch.tensor(
        metadata[STUDENT_ACTION_KEY][index], dtype=torch.long
    )
    observation[INTERVENTION_KEY] = torch.tensor(
        metadata[INTERVENTION_KEY][index], dtype=torch.bool
    )
    observation[STATE_ERROR_KEY] = torch.tensor(metadata[STATE_ERROR_KEY][index])


__all__ = [
    "RecoveryContract",
    "RecoveryLoadRequest",
    "RecoveryProvenance",
    "load_behavior_cloning_recovery",
    "save_behavior_cloning_recovery",
]
