from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.imitation_learning import (
    RECOVERY_DATASET_FORMAT,
    BehaviorCloningLap,
    RecoveryContract,
    RecoveryLoadRequest,
    RecoveryProvenance,
    load_behavior_cloning_recovery,
)
from trackmaniarl.trackmania.synthetic_recovery import (
    SyntheticRecoveryConfig,
    SyntheticRecoveryPathRequest,
    SyntheticRecoveryRequest,
    generate_synthetic_recovery,
    generate_synthetic_recovery_from_path,
)
from trackmaniarl.trackmania.synthetic_recovery_types import SyntheticRecoveryDataset

ACTION_IDS = (0, 1, 3, 39, 72, 73, 75)


def _demonstration_frames() -> np.ndarray:
    frames = np.zeros((7, 33), dtype=np.float32)
    frames[:, 3] = np.arange(10.0, 80.0, 10.0)
    frames[:, 4] = np.arange(7, dtype=np.float32)
    frames[:, 7] = 20.0
    frames[:, 10] = 1.0
    frames[:, 14] = 1.0
    frames[:, 16] = 20.0
    frames[:, 18] = 2.0
    frames[:, 29] = 1.0
    frames[:, 31] = 1.0
    frames[-1, 2] = 1.0
    return frames


def _demonstration() -> Demonstration:
    frames = _demonstration_frames()
    controls = np.tile(np.asarray([1.0, 0.0, 0.0], dtype=np.float32), (6, 1))
    return Demonstration(
        map_uid="map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=np.full(6, 39, dtype=np.int64),
        controls=controls,
        finish_time_s=0.07,
        decision_interval_ms=10.0,
    )


def _load_saved_recovery(dataset: SyntheticRecoveryDataset, path: Path) -> list[BehaviorCloningLap]:
    request = RecoveryLoadRequest(
        [path],
        _Pipeline(),
        ACTION_IDS,
        dataset.provenance.contract,
        frozenset({dataset.provenance.source_demonstration_sha256}),
    )
    return load_behavior_cloning_recovery(request)


def _native_recovery_request(
    tmp_path: Path,
) -> tuple[SyntheticRecoveryPathRequest, Demonstration, RecoveryContract]:
    demonstration = replace(_demonstration(), decision_interval_ms=None)
    source = save_demonstration(tmp_path / "native", demonstration)
    contract = RecoveryContract(
        map_uid="map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=20.0,
        control_alignment="frame_start",
    )
    request = SyntheticRecoveryPathRequest(
        source, ACTION_IDS, SyntheticRecoveryConfig(sample_stride=1), contract
    )
    return request, demonstration, contract


def test_synthetic_recovery_is_deterministic_and_keeps_monotonic_episodes() -> None:
    config = SyntheticRecoveryConfig(sample_stride=2)

    first = generate_synthetic_recovery(
        SyntheticRecoveryRequest(_demonstration(), ACTION_IDS, config)
    )
    second = generate_synthetic_recovery(
        SyntheticRecoveryRequest(_demonstration(), ACTION_IDS, config)
    )

    assert len(first.frames) == 33
    assert np.array_equal(first.frames, second.frames)
    assert np.array_equal(first.labels, second.labels)
    assert bool(np.all(first.episode_starts))
    assert np.count_nonzero(first.interventions) == 30


class _Pipeline:
    def reset_episode(self) -> None:
        return None

    def transform_observation(self, observation: object) -> dict[str, torch.Tensor]:
        values = np.asarray(observation, dtype=np.float32)
        return {"telemetry": torch.from_numpy(values[:26].copy())}


def test_synthetic_recovery_explicit_save_uses_bc_recovery_v3(tmp_path: Path) -> None:
    request = SyntheticRecoveryRequest(
        _demonstration(), ACTION_IDS, SyntheticRecoveryConfig(sample_stride=100)
    )
    dataset = generate_synthetic_recovery(request)

    path = dataset.save(tmp_path / "synthetic")
    laps = _load_saved_recovery(dataset, path)
    with np.load(path, allow_pickle=False) as data:
        format_name = str(data["format"].item())

    assert path.exists()
    assert format_name == RECOVERY_DATASET_FORMAT
    assert dataset.provenance.contract.map_uid == "map"
    assert len(laps) == len(dataset.frames)
    assert all(len(lap.labels) == 1 for lap in laps)


def test_synthetic_recovery_aligns_native_demo_to_target_decision_interval(
    tmp_path: Path,
) -> None:
    request, demonstration, contract = _native_recovery_request(tmp_path)
    dataset = generate_synthetic_recovery_from_path(request)

    expected_source = RecoveryProvenance.from_demonstration(
        demonstration,
        contract=contract,
    )
    assert dataset.provenance == expected_source
    assert dataset.frames[:3, 3].tolist() == [10.0, 30.0, 50.0]
    assert bool(np.all(dataset.episode_starts))
