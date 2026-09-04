"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trackmaniarl.trackmania.imitation_learning import (
    BehaviorCloningLap,
    RecoveryArrays,
    RecoveryContract,
    RecoveryMetadata,
    RecoveryProvenance,
    RecoverySaveRequest,
)


class _RecoveryPipeline:
    def reset_episode(self) -> None:
        return None

    def transform_observation(self, observation: object) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(observation, dtype=torch.float32)
        return {
            "lidar": torch.zeros((4, 8)),
            "lidar_mask": torch.ones(8, dtype=torch.bool),
            "telemetry": values[:26],
        }

    def collate(self, transitions: list[object]) -> list[object]:
        return transitions


@dataclass(frozen=True, slots=True)
class _RecoveryContractOverrides:
    map_uid: str = "test-map"
    geometry_sha256: str = "a" * 64
    action_repeat_frames: int = 1
    decision_interval_ms: float | None = 10.0


def _recovery_contract(
    overrides: _RecoveryContractOverrides | None = None,
) -> RecoveryContract:
    settings = overrides or _RecoveryContractOverrides()
    return RecoveryContract(
        map_uid=settings.map_uid,
        geometry_sha256=settings.geometry_sha256,
        action_repeat_frames=settings.action_repeat_frames,
        decision_interval_ms=settings.decision_interval_ms,
        control_alignment="frame_start",
    )


def _recovery_provenance() -> RecoveryProvenance:
    return RecoveryProvenance(
        _recovery_contract(),
        source_demonstration_sha256="b" * 64,
        source_checkpoint_sha256="c" * 64,
    )


def _recovery_save_request(
    path: Path,
    arrays: RecoveryArrays,
    metadata: RecoveryMetadata | None = None,
) -> RecoverySaveRequest:
    return RecoverySaveRequest(
        path,
        arrays,
        _recovery_provenance(),
        metadata or RecoveryMetadata(),
    )


def _rewrite_recovery_archive(
    path: Path,
    updates: dict[str, np.ndarray | None],
) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    for key, value in updates.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    np.savez_compressed(path, **payload)


def _observation(value: float) -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.full((4, 8), value),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.full((26,), value),
    }


class _CheckpointScaler:
    def __init__(self, scale: float) -> None:
        self.current_scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.current_scale = float(state["current_scale"])


def _lap(label: int) -> BehaviorCloningLap:
    return BehaviorCloningLap((_observation(float(label)),), torch.tensor([label]))
