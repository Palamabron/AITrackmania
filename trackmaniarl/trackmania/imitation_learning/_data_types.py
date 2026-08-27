from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from trackmaniarl.trackmania.demonstrations import Demonstration

RECOVERY_DATASET_FORMAT = "trackmaniarl-bc-recovery-v3"
SAMPLE_WEIGHT_KEY = "bc_sample_weight"
STUDENT_ACTION_KEY = "bc_student_action"
INTERVENTION_KEY = "bc_intervention"
STATE_ERROR_KEY = "bc_state_error"
ELITE_LAP_WEIGHT_TEMPERATURE_S = 0.35
MINIMUM_LAP_WEIGHT = 0.15


@dataclass(frozen=True, slots=True)
class BehaviorCloningLap:
    observations: tuple[Mapping[str, torch.Tensor], ...]
    labels: torch.Tensor
    quality_weight: float = 1.0
    source_id: str = ""


@dataclass(frozen=True, slots=True)
class RecoveryContract:
    map_uid: str
    geometry_sha256: str
    action_repeat_frames: int
    decision_interval_ms: float | None
    control_alignment: str

    def __post_init__(self) -> None:
        if not self.map_uid or not is_sha256(self.geometry_sha256):
            raise ValueError("recovery map and geometry identity are invalid")
        if self.action_repeat_frames < 1:
            raise ValueError("recovery action repeat must be positive")
        interval = self.decision_interval_ms
        if interval is not None and (not np.isfinite(interval) or interval <= 0.0):
            raise ValueError("recovery decision interval must be finite and positive")
        if interval is not None and self.action_repeat_frames != 1:
            raise ValueError("recovery decision interval requires action repeat one")
        if self.control_alignment != "frame_start":
            raise ValueError("recovery controls must use frame_start alignment")

    @classmethod
    def from_demonstration(cls, demonstration: Demonstration) -> RecoveryContract:
        return cls(
            demonstration.map_uid,
            demonstration.geometry_sha256,
            demonstration.action_repeat_frames,
            demonstration.decision_interval_ms,
            demonstration.control_alignment,
        )


@dataclass(frozen=True, slots=True)
class RecoveryProvenance:
    contract: RecoveryContract
    source_demonstration_sha256: str
    source_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        hashes = (self.source_demonstration_sha256, self.source_checkpoint_sha256)
        if any(value is not None and not is_sha256(value) for value in hashes):
            raise ValueError("recovery source hashes must be SHA-256 digests")

    @classmethod
    def from_demonstration(
        cls,
        demonstration: Demonstration,
        *,
        contract: RecoveryContract | None = None,
        source_checkpoint_sha256: str | None = None,
    ) -> RecoveryProvenance:
        return cls(
            contract or RecoveryContract.from_demonstration(demonstration),
            demonstration_sha256(demonstration),
            source_checkpoint_sha256,
        )


def is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def demonstration_sha256(demonstration: Demonstration) -> str:
    digest = hashlib.sha256()
    metadata = (
        demonstration.map_uid,
        demonstration.geometry_sha256,
        str(demonstration.action_repeat_frames),
        str(demonstration.decision_interval_ms),
        demonstration.control_alignment,
        f"{demonstration.finish_time_s:.12g}",
    )
    digest.update("\0".join(metadata).encode())
    for values in (demonstration.frames, demonstration.actions, demonstration.controls):
        array = np.ascontiguousarray(values)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()
