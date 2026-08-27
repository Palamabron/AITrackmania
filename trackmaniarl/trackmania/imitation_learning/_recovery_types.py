from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trackmaniarl.trackmania.imitation_learning._data_types import (
    RecoveryContract,
    RecoveryProvenance,
)


@dataclass(frozen=True, slots=True)
class RecoveryArchive:
    path: Path
    frames: np.ndarray
    labels: np.ndarray
    episode_starts: np.ndarray
    metadata: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class RecoveryArrays:
    frames: np.ndarray
    labels: np.ndarray
    episode_starts: np.ndarray
    action_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecoveryMetadata:
    sample_weights: np.ndarray | None = None
    student_actions: np.ndarray | None = None
    interventions: np.ndarray | None = None
    state_errors: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class RecoverySaveRequest:
    path: str | Path
    arrays: RecoveryArrays
    provenance: RecoveryProvenance
    metadata: RecoveryMetadata


@dataclass(frozen=True, slots=True)
class RecoveryReadRequest:
    path: Path
    action_ids: tuple[int, ...]
    expected_contract: RecoveryContract
    expected_source_hashes: frozenset[str]
