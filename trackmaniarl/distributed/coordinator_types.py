from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReplayRestoreMode(StrEnum):
    FULL = "full"
    LEARNER_ONLY = "learner_only"


@dataclass(frozen=True, slots=True)
class CoordinatorConfig:
    bind: str
    token: str
    fingerprint: str
    resume_checkpoint: Path | None = None
    restore_mode: ReplayRestoreMode = ReplayRestoreMode.FULL
    external_stop: Any | None = None
    demo_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class LearnerProcessRequest:
    config_path: str
    bind: str
    token: str
    resume_checkpoint: str | None = None
    restore_mode: ReplayRestoreMode = ReplayRestoreMode.FULL
    external_stop: Any | None = None
    demo_paths: tuple[str, ...] = ()
