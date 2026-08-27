"""Typed requests shared by distributed actor services."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.core.data import Transition


@dataclass(frozen=True, slots=True)
class ActorProcessRequest:
    config_path: str | Path
    target: str
    actor_id: str | None
    token: str
    external_stop: Any | None = None


@dataclass(frozen=True, slots=True)
class EnvironmentReset:
    environment: Any
    episode: int
    attempts: int = 5
    stop_on_failure: bool = True


@dataclass(frozen=True, slots=True)
class EvaluationEpisodeRequest:
    environment: Any
    pipeline: Any
    policy: ReplicablePolicy
    version: int


@dataclass(frozen=True, slots=True)
class TelemetryFailure:
    version: int
    transitions: int = 0
    reward: float = 0.0
    info: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class SpoolRequest:
    transitions: list[Transition]
    summaries: list[dict[str, Any]]
    policy_version: int
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    evaluation_snapshot: bytes = b""
