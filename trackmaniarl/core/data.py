"""Allocation-light data objects used in the TrackmaniaRL hot path.

Pydantic must never be used here.  ``observation`` and batch fields accept a
PyTree (including ``tensordict.TensorDict``) so image, telemetry, and custom
features do not require a new replay API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

from trackmaniarl.core.pytree import PyTree

type TransitionId = int


@dataclass(frozen=True, slots=True)
class Transition:
    """One environment transition, without runtime validation overhead."""

    observation: PyTree
    action: PyTree
    reward: float
    next_observation: PyTree
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    episode_id: str | None = None
    step: int | None = None


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """Sampling requirements supplied by a learner."""

    batch_size: int
    sequence_length: int = 1
    beta: float | None = None
    n_step: int = 1
    gamma: float = 0.99
    transition_count: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if self.beta is not None and not 0.0 <= self.beta <= 1.0:
            raise ValueError("beta must be between 0 and 1")
        if self.n_step < 1:
            raise ValueError("n_step must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if self.transition_count < 0:
            raise ValueError("transition_count must be non-negative")


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """A fully specified off-policy batch with explicit bootstrap semantics."""

    data: PyTree
    observations: PyTree
    actions: PyTree
    rewards: PyTree
    next_observations: PyTree
    terminated: PyTree
    truncated: PyTree
    bootstrap_discounts: PyTree
    transition_ids: Sequence[TransitionId]
    importance_weights: PyTree | None = None
    masks: PyTree | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PriorityUpdate:
    """Feedback from a learner to a prioritized replay sampler/store."""

    transition_ids: Sequence[TransitionId]
    priorities: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.transition_ids) != len(self.priorities):
            raise ValueError("transition_ids and priorities must have equal length")
        for priority in self.priorities:
            if not isinstance(priority, Real):
                raise TypeError("priorities must contain scalar real numbers")
            if not isfinite(float(priority)):
                raise ValueError("priorities must be finite")


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Ordered transitions from a single TrackMania episode."""

    episode_id: str
    transitions: Sequence[Transition]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EpisodeArtifact:
    """Portable episode summary; raw image arrays are deliberately excluded."""

    episode_id: str
    telemetry: Sequence[Mapping[str, Any]]
    actions: Sequence[Any]
    rewards: Sequence[float]
    observation_refs: Sequence[str]
    checkpoint_ref: str | None = None
    run_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
