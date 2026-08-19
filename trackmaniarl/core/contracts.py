"""Stable component contracts for external TrackmaniaRL projects."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from trackmaniarl.core.data import (
    BatchRequest,
    PriorityUpdate,
    TrainingBatch,
    Transition,
    TransitionId,
)


class ModelContract(StrEnum):
    """Train-time interface exposed by a model bundle."""

    CATEGORICAL_POLICY = "categorical_policy"
    CONTINUOUS_ACTOR_CRITIC = "continuous_actor_critic"
    CONTINUOUS_ACTOR_VALUE = "continuous_actor_value"
    CONTINUOUS_QUANTILE_ACTOR_CRITIC = "continuous_quantile_actor_critic"
    DISCRETE_ACTOR_CRITIC = "discrete_actor_critic"
    DISCRETE_QUANTILE = "discrete_quantile"
    ENSEMBLE_ACTOR_CRITIC = "ensemble_actor_critic"


@runtime_checkable
class Policy(Protocol):
    """Inference-only policy deployed to a rollout worker."""

    def act(self, observation: Any, *, deterministic: bool = False) -> Any: ...


@runtime_checkable
class BehaviorPolicy(Policy, Protocol):
    """Policy that records the behavior statistics required by on-policy learners."""

    def act_with_info(
        self, observation: Any, *, deterministic: bool = False
    ) -> tuple[Any, Mapping[str, Any]]: ...


@runtime_checkable
class ReplicablePolicy(Policy, Protocol):
    """Inference policy whose tensor state can be distributed safely."""

    def export_state(self) -> Mapping[str, Any]: ...

    def load_state(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class ExploratoryPolicy(Policy, Protocol):
    """Policy with actor-local exploration independent from learner weights."""

    def set_exploration_epsilon(self, epsilon: float) -> None: ...


@runtime_checkable
class Learner(Protocol):
    """Stateful optimisation algorithm running on the trainer."""

    def setup(self, context: Mapping[str, Any]) -> None: ...

    def update(
        self, batch: TrainingBatch
    ) -> Mapping[str, float] | tuple[Mapping[str, float], PriorityUpdate]: ...

    def policy(self) -> Policy: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@runtime_checkable
class ModelFactory(Protocol):
    """Creates one train-time model from an explicit user component configuration."""

    def build(self) -> Any: ...


@runtime_checkable
class ReplayStore(Protocol):
    """Stores transitions; it does not decide how to sample them."""

    def append(self, transition: Transition) -> TransitionId: ...

    def get(self, transition_ids: list[TransitionId]) -> list[Transition]: ...

    def available_ids(self) -> list[TransitionId]: ...

    def contains(self, transition_id: TransitionId) -> bool: ...

    def __len__(self) -> int: ...


@runtime_checkable
class EnvironmentFactory(Protocol):
    """Creates an isolated TrackMania environment for collection or evaluation."""

    def create(self, *, seed: int) -> Any: ...


@runtime_checkable
class Sampler(Protocol):
    """Chooses replay indices and collates a batch."""

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch: ...

    def update_priorities(self, update: PriorityUpdate) -> None: ...


@runtime_checkable
class FeaturePipeline(Protocol):
    """Transforms observations and collates sampled transitions."""

    def transform_observation(self, observation: Any) -> Any: ...

    def collate(self, transitions: list[Transition]) -> Any: ...


@runtime_checkable
class RunLogger(Protocol):
    """Receives neutral run events; implementations may be offline or remote."""

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None: ...

    def close(self) -> None: ...


Tracker = RunLogger


@runtime_checkable
class Evaluator(Protocol):
    """Runs a named TrackMania evaluation suite."""

    def evaluate(self, policy: Policy) -> Mapping[str, float]: ...


@runtime_checkable
class CheckpointCodec(Protocol):
    """Persists learner state without coupling trainers to a format."""

    def save(self, state: Mapping[str, Any], path: Path) -> None: ...

    def load(self, path: Path) -> Mapping[str, Any]: ...
