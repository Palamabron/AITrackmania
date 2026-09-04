"""Stable component contracts for external TrackmaniaRL projects."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    DISCRETE_VALUE = "discrete_value"
    ENSEMBLE_ACTOR_CRITIC = "ensemble_actor_critic"


class PolicyMode(StrEnum):
    ONLINE = "online"
    EVALUATION = "evaluation"


@dataclass(frozen=True, slots=True)
class ActionSelectionRequest:
    mode: PolicyMode
    epsilon: float


@dataclass(frozen=True, slots=True)
class EvaluatorRuntimeRequest:
    suite: Any | None
    environment_factory: EnvironmentFactory | None
    feature_pipeline: FeaturePipeline
    max_episode_steps: int = 2_000
    run_dir: str | Path | None = None


@runtime_checkable
class Policy(Protocol):
    """Inference-only policy deployed to a rollout worker."""

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> Any: ...


@runtime_checkable
class BehaviorPolicy(Policy, Protocol):
    """Policy that records the behavior statistics required by on-policy learners."""

    def act_with_info(
        self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE
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
class OfflineSupervisedLearner(Learner, Protocol):
    """Learner with a synthetic validation step distinct from replay training."""

    def validation_update(self, batch: TrainingBatch) -> Mapping[str, float]: ...


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
class EpisodePaceReplayStore(Protocol):
    """Replay capability for relabeling completed online episodes."""

    def label_episode_sampling_pace(self, episode_id: str, finish_time_s: float) -> int: ...


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


@runtime_checkable
class Evaluator(Protocol):
    """Runs a named TrackMania evaluation suite."""

    def evaluate(self, policy: Policy) -> Mapping[str, float]: ...


@runtime_checkable
class CheckpointCodec(Protocol):
    """Persists learner state without coupling trainers to a format."""

    def save(self, state: Mapping[str, Any], path: Path) -> None: ...

    def load(self, path: Path) -> Mapping[str, Any]: ...
