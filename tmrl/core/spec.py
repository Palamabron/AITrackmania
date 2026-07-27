"""Strict, serializable configuration boundary for a TMRL SDK run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from tmrl.core.data import BatchRequest

DEFAULT_EVALUATION_TIME_BUCKETS_S: tuple[float, ...] = (40.0, 38.0, 36.0)


class ComponentSpec(BaseModel):
    """A locally installed project component selected by import path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    class_path: str = Field(pattern=r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ComponentsSpec(BaseModel):
    """The required components for a complete training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learner: ComponentSpec
    environment: ComponentSpec | None = None
    model_factory: ComponentSpec | None = None
    replay_store: ComponentSpec
    sampler: ComponentSpec
    feature_pipeline: ComponentSpec
    logger: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(class_path="tmrl.core.builtins:JsonlRunLogger")
    )
    additional_loggers: tuple[ComponentSpec, ...] = ()
    checkpoint_codec: ComponentSpec = Field(
        default_factory=lambda: ComponentSpec(class_path="tmrl.core.builtins:TorchCheckpointCodec")
    )
    evaluator: ComponentSpec | None = None


class TrainingSpec(BaseModel):
    """Bounded off-policy training schedule executed by ``tmrl train``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_transitions: PositiveInt = 10_000
    max_episode_steps: PositiveInt = 2_000
    batch_size: PositiveInt = 256
    sequence_length: PositiveInt = 1
    n_step: PositiveInt = 1
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    beta: float | None = Field(default=None, ge=0.0, le=1.0)
    warmup_transitions: int = Field(default=1_000, ge=0)
    updates_per_transition: float = Field(default=1.0, gt=0.0)
    checkpoint_interval_updates: PositiveInt = 1_000
    metrics_interval_updates: PositiveInt = 50
    per_beta_final: float | None = Field(default=None, ge=0.0, le=1.0)
    per_beta_anneal_transitions: PositiveInt | None = None
    evaluate_every_episodes: PositiveInt | None = None
    max_episode_artifacts: PositiveInt = 100

    @field_validator("per_beta_final")
    @classmethod
    def _beta_final_requires_beta(cls, value: float | None, info: Any) -> float | None:
        if value is not None and info.data.get("beta") is None:
            raise ValueError("per_beta_final requires training.beta")
        return value

    def batch_request(
        self,
        *,
        batch_size: int | None = None,
        beta: float | None = None,
    ) -> BatchRequest:
        """Build the sole replay request used by the local runtime."""

        return BatchRequest(
            batch_size=self.batch_size if batch_size is None else batch_size,
            sequence_length=self.sequence_length,
            beta=self.beta if beta is None else beta,
            n_step=self.n_step,
            gamma=self.gamma,
        )

    def replay_beta(self, transitions: int) -> float | None:
        if self.beta is None or self.per_beta_final is None:
            return self.beta
        duration = self.per_beta_anneal_transitions or self.total_transitions
        fraction = min(1.0, max(0, transitions) / duration)
        return self.beta + fraction * (self.per_beta_final - self.beta)


class DistributedSpec(BaseModel):
    """Actor/learner exchange settings shared by local and remote runtimes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    port: int = Field(default=8787, ge=1, le=65535)
    rollout_chunk_transitions: PositiveInt = 128
    rollout_flush_s: float = Field(default=2.0, gt=0.0)
    policy_refresh_s: float = Field(default=5.0, gt=0.0)
    heartbeat_s: float = Field(default=5.0, gt=0.0)
    actor_timeout_s: float = Field(default=20.0, gt=0.0)
    max_inflight_chunks: PositiveInt = 4
    spool_max_bytes: PositiveInt = 2 * 1024**3
    max_message_bytes: PositiveInt = 16 * 1024**2
    soft_policy_lag_updates: PositiveInt = 1_000
    hard_policy_lag_updates: PositiveInt = 5_000
    max_update_credit: PositiveInt = 512
    epsilon_profiles: tuple[float, ...] = (1.0, 0.4, 0.1, 0.02)
    epsilon_start: float = Field(default=0.5, ge=0.0, le=1.0)
    epsilon_final: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay_transitions: PositiveInt = 1_500_000
    token_env: str = Field(default="TMRL_DISTRIBUTED_TOKEN", min_length=1)

    @field_validator("epsilon_profiles")
    @classmethod
    def _epsilon_profiles(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("epsilon_profiles must contain multipliers between 0 and 1")
        return values


class EvaluationMapSpec(BaseModel):
    """Immutable local map and geometry asset used by TrackMania evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    map_path: Path
    geometry_path: Path
    expected_map_uid: str = Field(min_length=1)


class EvaluationSuiteSpec(BaseModel):
    """Versioned local-map suite; game engine seeds are intentionally absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    maps: tuple[EvaluationMapSpec, ...] = ()
    trials_per_map: PositiveInt = 1
    time_buckets_s: tuple[float, ...] = DEFAULT_EVALUATION_TIME_BUCKETS_S
    target_median_s: float | None = None
    min_finish_rate: float = 0.9

    @field_validator("maps")
    @classmethod
    def _unique_map_ids(cls, maps: tuple[EvaluationMapSpec, ...]) -> tuple[EvaluationMapSpec, ...]:
        if len({item.id for item in maps}) != len(maps):
            raise ValueError("evaluation map ids must be unique")
        return maps

    @field_validator("time_buckets_s")
    @classmethod
    def _positive_time_buckets(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if any(value <= 0.0 for value in values):
            raise ValueError("time_buckets_s must contain positive finish times")
        return values

    @field_validator("target_median_s")
    @classmethod
    def _positive_target_median(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("target_median_s must be positive")
        return value

    @field_validator("min_finish_rate")
    @classmethod
    def _valid_finish_rate(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("min_finish_rate must be between 0 and 1")
        return value


class RunSpec(BaseModel):
    """All user-controlled configuration for one TrackMania RL run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = "1.2"
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seed: int = 0
    artifacts_dir: Path = Path("artifacts")
    components: ComponentsSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    distributed: DistributedSpec = Field(default_factory=DistributedSpec)
    evaluation: EvaluationSuiteSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_version")
    @classmethod
    def _api_version(cls, value: str) -> str:
        if value != "1.2":
            raise ValueError("RunSpec api_version must be '1.2'")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunSpec:
        """Load and validate a YAML run description without importing legacy config."""

        config_path = Path(path)
        with config_path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not isinstance(data, dict):
            raise TypeError(f"{config_path} must contain a YAML mapping")
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize the validated run specification as portable, deterministic YAML."""

        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
