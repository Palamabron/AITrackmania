"""Strict, serializable configuration boundary for a TMRL SDK run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator

from tmrl.core.data import BatchRequest


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
    evaluate_every_episodes: PositiveInt | None = None
    max_episode_artifacts: PositiveInt = 100

    def batch_request(self, *, batch_size: int | None = None) -> BatchRequest:
        """Build the sole replay request used by the local runtime."""

        return BatchRequest(
            batch_size=self.batch_size if batch_size is None else batch_size,
            sequence_length=self.sequence_length,
            beta=self.beta,
            n_step=self.n_step,
            gamma=self.gamma,
        )


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

    @field_validator("maps")
    @classmethod
    def _unique_map_ids(cls, maps: tuple[EvaluationMapSpec, ...]) -> tuple[EvaluationMapSpec, ...]:
        if len({item.id for item in maps}) != len(maps):
            raise ValueError("evaluation map ids must be unique")
        return maps


class RunSpec(BaseModel):
    """All user-controlled configuration for one TrackMania RL run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    api_version: str = "1.1"
    run_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    seed: int = 0
    artifacts_dir: Path = Path("artifacts")
    components: ComponentsSpec
    training: TrainingSpec = Field(default_factory=TrainingSpec)
    evaluation: EvaluationSuiteSpec | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("api_version")
    @classmethod
    def _api_version(cls, value: str) -> str:
        if value != "1.1":
            raise ValueError("RunSpec api_version must be '1.1'")
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
