"""Root configuration composed from Hydra defaults and optional user overrides."""

from __future__ import annotations

import re
import warnings

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tmrl.config.rtgym_boundary_iface import rtgym_discrete_boundary_lidar_family
from tmrl.config.schema.algorithm import AlgorithmConfig
from tmrl.config.schema.distributed import DistributedConfig
from tmrl.config.schema.environment import EnvironmentConfig
from tmrl.config.schema.interface import InterfaceConfig
from tmrl.config.schema.memory import MemoryConfig
from tmrl.config.schema.model import ModelConfig
from tmrl.config.schema.run_bundle import (
    ComputeConfig,
    DebuggerConfig,
    PlayerRunsConfig,
    RunConfig,
    WandbConfig,
)
from tmrl.config.schema.training import TrainingConfig


class MainConfig(BaseModel):
    """Validated TMRL experiment configuration (snake_case everywhere)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: str = Field(
        ...,
        min_length=5,
        max_length=32,
        description="Semantic version of this config schema; must satisfy the loader minimum.",
    )
    run: RunConfig = Field(..., description="Run naming, dataset path, and rollout caps.")
    wandb: WandbConfig = Field(default_factory=WandbConfig, description="Experiment tracking.")
    compute: ComputeConfig = Field(
        default_factory=ComputeConfig,
        description="Trainer/worker device policy.",
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Networking, security, and timeout knobs for tlspyo.",
    )
    training: TrainingConfig = Field(..., description="Optimization loop and replay sizes.")
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Replay buffer selection."
    )
    interface: InterfaceConfig = Field(
        default_factory=InterfaceConfig, description="Hydra interface preset selection."
    )
    model: ModelConfig = Field(..., description="Policy and value network layout.")
    algorithm: AlgorithmConfig = Field(..., description="RL algorithm hyperparameters.")
    environment: EnvironmentConfig = Field(..., description="Simulator interface and reward.")
    debugger: DebuggerConfig = Field(default_factory=DebuggerConfig, description="Diagnostics.")
    player_runs: PlayerRunsConfig = Field(
        default_factory=PlayerRunsConfig,
        description="Human demo ingestion for mixed replay.",
    )

    @field_validator("schema_version")
    @classmethod
    def _semver(cls, v: str) -> str:
        if not re.fullmatch(r"\d+\.\d+\.\d+", v.strip()):
            raise ValueError("schema_version must look like MAJOR.MINOR.PATCH (e.g. 0.6.0)")
        return v.strip()

    @model_validator(mode="after")
    def _validate_cross_field_runtime_constraints(self) -> MainConfig:
        iface = self.environment.rtgym_interface.upper()

        if self.model.use_rnn and not rtgym_discrete_boundary_lidar_family(iface):
            images_r2d2_or_world_telemetry = "TQCGRAB" in iface or iface.endswith("MTQC")
            if not (images_r2d2_or_world_telemetry and self.algorithm.name in ("IQN", "SDSAC")):
                raise ValueError(
                    "model.use_rnn=true is only validated for "
                    "(1) boundary-lidar rt-gym tokens (TM20LIDAR / *TRACKMAP*, "
                    "or tokens containing *LIDARIMAGES* / *TRACKMAPIMAGES*), "
                    "(SAC RNN actor-critic path), or (2) interfaces that use the "
                    "R2D2 sequence buffer or world-telemetry observation layout "
                    "(IQN or SDSAC only)."
                )

        if self.algorithm.n_steps > 1 and self.algorithm.n_steps >= self.training.batch_size:
            raise ValueError(
                "algorithm.n_steps must be smaller than training.batch_size when n_steps > 1 "
                "(path: algorithm.n_steps, training.batch_size)."
            )

        return self

    @model_validator(mode="after")
    def _check_discrete_algorithm_model_preset(self) -> MainConfig:
        """Reject IQN/SDSAC paired with continuous-only image actor-critic presets."""
        if self.algorithm.name not in ("IQN", "SDSAC"):
            return self
        if not type(self.model).discrete_action_compatible:
            raise ValueError(
                f"algorithm.name={self.algorithm.name!r} requires a "
                f"discrete-action-capable model preset; "
                f"model.type={self.model.type!r} is a continuous image actor-critic stack."
            )
        return self

    @model_validator(mode="after")
    def _warn_iqn_ignores_residual_actor_critic_depths(self) -> MainConfig:
        """IQN uses a single Q trunk depth (``residual_mlp_num_blocks``).

        Actor/critic split depths are not used.
        """
        if self.algorithm.name != "IQN":
            return self
        m = self.model
        nb = m.residual_mlp_num_blocks
        a, c = m.residual_mlp_num_blocks_actor, m.residual_mlp_num_blocks_critic
        confusing = (a != 0 and a != nb) or (c != 0 and c != nb)
        if confusing:
            warnings.warn(
                "IQN uses only model.residual_mlp_num_blocks for the shared Q-network trunk; "
                f"model.residual_mlp_num_blocks_actor={a} and residual_mlp_num_blocks_critic={c} "
                f"are ignored when they differ from residual_mlp_num_blocks={nb}. "
                "Use tmrl --explain-active-config to see effective model fields.",
                UserWarning,
                stacklevel=1,
            )
        return self
