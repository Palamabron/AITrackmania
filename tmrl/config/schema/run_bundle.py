"""Run identity, logging, compute devices, demos, profiling, and LR scheduler."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class SchedulerConfig(BaseModel):
    """Optional PyTorch cosine warm-restarts style schedule for the trainer optimizers."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="",
        description="Registered scheduler name; leave empty to skip installing this scheduler.",
    )
    t_0: PositiveInt = Field(
        default=100,
        description="Initial restart period (epochs) before the first cosine cycle resets.",
    )
    t_mult: PositiveInt = Field(
        default=2,
        description="Multiplicative factor applied to the period after each restart.",
    )
    eta_min: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-6,
        description="Minimum learning rate floor reached at the bottom of each cosine lobe.",
    )
    last_epoch: int = Field(
        default=-1,
        description="Optimizer epoch counter when resuming; -1 starts the schedule from scratch.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        name = v.strip().lower()
        allowed = {"", "cosine_annealing_warm_restarts", "cosine_warm_restarts"}
        if name not in allowed:
            raise ValueError(
                "training.scheduler.name must be one of: "
                "'' | 'cosine_annealing_warm_restarts' | 'cosine_warm_restarts'."
            )
        return name


class RunConfig(BaseModel):
    """Experiment identity and high-level rollout collection limits."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Unique run id used for checkpoints, weights filenames, and wandb run id.",
    )

    @field_validator("name")
    @classmethod
    def _safe_run_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("run.name must not be empty.")
        if re.search(r"[/\\]|\.\.", v):
            raise ValueError(
                f"run.name={v!r} contains path separators or '..' which could cause "
                "files to be written outside the expected directory. "
                "Use a plain identifier (e.g. 'my_experiment_01')."
            )
        return v

    reset_training: bool = Field(
        default=False,
        description=(
            "If true, discard resumed trainer state and re-instantiate networks from scratch."
        ),
    )
    dataset_path: str = Field(
        default="",
        description=(
            "Optional filesystem path for offline dataset; empty uses default TmrlData layout."
        ),
    )
    buffers_maxlen: PositiveInt = Field(
        default=500_000,
        description="Maximum transitions retained per rollout worker buffer before eviction.",
    )
    rw_max_samples_per_episode: PositiveInt = Field(
        default=1000,
        description="Hard cap on environment steps collected in a single worker episode.",
    )
    rw_test_episode_interval: PositiveInt = Field(
        default=5,
        description="How often (in episodes) workers enter deterministic eval mode briefly.",
    )
    rw_test_episodes_per_eval: PositiveInt = Field(
        default=10,
        description="Number of fixed-seed eval rollouts aggregated per evaluation window.",
    )


class WandbConfig(BaseModel):
    """Weights & Biases experiment tracking."""

    model_config = ConfigDict(extra="forbid")

    project: str = Field(default="tmrl", description="W&B project slug.")
    entity: str = Field(default="tmrl", description="W&B entity (user or team).")
    api_key: str = Field(
        default="",
        description="API key placeholder; prefer WANDB_API_KEY or WANDB_KEY environment variable.",
    )
    log_gradients: bool = Field(
        default=False,
        description="When true, upload histograms of gradients (verbose and slower).",
    )
    debug_reward: bool = Field(
        default=True,
        description="Log reward-debug time series when workers use wandb.",
    )
    log_from_worker: bool = Field(
        default=True,
        description="Allow worker processes to call wandb.init for side-by-side logging.",
    )


class ComputeConfig(BaseModel):
    """Device placement for trainer vs rollout workers."""

    model_config = ConfigDict(extra="forbid")

    cuda_training: bool = Field(
        default=True,
        description="Use CUDA for the central trainer if a GPU is available.",
    )
    cuda_inference: bool = Field(
        default=False,
        description="Use CUDA on rollout workers for policy forward (higher VRAM, lower latency).",
    )
    virtual_gamepad: bool = Field(
        default=True,
        description=(
            "Drive the game with a virtual gamepad; false falls back to keyboard injection."
        ),
    )


class PlayerRunsConfig(BaseModel):
    """Human demonstration replay for offline or mixed replay."""

    model_config = ConfigDict(extra="forbid")

    online_injection: bool = Field(
        default=False,
        description="Trainer periodically ingests new demos from disk during training.",
    )
    source_path: str = Field(
        default="",
        description="Directory of recorded runs; empty resolves to TmrlData/player_runs.",
    )
    consume_on_read: bool = Field(
        default=True,
        description="Delete or archive demo files after successful ingestion (pipeline-specific).",
    )
    max_files_per_update: PositiveInt = Field(
        default=1,
        description="Upper bound on demo files processed per trainer update tick.",
    )
    demo_injection_repeat: PositiveInt = Field(
        default=1,
        description="How many times each demo transition is duplicated into mixed batches.",
    )
    demo_sampling_weight: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Relative priority of demo vs policy data when building training batches.",
    )
    demo_weight_decay_samples: int = Field(
        default=0,
        ge=0,
        description="Environment-step horizon over which demo_weight is annealed toward baseline.",
    )
    demo_weight_decay_slowdown: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Scalar slowing the demo weight decay curve (1.0 = default speed).",
    )
    per_alpha: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.6,
        description="PER-style prioritization exponent when mixing demos with prioritized replay.",
    )
    demo_max_batch_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Maximum fraction of a minibatch that may consist of demo transitions.",
    )
    demo_min_batch_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Minimum demo fraction when online injection is active (floor).",
    )


class DebuggerConfig(BaseModel):
    """Diagnostics: profiling, checksums, and observation sanity checks."""

    model_config = ConfigDict(extra="forbid")

    debug_mode: bool = Field(
        default=False,
        description="Enable verbose asserts and extra logging across training code paths.",
    )
    profile_trainer: bool = Field(
        default=False,
        description="Run cProfile around each training epoch.",
    )
    pytorch_profiler: bool = Field(
        default=False,
        description="Enable torch.profiler for CUDA/CPU timeline capture.",
    )
    crc_debug: bool = Field(
        default=False,
        description="Verify checksums on tensors crossing the worker↔trainer boundary.",
    )
    crc_debug_samples: int = Field(
        default=0,
        ge=0,
        description="Number of batches to CRC-verify when crc_debug is enabled.",
    )
    wandb_debug: bool = Field(
        default=True,
        description="Emit additional debug scalars to W&B from the trainer.",
    )
    observation_bounds_check: bool = Field(
        default=False,
        description="Assert finite observations immediately before the policy forward pass.",
    )
