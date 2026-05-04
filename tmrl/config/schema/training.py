"""Training loop, replay capacity, checkpointing, and LR schedule."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from tmrl.config.schema.run_bundle import SchedulerConfig


class TrainingConfig(BaseModel):
    """How long to train, how often to sync, and replay/batch sizes."""

    model_config = ConfigDict(extra="forbid")

    max_epochs: PositiveInt = Field(
        default=10_000,
        description="Maximum outer training epochs before stopping the run.",
    )
    rounds_per_epoch: PositiveInt = Field(
        default=100,
        description="Rollout collection rounds executed within each epoch.",
    )
    training_steps_per_round: PositiveInt = Field(
        default=200,
        description="Optimizer (SGD) steps the trainer runs after each rollout round.",
    )
    max_training_steps_per_environment_step: Annotated[float, Field(gt=0)] = Field(
        default=4.0,
        description=(
            "Upper bound on gradient steps per single environment step (throttle async lag)."
        ),
    )
    environment_steps_before_training: int = Field(
        default=1000,
        ge=0,
        description="Collect this many environment transitions before the first optimizer step.",
    )
    update_model_interval: PositiveInt = Field(
        default=200,
        description="Broadcast fresh policy weights to workers every N environment steps.",
    )
    update_buffer_interval: PositiveInt = Field(
        default=200,
        description="Synchronize central replay with workers on this step cadence.",
    )
    save_model_every: int = Field(
        default=0,
        ge=0,
        description="Checkpoint interval in epochs; set 0 to disable periodic saves.",
    )
    best_checkpoint_criterion: str = Field(
        default="eval",
        description="Metric key used to compare checkpoints (e.g. eval return or lap statistic).",
    )
    best_checkpoint_lap_time: bool = Field(
        default=True,
        description=(
            "When true, prefer TMRL-style lap-time best model when eval finishes are clean."
        ),
    )
    best_checkpoint_min_finishes: int | None = Field(
        default=None,
        description="Minimum successful eval finishes required before lap-based best is eligible.",
    )
    competition_eval_crash_penalty_s: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Seconds penalizing the next finished lap after a crash in competition eval.",
    )
    competition_eval_max_crashes: int = Field(
        default=3,
        ge=0,
        description="Eval episodes terminate after this many crashes under competition rules.",
    )
    memory_size: PositiveInt = Field(
        default=1_000_000,
        description="Maximum transitions stored in the central replay buffer.",
    )
    batch_size: PositiveInt = Field(
        default=256,
        description="Minibatch size for each optimizer step on the trainer.",
    )
    batches_per_step: PositiveInt = Field(
        default=1,
        description="How many minibatches to draw per logical training step (UTD > 1).",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Cosine warm-restarts LR schedule parameters attached to the trainer.",
    )
