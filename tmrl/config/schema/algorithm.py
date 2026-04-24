"""RL algorithm hyperparameters (SAC family, TQC, REDQ, IQN, SDSAC)."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class AlgorithmConfig(BaseModel):
    """Trainer-specific optimization, exploration, and distributional RL knobs."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    hydra_target: str | None = Field(
        default=None,
        alias="_target_",
        description=(
            "Fully-qualified class path for the training agent. "
            "Used by the registry for validation and forward-compatible "
            "with hydra.utils.instantiate."
        ),
    )
    name: Literal["SAC", "TQC", "REDQSAC", "IQN", "SDSAC"] = Field(
        ...,
        description="Which learner class to construct (continuous SAC/TQC/REDQ or discrete IQN).",
    )
    learn_entropy_coef: bool = Field(
        default=False,
        description="Learn the entropy temperature alpha with a separate optimizer (SAC v2).",
    )
    lr_actor: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-5,
        description="Learning rate for policy (actor) parameters.",
    )
    lr_critic: Annotated[float, Field(ge=0.0)] = Field(
        default=5e-5,
        description="Learning rate for Q-network / critic parameters.",
    )
    lr_entropy: Annotated[float, Field(ge=0.0)] = Field(
        default=3e-4,
        description="Learning rate for log-alpha when learn_entropy_coef is true.",
    )
    gamma: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.99,
        description="Bellman discount factor for n-step and one-step targets.",
    )
    polyak: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.995,
        description="Soft-update interpolation τ for exponential moving average targets.",
    )
    target_entropy: float = Field(
        default=-0.5,
        description="Desired policy entropy for automatic alpha tuning; shape matches action dims.",
    )
    alpha: Annotated[float, Field(ge=0.0)] = Field(
        default=0.01,
        description="Initial or fixed entropy coefficient when not fully learned.",
    )
    redq_n: PositiveInt = Field(
        default=10,
        description="Total critics in the REDQ ensemble.",
    )
    redq_m: PositiveInt = Field(
        default=2,
        description="Subset size drawn from the ensemble for REDQ target minimization.",
    )
    redq_q_updates_per_policy_update: PositiveInt = Field(
        default=20,
        description="Critic gradient steps per actor update in REDQ (update-to-data ratio).",
    )
    top_quantiles_to_drop: int = Field(
        default=2,
        ge=0,
        description="TQC: drop this many largest quantile predictions from the target mixture.",
    )
    quantiles_number: PositiveInt = Field(
        default=1,
        description="Quantiles per critic output; must be 1 for vanilla SAC.",
    )
    n_steps: int = Field(
        default=1,
        ge=0,
        description="N-step return horizon; 0 or 1 reduces to one-step TD.",
    )
    r2d2_rewind: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description=(
            "R2D2-style replay only (MTQC / TQCGRAB interfaces): temporal rewind augmentation "
            "probability. Unused for LIDAR tuple memory."
        ),
    )
    r2d2_num_sequences: int = Field(
        default=0,
        ge=0,
        description=(
            "R2D2-style replay only: parallel sequences per minibatch; 0 uses i.i.d. transitions. "
            "Unused for LIDAR tuple memory."
        ),
    )
    r2d2_sequence_length: int = Field(
        default=0,
        ge=0,
        description=(
            "R2D2-style replay only: sequence length L for recurrent / stacked batches. "
            "IQN GNN+Lidar uses this only if USE_RNN_MODEL and batch layout matches."
        ),
    )
    r2d2_burn_in: int = Field(
        default=0,
        ge=0,
        description=(
            "R2D2-style replay / GRU path: burn-in prefix without full BPTT. "
            "Unused for plain LIDAR MLP IQN."
        ),
    )
    optimizer_actor: str = Field(
        default="adam",
        description="Torch optimizer class name for the actor (adam, adamw, sgd).",
    )
    optimizer_critic: str = Field(
        default="adam",
        description="Torch optimizer class name for critic parameters.",
    )
    betas_actor: list[float] | None = Field(
        default=None,
        description="Optional (beta1, beta2) for Adam on the actor; None uses PyTorch defaults.",
    )
    betas_critic: list[float] | None = Field(
        default=None,
        description="Optional (beta1, beta2) for Adam on the critic.",
    )
    weight_decay: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description=(
            "Single optimizer weight decay (AdamW semantics when using adamw): applied to actor "
            "and critic in SAC/TQC, to both SDSAC optimizers, and to the IQN Q-network. 0 disables."
        ),
    )
    grad_clip_actor: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Global L2 grad norm cap for the actor; 0 disables clipping.",
    )
    grad_clip_critic: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Global L2 grad norm cap for critics; 0 disables clipping.",
    )
    backup_clip_range: Annotated[float, Field(gt=0.0)] = Field(
        default=100.0,
        description="Symmetric clamp applied to TD targets for numerical stability.",
    )
    reward_normalize_scale: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Divide rewards by this constant before the Bellman backup (1.0 disables).",
    )
    mean_penalty_coef: Annotated[float, Field(ge=0.0)] = Field(
        default=0.05,
        description="Weight on mean-Q penalization used in some actor objectives.",
    )
    dynamic_truncation_enabled: bool = Field(
        default=False,
        description="TQC: adaptively truncate more quantiles when target variance spikes.",
    )
    dynamic_truncation_variance_pct: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.9,
        description="Percentile of running target variance that triggers extra truncation.",
    )
    vcse_enabled: bool = Field(
        default=False,
        description="Variance-conditioned Shannon entropy: modulate alpha with critic uncertainty.",
    )
    vcse_lambda: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Gain mapping critic standard deviation into the entropy bonus.",
    )
    vcse_alpha_base: Annotated[float, Field(ge=0.0)] = Field(
        default=0.03,
        description="Floor entropy coefficient before the VCSE uncertainty term is added.",
    )
    clipping_weights: bool = Field(
        default=False,
        description="Project all network weights into [-clip_weights_value, clip_weights_value].",
    )
    clip_weights_value: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Absolute bound used when clipping_weights is enabled.",
    )
    num_track_points: int = Field(
        default=0,
        ge=0,
        description="Fallback number of polyline samples when geometry-derived count is unknown.",
    )
    points_distance: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Nominal arc-length spacing between synthetic track samples.",
    )
    speed_bonus: float = Field(
        default=0.0,
        description="Legacy auxiliary speed-shaping scale still referenced in some reward bridges.",
    )
    speed_min_threshold: float = Field(
        default=0.0,
        description="Lower speed bound (km/h) used in legacy reward shaping hooks.",
    )
    speed_medium_threshold: float = Field(
        default=0.0,
        description="Mid speed bound (km/h) for piecewise reward shaping hooks.",
    )
    adam_eps: Annotated[float, Field(gt=0.0)] = Field(
        default=1e-8,
        description=(
            "Adam epsilon (SAC via sac_config, TQC optimizers, IQN Q-network Adam). "
            "REDQ-SAC uses its own hardcoded optimizers."
        ),
    )
    bc_lambda: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Static coefficient on behavior-cloning auxiliary loss.",
    )
    bc_lambda_start: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="BC loss weight at the beginning of a scheduled anneal.",
    )
    bc_lambda_end: Annotated[float, Field(ge=0.0)] = Field(
        default=0.01,
        description="BC loss weight at the end of the anneal window.",
    )
    bc_anneal_steps_start: int = Field(
        default=0,
        ge=0,
        description="Global environment step index where BC annealing starts.",
    )
    bc_anneal_steps_end: int = Field(
        default=2_000_000,
        ge=0,
        description="Global environment step index where BC annealing ends.",
    )
    mixed_precision: bool = Field(
        default=True,
        description="Enable torch.cuda.amp automatic mixed precision when hardware allows.",
    )
    mixed_precision_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Primary tensor dtype used inside autocast regions.",
    )
    horizontal_flip_p: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Probability of applying left-right mirroring to image observations.",
    )
    per_td_enabled: bool = Field(
        default=False,
        description="Enable prioritized replay for temporal-difference learning paths.",
    )
    per_td_alpha: Annotated[float, Field(ge=0.0)] = Field(
        default=0.6,
        description=(
            "PER prioritization exponent controlling how sharp the priority distribution is."
        ),
    )
    per_td_beta: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.4,
        description="Importance-sampling correction exponent annealed toward 1 during training.",
    )
    per_td_eps: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-6,
        description="Small constant added to priorities to avoid zero sampling probability.",
    )
    use_sde: bool = Field(
        default=True,
        description="Use generalized state-dependent exploration noise on continuous policies.",
    )
    log_std_init: float = Field(
        default=-3.0,
        description="Initial log-standard-deviation for the gSDE squashed Gaussian policy.",
    )
    sde_clip_mean: Annotated[float, Field(ge=0.0)] = Field(
        default=2.0,
        description=(
            "Clamp actor means into [-sde_clip_mean, sde_clip_mean] before sampling; 0 disables."
        ),
    )
    sde_sample_freq: PositiveInt = Field(
        default=100,
        description="Environment steps between resampling the gSDE noise matrix.",
    )
    entropy_floor: Annotated[float, Field(ge=0.0)] = Field(
        default=0.02,
        description="Minimum admissible entropy coefficient when using learned alpha.",
    )
    entropy_schedule: str = Field(
        default="learnable",
        description=(
            "Schedule identifier for entropy temperature (learnable, cosine, constant, ...)."
        ),
    )
    entropy_cosine_t0: PositiveInt = Field(
        default=300,
        description="First cosine half-period length (optimizer steps) for entropy annealing.",
    )
    entropy_cosine_tmult: Annotated[float, Field(gt=0.0)] = Field(
        default=1.5,
        description="Multiplicative growth of the cosine period after each restart.",
    )
    entropy_cosine_decay: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Per-cycle multiplicative shrink of the cosine entropy amplitude.",
    )
    fog_decay_temperature: Annotated[float, Field(gt=0.0)] = Field(
        default=3.0,
        description=(
            "FoG (forgetful observation gating) episode bias in R2D2-style memory sampling "
            "(tmrl.memory): >0 weights recent episodes; 0 disables. "
            "No effect on LIDAR tuple replay."
        ),
    )
    eder_oversample_ratio: int = Field(
        default=0,
        ge=0,
        description=(
            "IQN / SDSAC: batch diversity filter oversample ratio (>=2 enables EDER-style filter). "
            "0 disables. Unused for SAC/TQC/REDQ."
        ),
    )
    sdsac_avg_q: bool = Field(
        default=True,
        description="SDSAC: average ensemble Q values before the robust aggregation.",
    )
    sdsac_clip_q: bool = Field(
        default=True,
        description="SDSAC: clamp Q targets to stabilize critic bootstraps.",
    )
    sdsac_clip_q_epsilon: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Half-width of the trust region around the current Q estimate for clipping.",
    )
    sdsac_entropy_penalty: bool = Field(
        default=True,
        description="SDSAC: add an explicit entropy penalty term to the actor loss.",
    )
    sdsac_entropy_penalty_beta: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Scalar weight on the SDSAC entropy penalty term.",
    )
    iqn_n_steer_bins: PositiveInt = Field(
        default=13,
        description="Number of discrete steering bins in the IQN action space.",
    )
    iqn_lr: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-4,
        description="Adam learning rate for IQN quantile regression heads.",
    )
    iqn_epsilon_schedule_mode: str = Field(
        default="cosine",
        description="Schedule shape for epsilon-greedy decay (linear, cosine, ...).",
    )
    iqn_epsilon_start: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Starting exploration rate for epsilon-greedy action selection.",
    )
    iqn_epsilon_end: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.005,
        description="Terminal exploration rate after the decay horizon.",
    )
    iqn_explore_repeat_steps: PositiveInt = Field(
        default=4,
        description="Consecutive environment steps to repeat a sampled random action.",
    )
    iqn_epsilon_decay_steps: PositiveInt = Field(
        default=500_000,
        description="Environment-step horizon for epsilon decay in linear schedules.",
    )
    iqn_epsilon_cosine_t0: PositiveInt = Field(
        default=50_000,
        description="Initial cosine period for epsilon scheduling.",
    )
    iqn_epsilon_cosine_tmult: Annotated[float, Field(gt=0.0)] = Field(
        default=1.5,
        description="Cosine period multiplier after each epsilon schedule restart.",
    )
    iqn_epsilon_cosine_decay: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.8,
        description="Per-cycle shrink factor on the epsilon cosine amplitude.",
    )
    iqn_target_update_freq: PositiveInt = Field(
        default=1000,
        description="Environment steps between polyak or hard updates to the IQN target network.",
    )
    iqn_num_quantiles_train: PositiveInt = Field(
        default=64,
        description="Quantile samples drawn per training minibatch for the Huber loss.",
    )
    iqn_num_quantiles_target: PositiveInt = Field(
        default=64,
        description="Quantile samples when regressing toward the target distribution.",
    )
    iqn_num_quantiles_eval: PositiveInt = Field(
        default=32,
        description="Quantile samples for greedy action selection during evaluation rollouts.",
    )
    iqn_n_cos: PositiveInt = Field(
        default=64,
        description="Dimensionality of the cosine embedding for quantile fractions τ.",
    )
    iqn_dueling: bool = Field(
        default=True,
        description="Use dueling value decomposition inside the IQN trunk.",
    )
    iqn_double_dqn: bool = Field(
        default=True,
        description="Use double-Q reduction when constructing IQN bootstrap targets.",
    )
    iqn_grad_clip: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Global grad norm cap for IQN; set 0 to disable.",
    )
    iqn_huber_kappa: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Huber threshold κ for IQN quantile regression.",
    )
    iqn_use_value_rescaling: bool = Field(
        default=True,
        description=(
            "Apply signed value rescaling h(x) in IQN loss to damp large targets/TD errors."
        ),
    )
    iqn_value_rescaling_eps: Annotated[float, Field(gt=0.0)] = Field(
        default=1e-3,
        description="Epsilon term in signed value rescaling h(x).",
    )
    iqn_soft_target_tau: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.005,
        description=(
            "Polyak coefficient for IQN target network (set 0 to disable soft updates "
            "and keep hard updates by iqn_target_update_freq)."
        ),
    )
    iqn_log_target_stats: bool = Field(
        default=True,
        description="Log IQN target/TD distribution diagnostics to wandb.",
    )
    iqn_epsilon_cosine_initial_amplitude: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.1,
        description="Starting relative amplitude for cosine-shaped epsilon schedules.",
    )
    iqn_epsilon_cosine_floor_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.03,
        description="Lower bound expressed as a fraction of iqn_epsilon_start.",
    )
    iqn_epsilon_cosine_floor_steps: int = Field(
        default=0,
        ge=0,
        description="Flat exploration phase length before cosine epsilon kicks in.",
    )
    iqn_n_actions: PositiveInt = Field(
        default=78,
        description="Cardinality of the discrete action set (gas x brake x steer product).",
    )

    @model_validator(mode="after")
    def _consistency(self) -> AlgorithmConfig:
        from tmrl.config.enums import AlgorithmName

        for beta_name in ("betas_actor", "betas_critic"):
            betas = getattr(self, beta_name)
            if betas is not None and len(betas) != 2:
                raise ValueError(f"{beta_name} must contain exactly two floats when provided")

        from tmrl.custom.tm.utils.discrete_control import (
            BRAKE_TAP_TABLE_N_BRAKE,
            BRAKE_TAP_TABLE_N_GAS,
        )

        expected = self.iqn_n_steer_bins * BRAKE_TAP_TABLE_N_GAS * BRAKE_TAP_TABLE_N_BRAKE
        if self.iqn_n_actions != expected:
            raise ValueError(
                f"iqn_n_actions ({self.iqn_n_actions}) must equal "
                f"iqn_n_steer_bins * {BRAKE_TAP_TABLE_N_GAS} * {BRAKE_TAP_TABLE_N_BRAKE} "
                f"= {expected}"
            )

        try:
            alg = AlgorithmName(self.name)
        except ValueError:
            return self
        if (
            alg not in (AlgorithmName.TQC, AlgorithmName.IQN, AlgorithmName.SDSAC)
            and self.quantiles_number > 1
        ):
            raise ValueError("quantiles_number must be 1 unless using TQC, IQN, or SDSAC")
        if alg == AlgorithmName.SAC and self.quantiles_number != 1:
            raise ValueError("SAC requires quantiles_number == 1")
        return self
