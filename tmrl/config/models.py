"""Pydantic configuration schema for TMRL (Hydra-composed + optional config.json overlay)."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PositiveInt, model_validator

# -----------------------------------------------------------------------------
# Nested sections
# -----------------------------------------------------------------------------


class SchedulerConfig(BaseModel):
    """Learning-rate scheduler parameters (PyTorch CosineAnnealingWarmRestarts-style)."""

    model_config = ConfigDict(extra="forbid")

    NAME: str = Field(
        default="",
        description="Scheduler class name; empty string disables custom scheduler.",
    )
    T_0: PositiveInt = Field(
        default=100,
        description="Initial restart period (epochs) for cosine warm restarts.",
    )
    T_mult: PositiveInt = Field(
        default=2,
        description="Multiplicative factor for the restart period after each restart.",
    )
    eta_min: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-6,
        description="Minimum learning rate floor for the scheduler.",
    )
    last_epoch: int = Field(
        default=-1,
        description="Last epoch index for scheduler state resume; -1 starts fresh (may be negative).",
    )


class PlayerRunsConfig(BaseModel):
    """Optional human-demo injection from recorded player runs."""

    model_config = ConfigDict(extra="forbid")

    ONLINE_INJECTION: bool = Field(
        default=False,
        description="When True, trainer may pull demo transitions from SOURCE_PATH during training.",
    )
    SOURCE_PATH: str = Field(
        default="",
        description="Directory containing recorded runs; empty falls back to TmrlData player_runs.",
    )
    CONSUME_ON_READ: bool = Field(
        default=True,
        description="If True, delete or consume demo files after reading (pipeline-dependent).",
    )
    MAX_FILES_PER_UPDATE: PositiveInt = Field(
        default=1,
        description="Maximum demo files to ingest per trainer update cycle.",
    )
    DEMO_INJECTION_REPEAT: PositiveInt = Field(
        default=1,
        description="Repeat factor for injecting each demo transition into the batch pipeline.",
    )
    DEMO_SAMPLING_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Relative sampling weight for demo vs agent data in mixed replay.",
    )
    DEMO_WEIGHT_DECAY_SAMPLES: int = Field(
        default=0,
        ge=0,
        description="Over this many environment samples, decay demo sampling weight toward baseline.",
    )
    DEMO_WEIGHT_DECAY_SLOWDOWN: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Slowdown factor controlling how gradually demo weight decays.",
    )
    PER_ALPHA: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.6,
        description="PER-style alpha for prioritizing demo or mixed replay (when used).",
    )
    DEMO_MAX_BATCH_FRACTION: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Upper cap on fraction of a batch that may come from demos.",
    )
    DEMO_MIN_BATCH_FRACTION: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Lower floor on demo fraction in a batch (when online injection is on).",
    )


class DebuggerConfig(BaseModel):
    """Profiling, reproducibility checks, and verbose logging toggles."""

    model_config = ConfigDict(extra="forbid")

    DEBUG_MODE: bool = Field(
        default=False,
        description="Global debug flag for extra assertions and logging in training code.",
    )
    PROFILE_TRAINER: bool = Field(
        default=False,
        description="Run Python profiler around each training epoch when True.",
    )
    PYTORCH_PROFILER: bool = Field(
        default=False,
        description="Enable PyTorch autograd profiler for hot path diagnosis.",
    )
    CRC_DEBUG: bool = Field(
        default=False,
        description="Verify sample CRC checksums across worker/trainer for pipeline consistency.",
    )
    CRC_DEBUG_SAMPLES: int = Field(
        default=0,
        ge=0,
        description="Number of batches/samples to run CRC checks on when CRC_DEBUG is True.",
    )
    WANDB_DEBUG: bool = Field(
        default=True,
        description="Log additional debug scalars and artifacts to Weights & Biases.",
    )
    OBSERVATION_BOUNDS_CHECK: bool = Field(
        default=False,
        description="Assert observations are finite (no NaN/Inf) before network forward.",
    )


class RtGymInterfaceKwargsConfig(BaseModel):
    """Arguments forwarded to the rtgym interface implementation."""

    model_config = ConfigDict(extra="allow")

    save_replays: bool = Field(
        default=False,
        description="Ask the game interface to persist replays when supported.",
    )


class RtGymConfigSection(BaseModel):
    """real-time-gym time stepping, timeouts, and episode limits."""

    model_config = ConfigDict(extra="allow")

    time_step_duration: Annotated[float, Field(gt=0.0)] = Field(
        default=0.05,
        description="Nominal duration (seconds) of one environment step.",
    )
    start_obs_capture: Annotated[float, Field(ge=0.0)] = Field(
        default=0.04,
        description="Delay before capturing observations after step start (sync with game).",
    )
    time_step_timeout_factor: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Multiplier on time_step_duration before a step is considered timed out.",
    )
    act_buf_len: PositiveInt = Field(
        default=2,
        description="Number of past actions stacked in observations (RT-MDP delay coverage).",
    )
    reset_act_buf: bool = Field(
        default=True,
        description="Clear the action buffer on env.reset() to avoid stale actions.",
    )
    benchmark: bool = Field(
        default=False,
        description="Benchmark mode reduces overhead for throughput measurement.",
    )
    wait_on_done: bool = Field(
        default=True,
        description="Block until the done signal is fully processed before continuing.",
    )
    ep_max_length: PositiveInt = Field(
        default=1000,
        description="Maximum steps per episode before forced termination.",
    )
    interface_kwargs: RtGymInterfaceKwargsConfig = Field(
        default_factory=RtGymInterfaceKwargsConfig,
        description="Interface-specific keyword arguments (e.g. replay saving).",
    )


class RewardConfig(BaseModel):
    """TrackMania reward shaping, progress checks, and episode termination."""

    model_config = ConfigDict(extra="allow")

    MAX_TIME_NO_PROGRESS_SECONDS: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Terminate if no forward progress for this many seconds (0 disables).",
    )
    MIN_PROGRESS_RATE: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Minimum track progress per second averaged over SLOW_PROGRESS_WINDOW (0 off).",
    )
    SLOW_PROGRESS_WINDOW_SECONDS: Annotated[float, Field(gt=0.0)] = Field(
        default=5.0,
        description="Window length for slow-progress rate computation.",
    )
    DEBUG_REWARD_COMPONENTS: bool = Field(
        default=False,
        description="Log per-component reward breakdown at DEBUG_LOG_INTERVAL.",
    )
    DEBUG_LOG_INTERVAL: PositiveInt = Field(
        default=100,
        description="Environment steps between reward debug logs when DEBUG_REWARD_COMPONENTS.",
    )
    CONSTANT_PENALTY: float = Field(
        default=0.0,
        description="Small per-step penalty encouraging forward motion / efficiency.",
    )
    CHECK_FORWARD: PositiveInt = Field(
        default=500,
        description="Number of polyline points ahead to evaluate forward progress.",
    )
    CHECK_BACKWARD: PositiveInt = Field(
        default=10,
        description="Points behind the car used to detect backward movement / rewind.",
    )
    MIN_STEPS: PositiveInt = Field(
        default=70,
        description="Grace steps before progress / failure rules can end the episode.",
    )
    MAX_STRAY: Annotated[float, Field(gt=0.0)] = Field(
        default=50.0,
        description="Maximum lateral distance from reference trajectory before failure (meters).",
    )
    PROGRESS_REWARD_FULL_LAP: Annotated[float, Field(ge=0.0)] = Field(
        default=200.0,
        description="Total bonus for completing a full lap of progress along the track.",
    )
    SPEED_REWARD_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Scale for reward proportional to speed aligned with the track.",
    )
    SPEED_REWARD_EXPONENT: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Exponent on normalized speed; 1 linear, >1 emphasizes high speed.",
    )
    SPEED_REWARD_ALIGNMENT_FLOOR: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Minimum heading/track alignment factor applied to speed reward.",
    )
    MAX_SPEED_KMH: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Reference speed (km/h) for normalizing speed reward.",
    )
    MAX_TRACK_WIDTH: Annotated[float, Field(gt=0.0)] = Field(
        default=65.0,
        description="Half-width scale for boundary / CTE style penalties.",
    )
    CRASH_PENALTY: Annotated[float, Field(ge=0.0)] = Field(
        default=2.0,
        description="Penalty applied when a crash or reset-from-failure is triggered.",
    )
    REWARD_CLIP_FLOOR: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Clip negative rewards to -floor (magnitude) when non-zero clipping is used.",
    )
    REWARD_SCALE: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Global multiplier on shaped reward before logging and Bellman backup.",
    )
    END_OF_TRACK_REWARD: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Bonus when the end of the track / finish line is reached.",
    )
    TIME_BONUS_SCALE: float = Field(
        default=0.0,
        description="Scale for optional time-based completion bonus (0 disables).",
    )
    PROJECTED_VELOCITY_SCALE: float = Field(
        default=0.0,
        description="Weight for reward from velocity projected on track tangent.",
    )
    TRACK_LOOK_AHEAD_PCT: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Percent of track length used to place lookahead observation points.",
    )
    TRACK_POINT_SPACING_M: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Spacing in meters between lookahead polyline samples.",
    )
    TRACK_LOCAL_FRAME: bool = Field(
        default=False,
        description="Express observations in a track-aligned local frame when True.",
    )
    TRACK_CURVATURE_OBS: bool = Field(
        default=False,
        description="Append curvature features at lookahead to the observation vector.",
    )
    MIN_EPISODE_LENGTH_GUARANTEED: PositiveInt = Field(
        default=100,
        description="Never terminate before this many steps except hard crashes if implemented.",
    )
    DRIFT_REWARD_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Weight for Gaussian-shaped drift angle reward near optimal slip.",
    )
    DRIFT_REWARD_WEIGHT_START: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Initial drift reward weight before annealing (defaults to DRIFT_REWARD_WEIGHT).",
    )
    DRIFT_REWARD_WEIGHT_END: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Final drift weight after DRIFT_ANNEAL_STEPS environment steps.",
    )
    DRIFT_ANNEAL_STEPS: int = Field(
        default=0,
        ge=0,
        description="Linear anneal duration for drift reward weight (0 = no anneal).",
    )
    DRIFT_OPTIMAL_ANGLE_DEG: Annotated[float, Field(gt=0.0)] = Field(
        default=12.0,
        description="Peak slip angle in degrees for drift reward Gaussian.",
    )
    DRIFT_SIGMA_DEG: Annotated[float, Field(gt=0.0)] = Field(
        default=8.0,
        description="Angular sigma (degrees) for drift reward falloff.",
    )
    DRIFT_THRESHOLD_KMH: Annotated[float, Field(ge=0.0)] = Field(
        default=80.0,
        description="Minimum speed (km/h) before drift reward contributes.",
    )
    PROGRESS_MIN_ALIGNMENT: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Minimum velocity–track alignment to count progress reward (0 disables gate).",
    )
    VELOCITY_ALIGNMENT_REWARD_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Bonus weight for explicit dot(velocity, track_tangent) shaping.",
    )
    BARRIER_TOUCH_PENALTY: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Penalty when within BARRIER_TOUCH_RADIUS of wall above min speed.",
    )
    BARRIER_TOUCH_RADIUS: Annotated[float, Field(gt=0.0)] = Field(
        default=0.25,
        description="Distance to barrier polyline triggering touch penalty.",
    )
    BARRIER_TOUCH_MIN_SPEED_KMH: Annotated[float, Field(ge=0.0)] = Field(
        default=5.0,
        description="Speed threshold (km/h) below which barrier touch is ignored.",
    )
    SPEED_SAFE_DEVIATION_RATIO: Annotated[float, Field(ge=0.0)] = Field(
        default=0.15,
        description="Legacy speed deviation tolerance for safety-style penalties.",
    )
    WALL_HUG_SPEED_THRESHOLD: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Speed below which wall-hug penalty may apply (km/h).",
    )
    WALL_HUG_PENALTY_FACTOR: Annotated[float, Field(ge=0.0)] = Field(
        default=0.005,
        description="Scale for wall proximity / hug penalty when enabled in reward code.",
    )
    BOUNDARY_PENALTY_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=4.0,
        description="Weight for soft boundary distance penalty.",
    )
    BOUNDARY_CRASH_PENALTY: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Large penalty when leaving drivable corridor (off-track).",
    )
    CONDITIONAL_PENALTY_WHEN_BRAKING: bool = Field(
        default=False,
        description="Apply an extra penalty only when brake input exceeds BRAKE_THRESHOLD.",
    )
    BRAKE_THRESHOLD: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.3,
        description="Normalized brake input above which conditional braking penalty applies.",
    )
    CTE_PENALTY_WEIGHT: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Cross-track error penalty weight (0 disables).",
    )
    CTE_PENALTY_EXPONENT: Annotated[float, Field(gt=0.0)] = Field(
        default=2.0,
        description="Exponent on normalized CTE for penalty sharpness.",
    )
    PROXIMITY_REWARD_SHAPING: float = Field(
        default=0.0,
        description="Legacy proximity shaping coefficient when used by reward implementation.",
    )
    PROGRESS_REWARD_EXPONENT: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Exponent on progress increment reward.",
    )
    SPEED_REWARD_THRESHOLD_KMH: Annotated[float, Field(ge=0.0)] = Field(
        default=50.0,
        description="Speed threshold for legacy speed reward gating.",
    )
    SPEED_TERMINAL_SCALE: float = Field(
        default=0.0,
        description="Terminal-state speed bonus scale when implemented.",
    )


class EnvConfig(BaseModel):
    """TM2020 rtgym interface, rendering, and environment reward wiring."""

    model_config = ConfigDict(extra="allow")

    SEED: int = Field(default=0, description="RNG seed forwarded to the environment and numpy/torch.")
    RTGYM_INTERFACE: str = Field(
        ...,
        description=(
            "Interface id: LIDAR, LIDARPROGRESS, TRACKMAP, FULL, MOBILEV3, TQCGRAB_IMAGES, MTQC, etc."
        ),
    )
    INIT_GAS_BIAS: Annotated[float, Field(ge=-1.0, le=1.0)] = Field(
        default=0.0,
        description="Added to actor gas logit before tanh to bias initial acceleration.",
    )
    MAP_NAME: str = Field(default="", description="Map / campaign name for paths and logging.")
    END_OF_TRACK_REWARD: float = Field(
        default=0.0,
        description="Environment-level finish bonus (may duplicate REWARD_CONFIG.END_OF_TRACK_REWARD).",
    )
    WINDOW_WIDTH: PositiveInt = Field(default=640, description="Game window width in pixels.")
    WINDOW_HEIGHT: PositiveInt = Field(default=480, description="Game window height in pixels.")
    IMG_WIDTH: PositiveInt = Field(default=64, description="Resized observation image width.")
    IMG_HEIGHT: PositiveInt = Field(default=64, description="Resized observation image height.")
    USE_IMAGES: bool = Field(default=True, description="Whether CNN image observations are enabled.")
    IMG_GRAYSCALE: bool = Field(default=True, description="Convert captures to single-channel images.")
    SLEEP_TIME_AT_RESET: Annotated[float, Field(ge=0.0)] = Field(
        default=1.5,
        description="Seconds to sleep after reset to let the game stabilize.",
    )
    IMG_HIST_LEN: PositiveInt = Field(
        default=4,
        description="Number of past frames stacked as the image observation history.",
    )
    MIN_NB_ZERO_REW_BEFORE_FAILURE: int = Field(
        default=0,
        ge=0,
        description="Terminate after this many consecutive zero-reward steps (0 disables).",
    )
    MAX_NB_ZERO_REW_BEFORE_FAILURE: int = Field(
        default=0,
        ge=0,
        description="Upper bound paired with zero-reward failure heuristic.",
    )
    MIN_NB_STEPS_BEFORE_FAILURE: int = Field(
        default=0,
        ge=0,
        description="Minimum steps before failure heuristics may trigger.",
    )
    OSCILLATION_PERIOD: int = Field(
        default=0,
        ge=0,
        description="Oscillation-detection window length; 0 disables that failure mode.",
    )
    NB_OBS_FORWARD: int = Field(
        default=0,
        ge=0,
        description="Forward observation count used in legacy reward interfaces.",
    )
    CRASH_PENALTY: float = Field(default=0.0, description="Penalty scalar on crash events.")
    CRASH_COOLDOWN: int = Field(
        default=0,
        ge=0,
        description="Steps to suppress repeated crash detections after a crash.",
    )
    CONSTANT_PENALTY: float = Field(
        default=0.0,
        description="Per-step constant penalty in environment reward (e.g. time cost).",
    )
    LAP_REWARD: float = Field(default=0.0, description="Bonus for completing a lap.")
    LAP_COOLDOWN: int = Field(default=0, ge=0, description="Steps to ignore further lap rewards.")
    CHECKPOINT_REWARD: float = Field(default=0.0, description="Reward for passing intermediate CPs.")
    LINUX_X_OFFSET: int = Field(default=64, description="Linux window capture X offset in pixels.")
    LINUX_Y_OFFSET: int = Field(default=70, description="Linux window capture Y offset in pixels.")
    IMG_SCALE_CHECK_ENV: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Scale factor when sanity-checking image dimensions at startup.",
    )
    OBS_SPEED_SCALE: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Multiplier on speed features in the observation vector.",
    )
    OBS_TRACK_SCALE: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Multiplier on track geometry features in observations.",
    )
    REWARD_CONFIG: RewardConfig = Field(
        default_factory=RewardConfig,
        description="Nested reward hyperparameters consumed by compute_reward.",
    )
    RTGYM_CONFIG: RtGymConfigSection = Field(
        default_factory=RtGymConfigSection,
        description="real-time-gym timestep and episode configuration.",
    )


class ModelConfig(BaseModel):
    """Trainer loop, replay, and neural architecture options."""

    model_config = ConfigDict(extra="forbid")

    MAX_EPOCHS: PositiveInt = Field(default=10000, description="Maximum training epochs.")
    ROUNDS_PER_EPOCH: PositiveInt = Field(
        default=100,
        description="Number of rollout rounds collected per epoch.",
    )
    TRAINING_STEPS_PER_ROUND: PositiveInt = Field(
        default=200,
        description="Gradient update steps executed per round on the trainer.",
    )
    MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP: Annotated[float, Field(gt=0.0)] = Field(
        default=4.0,
        description="Cap on optimizer steps per single env step (async training throttle).",
    )
    ENVIRONMENT_STEPS_BEFORE_TRAINING: int = Field(
        default=1000,
        ge=0,
        description="Collect this many env steps before the first training update.",
    )
    UPDATE_MODEL_INTERVAL: PositiveInt = Field(
        default=200,
        description="Env steps between policy weight broadcasts to workers.",
    )
    UPDATE_BUFFER_INTERVAL: PositiveInt = Field(
        default=200,
        description="Steps between central buffer synchronization events.",
    )
    SAVE_MODEL_EVERY: int = Field(
        default=0,
        ge=0,
        description="Save checkpoint every N epochs; 0 disables periodic saves.",
    )
    BEST_CHECKPOINT_CRITERION: str = Field(
        default="eval",
        description="Metric name used to pick best checkpoint (eval return, lap time, etc.).",
    )
    BEST_CHECKPOINT_LAP_TIME: bool = Field(
        default=True,
        description="Prefer competition-style mean lap checkpointing when enough clean finishes.",
    )
    BEST_CHECKPOINT_MIN_FINISHES: int | None = Field(
        default=None,
        description="Minimum successful finishes in eval window for lap-based best model; null=all.",
    )
    COMPETITION_EVAL_CRASH_PENALTY_S: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Seconds added to next lap time per crash in competition eval.",
    )
    COMPETITION_EVAL_MAX_CRASHES: int = Field(
        default=3,
        ge=0,
        description="Maximum crashes allowed in an eval episode before disqualification.",
    )
    MEMORY_SIZE: PositiveInt = Field(default=1_000_000, description="Replay buffer capacity in transitions.")
    BATCH_SIZE: PositiveInt = Field(default=256, description="SGD minibatch size for learner updates.")
    BATCHES_PER_STEP: PositiveInt = Field(
        default=1,
        description="Number of minibatches per learner step (UTD-style repetition).",
    )
    SCHEDULER: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Optional LR scheduler configuration for the trainer.",
    )
    NOISY_LINEAR_CRITIC: bool = Field(
        default=False,
        description="Use factorized Gaussian noise on critic linear layers (NoisyNet).",
    )
    NOISY_LINEAR_ACTOR: bool = Field(
        default=False,
        description="Use factorized Gaussian noise on actor linear layers (NoisyNet).",
    )
    OUTPUT_DROPOUT: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Dropout on policy/value heads before output.",
    )
    RNN_DROPOUT: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.0,
        description="Dropout inside recurrent cores when USE_RNN is True.",
    )
    CNN_FILTERS: list[PositiveInt] = Field(
        default_factory=lambda: [32, 64, 64, 64],
        description="Per-layer channel widths for the vanilla CNN encoder.",
    )
    CNN_OUTPUT_SIZE: PositiveInt = Field(
        default=256,
        description="Flattened CNN embedding dimension before MLP heads.",
    )
    RNN_LENS: list[PositiveInt] = Field(
        default_factory=lambda: [1],
        description="Sequence chunk lengths for recurrent training when used.",
    )
    RNN_SIZES: list[PositiveInt] = Field(
        default_factory=lambda: [64],
        description="Hidden sizes per recurrent layer.",
    )
    API_MLP_SIZES: list[PositiveInt] = Field(
        default_factory=lambda: [256, 256],
        description="Hidden layer sizes for non-image API MLP trunk.",
    )
    API_LAYERNORM: bool = Field(default=True, description="Apply LayerNorm in API MLP blocks.")
    MLP_LAYERNORM: bool = Field(default=False, description="Apply LayerNorm inside residual MLP stacks.")
    USE_RESIDUAL_MLP: bool = Field(
        default=False,
        description="Use residual MLP actor/critic backbones for lidar / vector obs.",
    )
    RESIDUAL_MLP_HIDDEN_DIM: PositiveInt = Field(
        default=256,
        description="Hidden width of each residual MLP block.",
    )
    RESIDUAL_MLP_NUM_BLOCKS: PositiveInt = Field(
        default=6,
        description="Default residual block count when actor/critic-specific counts are 0.",
    )
    RESIDUAL_MLP_NUM_BLOCKS_ACTOR: int = Field(
        default=0,
        ge=0,
        description="Actor-specific residual depth; 0 means use RESIDUAL_MLP_NUM_BLOCKS.",
    )
    RESIDUAL_MLP_NUM_BLOCKS_CRITIC: int = Field(
        default=0,
        ge=0,
        description="Critic-specific residual depth; 0 means use RESIDUAL_MLP_NUM_BLOCKS.",
    )
    USE_RESIDUAL_SOPHY: bool = Field(
        default=False,
        description="Use Sophy-style residual CNN trunk for image policies.",
    )
    USE_TRACK_CONV1D: bool = Field(
        default=True,
        description="Enable 1D conv processing of track polyline features.",
    )
    USE_SIMBAV2: bool = Field(
        default=False,
        description="Enable SimbaV2-specific architecture paths in the policy network.",
    )
    TRACK_ENCODER: str = Field(
        default="conv1d",
        description="Track encoder type: conv1d, gnn, etc.",
    )
    GNN_LAYERS: PositiveInt = Field(default=3, description="Number of message-passing layers in GNN encoder.")
    GNN_HIDDEN: PositiveInt = Field(default=64, description="Hidden dimension for GNN track encoder.")
    BINARY_BRAKE: bool = Field(
        default=False,
        description="Discretize brake output to {0,1} at the policy head when True.",
    )
    USE_RNN: bool = Field(
        default=False,
        description="Insert an LSTM/GRU core in the policy when True (advanced setups).",
    )
    RNN_HIDDEN_SIZE: int = Field(
        default=0,
        ge=0,
        description="Recurrent hidden size; 0 falls back to RESIDUAL_MLP_HIDDEN_DIM in trainers.",
    )
    USE_EFFICIENTNET: bool = Field(
        default=True,
        description="Use EfficientNet backbone for image encoders when applicable.",
    )
    USE_FROZEN_EFFNET: bool = Field(
        default=False,
        description="Freeze EfficientNet weights and train only heads / fusion MLP.",
    )
    FROZEN_EFFNET_EMBED_DIM: PositiveInt = Field(
        default=256,
        description="Embedding dimension after frozen EfficientNet pooling.",
    )
    FROZEN_EFFNET_WIDTH_MULT: Annotated[float, Field(gt=0.0)] = Field(
        default=0.5,
        description="EfficientNet width multiplier for parameter/perf tradeoff.",
    )
    FROZEN_EFFNET_VARIANT: str = Field(
        default="xs",
        description="Which EfficientNet variant string the builder loads.",
    )
    FROZEN_EFFNET_USE_DW_STEM: bool = Field(
        default=False,
        description="Use depthwise separable stem in frozen EfficientNet wrapper.",
    )


class AlgConfig(BaseModel):
    """Algorithm hyperparameters (SAC, TQC, REDQ, IQN, SDSAC extensions)."""

    model_config = ConfigDict(extra="forbid")

    ALGORITHM: Literal["SAC", "TQC", "REDQSAC", "IQN", "SDSAC"] = Field(
        ...,
        description="Which learner implementation to instantiate.",
    )
    LEARN_ENTROPY_COEF: bool = Field(
        default=False,
        description="Learn entropy coefficient alpha automatically (SAC v2 style).",
    )
    LR_ACTOR: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-5,
        description="Policy network learning rate.",
    )
    LR_CRITIC: Annotated[float, Field(ge=0.0)] = Field(
        default=5e-5,
        description="Q / critic learning rate.",
    )
    LR_ENTROPY: Annotated[float, Field(ge=0.0)] = Field(
        default=3e-4,
        description="Learning rate for log-alpha when LEARN_ENTROPY_COEF is True.",
    )
    GAMMA: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.99,
        description="Discount factor for n-step and Bellman targets.",
    )
    POLYAK: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.995,
        description="Soft-update interpolation factor for target networks.",
    )
    TARGET_ENTROPY: float = Field(
        default=-0.5,
        description="Target entropy for automatic temperature tuning.",
    )
    ALPHA: Annotated[float, Field(ge=0.0)] = Field(
        default=0.01,
        description="Initial or fixed entropy temperature.",
    )
    REDQ_N: PositiveInt = Field(default=10, description="Ensemble size for REDQ critics.")
    REDQ_M: PositiveInt = Field(default=2, description="Subset size sampled for REDQ target minimization.")
    REDQ_Q_UPDATES_PER_POLICY_UPDATE: PositiveInt = Field(
        default=20,
        description="Critic gradient steps per actor update in REDQ (UTD).",
    )
    TOP_QUANTILES_TO_DROP: int = Field(
        default=2,
        ge=0,
        description="TQC: number of highest quantile predictions dropped in robust update.",
    )
    QUANTILES_NUMBER: PositiveInt = Field(
        default=1,
        description="Number of quantiles per critic output; must be 1 for plain SAC.",
    )
    N_STEPS: int = Field(
        default=1,
        ge=0,
        description="N-step return horizon; 0 or 1 treated as one-step in parts of the code.",
    )
    R2D2_REWIND: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Probability / fraction controlling R2D2 sequence rewind augmentation.",
    )
    R2D2_NUM_SEQUENCES: int = Field(
        default=0,
        ge=0,
        description="Sequences per batch for R2D2 replay; 0 uses i.i.d. sampling mode.",
    )
    R2D2_SEQUENCE_LENGTH: int = Field(
        default=0,
        ge=0,
        description="Recurrent sequence length L for R2D2; B*L should match batch layout when set.",
    )
    R2D2_BURN_IN: int = Field(
        default=0,
        ge=0,
        description="Prefix length without BPTT for hidden state burn-in in recurrent replay.",
    )
    OPTIMIZER_ACTOR: str = Field(default="adam", description="Optimizer name for the actor.")
    OPTIMIZER_CRITIC: str = Field(default="adam", description="Optimizer name for critics.")
    BETAS_ACTOR: list[float] | None = Field(
        default=None,
        description="Adam beta1,beta2 for actor; None uses PyTorch defaults.",
    )
    BETAS_CRITIC: list[float] | None = Field(
        default=None,
        description="Adam beta1,beta2 for critic; None uses PyTorch defaults.",
    )
    L2_ACTOR: float | None = Field(default=None, description="Optional AdamW weight decay for actor.")
    L2_CRITIC: float | None = Field(default=None, description="Optional AdamW weight decay for critic.")
    GRAD_CLIP_ACTOR: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Max grad norm for actor; 0 disables clipping.",
    )
    GRAD_CLIP_CRITIC: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Max grad norm for critic; 0 disables clipping.",
    )
    BACKUP_CLIP_RANGE: Annotated[float, Field(gt=0.0)] = Field(
        default=100.0,
        description="Symmetric clip on TD targets for stability (TQC / distributional).",
    )
    REWARD_NORMALIZE_SCALE: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Divide rewards by this factor before Bellman backup when not 1.0.",
    )
    MEAN_PENALTY_COEF: Annotated[float, Field(ge=0.0)] = Field(
        default=0.05,
        description="Coefficient for mean Q regularization in some actor losses.",
    )
    DYNAMIC_TRUNCATION_ENABLED: bool = Field(
        default=False,
        description="TQC: adaptively drop more quantiles when target variance is high.",
    )
    DYNAMIC_TRUNCATION_VARIANCE_PCT: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.9,
        description="Variance percentile threshold for dynamic quantile truncation.",
    )
    VCSE_ENABLED: bool = Field(
        default=False,
        description="Variance-conditioned entropy exploration modulation.",
    )
    VCSE_LAMBDA: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Scales critic uncertainty into entropy bonus when VCSE is enabled.",
    )
    VCSE_ALPHA_BASE: Annotated[float, Field(ge=0.0)] = Field(
        default=0.03,
        description="Base entropy offset added before VCSE uncertainty term.",
    )
    CLIPPING_WEIGHTS: bool = Field(default=False, description="Globally clip network weight magnitudes.")
    CLIP_WEIGHTS_VALUE: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Absolute max weight value when CLIPPING_WEIGHTS is True.",
    )
    ACTOR_WEIGHT_DECAY: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="L2-style weight decay on actor parameters (legacy path).",
    )
    CRITIC_WEIGHT_DECAY: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="L2-style weight decay on critic parameters (legacy path).",
    )
    NUMBER_OF_POINTS: int = Field(
        default=0,
        ge=0,
        description="Fallback polyline point count when trajectory-derived count is unavailable.",
    )
    POINTS_DISTANCE: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Nominal spacing between discretized track points when used.",
    )
    SPEED_BONUS: float = Field(default=0.0, description="Legacy speed bonus scale in algorithm config.")
    SPEED_MIN_THRESHOLD: float = Field(
        default=0.0,
        description="Lower speed threshold for shaped returns in legacy code paths.",
    )
    SPEED_MEDIUM_THRESHOLD: float = Field(
        default=0.0,
        description="Mid speed threshold for shaped returns in legacy code paths.",
    )
    ADAM_EPS: Annotated[float, Field(gt=0.0)] = Field(
        default=1e-8,
        description="Numerical epsilon for Adam-family optimizers.",
    )
    BC_LAMBDA: Annotated[float, Field(ge=0.0)] = Field(
        default=0.0,
        description="Behavior cloning auxiliary loss weight (static).",
    )
    BC_LAMBDA_START: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="BC loss weight at the start of annealing schedule.",
    )
    BC_LAMBDA_END: Annotated[float, Field(ge=0.0)] = Field(
        default=0.01,
        description="BC loss weight at the end of annealing schedule.",
    )
    BC_ANNEAL_STEPS_START: int = Field(
        default=0,
        ge=0,
        description="Environment step at which BC annealing begins.",
    )
    BC_ANNEAL_STEPS_END: int = Field(
        default=2_000_000,
        ge=0,
        description="Environment step at which BC annealing ends.",
    )
    MIXED_PRECISION: bool = Field(default=True, description="Use AMP on supported GPUs.")
    MIXED_PRECISION_DTYPE: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Torch AMP dtype selection.",
    )
    HORIZONTAL_FLIP_P: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.5,
        description="Probability of left-right image flip for data augmentation.",
    )
    PER_TD_ENABLED: bool = Field(default=False, description="Enable prioritized replay for TD learning.")
    PER_TD_ALPHA: Annotated[float, Field(ge=0.0)] = Field(
        default=0.6,
        description="PER prioritization exponent (how much prioritization matters).",
    )
    PER_TD_BETA: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.4,
        description="Importance-sampling correction exponent for PER.",
    )
    PER_TD_EPS: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-6,
        description="Small positive offset added to priorities for numerical stability.",
    )
    USE_SDE: bool = Field(
        default=True,
        description="Use state-dependent exploration noise instead of independent Gaussian.",
    )
    LOG_STD_INIT: float = Field(default=-3.0, description="Initial log std for gSDE noise parameterization.")
    SDE_CLIP_MEAN: Annotated[float, Field(ge=0.0)] = Field(
        default=2.0,
        description="Clip actor mean to [-SDE_CLIP_MEAN, SDE_CLIP_MEAN] under gSDE; 0 disables.",
    )
    SDE_SAMPLE_FREQ: PositiveInt = Field(
        default=100,
        description="Re-draw gSDE noise every this many environment steps.",
    )
    ENTROPY_FLOOR: Annotated[float, Field(ge=0.0)] = Field(
        default=0.02,
        description="Minimum entropy coefficient to prevent collapse.",
    )
    ENTROPY_SCHEDULE: str = Field(
        default="learnable",
        description="Entropy schedule id: learnable, cosine, constant, etc.",
    )
    ENTROPY_COSINE_T0: PositiveInt = Field(
        default=300,
        description="First half-period length for cosine entropy schedule.",
    )
    ENTROPY_COSINE_TMULT: Annotated[float, Field(gt=0.0)] = Field(
        default=1.5,
        description="Period growth factor for cosine entropy restarts.",
    )
    ENTROPY_COSINE_DECAY: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Amplitude decay factor each cosine cycle.",
    )
    FOG_DECAY_TEMPERATURE: Annotated[float, Field(gt=0.0)] = Field(
        default=3.0,
        description="Temperature for future observation gating (FOG) when used.",
    )
    EDER_OVERSAMPLE_RATIO: int = Field(
        default=0,
        ge=0,
        description="EDER oversampling ratio hyperparameter when that path is enabled.",
    )
    SDSAC_AVG_Q: bool = Field(default=True, description="SDSAC: average ensemble Q estimates.")
    SDSAC_CLIP_Q: bool = Field(default=True, description="SDSAC: clip Q targets for stability.")
    SDSAC_CLIP_Q_EPSILON: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Epsilon width for SDSAC Q clipping.",
    )
    SDSAC_ENTROPY_PENALTY: bool = Field(
        default=True,
        description="SDSAC: add entropy penalty term to objective.",
    )
    SDSAC_ENTROPY_PENALTY_BETA: Annotated[float, Field(ge=0.0)] = Field(
        default=0.5,
        description="Weight of SDSAC entropy penalty term.",
    )
    IQN_N_STEER_BINS: PositiveInt = Field(
        default=13,
        description="Discretized steering bins for IQN action space.",
    )
    IQN_LR: Annotated[float, Field(ge=0.0)] = Field(
        default=1e-4,
        description="IQN learner learning rate.",
    )
    IQN_EPSILON_SCHEDULE_MODE: str = Field(
        default="cosine",
        description="Exploration epsilon schedule: linear, cosine, etc.",
    )
    IQN_EPSILON_START: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=1.0,
        description="Initial epsilon for epsilon-greedy exploration.",
    )
    IQN_EPSILON_END: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.005,
        description="Final epsilon after decay schedule.",
    )
    IQN_EXPLORE_REPEAT_STEPS: PositiveInt = Field(
        default=4,
        description="Hold random exploratory action for this many env steps when exploring.",
    )
    IQN_EPSILON_DECAY_STEPS: PositiveInt = Field(
        default=500_000,
        description="Environment steps over which linear epsilon decays (mode-dependent).",
    )
    IQN_EPSILON_COSINE_T0: PositiveInt = Field(
        default=50_000,
        description="First cosine period length for epsilon schedule.",
    )
    IQN_EPSILON_COSINE_TMULT: Annotated[float, Field(gt=0.0)] = Field(
        default=1.5,
        description="Cosine period multiplier for epsilon schedule restarts.",
    )
    IQN_EPSILON_COSINE_DECAY: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.8,
        description="Per-cycle amplitude decay for cosine epsilon schedule.",
    )
    IQN_TARGET_UPDATE_FREQ: PositiveInt = Field(
        default=1000,
        description="Env steps between hard or soft IQN target network updates.",
    )
    IQN_NUM_QUANTILES_TRAIN: PositiveInt = Field(
        default=64,
        description="Number of quantile samples per batch for IQN loss.",
    )
    IQN_NUM_QUANTILES_TARGET: PositiveInt = Field(
        default=64,
        description="Quantile count for target value distribution in IQN.",
    )
    IQN_NUM_QUANTILES_EVAL: PositiveInt = Field(
        default=32,
        description="Quantile count when estimating greedy action for deployment.",
    )
    IQN_N_COS: PositiveInt = Field(
        default=64,
        description="Embedding dimension for IQN cosine basis functions.",
    )
    IQN_DUELING: bool = Field(default=True, description="Use dueling architecture in IQN backbone.")
    IQN_DOUBLE_DQN: bool = Field(default=True, description="Use double Q reduction for IQN targets.")
    IQN_GRAD_CLIP: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Max grad norm for IQN optimizer; 0 disables.",
    )
    IQN_EPSILON_COSINE_INITIAL_AMPLITUDE: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.1,
        description="Starting amplitude for cosine epsilon exploration envelope.",
    )
    IQN_EPSILON_COSINE_FLOOR_FRACTION: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.03,
        description="Floor as fraction of IQN_EPSILON_START for cosine epsilon schedule.",
    )
    IQN_EPSILON_COSINE_FLOOR_STEPS: int = Field(
        default=0,
        ge=0,
        description="Optional flat epsilon floor phase length before cosine decay.",
    )
    IQN_N_ACTIONS: PositiveInt = Field(
        default=78,
        description="Total discrete actions including gas/brake/steer product space size.",
    )

    @model_validator(mode="after")
    def _quantiles_consistent(self) -> AlgConfig:
        from tmrl.config.enums import AlgorithmName

        for beta_name in ("BETAS_ACTOR", "BETAS_CRITIC"):
            betas = getattr(self, beta_name)
            if betas is not None and len(betas) != 2:
                raise ValueError(f"{beta_name} must have length 2 when set")

        try:
            name = AlgorithmName(self.ALGORITHM)
        except ValueError:
            return self
        if name not in (AlgorithmName.TQC, AlgorithmName.IQN, AlgorithmName.SDSAC) and self.QUANTILES_NUMBER > 1:
            raise ValueError("QUANTILES_NUMBER must be 1 unless using TQC, IQN, or SDSAC")
        if name == AlgorithmName.SAC and self.QUANTILES_NUMBER != 1:
            raise ValueError("SAC requires QUANTILES_NUMBER == 1")
        return self


class MainConfig(BaseModel):
    """Full TMRL configuration: run, networking, model, algorithm, environment, debugger."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    config_schema_version: str = Field(
        ...,
        validation_alias=AliasChoices("__VERSION__", "config_schema_version"),
        serialization_alias="__VERSION__",
        description="Config schema version; must be >= loader minimum for compatibility.",
    )
    RUN_NAME: str = Field(..., description="Experiment / run id for paths, checkpoints, and wandb.")
    RESET_TRAINING: bool = Field(
        default=False,
        description="When True, discard resumed checkpoints and start weights from scratch.",
    )
    DATASET_PATH: str = Field(
        default="",
        description="Optional override path for offline dataset folder; empty uses TmrlData layout.",
    )
    BUFFERS_MAXLEN: PositiveInt = Field(
        default=500_000,
        description="Max stored rollout samples per worker buffer before eviction.",
    )
    RW_MAX_SAMPLES_PER_EPISODE: PositiveInt = Field(
        default=1000,
        description="Hard cap on timesteps collected per rollout episode.",
    )
    RW_TEST_EPISODE_INTERVAL: PositiveInt = Field(
        default=5,
        description="Episodes between periodic test-mode eval rollouts on workers.",
    )
    RW_TEST_EPISODES_PER_EVAL: PositiveInt = Field(
        default=10,
        description="Number of deterministic eval episodes per evaluation signal.",
    )
    CUDA_TRAINING: bool = Field(default=True, description="Use GPU for the central trainer process.")
    CUDA_INFERENCE: bool = Field(default=False, description="Use GPU on rollout worker inference.")
    VIRTUAL_GAMEPAD: bool = Field(
        default=True,
        description="Use virtual gamepad device; False falls back to keyboard controls.",
    )
    LOCALHOST_WORKER: bool = Field(
        default=True,
        description="Worker connects to 127.0.0.1 when colocated with the server.",
    )
    LOCALHOST_TRAINER: bool = Field(
        default=True,
        description="Trainer connects to 127.0.0.1 when colocated with the server.",
    )
    PUBLIC_IP_SERVER: str = Field(
        default="0.0.0.0",
        description="Advertised server IP for remote workers and trainers.",
    )
    PASSWORD: str = Field(
        default="YourRandomPasswordHere",
        description="Shared secret for tlspyo / TMRL networking (override via TMRL_PASSWORD env).",
    )
    TLS: bool = Field(default=False, description="Enable TLS on tlspyo transports when True.")
    TLS_HOSTNAME: str = Field(default="default", description="TLS server name indication hostname.")
    TLS_CREDENTIALS_DIRECTORY: str = Field(
        default="",
        description="Directory with TLS certs; empty disables custom credential loading.",
    )
    NB_WORKERS: int = Field(
        default=-1,
        description="Requested worker count; negative means autodetect / unlimited pool.",
    )
    WANDB_PROJECT: str = Field(default="tmrl", description="Weights & Biases project name.")
    WANDB_ENTITY: str = Field(default="tmrl", description="Weights & Biases entity / team.")
    WANDB_KEY: str = Field(
        default="YourWandbApiKey",
        description="API key placeholder; prefer WANDB_API_KEY environment variable.",
    )
    WANDB_GRADIENTS: bool = Field(default=False, description="Log weight histograms / gradients to wandb.")
    WANDB_DEBUG_REWARD: bool = Field(default=True, description="Log reward-debug series to wandb.")
    WANDB_WORKER: bool = Field(default=True, description="Enable wandb logging from worker processes.")
    PORT: int = Field(default=55555, ge=1, le=65535, description="Primary tlspyo server port.")
    LOCAL_PORT_SERVER: int = Field(
        default=55556,
        ge=1,
        le=65535,
        description="Secondary local server bind port.",
    )
    LOCAL_PORT_TRAINER: int = Field(
        default=55557,
        ge=1,
        le=65535,
        description="Trainer-side local port allocation.",
    )
    LOCAL_PORT_WORKER: int = Field(
        default=55558,
        ge=1,
        le=65535,
        description="Worker-side local port allocation.",
    )
    BUFFER_SIZE: PositiveInt = Field(
        default=536_870_912,
        description="Socket buffer size in bytes for high-throughput tensor streaming.",
    )
    HEADER_SIZE: PositiveInt = Field(
        default=12,
        description="Binary message header size in bytes on the wire protocol.",
    )
    SOCKET_TIMEOUT_CONNECT_TRAINER: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Seconds to wait for trainer socket connect.",
    )
    SOCKET_TIMEOUT_ACCEPT_TRAINER: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Seconds to wait accepting trainer connections.",
    )
    SOCKET_TIMEOUT_CONNECT_ROLLOUT: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Seconds to wait for rollout worker connect.",
    )
    SOCKET_TIMEOUT_ACCEPT_ROLLOUT: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Seconds to wait accepting rollout connections.",
    )
    SOCKET_TIMEOUT_COMMUNICATE: Annotated[float, Field(gt=0.0)] = Field(
        default=30.0,
        description="Per-message socket timeout during active communication.",
    )
    SELECT_TIMEOUT_OUTBOUND: Annotated[float, Field(gt=0.0)] = Field(
        default=30.0,
        description="select/poll timeout for outbound queues.",
    )
    ACK_TIMEOUT_WORKER_TO_SERVER: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="ACK timeout for worker→server control messages.",
    )
    ACK_TIMEOUT_TRAINER_TO_SERVER: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="ACK timeout for trainer→server control messages.",
    )
    ACK_TIMEOUT_SERVER_TO_WORKER: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="ACK timeout for server→worker control messages.",
    )
    ACK_TIMEOUT_SERVER_TO_TRAINER: Annotated[float, Field(gt=0.0)] = Field(
        default=7200.0,
        description="ACK timeout for server→trainer bulk operations.",
    )
    RECV_TIMEOUT_TRAINER_FROM_SERVER: Annotated[float, Field(gt=0.0)] = Field(
        default=7200.0,
        description="Blocking recv timeout for trainer waiting on server payloads.",
    )
    RECV_TIMEOUT_WORKER_FROM_SERVER: Annotated[float, Field(gt=0.0)] = Field(
        default=600.0,
        description="Blocking recv timeout for worker waiting on server payloads.",
    )
    WAIT_BEFORE_RECONNECTION: Annotated[float, Field(ge=0.0)] = Field(
        default=10.0,
        description="Cooldown seconds before retrying a dropped connection.",
    )
    LOOP_SLEEP_TIME: Annotated[float, Field(ge=0.0)] = Field(
        default=1.0,
        description="Idle sleep duration in dispatcher loops to reduce CPU spin.",
    )
    PLAYER_RUNS: PlayerRunsConfig = Field(
        default_factory=PlayerRunsConfig,
        description="Human demonstration ingestion settings.",
    )
    MODEL: ModelConfig = Field(..., description="Training loop and neural network architecture.")
    ALG: AlgConfig = Field(..., description="RL algorithm hyperparameters.")
    ENV: EnvConfig = Field(..., description="Environment interface and observation preprocessing.")
    DEBUGGER: DebuggerConfig = Field(
        default_factory=DebuggerConfig,
        description="Debugging and profiling switches.",
    )
