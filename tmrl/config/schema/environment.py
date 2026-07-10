"""Simulator interface, observations, rt-gym timing, and reward shaping."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

_REMOVED_REWARD_FIELDS = frozenset(
    {
        "barrier_touch_penalty",
        "barrier_touch_radius",
        "barrier_touch_min_speed_kmh",
        "steering_delta_penalty",
    }
)


class RtGymInterfaceKwargs(BaseModel):
    """Passthrough kwargs for the concrete TM2020 interface implementation."""

    model_config = ConfigDict(extra="allow")

    save_replays: bool = Field(
        default=False,
        description="Request that the game interface persist replay files when supported.",
    )


class RtGymConfig(BaseModel):
    """real-time-gym control loop: timestep duration, action buffer, episode cap."""

    model_config = ConfigDict(extra="allow")

    time_step_duration: float = Field(
        default=0.05,
        gt=0.0,
        le=10.0,
        description="Nominal seconds simulated per env.step (controls control frequency).",
    )
    start_obs_capture: float = Field(
        default=0.04,
        ge=0.0,
        le=10.0,
        description="Delay after step start before grabbing observations (sync with render).",
    )
    time_step_timeout_factor: float = Field(
        default=1.0,
        gt=0.0,
        le=100.0,
        description="Multiplier on time_step_duration before a step is treated as timed out.",
    )
    act_buf_len: PositiveInt = Field(
        default=2,
        description="Number of past actions concatenated to observations (RT-MDP delay model).",
    )
    reset_act_buf: bool = Field(
        default=True,
        description="Clear stored actions on reset so stale commands are not replayed.",
    )
    benchmark: bool = Field(
        default=False,
        description="Skip nonessential synchronization when benchmarking throughput.",
    )
    wait_on_done: bool = Field(
        default=True,
        description="Block until the simulator acknowledges episode termination.",
    )
    ep_max_length: PositiveInt = Field(
        default=1000,
        description="Forced episode truncation after this many environment steps.",
    )
    interface_kwargs: RtGymInterfaceKwargs = Field(
        default_factory=RtGymInterfaceKwargs,
        description="Additional keyword arguments forwarded into the interface constructor.",
    )


class RewardConfig(BaseModel):
    """Dense and sparse rewards, progress checks, and episode cutoffs used by compute_reward."""

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_reward_fields(cls, data: Any) -> Any:
        """Reject config fields that were removed from the reward schema.

        Raises early with a descriptive error rather than silently ignoring legacy
        fields that are no longer implemented in the reward function.

        Args:
            data: Raw input value before Pydantic field coercion.

        Returns:
            The original *data* unchanged when no removed fields are present.

        Raises:
            ValueError: If any field name from ``_REMOVED_REWARD_FIELDS`` appears in *data*.
        """
        if isinstance(data, dict):
            removed = sorted(_REMOVED_REWARD_FIELDS.intersection(data))
            if removed:
                names = ", ".join(removed)
                raise ValueError(
                    f"Removed reward config field(s): {names}. "
                    "These reward shaping terms are no longer implemented."
                )
        return data

    min_seconds_before_failure: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "End the episode after this many sim seconds without forward progress along the "
            "reference trajectory (0 disables this cutoff)."
        ),
    )
    off_track_seconds_before_failure: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Sim seconds after each reset before off-track termination can trigger (spawn grace). "
            "0 allows off-track from the first env step."
        ),
    )
    min_progress_rate: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Terminate ('slow_progress') when track distance gained averaged over "
            "slow_progress_window_seconds falls below this rate in m/s (0 disables). Catches "
            "slow creep that evades the binary no-progress timeout, which is reset by any "
            "forward gain however small."
        ),
    )
    slow_progress_window_seconds: Annotated[float, Field(gt=0.0)] = Field(
        default=5.0,
        description="Sliding-window length (sim seconds) for the min_progress_rate check.",
    )
    debug_reward_components: bool = Field(
        default=False,
        description="Print or log per-component reward decomposition during rollout.",
    )
    debug_log_interval: PositiveInt = Field(
        default=100,
        description="Environment steps between debug logs when debug_reward_components is true.",
    )
    constant_penalty: float = Field(
        default=0.0,
        description="Small per-step penalty discouraging stagnation or encouraging efficiency.",
    )
    terminal_failure_penalty: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "One-time penalty (pre reward_scale) applied when an episode ends in a failure "
            "termination (no_progress_timeout / off_track / boundary_crash). Without it, "
            "'creep then stall' bootstraps to V~0 and is a free local optimum. Applied after "
            "reward_clip_floor on purpose (the floor bounds per-step shaping, not terminal events)."
        ),
    )
    check_forward: PositiveInt = Field(
        default=500,
        description="Number of polyline vertices ahead used for forward progress estimation.",
    )
    check_backward: PositiveInt = Field(
        default=10,
        description="Vertices behind the car used to detect rollback or oscillation.",
    )
    min_steps: PositiveInt = Field(
        default=70,
        description=(
            "Legacy W&B / snapshots only; stagnant-progress cutoff uses "
            "min_seconds_before_failure (seconds), not this field."
        ),
    )
    max_stray: Annotated[float, Field(gt=0.0)] = Field(
        default=50.0,
        description="Maximum lateral distance (m) from the reference trajectory before failure.",
    )
    progress_reward_full_lap: float = Field(
        default=200.0,
        ge=0.0,
        description="Bonus awarded when the agent advances one full lap along the spline.",
    )
    speed_reward_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Scale for aligning speed with the track tangent. "
            "Credited only when reward_progress > 0 (new ground covered)."
        ),
    )
    speed_reward_exponent: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Exponent on normalized speed; >1 emphasizes high-speed segments.",
    )
    speed_reward_alignment_floor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum heading alignment factor applied before speed reward is credited.",
    )
    max_speed_kmh: Annotated[float, Field(gt=0.0)] = Field(
        default=300.0,
        description="Reference velocity (km/h) for normalizing speed-based rewards.",
    )
    max_track_width: Annotated[float, Field(gt=0.0)] = Field(
        default=65.0,
        description="Half-width scale (m) for boundary and cross-track penalties.",
    )
    crash_penalty: float = Field(
        default=0.5,
        ge=0.0,
        description=(
            "Penalty applied when a crash or hard reset is triggered (RewardFunction and "
            "interface crash subtraction). Merged at load: non-zero environment.crash_penalty "
            "overrides this value for the effective runtime scalar."
        ),
    )
    reward_clip_floor: float = Field(
        default=10.0,
        ge=0.0,
        description="Magnitude floor when clipping negative shaped rewards.",
    )
    reward_scale: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Global multiplier applied to the shaped reward before logging and RL backup.",
    )
    end_of_track_reward: float = Field(
        default=10.0,
        ge=0.0,
        description="Sparse bonus for crossing the finish line / end of authored track.",
    )
    time_bonus_scale: float = Field(
        default=0.0,
        description="Optional reward for finishing quickly (0 disables time-based bonus).",
    )
    projected_velocity_scale: float = Field(
        default=0.0,
        description=(
            "Weight on velocity projected onto the local track tangent (v*cos(theta)*dt). "
            "Dense per-step speed-along-track signal; main lap-time lever in the SOTA reward. "
            "Credited only when reward_progress > 0 (new ground covered)."
        ),
    )
    track_look_ahead_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Percent of total track length used to place lookahead observation samples.",
    )
    track_point_spacing_m: float = Field(
        default=0.0,
        ge=0.0,
        description="Target arc-length spacing (m) between lookahead polyline samples.",
    )
    track_local_frame: bool = Field(
        default=False,
        description="Express vectors in a track-aligned frame when true.",
    )
    track_curvature_obs: bool = Field(
        default=False,
        description="Append curvature features at lookahead points to the observation.",
    )
    min_episode_length_guaranteed: PositiveInt = Field(
        default=100,
        description="Hard minimum episode length before non-crash terminations may fire.",
    )
    drift_reward_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Base drift-shaping weight.",
    )
    drift_reward_weight_start: float = Field(
        default=0.0,
        ge=0.0,
        description="Initial drift weight before annealing (defaults to drift_reward_weight).",
    )
    drift_reward_weight_end: float = Field(
        default=0.0,
        ge=0.0,
        description="Terminal drift weight after annealing completes.",
    )
    drift_anneal_steps: int = Field(
        default=0,
        ge=0,
        description="Environment steps over which drift weights linearly interpolate.",
    )
    drift_optimal_angle_deg: Annotated[float, Field(gt=0.0)] = Field(
        default=12.0,
        description="Slip angle (deg) that maximizes the Gaussian drift reward.",
    )
    drift_sigma_deg: Annotated[float, Field(gt=0.0)] = Field(
        default=8.0,
        description="Angular standard deviation of the drift reward Gaussian.",
    )
    drift_threshold_kmh: float = Field(
        default=80.0,
        ge=0.0,
        description="Minimum speed (km/h) before drift reward contributes.",
    )
    progress_min_alignment: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Legacy/no-op: progress reward is not gated by velocity alignment.",
    )
    velocity_alignment_reward_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Legacy/no-op: no separate velocity-alignment reward is applied.",
    )
    speed_safe_deviation_ratio: float = Field(
        default=0.15,
        ge=0.0,
        description="Legacy speed slack ratio.",
    )
    wall_hug_speed_threshold: float = Field(
        default=10.0,
        ge=0.0,
        description="Speed threshold (km/h) for proximity / wall-hug penalties.",
    )
    wall_hug_penalty_factor: float = Field(
        default=0.005,
        ge=0.0,
        description=(
            "Scale for wall-hug shaping when lateral_ratio exceeds "
            "wall_hug_lateral_threshold at speed >= wall_hug_speed_threshold (0 disables)."
        ),
    )
    boundary_penalty_weight: float = Field(
        default=4.0,
        ge=0.0,
        description=(
            "Soft quadratic penalty when lateral_ratio exceeds boundary_penalty_start "
            "(requires loaded boundary geometry; 0 disables)."
        ),
    )
    boundary_crash_penalty: float = Field(
        default=10.0,
        ge=0.0,
        description=(
            "Terminate and subtract this amount when lateral_ratio > 1.0 "
            "(requires loaded boundary geometry; 0 disables)."
        ),
    )
    conditional_penalty_when_braking: bool = Field(
        default=False,
        description="Apply an extra penalty only when brake input exceeds brake_threshold.",
    )
    brake_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Normalized brake input threshold.",
    )
    cte_penalty_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Cross-track error penalty: weight * (norm_dist ** cte_penalty_exponent) "
            "where norm_dist is distance from trajectory / half track width (0 disables)."
        ),
    )
    cte_penalty_exponent: Annotated[float, Field(gt=0.0)] = Field(
        default=2.0,
        description="Exponent on normalized cross-track distance for cte_penalty_weight.",
    )
    proximity_reward_shaping: float = Field(
        default=0.0,
        description="Legacy proximity shaping coefficient.",
    )
    progress_reward_exponent: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0, description="Exponent on incremental progress."
    )
    speed_reward_threshold_kmh: float = Field(
        default=50.0,
        ge=0.0,
        description="Speed gate for legacy shaping.",
    )
    speed_terminal_scale: float = Field(
        default=0.0,
        description="Terminal-state speed bonus scale.",
    )


class EnvironmentConfig(BaseModel):
    """High-level TM2020 environment: visuals, interface id, reward, and rt-gym settings."""

    model_config = ConfigDict(extra="allow")

    seed: int = Field(
        default=0,
        description="RNG seed forwarded to sim, numpy, and torch where applicable.",
    )
    rtgym_interface: str = Field(
        ...,
        description=(
            "Case-insensitive interface token "
            "(e.g. TM20LIDAR, TM20TRACKMAP, TM20TRACKMAPIMAGES, TM20LIDARIMAGES, "
            "TQCGRAB_IMAGES, MTQC). "
            "Selects observation layout and preprocessor pipeline."
        ),
    )
    init_gas_bias: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Additive bias on the gas logit before tanh to encourage rolling starts.",
    )
    map_name: str = Field(
        default="",
        description="Filesystem-safe map id used for reward/track pickles.",
    )
    end_of_track_reward: float = Field(
        default=0.0,
        description="Environment-level finish bonus (may duplicate reward.end_of_track_reward).",
    )
    window_width: PositiveInt = Field(
        default=640,
        description="Captured game window width in pixels.",
    )
    window_height: PositiveInt = Field(
        default=480,
        description="Captured game window height in pixels.",
    )
    img_width: PositiveInt = Field(
        default=64,
        description="Downsampled observation image width.",
    )
    img_height: PositiveInt = Field(
        default=64,
        description="Downsampled observation image height.",
    )
    use_images: bool = Field(
        default=True,
        description="Whether CNN image tensors are part of the observation.",
    )
    img_grayscale: bool = Field(
        default=True,
        description="Convert RGB captures to single-channel tensors.",
    )
    sleep_time_at_reset: float = Field(
        default=1.5,
        ge=0.0,
        description="Seconds to sleep after reset to let shaders and physics settle.",
    )
    img_hist_len: PositiveInt = Field(
        default=4,
        description="Number of consecutive frames stacked along the channel/time axis.",
    )
    min_zero_reward_steps_before_failure: int = Field(
        default=0,
        ge=0,
        description="Terminate after this many consecutive zero-reward steps (0 disables).",
    )
    max_zero_reward_steps_before_failure: int = Field(
        default=0,
        ge=0,
        description="Upper companion bound for zero-reward-based termination heuristics.",
    )
    min_seconds_before_failure: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Optional floor/raise for stagnant-progress cutoff: merged as "
            "max(reward.min_seconds_before_failure, this). Both are seconds without forward "
            "progress before episode end (0 in both disables)."
        ),
    )
    off_track_seconds_before_failure: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Optional floor/raise for off-track spawn grace: merged as "
            "max(reward.off_track_seconds_before_failure, this). 0 means use reward section only."
        ),
    )
    oscillation_period: int = Field(
        default=0,
        ge=0,
        description="Window for oscillation-based failure; 0 disables that detector.",
    )
    forward_obs_count: int = Field(
        default=0,
        ge=0,
        description="Legacy forward-looking observation slots used by select reward interfaces.",
    )
    crash_penalty: float = Field(
        default=0.0,
        description=(
            "Optional override for crash penalty magnitude. When non-zero, replaces "
            "environment.reward.crash_penalty for the effective value (logged as "
            "CRASH_PENALTY and written back into reward config at load). When zero "
            "(default), use reward.crash_penalty only."
        ),
    )
    crash_cooldown: int = Field(
        default=0,
        ge=0,
        description="Steps to suppress duplicate crash detections.",
    )
    constant_penalty: float = Field(
        default=0.0,
        description="Per-step penalty wired into the interface reward.",
    )
    lap_reward: float = Field(
        default=0.0,
        description="Bonus for crossing intermediate lap triggers.",
    )
    lap_cooldown: int = Field(
        default=0,
        ge=0,
        description="Steps to ignore further lap bonuses.",
    )
    checkpoint_reward: float = Field(
        default=0.0,
        description="Reward for intermediate checkpoint triggers.",
    )
    linux_x_offset: int = Field(default=64, description="X offset for Linux window capture.")
    linux_y_offset: int = Field(default=70, description="Y offset for Linux window capture.")
    img_scale_check_env: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0,
        description="Scale factor when validating observation resolutions at startup.",
    )
    obs_speed_scale: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0, description="Multiplier on speed channels in obs."
    )
    obs_track_scale: Annotated[float, Field(gt=0.0)] = Field(
        default=1.0, description="Multiplier on track geometry channels."
    )
    track_coords_divisor: Annotated[float, Field(gt=0.0)] = Field(
        default=100.0,
        description=(
            "For world-telemetry observations: preprocessor divides track tensor obs[0] by this "
            "value (then clips to [-1, 1]). Lower values amplify small local-frame coordinates for "
            "GNN/Conv encoders; use ~100 for very large world-frame extents. Preset algorithm=iqn "
            "sets a lower default for typical local-track setups."
        ),
    )
    reward: RewardConfig = Field(
        default_factory=RewardConfig,
        description="Reward function weights, gates, and termination thresholds.",
    )
    rtgym: RtGymConfig = Field(
        default_factory=RtGymConfig,
        description="Low-level stepping and buffering parameters forwarded to real-time-gym.",
    )

    @model_validator(mode="after")
    def _reject_legacy_screen_ray_rangefinder_tokens(self) -> EnvironmentConfig:
        """Reject legacy ``TM20*LIDARPROGRESS*`` interface tokens that were removed.

        Returns:
            The validated :class:`EnvironmentConfig` instance.

        Raises:
            ValueError: If ``rtgym_interface`` contains the string ``LIDARPROGRESS``.
        """
        rt = str(self.rtgym_interface).upper()
        if "LIDARPROGRESS" in rt:
            raise ValueError(
                "Legacy TM20*LIDAR*PROGRESS screen-ray rangefinder tokens were removed; "
                "use TM20LIDAR / TM20TRACKMAP or a token with *LIDARIMAGES* / *TRACKMAPIMAGES*."
            )
        return self

    def rtgym_config_dict(self) -> dict[str, Any]:
        """Return a plain dict of rt-gym stepping parameters for mutating ``DEFAULT_CONFIG_DICT``.

        Returns:
            dict: All :class:`RtGymConfig` fields serialized to Python-native types.
        """
        return self.rtgym.model_dump(mode="python")
