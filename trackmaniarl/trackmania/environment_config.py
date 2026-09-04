"""Validated configuration for the TrackMania environment."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.pace import ReferencePaceProfile
from trackmaniarl.trackmania.reward_config import RewardConfig
from trackmaniarl.trackmania.telemetry import (
    DEFAULT_POSITION_INDICES,
    DEFAULT_TELEMETRY_FIELD_COUNT,
    DEFAULT_VELOCITY_INDICES,
)

_FINITE_REWARD_NAMES = (
    "crash_distance",
    "finish_progress",
    "minimum_progress_per_window_m",
    "terminal_failure_penalty",
    "collision_penalty",
    "time_penalty_per_second",
    "max_time_delta_s",
    "progress_reward_full_lap",
    "finish_reward",
    "potential_progress_weight",
    "max_projected_speed_mps",
    "velocity_to_mps_scale",
    "projected_velocity_scale",
    "projected_speed_bonus_scale",
    "steering_delta_penalty",
    "time_attack_bonus_scale",
    "time_attack_linear_scale",
    "pace_reward_scale",
    "pace_debt_clip_s",
    "collision_cooldown_s",
    "reward_gamma",
)

_REWARD_KWARG_NAMES = (
    "crash_distance",
    "finish_progress",
    "no_progress_steps",
    "slow_progress_window_steps",
    "minimum_progress_per_window_m",
    "terminal_failure_penalty",
    "collision_penalty",
    "collision_cooldown_s",
    "minimum_finish_steps",
    "nearest_forward_points",
    "nearest_backward_points",
    "limit_progress_by_kinematics",
    "time_penalty_per_second",
    "max_time_delta_s",
    "maximum_race_time_s",
    "progress_reward_full_lap",
    "finish_reward",
    "potential_progress_weight",
    "max_projected_speed_mps",
    "velocity_to_mps_scale",
    "projected_velocity_scale",
    "projected_speed_bonus_scale",
    "steering_delta_penalty",
    "time_attack_target_s",
    "time_attack_bonus_scale",
    "time_attack_linear_scale",
    "pace_reward_scale",
    "pace_debt_clip_s",
    "reward_gamma",
)


class TrackmaniaEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    geometry_path: Path
    use_racing_line: bool = False
    expected_map_uid: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=9000, ge=1, le=65535)
    session_port: int = Field(default=9001, ge=1, le=65535)
    timeout_s: float = Field(default=10.0, gt=0)
    reset_settle_s: float = Field(default=0.0, ge=0.0)
    start_timeout_s: float = Field(default=15.0, gt=0.0)
    start_poll_s: float = Field(default=0.01, ge=0.0)
    confirm_finish_before_reset: bool = True
    restart_input: Literal["gamepad", "keyboard"] = "gamepad"
    action_repeat_frames: int = Field(default=4, ge=1, le=20)
    decision_interval_ms: float | None = Field(default=None, gt=0.0, le=250.0)
    demonstration_action_lead_ms: float = Field(default=0.0, ge=0.0, le=250.0)
    demonstration_control_aggregation: bool = False
    compact_action_ids: tuple[int, ...] | None = None
    control_backend: Literal["gamepad", "keyboard"] = "gamepad"
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES
    velocity_indices: tuple[int, int, int] = DEFAULT_VELOCITY_INDICES
    crash_distance: float = Field(default=25.0, gt=0)
    finish_progress: float = Field(default=0.995, gt=0.0, le=1.0)
    no_progress_steps: int = Field(default=200, ge=1)
    slow_progress_window_steps: int = Field(default=80, ge=2)
    minimum_progress_per_window_m: float = Field(default=2.0, ge=0.0)
    terminal_failure_penalty: float = Field(default=1.0, ge=0.0)
    collision_penalty: float = Field(default=0.05, ge=0.0)
    collision_cooldown_s: float = Field(default=0.0, ge=0.0)
    minimum_finish_steps: int = Field(default=50, ge=1)
    nearest_forward_points: int = Field(default=500, ge=1)
    nearest_backward_points: int = Field(default=10, ge=0)
    limit_progress_by_kinematics: bool = True
    time_penalty_per_second: float = Field(default=0.1, ge=0.0)
    max_time_delta_s: float = Field(default=1.0, gt=0.0)
    maximum_race_time_s: float | None = Field(default=None, gt=0.0)
    progress_reward_full_lap: float = Field(default=10.0, ge=0.0)
    finish_reward: float = Field(default=30.0, ge=0.0)
    potential_progress_weight: float = Field(default=2.0, ge=0.0)
    max_projected_speed_mps: float = Field(default=100.0, gt=0.0)
    velocity_to_mps_scale: float = Field(default=0.001, gt=0.0)
    projected_velocity_scale: float = Field(default=0.0, ge=0.0)
    projected_speed_bonus_scale: float = Field(default=0.0, ge=0.0)
    steering_delta_penalty: float = Field(default=0.0, ge=0.0)
    time_attack_target_s: float | None = Field(default=None, gt=0.0)
    time_attack_bonus_scale: float = Field(default=0.0, ge=0.0)
    time_attack_linear_scale: float = Field(default=0.0, ge=0.0)
    pace_reference_path: Path | None = None
    pace_reward_scale: float = Field(default=0.0, ge=0.0)
    pace_debt_clip_s: float = Field(default=10.0, gt=0.0)
    reward_gamma: float = Field(default=0.995, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _reward_contract_is_valid(self) -> TrackmaniaEnvironmentConfig:
        self._validate_indices()
        self._validate_reward_values()
        self._validate_time_attack()
        self._validate_decision_interval()
        self._validate_pace()
        select_brake_tap_actions(self.compact_action_ids)
        return self

    def _validate_indices(self) -> None:
        for name, indices in (
            ("position_indices", self.position_indices),
            ("velocity_indices", self.velocity_indices),
        ):
            if len(set(indices)) != 3 or any(
                index < 0 or index >= DEFAULT_TELEMETRY_FIELD_COUNT for index in indices
            ):
                raise ValueError(f"{name} must be three unique indices inside the telemetry packet")

    def _validate_reward_values(self) -> None:
        values = (getattr(self, name) for name in _FINITE_REWARD_NAMES)
        optional_values = (self.maximum_race_time_s, self.time_attack_target_s)
        if not all(isfinite(value) for value in values) or not all(
            value is None or isfinite(value) for value in optional_values
        ):
            raise ValueError("reward values must be finite")

    def _validate_time_attack(self) -> None:
        if (
            self.time_attack_bonus_scale or self.time_attack_linear_scale
        ) and self.time_attack_target_s is None:
            raise ValueError("time-attack reward scales require time_attack_target_s")

    def _validate_decision_interval(self) -> None:
        if self.decision_interval_ms is not None and self.action_repeat_frames != 1:
            raise ValueError("decision_interval_ms requires action_repeat_frames=1")
        if self.demonstration_control_aggregation and self.decision_interval_ms is None:
            raise ValueError("demonstration control aggregation requires decision_interval_ms")
        if self.demonstration_control_aggregation and self.control_backend != "gamepad":
            raise ValueError("demonstration control aggregation requires the gamepad backend")

    def _validate_pace(self) -> None:
        if self.pace_reward_scale and self.pace_reference_path is None:
            raise ValueError("pace_reward_scale requires pace_reference_path")

    def reward_config(self, pace_profile: ReferencePaceProfile | None = None) -> RewardConfig:
        values = self.model_dump(include=set(_REWARD_KWARG_NAMES))
        return RewardConfig(**values, pace_profile=pace_profile)
