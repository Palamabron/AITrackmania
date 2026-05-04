"""Interface preset configuration loaded from Hydra ``interface/*`` defaults."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InterfaceConfig(BaseModel):
    """Hydra-facing interface preset used for explicit config validation."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    hydra_target: str | None = Field(default=None, alias="_target_")
    name: str = Field(default="vision")
    img_hist_len: int = Field(default=4, ge=1)
    gamepad: bool = Field(default=True)
    grayscale: bool = Field(default=True)
    resize_to: tuple[int, int] = Field(default=(64, 64))
    finish_reward: float = Field(default=1.0)
    constant_penalty: float = Field(default=0.0)
    crash_penalty: float = Field(default=10.0)
    reward_check_forward: int = Field(default=500, ge=1)
    reward_check_backward: int = Field(default=10, ge=0)
    reward_max_stray: float = Field(default=50.0, gt=0.0)
    sleep_time_at_reset: float = Field(default=1.5, ge=0.0)
    window_width: int = Field(default=640, ge=1)
    window_height: int = Field(default=480, ge=1)
    discrete_n_steer_bins: int = Field(default=0, ge=0)
    points_number: int = Field(default=5, ge=0)
    checkpoint_reward: float = Field(default=0.0)
    lap_reward: float = Field(default=0.0)
    track_local_frame: bool = Field(default=False)
    obs_speed_scale: float = Field(default=1.0, gt=0.0)
    obs_track_scale: float = Field(default=1.0, gt=0.0)
    min_steps_end_of_track: int = Field(default=50, ge=1)
    min_episode_length_guaranteed: int = Field(default=100, ge=1)
