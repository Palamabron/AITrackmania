"""Validated configuration and trajectory data for TrackMania rewards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trackmaniarl.trackmania.pace import ReferencePaceProfile


@dataclass(frozen=True, slots=True)
class RewardConfig:
    crash_distance: float = 25.0
    finish_progress: float = 0.995
    no_progress_steps: int = 200
    slow_progress_window_steps: int = 80
    minimum_progress_per_window_m: float = 2.0
    terminal_failure_penalty: float = 1.0
    collision_penalty: float = 0.05
    collision_cooldown_s: float = 0.0
    minimum_finish_steps: int = 50
    nearest_forward_points: int = 500
    nearest_backward_points: int = 10
    time_penalty_per_second: float = 0.1
    max_time_delta_s: float = 1.0
    maximum_race_time_s: float | None = None
    progress_reward_full_lap: float = 10.0
    finish_reward: float = 30.0
    potential_progress_weight: float = 2.0
    max_projected_speed_mps: float = 100.0
    velocity_to_mps_scale: float = 0.001
    projected_velocity_scale: float = 0.0
    projected_speed_bonus_scale: float = 0.0
    steering_delta_penalty: float = 0.0
    time_attack_target_s: float | None = None
    time_attack_bonus_scale: float = 0.0
    time_attack_linear_scale: float = 0.0
    pace_profile: ReferencePaceProfile | None = None
    pace_reward_scale: float = 0.0
    pace_debt_clip_s: float = 10.0
    reward_gamma: float = 0.995


@dataclass(frozen=True, slots=True)
class RewardTrajectory:
    points: np.ndarray
    segment_directions: np.ndarray
    segment_lengths: np.ndarray


def build_reward_trajectory(trajectory: np.ndarray) -> RewardTrajectory:
    points = np.asarray(trajectory, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 3:
        raise ValueError("trajectory must have shape (points >= 2, coordinates >= 3)")
    if not np.isfinite(points).all():
        raise ValueError("trajectory points must be finite")
    points = points[:, :3]
    directions = np.diff(points, axis=0)
    lengths = np.linalg.norm(directions, axis=1)
    _validate_segments(directions, lengths)
    return RewardTrajectory(points, directions / lengths[:, None], lengths)


def _validate_segments(directions: np.ndarray, lengths: np.ndarray) -> None:
    if np.any(lengths <= 0.0):
        raise ValueError("trajectory must not contain adjacent duplicate points")
    units = directions / lengths[:, None]
    if len(units) > 1 and np.any(np.linalg.norm(units[:-1] + units[1:], axis=1) <= 1.0e-6):
        raise ValueError("trajectory must not contain a zero-length local tangent")


def validate_reward_config(config: RewardConfig, point_count: int) -> None:
    _validate_finite_limits(config)
    _validate_progress_limits(config)
    _validate_non_negative_limits(config)
    _validate_optional_limits(config)
    _validate_reference_config(config, point_count)


def _validate_finite_limits(config: RewardConfig) -> None:
    limits = (*_finite_progress_limits(config), *_finite_shaping_limits(config))
    limits = (*limits, *_finite_reference_limits(config))
    optional = (config.maximum_race_time_s, config.time_attack_target_s)
    if not all(np.isfinite(value) for value in limits):
        raise ValueError("reward limits must be finite")
    if not all(value is None or np.isfinite(value) for value in optional):
        raise ValueError("reward limits must be finite")


def _finite_progress_limits(config: RewardConfig) -> tuple[float, ...]:
    return (
        config.crash_distance,
        config.finish_progress,
        config.minimum_progress_per_window_m,
        config.terminal_failure_penalty,
        config.collision_penalty,
        config.collision_cooldown_s,
        config.time_penalty_per_second,
        config.max_time_delta_s,
        config.progress_reward_full_lap,
        config.finish_reward,
    )


def _finite_shaping_limits(config: RewardConfig) -> tuple[float, ...]:
    return (
        config.potential_progress_weight,
        config.max_projected_speed_mps,
        config.velocity_to_mps_scale,
        config.projected_velocity_scale,
        config.projected_speed_bonus_scale,
        config.steering_delta_penalty,
    )


def _finite_reference_limits(config: RewardConfig) -> tuple[float, ...]:
    return (
        config.time_attack_bonus_scale,
        config.time_attack_linear_scale,
        config.pace_reward_scale,
        config.pace_debt_clip_s,
        config.reward_gamma,
    )


def _validate_progress_limits(config: RewardConfig) -> None:
    if config.crash_distance <= 0.0:
        raise ValueError("crash_distance must be positive")
    if not 0.0 < config.finish_progress <= 1.0:
        raise ValueError("finish_progress must be in (0, 1]")
    if config.no_progress_steps < 1 or config.slow_progress_window_steps < 2:
        raise ValueError("progress timeout windows must be positive")


def _validate_non_negative_limits(config: RewardConfig) -> None:
    if any(value < 0.0 for value in _non_negative_limits(config)):
        raise ValueError("reward limits must be non-negative")
    if not _positive_reward_scales(config):
        raise ValueError("reward limits must be non-negative")


def _non_negative_limits(config: RewardConfig) -> tuple[float, ...]:
    return (
        config.minimum_progress_per_window_m,
        config.terminal_failure_penalty,
        config.collision_penalty,
        config.collision_cooldown_s,
        config.time_penalty_per_second,
        config.progress_reward_full_lap,
        config.finish_reward,
        config.potential_progress_weight,
        config.projected_velocity_scale,
        config.projected_speed_bonus_scale,
        config.steering_delta_penalty,
        config.time_attack_bonus_scale,
        config.time_attack_linear_scale,
        config.pace_reward_scale,
    )


def _positive_reward_scales(config: RewardConfig) -> bool:
    counts_valid = config.minimum_finish_steps >= 1 and config.nearest_forward_points >= 1
    counts_valid = counts_valid and config.nearest_backward_points >= 0
    scales_valid = config.max_time_delta_s > 0.0 and config.max_projected_speed_mps > 0.0
    scales_valid = scales_valid and config.velocity_to_mps_scale > 0.0
    clips_valid = config.pace_debt_clip_s > 0.0 and 0.0 <= config.reward_gamma <= 1.0
    return all((counts_valid, scales_valid, clips_valid))


def _validate_optional_limits(config: RewardConfig) -> None:
    if config.time_attack_target_s is not None and config.time_attack_target_s <= 0.0:
        raise ValueError("time_attack_target_s must be positive")
    if config.maximum_race_time_s is not None and config.maximum_race_time_s <= 0.0:
        raise ValueError("maximum_race_time_s must be positive")
    has_time_attack_reward = config.time_attack_bonus_scale or config.time_attack_linear_scale
    if has_time_attack_reward and config.time_attack_target_s is None:
        raise ValueError("time-attack reward scales require time_attack_target_s")


def _validate_reference_config(config: RewardConfig, point_count: int) -> None:
    if config.pace_reward_scale and config.pace_profile is None:
        raise ValueError("pace_reward_scale requires a pace_profile")
    if config.pace_profile is None:
        return
    if len(config.pace_profile.reference_times_s) != point_count:
        raise ValueError("pace profile length must match trajectory length")
