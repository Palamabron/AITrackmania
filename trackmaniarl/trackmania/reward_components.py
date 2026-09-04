"""Reusable geometry, pace, and terminal reward calculations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np

from trackmaniarl.trackmania.reward_types import (
    AdvanceRequest,
    CollisionRequest,
    PaceValues,
    RewardResult,
    TerminalReason,
    TerminalRequest,
)

if TYPE_CHECKING:
    from trackmaniarl.trackmania.reward import TrajectoryReward


def apply_collision(reward: TrajectoryReward, request: CollisionRequest) -> RewardResult:
    if not request.collision:
        return request.result
    if (
        request.race_time_s is not None
        and reward._last_penalized_collision_s is not None
        and request.race_time_s - reward._last_penalized_collision_s < reward.collision_cooldown_s
    ):
        return replace(request.result, collision_detected=True)
    reward._last_penalized_collision_s = request.race_time_s
    collision_reward = -reward.collision_penalty
    return replace(
        request.result,
        reward=request.result.reward + collision_reward,
        collision_reward=collision_reward,
        collided=True,
        collision_detected=True,
    )


def nearest_point(reward: TrajectoryReward, point: np.ndarray) -> tuple[int, float]:
    window_start = max(0, reward._index - reward.nearest_backward_points)
    window_stop = min(len(reward.points), reward._index + reward.nearest_forward_points + 1)
    distances = np.linalg.norm(reward.points[window_start:window_stop] - point, axis=1)
    nearest = window_start + int(np.argmin(distances))
    return nearest, float(distances[nearest - window_start])


def bounded_advance(reward: TrajectoryReward, request: AdvanceRequest) -> int:
    previous = reward._previous_position
    reward._previous_position = request.point
    if previous is None:
        return reward._index
    displacement_m = float(np.linalg.norm(request.point - previous))
    accepted_motion_m = min(displacement_m, reward.max_projected_speed_mps * request.time_budget_s)
    reward._reachable_progress_m += accepted_motion_m
    reachable = (
        int(
            np.searchsorted(reward._cumulative_distance, reward._reachable_progress_m, side="right")
        )
        - 1
    )
    return min(request.nearest, max(reachable, reward._index))


def potential(reward: TrajectoryReward) -> float:
    return (
        reward.potential_progress_weight
        * reward.progress_m
        / max(1.0, float(reward._cumulative_distance[-1]))
    )


def progress_reward(reward: TrajectoryReward, previous_index: int) -> float:
    progress_delta = (
        reward._cumulative_distance[reward._index] - reward._cumulative_distance[previous_index]
    )
    return float(
        reward.progress_reward_full_lap
        * progress_delta
        / max(1.0, float(reward._cumulative_distance[-1]))
    )


def projected_velocity_mps(reward: TrajectoryReward, velocity: np.ndarray | None) -> float:
    if velocity is None:
        return 0.0
    tangent = reward._path_tangent()
    velocity_mps = reward._vector3("velocity", velocity) * reward.velocity_to_mps_scale
    projected = float(np.dot(velocity_mps, tangent))
    return float(
        np.clip(projected, -reward.max_projected_speed_mps, reward.max_projected_speed_mps)
    )


def steering_delta_reward(reward: TrajectoryReward, steering: float | None) -> float:
    if steering is None:
        return 0.0
    current = float(np.clip(steering, -1.0, 1.0))
    delta = abs(current - reward._previous_steering)
    reward._previous_steering = current
    return -reward.steering_delta_penalty * delta


def time_attack_terminal_reward(reward: TrajectoryReward, race_time_s: float | None) -> float:
    if reward.time_attack_target_s is None or race_time_s is None:
        return 0.0
    improvement_s = reward.time_attack_target_s - race_time_s
    raw_reward = (
        reward.time_attack_bonus_scale * max(0.0, improvement_s) ** 2
        + reward.time_attack_linear_scale * improvement_s
    )
    return max(-reward.finish_reward, raw_reward)


def below_progress_threshold(reward: TrajectoryReward) -> bool:
    threshold = reward.minimum_progress_per_window_m
    return reward._window_progress_m < threshold and not np.isclose(
        reward._window_progress_m,
        threshold,
        rtol=1.0e-3,
        atol=1.0e-3,
    )


def time_debt(
    reward: TrajectoryReward,
    race_time_ms: float | None,
    terminal_reason: TerminalReason | None = None,
) -> tuple[float, float]:
    if reward.pace_profile is None or race_time_ms is None:
        return 0.0, 0.0
    reference_time_s = (
        float(reward.pace_profile.reference_times_s[-1])
        if terminal_reason == "finished"
        else reward.pace_profile.time_at_index(reward._index)
    )
    race_time_s = reward._race_time_s(race_time_ms)
    assert race_time_s is not None
    debt = float(
        np.clip(race_time_s - reference_time_s, -reward.pace_debt_clip_s, reward.pace_debt_clip_s)
    )
    return reference_time_s, debt


def pace_reward(
    reward: TrajectoryReward,
    race_time_ms: float | None,
    terminal_reason: TerminalReason | None = None,
) -> PaceValues:
    reference_time_s, time_debt_s = reward._time_debt(race_time_ms, terminal_reason)
    previous = reward._previous_time_debt_s
    reward._previous_time_debt_s = time_debt_s
    if reward.pace_profile is None or previous is None:
        return PaceValues(0.0, reference_time_s, time_debt_s)
    previous_potential = -reward.pace_reward_scale * previous
    current_potential = -reward.pace_reward_scale * time_debt_s
    shaping = (
        -previous_potential
        if terminal_reason is not None
        else reward.reward_gamma * current_potential - previous_potential
    )
    return PaceValues(shaping, reference_time_s, time_debt_s)


def with_pace(reward: TrajectoryReward, result: RewardResult, pace: PaceValues) -> RewardResult:
    return replace(
        result,
        reward=result.reward + pace.reward,
        pace_reward=pace.reward,
        reference_time_s=pace.reference_time_s,
        time_debt_s=pace.time_debt_s,
        nearest_distance_m=reward._nearest_distance_m,
        accepted_progress_delta_m=reward._accepted_progress_delta_m,
        window_progress_m=reward._window_progress_m,
        steps_since_progress=reward._step - reward._last_progress_step,
    )


def path_tangent(reward: TrajectoryReward) -> np.ndarray:
    if reward._index == 0:
        direction = reward._segment_directions[0]
    elif reward._index == len(reward.points) - 1:
        direction = reward._segment_directions[-1]
    else:
        direction = (
            reward._segment_directions[reward._index - 1]
            + reward._segment_directions[reward._index]
        )
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("trajectory must not contain a zero-length local tangent")
    return cast(np.ndarray, direction / norm)


def time_reward(reward: TrajectoryReward, race_time_ms: float | None) -> tuple[float, float]:
    race_time_s = reward._race_time_s(race_time_ms)
    if race_time_s is None or reward._previous_race_time_s is None:
        reward._previous_race_time_s = race_time_s
        return 0.0, 0.0
    elapsed_s = race_time_s - reward._previous_race_time_s
    if elapsed_s < 0.0:
        raise ValueError("race time must be monotonic within an episode")
    reward._previous_race_time_s = race_time_s
    bounded_elapsed_s = min(elapsed_s, reward.max_time_delta_s)
    return -reward.time_penalty_per_second * bounded_elapsed_s, bounded_elapsed_s


def race_time_s(race_time_ms: float | None) -> float | None:
    if race_time_ms is None:
        return None
    value = float(race_time_ms)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("race time must be finite and non-negative")
    return value / 1_000.0


def vector3(name: str, value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    return vector


def terminal(reward: TrajectoryReward, request: TerminalRequest) -> RewardResult:
    pbrs_reward = -float(reward._previous_potential or 0.0)
    reward._previous_potential = 0.0
    total = _terminal_total(request, pbrs_reward)
    return _terminal_result(request, pbrs_reward, total)


def _terminal_total(request: TerminalRequest, pbrs_reward: float) -> float:
    terms = request.terms
    return (
        terms.time_reward
        + pbrs_reward
        + terms.progress_reward
        + terms.projected_velocity_reward
        + terms.projected_speed_reward
        + terms.steering_delta_reward
        + request.terminal_reward
        + request.time_attack_terminal_reward
    )


def _terminal_result(request: TerminalRequest, pbrs_reward: float, total: float) -> RewardResult:
    terms = request.terms
    return RewardResult(
        reward=total,
        terminated=True,
        reason=request.reason,
        time_reward=terms.time_reward,
        pbrs_reward=pbrs_reward,
        progress_reward=terms.progress_reward,
        projected_velocity_reward=terms.projected_velocity_reward,
        projected_speed_reward=terms.projected_speed_reward,
        steering_delta_reward=terms.steering_delta_reward,
        time_attack_terminal_reward=request.time_attack_terminal_reward,
        terminal_reward=request.terminal_reward,
        potential_progress=terms.progress_potential,
        projected_velocity_mps=terms.projected_velocity_mps,
        projected_velocity_ratio=terms.projected_velocity_ratio,
    )
