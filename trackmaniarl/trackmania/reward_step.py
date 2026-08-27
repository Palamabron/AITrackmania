"""Transition scoring for geometry-based TrackMania rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from trackmaniarl.trackmania.reward_types import (
    AdvanceRequest,
    CollisionRequest,
    RewardResult,
    RewardTerms,
    TerminalReason,
    TerminalRequest,
    TransitionInput,
)

if TYPE_CHECKING:
    from trackmaniarl.trackmania.reward import TrajectoryReward


@dataclass(frozen=True, slots=True)
class _StepInput:
    transition: TransitionInput
    race_time_s: float | None
    time_reward: float
    elapsed_s: float
    point: np.ndarray


@dataclass(frozen=True, slots=True)
class _ProgressState:
    nearest_distance: float
    progress_reward: float
    valid_finish: bool


@dataclass(frozen=True, slots=True)
class _NearestState:
    index: int
    distance: float


@dataclass(frozen=True, slots=True)
class _FinishState:
    progress_reward: float
    progress_m: float
    valid: bool


@dataclass(frozen=True, slots=True)
class _ScoredState:
    terms: RewardTerms
    reason: TerminalReason | None
    race_time_s: float | None


def score_transition(reward: TrajectoryReward, transition: TransitionInput) -> RewardResult:
    step = _validated_step(reward, transition)
    progress = _advance_progress(reward, step)
    terms = _reward_terms(reward, step, progress)
    reason = _terminal_reason(reward, progress, step.race_time_s)
    pace = reward._pace_reward(transition.race_time_ms, reason)
    result = _scored_result(reward, _ScoredState(terms, reason, step.race_time_s))
    collided = reward._apply_collision(
        CollisionRequest(result, transition.collision, step.race_time_s)
    )
    return reward._with_pace(collided, pace)


def _validated_step(reward: TrajectoryReward, transition: TransitionInput) -> _StepInput:
    race_time_s = reward._race_time_s(transition.race_time_ms)
    time_reward, elapsed_s = reward._time_reward(transition.race_time_ms)
    point = reward._vector3("position", transition.position)
    if transition.velocity is not None:
        reward._vector3("velocity", transition.velocity)
    if transition.steering is not None and not np.isfinite(transition.steering):
        raise ValueError("steering must be finite")
    return _StepInput(transition, race_time_s, time_reward, elapsed_s, point)


def _advance_progress(reward: TrajectoryReward, step: _StepInput) -> _ProgressState:
    nearest, nearest_distance = reward._nearest_point(step.point)
    previous_index = reward._index
    _accept_progress(reward, step, _NearestState(nearest, nearest_distance))
    reward._step += 1
    progress_m = float(reward._cumulative_distance[reward._index])
    previous_progress_m = float(reward._cumulative_distance[previous_index])
    reward._nearest_distance_m = nearest_distance
    reward._accepted_progress_delta_m = progress_m - previous_progress_m
    progress_reward = reward._progress_reward(previous_index)
    if reward._index > previous_index:
        reward._last_progress_step = reward._step
    _update_progress_history(reward, progress_m)
    valid_finish = _valid_finish(reward, step.transition, nearest_distance)
    finish = _FinishState(progress_reward, progress_m, valid_finish)
    progress_reward = _complete_finish(reward, finish)
    return _ProgressState(nearest_distance, progress_reward, valid_finish)


def _accept_progress(reward: TrajectoryReward, step: _StepInput, nearest: _NearestState) -> None:
    if nearest.distance > reward.crash_distance:
        reward._previous_position = step.point
        return
    time_budget_s = (
        step.elapsed_s if step.transition.race_time_ms is not None else reward.max_time_delta_s
    )
    request = AdvanceRequest(nearest.index, step.point, time_budget_s)
    reward._index = max(reward._index, reward._bounded_advance(request))


def _update_progress_history(reward: TrajectoryReward, progress_m: float) -> None:
    reward._progress_history.append((reward._step, progress_m))
    while reward._progress_history and (
        reward._step - reward._progress_history[0][0] > reward.slow_progress_window_steps
    ):
        reward._progress_history.popleft()
    reward._window_progress_m = (
        progress_m - reward._progress_history[0][1] if reward._progress_history else 0.0
    )


def _valid_finish(
    reward: TrajectoryReward, transition: TransitionInput, nearest_distance: float
) -> bool:
    near_finish = (
        reward.progress_m / max(1.0, float(reward._cumulative_distance[-1]))
        >= reward.finish_progress
    )
    return bool(
        transition.finish_ui_active
        and near_finish
        and reward._step >= reward.minimum_finish_steps
        and nearest_distance <= reward.crash_distance
    )


def _complete_finish(reward: TrajectoryReward, finish: _FinishState) -> float:
    if not finish.valid or reward._index >= len(reward.points) - 1:
        return finish.progress_reward
    remaining_m = float(reward._cumulative_distance[-1]) - finish.progress_m
    completion = reward.progress_reward_full_lap * remaining_m
    reward._index = len(reward.points) - 1
    return finish.progress_reward + completion / max(1.0, float(reward._cumulative_distance[-1]))


def _reward_terms(
    reward: TrajectoryReward, step: _StepInput, progress: _ProgressState
) -> RewardTerms:
    potential = reward._potential()
    projected_mps, ratio, velocity_reward, speed_reward = _velocity_terms(reward, step)
    steering_reward = reward._steering_delta_reward(step.transition.steering)
    _initialize_potential(reward, potential)
    return RewardTerms(
        step.time_reward,
        potential,
        progress.progress_reward,
        velocity_reward,
        speed_reward,
        steering_reward,
        projected_mps,
        ratio,
    )


def _velocity_terms(
    reward: TrajectoryReward, step: _StepInput
) -> tuple[float, float, float, float]:
    projected_mps = reward._projected_velocity_mps(step.transition.velocity)
    ratio = float(np.clip(projected_mps / reward.max_projected_speed_mps, -1.0, 1.0))
    velocity_reward = reward.projected_velocity_scale * projected_mps * step.elapsed_s
    speed_reward = reward.projected_speed_bonus_scale * max(0.0, ratio) ** 2 * step.elapsed_s
    return projected_mps, ratio, velocity_reward, speed_reward


def _initialize_potential(reward: TrajectoryReward, potential: float) -> None:
    if reward._previous_potential is None:
        reward._previous_potential = potential


def _terminal_reason(
    reward: TrajectoryReward, progress: _ProgressState, race_time_s: float | None
) -> TerminalReason | None:
    if progress.nearest_distance > reward.crash_distance:
        return "off_track"
    if progress.valid_finish:
        return "finished"
    if _time_limit_reached(reward, race_time_s):
        return "time_limit"
    if reward._step - reward._last_progress_step >= reward.no_progress_steps:
        return "no_progress"
    if _slow_progress(reward):
        return "slow_progress"
    return None


def _time_limit_reached(reward: TrajectoryReward, race_time_s: float | None) -> bool:
    return bool(
        reward.maximum_race_time_s is not None
        and race_time_s is not None
        and race_time_s >= reward.maximum_race_time_s
    )


def _slow_progress(reward: TrajectoryReward) -> bool:
    return bool(
        len(reward._progress_history) >= 2
        and reward._step >= reward.slow_progress_window_steps
        and reward._below_progress_threshold()
    )


def _scored_result(reward: TrajectoryReward, state: _ScoredState) -> RewardResult:
    if state.reason is None:
        return _ongoing_result(reward, state.terms)
    terminal_reward = (
        reward.finish_reward
        if state.reason == "finished"
        else -abs(reward.terminal_failure_penalty)
    )
    time_attack = (
        reward._time_attack_terminal_reward(state.race_time_s)
        if state.reason == "finished"
        else 0.0
    )
    return reward._terminal(
        TerminalRequest(state.reason, state.terms, terminal_reward, time_attack)
    )


def _ongoing_result(reward: TrajectoryReward, terms: RewardTerms) -> RewardResult:
    previous_potential = reward._previous_potential
    assert previous_potential is not None
    pbrs_reward = reward.reward_gamma * terms.progress_potential - previous_potential
    reward._previous_potential = terms.progress_potential
    return RewardResult(
        reward=_ongoing_reward(terms, pbrs_reward),
        terminated=False,
        reason=None,
        time_reward=terms.time_reward,
        pbrs_reward=pbrs_reward,
        progress_reward=terms.progress_reward,
        projected_velocity_reward=terms.projected_velocity_reward,
        projected_speed_reward=terms.projected_speed_reward,
        steering_delta_reward=terms.steering_delta_reward,
        potential_progress=terms.progress_potential,
        projected_velocity_mps=terms.projected_velocity_mps,
        projected_velocity_ratio=terms.projected_velocity_ratio,
    )


def _ongoing_reward(terms: RewardTerms, pbrs_reward: float) -> float:
    return (
        terms.time_reward
        + pbrs_reward
        + terms.progress_reward
        + terms.projected_velocity_reward
        + terms.projected_speed_reward
        + terms.steering_delta_reward
    )
