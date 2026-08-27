"""One-step execution for the OpenPlanet TrackMania environment."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL
from trackmaniarl.trackmania.reward_types import RewardResult, TransitionInput
from trackmaniarl.trackmania.telemetry import TelemetryFrame

if TYPE_CHECKING:
    from trackmaniarl.trackmania.environment import OpenPlanetEnvironment


@dataclass(frozen=True, slots=True)
class _AppliedAction:
    control: np.ndarray
    duration_ms: float


@dataclass(frozen=True, slots=True)
class _StepTelemetry:
    frame: TelemetryFrame
    skipped_frames: int
    wait_ms: float
    decision_interval_ms: float | None


@dataclass(frozen=True, slots=True)
class _RewardedStep:
    position: np.ndarray
    race_time_ms: float
    step_race_time_ms: float
    brake_tap: bool
    applied_brake: float
    result: RewardResult


@dataclass(frozen=True, slots=True)
class _StepReport:
    applied: _AppliedAction
    telemetry: _StepTelemetry
    outcome: _RewardedStep


def step(
    environment: OpenPlanetEnvironment, action: Any, clock: Callable[[], float]
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    applied = _apply_action(environment, action, clock)
    telemetry = _read_step_telemetry(environment, clock)
    outcome = _score_step(environment, applied.control, telemetry.frame)
    if outcome.result.reason == "finished":
        environment._finish_confirmation_pending = True
    return (
        telemetry.frame.values,
        outcome.result.reward,
        outcome.result.terminated,
        False,
        _step_info(environment, _StepReport(applied, telemetry, outcome)),
    )


def _apply_action(
    environment: OpenPlanetEnvironment, action: Any, clock: Callable[[], float]
) -> _AppliedAction:
    if isinstance(action, (int, np.integer)):
        return _apply_discrete_action(environment, int(action), clock)
    return _apply_continuous_action(environment, action, clock)


def _apply_discrete_action(
    environment: OpenPlanetEnvironment, index: int, clock: Callable[[], float]
) -> _AppliedAction:
    if not 0 <= index < environment._action_count:
        raise ValueError(f"TrackMania discrete action must be in [0, {environment._action_count})")
    control = environment._action_table[index]
    started = clock()
    environment.controller.apply_discrete(control)
    return _AppliedAction(control, (clock() - started) * 1_000.0)


def _apply_continuous_action(
    environment: OpenPlanetEnvironment, action: Any, clock: Callable[[], float]
) -> _AppliedAction:
    control = np.asarray(action, dtype=np.float32).reshape(-1)
    if control.shape != (3,):
        raise ValueError("TrackMania action must be index or [gas, brake, steer]")
    started = clock()
    environment.controller.apply(control)
    return _AppliedAction(control, (clock() - started) * 1_000.0)


def _read_step_telemetry(
    environment: OpenPlanetEnvironment, clock: Callable[[], float]
) -> _StepTelemetry:
    started = clock()
    frame, skipped = _repeated_frame(environment)
    frame, skipped = _reach_decision_interval(environment, frame, skipped)
    return _StepTelemetry(
        frame,
        skipped,
        (clock() - started) * 1_000.0,
        environment.config.decision_interval_ms,
    )


def _repeated_frame(environment: OpenPlanetEnvironment) -> tuple[TelemetryFrame, int]:
    frame = None
    skipped = 0
    for _ in range(environment.config.action_repeat_frames):
        frame = environment.client.read()
        skipped += frame.skipped_frames
    assert frame is not None
    return frame, skipped


def _reach_decision_interval(
    environment: OpenPlanetEnvironment, frame: TelemetryFrame, skipped: int
) -> tuple[TelemetryFrame, int]:
    interval = environment.config.decision_interval_ms
    if interval is None:
        return frame, skipped
    target_race_time_ms = _last_race_time_ms(environment) + interval
    for _ in range(64):
        if float(frame.values[3]) >= target_race_time_ms or bool(frame.values[2]):
            return frame, skipped
        frame = environment.client.read()
        skipped += frame.skipped_frames
    raise TimeoutError("TrackMania telemetry did not reach the configured decision interval")


def _score_step(
    environment: OpenPlanetEnvironment, control: np.ndarray, frame: TelemetryFrame
) -> _RewardedStep:
    position = frame.values[list(environment.config.position_indices)]
    race_time_ms = float(frame.values[3])
    step_race_time_ms = max(0.0, race_time_ms - _last_race_time_ms(environment))
    environment._last_race_time_ms = race_time_ms
    brake_tap = float(control[1]) == BRAKE_TAP_SENTINEL
    applied_brake = _applied_brake(control, step_race_time_ms)
    transition = _transition_input(environment, control, frame)
    result = environment.reward.step(transition)
    return _RewardedStep(
        position, race_time_ms, step_race_time_ms, brake_tap, applied_brake, result
    )


def _last_race_time_ms(environment: OpenPlanetEnvironment) -> float:
    race_time_ms = environment._last_race_time_ms
    if race_time_ms is None:
        raise RuntimeError("TrackMania environment must be reset before stepping")
    return race_time_ms


def _transition_input(
    environment: OpenPlanetEnvironment,
    control: np.ndarray,
    frame: TelemetryFrame,
) -> TransitionInput:
    return TransitionInput(
        frame.values[list(environment.config.position_indices)],
        bool(frame.values[2]),
        frame.values[list(environment.config.velocity_indices)],
        float(frame.values[3]),
        environment.controller.consume_collision(),
        float(control[2]),
    )


def _applied_brake(control: np.ndarray, step_race_time_ms: float) -> float:
    if float(control[1]) != BRAKE_TAP_SENTINEL:
        return float(np.clip(control[1], 0.0, 1.0))
    return min(1.0, BRAKE_TAP_DURATION_S * 1_000.0 / max(step_race_time_ms, 1.0))


def _step_info(environment: OpenPlanetEnvironment, report: _StepReport) -> dict[str, Any]:
    return {
        **_timing_info(environment, report),
        **_control_info(report.applied.control, report.outcome),
        **_reward_info(environment, report.outcome),
    }


def _timing_info(environment: OpenPlanetEnvironment, report: _StepReport) -> dict[str, Any]:
    applied, telemetry, outcome = report.applied, report.telemetry, report.outcome
    interval = telemetry.decision_interval_ms
    return {
        "termination_reason": outcome.result.reason,
        "telemetry_health": "ok",
        "position": outcome.position.tolist(),
        "race_time_ms": outcome.race_time_ms,
        "step_race_time_ms": outcome.step_race_time_ms,
        "decision_interval_ms": float(interval or 0.0),
        "decision_interval_error_ms": outcome.step_race_time_ms - interval if interval else 0.0,
        "controller_apply_ms": applied.duration_ms,
        "telemetry_wait_ms": telemetry.wait_ms,
        "telemetry_skipped_frames": telemetry.skipped_frames,
        "episode_elapsed_s": monotonic() - environment._episode_started_at,
    }


def _control_info(control: np.ndarray, outcome: _RewardedStep) -> dict[str, float]:
    return {
        "control_gas": float(control[0]),
        "control_brake": outcome.applied_brake,
        "control_brake_tap": float(outcome.brake_tap),
        "control_steer": float(control[2]),
    }


def _reward_info(environment: OpenPlanetEnvironment, outcome: _RewardedStep) -> dict[str, Any]:
    return {
        "progress_m": environment.reward.progress_m,
        "progress_pct": environment.reward.progress_pct,
        **_reward_component_info(outcome.result),
        **_reward_progress_info(outcome.result),
    }


def _reward_component_info(result: RewardResult) -> dict[str, Any]:
    return {
        "reward_time": result.time_reward,
        "reward_pbrs": result.pbrs_reward,
        "reward_progress": result.progress_reward,
        "reward_projected_velocity": result.projected_velocity_reward,
        "reward_projected_speed": result.projected_speed_reward,
        "reward_steering_delta": result.steering_delta_reward,
        "reward_time_attack_terminal": result.time_attack_terminal_reward,
        "reward_pace": result.pace_reward,
        "reward_collision": result.collision_reward,
        "collision": result.collided,
        "collision_detected": result.collision_detected,
        "reward_terminal": result.terminal_reward,
    }


def _reward_progress_info(result: RewardResult) -> dict[str, Any]:
    return {
        "potential_progress": result.potential_progress,
        "reference_time_s": result.reference_time_s,
        "time_debt_s": result.time_debt_s,
        "nearest_distance_m": result.nearest_distance_m,
        "accepted_progress_delta_m": result.accepted_progress_delta_m,
        "window_progress_m": result.window_progress_m,
        "steps_since_progress": result.steps_since_progress,
        "projected_velocity_mps": result.projected_velocity_mps,
        "projected_velocity_ratio": result.projected_velocity_ratio,
    }
