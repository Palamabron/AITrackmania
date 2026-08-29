"""Demonstration conversion into validated replay transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import build_brake_tap_action_table
from trackmaniarl.trackmania.demonstration_data import Demonstration, load_demonstration
from trackmaniarl.trackmania.demonstration_processing import (
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    resample_demonstration,
    validate_demonstration,
)
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry, file_sha256
from trackmaniarl.trackmania.pace import PaceDemonstrationRequest, ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.reward_types import RewardResult, TransitionInput


@dataclass(frozen=True, slots=True)
class DemonstrationTransitionContext:
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry


@dataclass(frozen=True, slots=True)
class _TransitionRequest:
    path: str | Path
    pipeline: FeaturePipeline
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry


@dataclass(frozen=True, slots=True)
class _TransitionContext:
    pipeline: FeaturePipeline
    config: TrackmaniaEnvironmentConfig
    reward: TrajectoryReward
    episode_id: str
    finish_time_s: float
    action_table: list[np.ndarray]
    compact_indices: dict[int, int] | None
    last_step: int


@dataclass(frozen=True, slots=True)
class _TransitionBatch:
    context: _TransitionContext
    frames: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True, slots=True)
class _TransitionStep:
    index: int
    source_action: int
    steering_switch: bool
    steering_switch_distance: int
    next_frame: np.ndarray
    observation: Any


@dataclass(frozen=True, slots=True)
class _ScoredStep:
    step: _TransitionStep
    action: int
    next_observation: Any
    result: RewardResult


@dataclass(frozen=True, slots=True)
class _TransitionResources:
    reward: TrajectoryReward
    action_table: list[np.ndarray]
    action_count: int


def demonstration_transitions(
    path: str | Path,
    pipeline: FeaturePipeline,
    context: DemonstrationTransitionContext,
) -> list[Transition]:
    request = _TransitionRequest(path, pipeline, context.config, context.geometry)
    demonstration = load_demonstration(path)
    validate_demonstration(demonstration, request.config, request.geometry)
    batch = _prepare_transition_batch(request, demonstration)
    transitions, final_reason = _build_transitions(batch)
    if not transitions[-1].terminated or final_reason != "finished":
        raise ValueError("demonstration does not satisfy the configured finish contract")
    return transitions


def _prepare_transition_batch(
    request: _TransitionRequest, demonstration: Demonstration
) -> _TransitionBatch:
    frames, actions = resample_demonstration_for_environment(demonstration, request.config)
    reward = _reward(request.config, request.geometry, len(actions))
    _reset_transition_state(request, reward, frames[0])
    _, table = build_brake_tap_action_table()
    resources = _TransitionResources(reward, table, len(actions))
    context = _transition_context(request, demonstration, resources)
    return _TransitionBatch(context, frames, actions)


def _transition_context(
    request: _TransitionRequest,
    demonstration: Demonstration,
    resources: _TransitionResources,
) -> _TransitionContext:
    return _TransitionContext(
        request.pipeline,
        request.config,
        resources.reward,
        f"demo-{file_sha256(request.path)[:16]}",
        demonstration.finish_time_s,
        resources.action_table,
        _compact_indices(request.config.compact_action_ids),
        resources.action_count - 1,
    )


def resample_demonstration_for_environment(
    demonstration: Demonstration, config: TrackmaniaEnvironmentConfig
) -> tuple[np.ndarray, np.ndarray]:
    return resample_demonstration(
        DemonstrationResamplingRequest(
            demonstration,
            config.decision_interval_ms,
            DemonstrationResamplingConfig(
                config.demonstration_action_lead_ms,
                config.demonstration_control_aggregation,
            ),
        )
    )


def _reward(
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
    demonstration_steps: int,
) -> TrajectoryReward:
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    pace_profile = _pace_profile(config, geometry, reference)
    reward_config = replace(
        config.reward_config(pace_profile),
        no_progress_steps=demonstration_steps + 1,
        slow_progress_window_steps=demonstration_steps + 1,
    )
    return TrajectoryReward(reference, reward_config)


def _pace_profile(
    config: TrackmaniaEnvironmentConfig, geometry: BoundaryGeometry, reference: np.ndarray
) -> ReferencePaceProfile | None:
    if config.pace_reference_path is None:
        return None
    request = PaceDemonstrationRequest(config.pace_reference_path, geometry, reference)
    return ReferencePaceProfile.from_demonstration(request)


def _reset_transition_state(
    request: _TransitionRequest, reward: TrajectoryReward, frame: np.ndarray
) -> None:
    reset_pipeline = getattr(request.pipeline, "reset_episode", None)
    if callable(reset_pipeline):
        reset_pipeline()
    position = frame[list(request.config.position_indices)]
    velocity = frame[list(request.config.velocity_indices)]
    reward.reset(position, velocity=velocity, race_time_ms=float(frame[3]))


def _compact_indices(action_ids: tuple[int, ...] | None) -> dict[int, int] | None:
    if action_ids is None:
        return None
    return {action: index for index, action in enumerate(action_ids)}


def _build_transitions(batch: _TransitionBatch) -> tuple[list[Transition], str | None]:
    prepared = batch.context.pipeline.transform_observation(batch.frames[0])
    transitions: list[Transition] = []
    final_reason: str | None = None
    steering = batch.actions // 6
    switches = np.r_[False, steering[1:] != steering[:-1]]
    distances = _switch_distances(switches)
    pairs = zip(batch.actions, switches, distances, batch.frames[1:], strict=True)
    for index, (action, steering_switch, distance, next_frame) in enumerate(pairs):
        step = _TransitionStep(
            index, int(action), bool(steering_switch), int(distance), next_frame, prepared
        )
        transition, prepared, result = _convert_transition(batch.context, step)
        transitions.append(transition)
        final_reason = result.reason
    return transitions, final_reason


def _switch_distances(switches: np.ndarray) -> np.ndarray:
    count = len(switches)
    indices = np.flatnonzero(switches)
    if not len(indices):
        return np.full(count, count, dtype=np.int64)
    positions = np.arange(count)
    insertion = np.searchsorted(indices, positions)
    left = indices[np.maximum(insertion - 1, 0)]
    right = indices[np.minimum(insertion, len(indices) - 1)]
    distances = np.minimum(np.abs(positions - left), np.abs(right - positions))
    return cast(np.ndarray, distances)


def _convert_transition(
    context: _TransitionContext, step: _TransitionStep
) -> tuple[Transition, Any, RewardResult]:
    action = _compact_action(context, step.source_action)
    result = _score_transition(context, step)
    if result.terminated and step.index != context.last_step:
        raise ValueError(f"demonstration reward terminated early: {result.reason}")
    next_observation = context.pipeline.transform_observation(step.next_frame)
    scored = _ScoredStep(step, action, next_observation, result)
    return _transition_record(context, scored), next_observation, result


def _compact_action(context: _TransitionContext, source_action: int) -> int:
    indices = context.compact_indices
    if indices is None:
        return source_action
    if source_action not in indices:
        raise ValueError(f"demonstration action {source_action} is outside compact action IDs")
    return indices[source_action]


def _score_transition(context: _TransitionContext, step: _TransitionStep) -> RewardResult:
    config = context.config
    frame = step.next_frame
    control = context.action_table[step.source_action]
    transition = TransitionInput(
        frame[list(config.position_indices)],
        bool(frame[2]),
        frame[list(config.velocity_indices)],
        float(frame[3]),
        False,
        float(control[2]),
    )
    return context.reward.step(transition)


def _transition_record(context: _TransitionContext, scored: _ScoredStep) -> Transition:
    return Transition(
        observation=scored.step.observation,
        action=scored.action,
        reward=scored.result.reward,
        next_observation=scored.next_observation,
        terminated=scored.result.terminated,
        truncated=False,
        info=_transition_info(context, scored),
        episode_id=context.episode_id,
        step=scored.step.index,
    )


def _transition_info(context: _TransitionContext, scored: _ScoredStep) -> dict[str, object]:
    return {
        "source": "demo",
        "is_demo": True,
        "demonstration_steering_switch": scored.step.steering_switch,
        "demonstration_steering_switch_distance": scored.step.steering_switch_distance,
        "sampling/projected_lap_time_s": context.finish_time_s,
    }
