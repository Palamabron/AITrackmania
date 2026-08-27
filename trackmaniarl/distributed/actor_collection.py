"""Rollout collection and deterministic evaluation for distributed actors."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Any, cast

import numpy as np
import torch

from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import tree_map
from trackmaniarl.distributed.actor_errors import ActorEnvironmentError
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics, summarize_episode
from trackmaniarl.distributed.actor_protocols import CollectionRuntime
from trackmaniarl.distributed.actor_requests import EnvironmentReset, SpoolRequest

logger = logging.getLogger(__name__)

_TELEMETRY_RESET_ATTEMPTS = 5
_TELEMETRY_RETRY_INITIAL_S = 5.0
_TELEMETRY_RETRY_MAX_S = 60.0


@dataclass(slots=True)
class CollectionBuffers:
    transitions: list[Transition] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    started: float = field(default_factory=monotonic)

    def flush(self, runtime: CollectionRuntime, version: int) -> None:
        runtime._spool(SpoolRequest(self.transitions, self.summaries, version))
        self.transitions = []
        self.summaries = []
        self.started = monotonic()


@dataclass(slots=True)
class CollectionContext:
    runtime: CollectionRuntime
    environment: Any
    pipeline: Any
    buffers: CollectionBuffers = field(default_factory=CollectionBuffers)


@dataclass(slots=True)
class TrainingEpisode:
    prepared: Any
    policy: ReplicablePolicy
    epsilon: float
    version: int
    episode_id: str
    metrics: EpisodeMetrics
    steps: int = 0


@dataclass(slots=True)
class PolicyStep:
    index: int
    action: Any
    policy_info: Mapping[str, Any]
    next_observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    inference_s: float


@dataclass(frozen=True, slots=True)
class ResetRetry:
    exception: TimeoutError | ConnectionError
    attempt: int
    attempts: int
    delay: float


def collect(runtime: CollectionRuntime, environment: Any, pipeline: Any) -> None:
    context = CollectionContext(runtime, environment, pipeline)
    episode = 0
    while not runtime.stop.is_set():
        if runtime.evaluate.is_set():
            runtime.evaluate.clear()
            runtime._evaluate(environment, pipeline)
        state = _start_training_episode(context, episode)
        if state is None:
            break
        try:
            _run_training_episode(context, state)
        except (TimeoutError, ConnectionError) as exc:
            _handle_training_interruption(context, exc)
            episode += 1
            continue
        _finish_training_episode(context, state)
        episode += 1


def _start_training_episode(context: CollectionContext, episode: int) -> TrainingEpisode | None:
    runtime = context.runtime
    observation = runtime._reset_environment(EnvironmentReset(context.environment, episode))
    if observation is None:
        return None
    reset_pipeline(context.pipeline)
    prepared = context.pipeline.transform_observation(observation)
    policy, epsilon, version = runtime._policy()
    reset_policy(policy)
    episode_id = f"{runtime.actor_id}/{runtime.session_id}/{episode:08d}"
    return TrainingEpisode(
        prepared,
        policy,
        epsilon,
        version,
        episode_id,
        EpisodeMetrics.from_policy(policy),
    )


def _run_training_episode(context: CollectionContext, state: TrainingEpisode) -> None:
    for step_index in range(context.runtime.spec.training.max_episode_steps):
        step = _take_training_step(context, state, step_index)
        _record_training_step(context, state, step)
        if step.terminated or step.truncated or context.runtime.stop.is_set():
            break


def _take_training_step(
    context: CollectionContext, state: TrainingEpisode, step_index: int
) -> PolicyStep:
    started = monotonic()
    action, policy_info = _sample_policy(state.policy, state.prepared)
    inference_s = monotonic() - started
    next_observation, reward, terminated, truncated, info = context.environment.step(action)
    final_step = step_index == context.runtime.spec.training.max_episode_steps - 1
    return PolicyStep(
        step_index,
        action,
        policy_info,
        next_observation,
        float(reward),
        bool(terminated),
        bool(truncated or (final_step and not terminated)),
        cast(Mapping[str, Any], info),
        inference_s,
    )


def _sample_policy(policy: ReplicablePolicy, observation: Any) -> tuple[Any, Mapping[str, Any]]:
    sample = getattr(policy, "act_with_info", None)
    if not callable(sample):
        return policy.act(observation), {}
    action, policy_info = sample(observation)
    return action, cast(Mapping[str, Any], policy_info)


def _record_training_step(
    context: CollectionContext, state: TrainingEpisode, step: PolicyStep
) -> None:
    metrics = state.metrics
    metrics.record_inference(step.inference_s)
    metrics.record_policy(state.policy, step.index)
    metrics.record_diagnostics(step.action, state.policy, step.info)
    next_prepared = context.pipeline.transform_observation(step.next_observation)
    context.buffers.transitions.append(_transition(state, step, next_prepared))
    state.prepared = next_prepared
    state.steps = step.index + 1
    metrics.record_reward(step.reward, step.info)
    if context.runtime._should_flush(context.buffers.transitions, context.buffers.started):
        context.buffers.flush(context.runtime, state.version)


def _transition(
    state: TrainingEpisode,
    step: PolicyStep,
    next_prepared: Any,
) -> Transition:
    return Transition(
        observation=snapshot_observation(state.prepared),
        action=step.action,
        reward=step.reward,
        next_observation=snapshot_observation(next_prepared),
        terminated=step.terminated,
        truncated=step.truncated,
        info=_transition_info(state, step),
        episode_id=state.episode_id,
        step=step.index,
    )


def _transition_info(state: TrainingEpisode, step: PolicyStep) -> dict[str, Any]:
    return {
        **dict(step.info),
        "policy_version": state.version,
        "actor_epsilon": state.epsilon,
        **step.policy_info,
    }


def _handle_training_interruption(
    context: CollectionContext, exc: TimeoutError | ConnectionError
) -> None:
    runtime = context.runtime
    logger.warning(
        "Actor %s telemetry stalled mid-episode (%s: %s); closing the available "
        "rollout as a bootstrappable truncation",
        runtime.actor_id,
        type(exc).__name__,
        exc,
    )
    _truncate_interrupted_rollout(context.buffers)
    _, _, version = runtime._policy()
    context.buffers.flush(runtime, version)


def _truncate_interrupted_rollout(buffers: CollectionBuffers) -> None:
    if not buffers.transitions:
        return
    last = buffers.transitions[-1]
    buffers.transitions[-1] = replace(
        last,
        terminated=False,
        truncated=True,
        info={
            **dict(last.info),
            "termination_reason": "telemetry_interruption",
            "telemetry_health": "interrupted",
        },
    )


def _finish_training_episode(context: CollectionContext, state: TrainingEpisode) -> None:
    info = state.metrics.summary_info(state.epsilon, state.version, state.steps)
    summary = context.runtime._summary(state.metrics.total_reward, info, state.steps)
    context.buffers.summaries.append(summary)
    _, _, version = context.runtime._policy()
    context.buffers.flush(context.runtime, version)


def reset_environment(runtime: CollectionRuntime, request: EnvironmentReset) -> Any:
    attempts = request.attempts
    if attempts < 1:
        raise ValueError("telemetry reset attempts must be positive")
    delay = _TELEMETRY_RETRY_INITIAL_S
    for attempt in range(attempts):
        try:
            observation, _ = request.environment.reset(seed=runtime._actor_seed() + request.episode)
            return observation
        except (TimeoutError, ConnectionError) as exc:
            if attempt == attempts - 1:
                return _handle_reset_exhaustion(runtime, exc, request)
            _log_reset_retry(runtime, ResetRetry(exc, attempt, attempts, delay))
            if runtime.stop.wait(delay):
                return None
            delay = min(delay * 2.0, _TELEMETRY_RETRY_MAX_S)
    raise AssertionError("unreachable")


def _handle_reset_exhaustion(
    runtime: CollectionRuntime,
    exc: TimeoutError | ConnectionError,
    request: EnvironmentReset,
) -> None:
    reason = (
        f"telemetry unavailable after {request.attempts} reset attempts: "
        f"{type(exc).__name__}: {exc}"
    )
    if not request.stop_on_failure:
        logger.warning("Actor %s evaluation %s", runtime.actor_id, reason)
        return None
    runtime.stop_reason = reason
    runtime.stop.set()
    raise ActorEnvironmentError(reason) from exc


def _log_reset_retry(
    runtime: CollectionRuntime,
    retry: ResetRetry,
) -> None:
    logger.warning(
        "Actor %s environment reset failed (%s: %s); retry %d/%d in %.0fs",
        runtime.actor_id,
        type(retry.exception).__name__,
        retry.exception,
        retry.attempt + 1,
        retry.attempts - 1,
        retry.delay,
    )


def reset_pipeline(pipeline: Any) -> None:
    reset = getattr(pipeline, "reset_episode", None)
    if callable(reset):
        reset()


def reset_policy(policy: Any) -> None:
    reset = getattr(policy, "reset_episode", None)
    if callable(reset):
        reset()


def snapshot_observation(observation: Any) -> Any:
    return tree_map(_copy_observation_leaf, observation)


def _copy_observation_leaf(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def should_flush(runtime: CollectionRuntime, transitions: list[Transition], started: float) -> bool:
    return len(transitions) >= runtime.spec.distributed.rollout_chunk_transitions or (
        bool(transitions) and monotonic() - started >= runtime.spec.distributed.rollout_flush_s
    )


def summary(reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
    return summarize_episode(reward, info, transitions)
