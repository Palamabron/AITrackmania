"""Deterministic evaluation episodes run by distributed actors."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic
from typing import Any, cast

from trackmaniarl.core.contracts import ExploratoryPolicy, PolicyMode, ReplicablePolicy
from trackmaniarl.distributed.actor_collection import reset_pipeline, reset_policy
from trackmaniarl.distributed.actor_metrics import EpisodeMetrics
from trackmaniarl.distributed.actor_protocols import CollectionRuntime
from trackmaniarl.distributed.actor_requests import (
    EnvironmentReset,
    EvaluationEpisodeRequest,
    SpoolRequest,
    TelemetryFailure,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvaluationPlan:
    runtime: CollectionRuntime
    environment: Any
    pipeline: Any
    policy: ReplicablePolicy
    version: int


def evaluate(runtime: CollectionRuntime, environment: Any, pipeline: Any) -> None:
    suite = runtime.spec.evaluation
    trials = 1 if suite is None else suite.trials_per_map
    policy, version = runtime._evaluation_policy()
    plan = EvaluationPlan(runtime, environment, pipeline, policy, version)
    summaries = _evaluation_trials(plan, trials)
    snapshot = runtime.codec.encode(dict(policy.export_state()))
    runtime._spool(SpoolRequest([], [], version, summaries, snapshot))


def _evaluation_trials(plan: EvaluationPlan, trials: int) -> list[dict[str, Any]]:
    if not _prewarm_evaluation_policy(plan):
        return _evaluation_failures(plan, trials)
    summaries: list[dict[str, Any]] = []
    for trial in range(trials):
        request = EvaluationEpisodeRequest(
            plan.environment, plan.pipeline, plan.policy, plan.version
        )
        summary = plan.runtime._evaluate_episode(request)
        summaries.append(summary)
        if summary["termination"] == "telemetry_error":
            summaries.extend(_evaluation_failures(plan, trials - trial - 1))
            break
    return summaries


def _evaluation_failures(plan: EvaluationPlan, count: int) -> list[dict[str, Any]]:
    failure = TelemetryFailure(plan.version)
    return [plan.runtime._evaluation_telemetry_failure(failure) for _ in range(count)]


def _prewarm_evaluation_policy(plan: EvaluationPlan) -> bool:
    reset = EnvironmentReset(
        plan.environment,
        1_000_000 + plan.runtime._evaluation_index,
        attempts=1,
        stop_on_failure=False,
    )
    observation = plan.runtime._reset_environment(reset)
    if observation is None:
        return False
    reset_pipeline(plan.pipeline)
    reset_policy(plan.policy)
    try:
        prepared = plan.pipeline.transform_observation(observation)
        plan.policy.act(prepared, PolicyMode.EVALUATION)
    finally:
        reset_pipeline(plan.pipeline)
        reset_policy(plan.policy)
    return True


def evaluation_policy(runtime: CollectionRuntime) -> tuple[ReplicablePolicy, int]:
    with runtime._evaluation_request_lock:
        request = runtime._evaluation_request
        runtime._evaluation_request = None
    if request is None:
        policy, _, version = runtime._policy()
        return policy, version
    return _policy_from_request(runtime, request)


def _policy_from_request(
    runtime: CollectionRuntime, request: tuple[bytes, int]
) -> tuple[ReplicablePolicy, int]:
    snapshot, version = request
    state = runtime.codec.decode(snapshot)
    if not isinstance(state, Mapping):
        raise ValueError("evaluation policy snapshot must decode to a mapping")
    policy = runtime._new_policy()
    policy.load_state(state)
    if isinstance(policy, ExploratoryPolicy):
        policy.set_exploration_epsilon(0.0)
    return policy, version


@dataclass(slots=True)
class EvaluationContext:
    runtime: CollectionRuntime
    environment: Any
    pipeline: Any
    policy: Any
    version: int
    prepared: Any
    metrics: EpisodeMetrics


@dataclass(slots=True)
class EvaluationStep:
    index: int
    action: Any
    inference_s: float
    observation: Any
    reward: float
    info: Mapping[str, Any]
    stopped: bool


@dataclass(frozen=True, slots=True)
class EvaluatedAction:
    value: Any
    inference_s: float


def evaluate_episode(
    runtime: CollectionRuntime, request: EvaluationEpisodeRequest
) -> dict[str, Any]:
    plan = EvaluationPlan(
        runtime,
        request.environment,
        request.pipeline,
        request.policy,
        request.version,
    )
    return _evaluate_episode(plan)


def _evaluate_episode(plan: EvaluationPlan) -> dict[str, Any]:
    runtime = plan.runtime
    reset = EnvironmentReset(
        plan.environment,
        1_000_000 + runtime._evaluation_index,
        attempts=1,
        stop_on_failure=False,
    )
    observation = runtime._reset_environment(reset)
    if observation is None:
        return runtime._evaluation_telemetry_failure(TelemetryFailure(plan.version))
    context = _evaluation_context(plan, observation)
    failure = _run_evaluation_episode(context)
    if failure is not None:
        return failure
    return _evaluation_summary(context)


def _evaluation_context(plan: EvaluationPlan, observation: Any) -> EvaluationContext:
    reset_pipeline(plan.pipeline)
    reset_policy(plan.policy)
    return EvaluationContext(
        plan.runtime,
        plan.environment,
        plan.pipeline,
        plan.policy,
        plan.version,
        plan.pipeline.transform_observation(observation),
        EpisodeMetrics.from_policy(plan.policy),
    )


def _evaluation_summary(context: EvaluationContext) -> dict[str, Any]:
    runtime = context.runtime
    transitions = context.metrics.controls.samples
    summary = runtime._summary(
        context.metrics.total_reward,
        context.metrics.summary_info(0.0, context.version, transitions),
        transitions,
    )
    summary["deterministic"] = 1.0
    runtime._evaluation_index += 1
    return summary


def _run_evaluation_episode(context: EvaluationContext) -> dict[str, Any] | None:
    for step_index in range(context.runtime.spec.training.max_episode_steps):
        failure, stopped = _take_evaluation_step(context, step_index)
        if failure is not None:
            return failure
        if stopped:
            break
    return None


def _take_evaluation_step(
    context: EvaluationContext, step_index: int
) -> tuple[dict[str, Any] | None, bool]:
    evaluated = _evaluation_action(context)
    try:
        step = _evaluation_step(context, step_index, evaluated)
    except (TimeoutError, ConnectionError) as exc:
        return _evaluation_interruption(context, exc), True
    _record_evaluation_step(context, step)
    return None, step.stopped


def _evaluation_action(context: EvaluationContext) -> EvaluatedAction:
    started = monotonic()
    action = context.policy.act(context.prepared, PolicyMode.EVALUATION)
    return EvaluatedAction(action, monotonic() - started)


def _evaluation_step(
    context: EvaluationContext, step_index: int, evaluated: EvaluatedAction
) -> EvaluationStep:
    observation, reward, terminated, truncated, info = context.environment.step(evaluated.value)
    stopped = bool(terminated or truncated or context.runtime.stop.is_set())
    return EvaluationStep(
        step_index,
        evaluated.value,
        evaluated.inference_s,
        observation,
        float(reward),
        cast(Mapping[str, Any], info),
        stopped,
    )


def _record_evaluation_step(context: EvaluationContext, step: EvaluationStep) -> None:
    metrics = context.metrics
    metrics.record_inference(step.inference_s)
    metrics.record_policy(context.policy, step.index)
    metrics.record_diagnostics(step.action, context.policy, step.info)
    context.prepared = context.pipeline.transform_observation(step.observation)
    metrics.record_reward(step.reward, step.info)


def _evaluation_interruption(
    context: EvaluationContext, exc: TimeoutError | ConnectionError
) -> dict[str, Any]:
    runtime = context.runtime
    logger.warning(
        "Actor %s deterministic evaluation telemetry failed (%s: %s)",
        runtime.actor_id,
        type(exc).__name__,
        exc,
    )
    transitions = context.metrics.controls.samples
    failure = TelemetryFailure(
        context.version,
        transitions,
        context.metrics.total_reward,
        context.metrics.summary_info(0.0, context.version, transitions),
    )
    return runtime._evaluation_telemetry_failure(failure)


def evaluation_telemetry_failure(
    runtime: CollectionRuntime, failure: TelemetryFailure
) -> dict[str, Any]:
    return _telemetry_failure_summary(runtime, failure)


def _telemetry_failure_summary(
    runtime: CollectionRuntime, failure: TelemetryFailure
) -> dict[str, Any]:
    summary = runtime._summary(
        failure.reward,
        {
            **dict(failure.info or {}),
            "termination_reason": "telemetry_error",
            "telemetry_error": 1.0,
            "actor_epsilon": 0.0,
            "policy_version": failure.version,
        },
        failure.transitions,
    )
    summary["deterministic"] = 1.0
    runtime._evaluation_index += 1
    return summary
