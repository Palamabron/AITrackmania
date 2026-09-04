"""Deterministic no-game validation for resolved runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trackmaniarl.core.contracts import OfflineSupervisedLearner, PolicyMode
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.runtime import ResolvedRun, prepare_run, record_run_attempt


def validate_resolved_run(run: ResolvedRun) -> dict[str, float]:
    """Execute a deterministic no-game smoke update for ``trackmaniarl validate``."""

    prepare_run(run)
    _setup_validation_learner(run)
    record_run_attempt(run)
    request = _validation_request(run)
    _populate_validation_replay(run, request)
    output = _run_validation_update(run, request)
    run.logger.log("validation/update", output, step=1)
    _round_trip_validation_checkpoint(run)
    return output


def _setup_validation_learner(run: ResolvedRun) -> None:
    run.learner.setup(
        {"seed": run.spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
    )


def _validation_request(run: ResolvedRun) -> BatchRequest:
    request = run.spec.training.batch_request()
    if not getattr(run.learner, "on_policy", False):
        return request
    return BatchRequest(
        batch_size=1,
        sequence_length=max(2, request.sequence_length),
        gamma=request.gamma,
    )


@dataclass(frozen=True, slots=True)
class _ValidationReplay:
    run: ResolvedRun
    policy: Any
    transition_count: int


def _populate_validation_replay(run: ResolvedRun, request: BatchRequest) -> None:
    count = max(8, request.batch_size + request.sequence_length - 1)
    context = _ValidationReplay(run, run.learner.policy(), count)
    for step in range(count):
        run.replay_store.append(_validation_transition(context, step))


def _validation_transition(context: _ValidationReplay, step: int) -> Transition:
    synthetic = getattr(context.run.feature_pipeline, "synthetic_observation", None)
    raw = synthetic() if callable(synthetic) else {"speed": float(step)}
    observation = context.run.feature_pipeline.transform_observation(raw)
    action, policy_info = _validation_action(context.policy, observation)
    return Transition(
        observation=observation,
        action=action,
        reward=float(step),
        next_observation=observation,
        terminated=step == context.transition_count - 1,
        truncated=False,
        info=_validation_info(context, step, policy_info),
        episode_id="validation",
        step=step,
    )


def _validation_action(policy: Any, observation: Any) -> tuple[Any, dict[str, Any]]:
    sample = getattr(policy, "act_with_info", None)
    if callable(sample):
        action, info = sample(observation, mode=PolicyMode.EVALUATION)
        return action, dict(info)
    return policy.act(observation, mode=PolicyMode.EVALUATION), {}


def _validation_info(
    context: _ValidationReplay, step: int, policy_info: dict[str, Any]
) -> dict[str, Any]:
    is_demo = step < context.transition_count // 2
    return {
        "is_demo": is_demo,
        "sampling/projected_lap_time_s": 1.0 if is_demo else float("inf"),
        **policy_info,
    }


def _run_validation_update(run: ResolvedRun, request: BatchRequest) -> dict[str, float]:
    batch = run.sampler.sample(run.replay_store, request)
    update = (
        run.learner.validation_update(batch)
        if isinstance(run.learner, OfflineSupervisedLearner)
        else run.learner.update(batch)
    )
    metrics, priority_update = update if isinstance(update, tuple) else (update, None)
    if priority_update is not None:
        run.sampler.update_priorities(priority_update)
    return {key: float(value) for key, value in metrics.items()}


def _round_trip_validation_checkpoint(run: ResolvedRun) -> None:
    checkpoint = run.run_dir / "checkpoints" / "validation.json"
    run.checkpoint_codec.save(run.learner.state_dict(), checkpoint)
    run.learner.load_state_dict(run.checkpoint_codec.load(checkpoint))
