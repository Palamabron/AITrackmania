"""Bounded collection and learner-update lifecycle for local training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trackmaniarl.core.collector import (
    CollectionResult,
    EpisodeCollector,
    FixedStepRolloutCollector,
    RolloutCollectionConfig,
)
from trackmaniarl.core.data import BatchRequest
from trackmaniarl.core.runtime import prepare_run, record_run_attempt
from trackmaniarl.core.training_support import TrainingCounters, TrainingResult
from trackmaniarl.observability.artifacts import AsyncEpisodeWriter

if TYPE_CHECKING:
    from trackmaniarl.core.training import Trainer


@dataclass(slots=True)
class _TrainingSession:
    writer: AsyncEpisodeWriter
    counters: TrainingCounters = field(default_factory=TrainingCounters)
    checkpoints: list[Path] = field(default_factory=list)
    evaluation: Mapping[str, float] | None = None
    rollout_environment: Any | None = None
    rollout_collector: FixedStepRolloutCollector | None = None


@dataclass(frozen=True, slots=True)
class _Failure:
    event: str
    exception: BaseException


def run_training(trainer: Trainer) -> TrainingResult:
    _initialize_run(trainer)
    session = _start_session(trainer)
    return _execute_session(trainer, session)


def _initialize_run(trainer: Trainer) -> None:
    training = trainer.run.spec.training
    prepare_run(trainer.run)
    trainer.run.learner.setup(
        {
            "seed": trainer.run.spec.seed,
            "run_dir": trainer.run.run_dir,
            "model_factory": trainer.run.model_factory,
            "total_transitions": training.total_transitions,
            "restoring_checkpoint": trainer.resume_checkpoint is not None,
        }
    )
    record_run_attempt(trainer.run)
    print(
        f"Training started: run_id={trainer.run.spec.run_id}, "
        f"target_transitions={training.total_transitions}, artifacts={trainer.run.run_dir}",
        flush=True,
    )


def _start_session(trainer: Trainer) -> _TrainingSession:
    training = trainer.run.spec.training
    writer = AsyncEpisodeWriter(
        trainer.run.run_dir / "episodes", max_artifacts=training.max_episode_artifacts
    )
    session = _TrainingSession(writer)
    _resume_session(trainer, session)
    if trainer.on_policy:
        _start_on_policy_collection(trainer, session)
    return session


def _resume_session(trainer: Trainer, session: _TrainingSession) -> None:
    if trainer.resume_checkpoint is None:
        return
    state = trainer.run.checkpoint_codec.load(trainer.resume_checkpoint)
    restored = trainer._restore_checkpoint(state)
    session.counters = _restored_counters(restored)
    trainer._log("train/resumed", {"checkpoint": str(trainer.resume_checkpoint)}, session.counters)


def _restored_counters(state: Mapping[str, Any]) -> TrainingCounters:
    episodes = int(state["episodes"])
    return TrainingCounters(
        transitions=int(state["transitions"]),
        updates=int(state["updates"]),
        episodes=episodes,
        next_episode_index=int(state["next_episode_index"]),
        fractional_updates=float(state["fractional_updates"]),
    )


def _start_on_policy_collection(trainer: Trainer, session: _TrainingSession) -> None:
    reset = getattr(trainer.run.learner, "reset_environment_state", None)
    if callable(reset):
        reset()
    environment = trainer.environment_factory.create(seed=trainer.run.spec.seed)
    session.rollout_environment = environment
    training = trainer.run.spec.training
    session.rollout_collector = FixedStepRolloutCollector(
        trainer.run.replay_store,
        trainer.run.feature_pipeline,
        RolloutCollectionConfig(
            trainer.run.learner.policy(),
            environment,
            training.max_episode_steps,
            trainer.run.spec.seed,
            session.counters.next_episode_index,
        ),
    )


def _execute_session(trainer: Trainer, session: _TrainingSession) -> TrainingResult:
    try:
        return _run_to_completion(trainer, session)
    except KeyboardInterrupt as exc:
        _record_interruption(trainer, session, exc)
        raise
    except BaseException as exc:
        _record_failure(trainer, session, exc)
        raise
    finally:
        _close_session(session)


def _run_to_completion(trainer: Trainer, session: _TrainingSession) -> TrainingResult:
    total = trainer.run.spec.training.total_transitions
    while session.counters.transitions < total:
        _training_cycle(trainer, session)
    _save_final_checkpoint(trainer, session)
    _run_final_evaluation(trainer, session)
    counters = session.counters
    print(
        f"Training finished: transitions={counters.transitions}, "
        f"updates={counters.updates}, episodes={counters.episodes}",
        flush=True,
    )
    return TrainingResult(
        counters.episodes,
        counters.transitions,
        counters.updates,
        tuple(path for path in session.checkpoints if path.is_file()),
        session.evaluation,
    )


def _training_cycle(trainer: Trainer, session: _TrainingSession) -> None:
    counters = session.counters
    previous_transitions = counters.transitions
    previous_episodes = counters.episodes
    result = _collect(trainer, session)
    _record_collection(trainer, session, result)
    _schedule_updates(trainer, session, previous_transitions)
    _perform_updates(trainer, session, result)
    _run_periodic_evaluation(trainer, session, previous_episodes)


def _collect(trainer: Trainer, session: _TrainingSession) -> CollectionResult:
    remaining = trainer.run.spec.training.total_transitions - session.counters.transitions
    collector = session.rollout_collector
    if collector is None:
        return _collect_episode(trainer, session.counters, remaining)
    collector.set_policy(trainer.run.learner.policy())
    length = min(trainer.run.spec.training.sequence_length, remaining)
    return collector.collect(length, rollout_id=f"rollout-{session.counters.updates:08d}")


def _collect_episode(
    trainer: Trainer, counters: TrainingCounters, remaining: int
) -> CollectionResult:
    environment = trainer.environment_factory.create(seed=trainer.run.spec.seed + counters.episodes)
    collector = EpisodeCollector(
        trainer.run.replay_store, trainer.run.feature_pipeline, trainer.run.learner.policy()
    )
    try:
        return collector.collect(
            environment,
            episode_id=f"episode-{counters.episodes:08d}",
            max_steps=min(trainer.run.spec.training.max_episode_steps, remaining),
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()


def _record_collection(
    trainer: Trainer, session: _TrainingSession, result: CollectionResult
) -> None:
    session.writer.submit(result.artifact)
    session.counters.transitions += result.transitions
    session.counters.episodes += result.completed_episodes
    if result.transitions == 0:
        raise RuntimeError("Environment returned an empty episode; refusing to spin forever")
    trainer._log("train/episode", _episode_log_payload(trainer, result), session.counters)
    if result.completed_episodes:
        _print_episode(trainer, result, session.counters)


def _episode_log_payload(trainer: Trainer, result: CollectionResult) -> dict[str, object]:
    return {
        "reward": result.total_reward,
        "transitions": result.transitions,
        "replay_size": len(trainer.run.replay_store),
        "termination": result.artifact.metadata.get("termination", "unknown"),
        **trainer._episode_metrics(result),
    }


def _print_episode(trainer: Trainer, result: CollectionResult, counters: TrainingCounters) -> None:
    metrics = trainer._episode_metrics(result)
    termination = result.artifact.metadata.get("termination", "unknown")
    print(
        f"Episode {counters.episodes}: progress={metrics['progress_pct']:.1f}%, "
        f"reward={result.total_reward:.3f}, time={metrics['episode_elapsed_s']:.2f}s, "
        f"race={metrics['race_time_s']:.2f}s, termination={termination}; "
        f"transitions={counters.transitions}/{trainer.run.spec.training.total_transitions}, "
        f"updates={counters.updates}",
        flush=True,
    )


def _schedule_updates(
    trainer: Trainer,
    session: _TrainingSession,
    previous_transitions: int,
) -> None:
    training = trainer.run.spec.training
    footprint = training.batch_size * training.sequence_length + training.n_step - 1
    ready = 1 if trainer.on_policy else max(training.warmup_transitions, footprint)
    newly_ready = max(0, session.counters.transitions - ready)
    newly_ready -= max(0, previous_transitions - ready)
    earned = 1.0 if trainer.on_policy else newly_ready * training.updates_per_transition
    session.counters.fractional_updates += earned


def _perform_updates(trainer: Trainer, session: _TrainingSession, result: CollectionResult) -> None:
    training = trainer.run.spec.training
    footprint = training.batch_size * training.sequence_length + training.n_step - 1
    ready = 1 if trainer.on_policy else max(training.warmup_transitions, footprint)
    while len(trainer.run.replay_store) >= ready and session.counters.fractional_updates >= 1:
        metrics = _update_once(trainer, session, result)
        _maybe_checkpoint(trainer, session)
        _maybe_print_update(trainer, session.counters, metrics)


def _update_once(
    trainer: Trainer, session: _TrainingSession, result: CollectionResult
) -> Mapping[str, float]:
    request = _batch_request(trainer, session.counters, result)
    batch = trainer.run.sampler.sample(trainer.run.replay_store, request)
    update = trainer.run.learner.update(batch)
    metrics, priorities = update if isinstance(update, tuple) else (update, None)
    if priorities is not None:
        trainer._update_priorities(priorities)
    session.counters.updates += 1
    session.counters.fractional_updates -= 1
    payload = {**metrics, "replay_size": len(trainer.run.replay_store)}
    trainer._log("train/update", payload, session.counters)
    return metrics


def _batch_request(
    trainer: Trainer, counters: TrainingCounters, result: CollectionResult
) -> BatchRequest:
    training = trainer.run.spec.training
    if trainer.on_policy:
        return BatchRequest(batch_size=1, sequence_length=result.transitions, gamma=training.gamma)
    return training.batch_request(
        beta=training.replay_beta(counters.transitions),
        transition_count=counters.transitions,
    )


def _maybe_checkpoint(trainer: Trainer, session: _TrainingSession) -> None:
    interval = trainer.run.spec.training.checkpoint_interval_updates
    if interval is None or session.counters.updates % interval:
        return
    session.checkpoints.append(trainer._write_checkpoint(session.counters))


def _maybe_print_update(
    trainer: Trainer, counters: TrainingCounters, metrics: Mapping[str, float]
) -> None:
    if counters.updates != 1 and counters.updates % 100:
        return
    print(
        f"Training progress: transitions={counters.transitions}/"
        f"{trainer.run.spec.training.total_transitions}, updates={counters.updates}, "
        f"loss={_primary_loss(metrics)}",
        flush=True,
    )


def _primary_loss(metrics: Mapping[str, float]) -> float | str:
    return next((value for name, value in metrics.items() if name.startswith("loss/")), "n/a")


def _run_periodic_evaluation(
    trainer: Trainer, session: _TrainingSession, previous_episodes: int
) -> None:
    evaluator = trainer.run.evaluator
    interval = trainer.run.spec.training.evaluate_every_episodes
    if evaluator is None or interval is None:
        return
    counters = session.counters
    crossed_interval = counters.episodes // interval > previous_episodes // interval
    if not crossed_interval or counters.transitions >= trainer.run.spec.training.total_transitions:
        return
    trainer._checkpoint_for_evaluation(session.checkpoints, counters)
    session.evaluation = evaluator.evaluate(trainer.run.learner.policy())
    trainer._log("eval/suite", session.evaluation, counters)


def _save_final_checkpoint(trainer: Trainer, session: _TrainingSession) -> None:
    if not trainer.run.spec.training.save_final_checkpoint:
        return
    checkpoint = trainer._write_checkpoint(session.counters)
    if not session.checkpoints or session.checkpoints[-1] != checkpoint:
        session.checkpoints.append(checkpoint)


def _run_final_evaluation(trainer: Trainer, session: _TrainingSession) -> None:
    evaluator = trainer.run.evaluator
    if evaluator is None:
        return
    trainer._set_evaluation_checkpoint(session.checkpoints)
    session.evaluation = evaluator.evaluate(trainer.run.learner.policy())
    trainer._log("eval/suite", session.evaluation, session.counters)


def _record_interruption(
    trainer: Trainer, session: _TrainingSession, exc: KeyboardInterrupt
) -> None:
    if trainer.run.spec.training.save_final_checkpoint:
        checkpoint = trainer._write_checkpoint(session.counters)
        if not session.checkpoints or session.checkpoints[-1] != checkpoint:
            session.checkpoints.append(checkpoint)
    _log_exception(trainer, session, _Failure("run/interrupted", exc))


def _record_failure(trainer: Trainer, session: _TrainingSession, exc: BaseException) -> None:
    _log_exception(trainer, session, _Failure("run/failure", exc))


def _log_exception(trainer: Trainer, session: _TrainingSession, failure: _Failure) -> None:
    payload = {
        "exception_type": type(failure.exception).__name__,
        "message": str(failure.exception),
    }
    trainer._log(failure.event, payload, session.counters)


def _close_session(session: _TrainingSession) -> None:
    if session.rollout_environment is not None:
        close = getattr(session.rollout_environment, "close", None)
        if callable(close):
            close()
    session.writer.close()
