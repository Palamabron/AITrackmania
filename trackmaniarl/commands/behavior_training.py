"""Behavior-cloning optimization and checkpoint lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch

from trackmaniarl.commands.behavior_types import (
    _BehaviorCheckpoints,
    _BehaviorCloningSelection,
    _BehaviorData,
    _BehaviorRuntime,
    _BehaviorTrainingRequest,
)
from trackmaniarl.commands.behavior_validation import (
    _SelectionCandidate,
    _validate_behavior_cloning,
)
from trackmaniarl.trackmania.imitation_learning import (
    class_weights,
    clone_state,
    collate_behavior_cloning,
    flatten_behavior_cloning_laps,
)


def _train_behavior_cloning(request: _BehaviorTrainingRequest) -> None:
    runtime = _behavior_runtime(request)
    _run_behavior_training(runtime)
    if runtime.selection.checkpoint_state is None:
        raise RuntimeError("behavior cloning completed without a validation checkpoint")
    runtime.run.learner.load_state_dict(runtime.selection.checkpoint_state)
    _print_training_result(runtime)


def _behavior_runtime(request: _BehaviorTrainingRequest) -> _BehaviorRuntime:
    data = _behavior_data(request)
    directory = request.run.run_dir / "checkpoints"
    checkpoints = _BehaviorCheckpoints(
        directory / "bc-best-validation.pt", directory / "bc-latest.pt"
    )
    runtime = _BehaviorRuntime(request.run, data, checkpoints)
    if request.resume is not None:
        _restore_training_runtime(runtime, request.resume)
    return runtime


def _behavior_data(request: _BehaviorTrainingRequest) -> _BehaviorData:
    learner = request.run.learner
    train_observations, train_labels = flatten_behavior_cloning_laps(request.training)
    validation_observations, validation_labels = flatten_behavior_cloning_laps(request.validation)
    train_tensors = collate_behavior_cloning(train_observations)
    validation_tensors = collate_behavior_cloning(validation_observations)
    request.training.clear()
    request.validation.clear()
    weights = class_weights(
        train_labels, learner.model.action_count, power=learner.class_weight_power
    )
    generator = torch.Generator().manual_seed(request.run.spec.seed)
    return _BehaviorData(
        train_tensors, train_labels, validation_tensors, validation_labels, weights, generator
    )


def _restore_training_runtime(runtime: _BehaviorRuntime, resume: Mapping[str, Any]) -> None:
    training = resume["training"]
    if resume["schema_version"] != "trackmaniarl-bc-training-v2" or not isinstance(
        training, Mapping
    ):
        raise ValueError("BC resume requires a complete v2 training checkpoint")
    runtime.run.learner.load_state_dict(resume["learner"])
    runtime.data.generator.set_state(training["batch_generator"])
    runtime.selection = _restore_behavior_cloning_selection(training["selection"])
    runtime.best_step = int(training["best_step"])
    runtime.start_step = int(training["step"]) + 1


def _run_behavior_training(runtime: _BehaviorRuntime) -> None:
    learner = runtime.run.learner
    for step in range(runtime.start_step, learner.max_steps + 1):
        metrics = _train_behavior_batch(runtime)
        interval = runtime.run.spec.training.metrics_interval_updates
        if step % interval == 0:
            runtime.run.logger.log("bc/train", metrics, step=step)
        if step % learner.validation_interval != 0:
            continue
        candidate = _validate_behavior_cloning(runtime, step)
        _save_validation_checkpoints(runtime, step, candidate)
        if runtime.selection.stale_validations >= learner.early_stopping_patience:
            print(
                f"Behavior cloning early-stopped at step {step}: "
                f"lr={learner.current_learning_rate():.2e}"
            )
            break


def _train_behavior_batch(runtime: _BehaviorRuntime) -> dict[str, float]:
    run = runtime.run
    data = runtime.data
    indices = torch.randint(
        len(data.train_labels),
        (run.spec.training.batch_size,),
        generator=data.generator,
    )
    labels = data.train_labels[indices]
    observations = {key: value[indices] for key, value in data.train_observations.items()}
    return cast(dict[str, float], run.learner.train_batch(observations, labels, data.weights))


def _save_validation_checkpoints(
    runtime: _BehaviorRuntime, step: int, candidate: _SelectionCandidate
) -> None:
    if candidate.improved:
        state = runtime.selection.checkpoint_state
        if state is None:
            raise RuntimeError("improved validation must capture learner state")
        runtime.best_step = step
        runtime.run.checkpoint_codec.save(
            {"schema_version": "trackmaniarl-bc-policy-v2", "learner": clone_state(state)},
            runtime.checkpoints.best,
        )
    runtime.run.checkpoint_codec.save(_latest_checkpoint(runtime, step), runtime.checkpoints.latest)


def _latest_checkpoint(runtime: _BehaviorRuntime, step: int) -> dict[str, Any]:
    return {
        "schema_version": "trackmaniarl-bc-training-v2",
        "learner": clone_state(runtime.run.learner.state_dict()),
        "training": {
            "step": step,
            "best_step": runtime.best_step,
            "batch_generator": runtime.data.generator.get_state(),
            "selection": _serialize_behavior_cloning_selection(runtime.selection),
        },
    }


def _print_training_result(runtime: _BehaviorRuntime) -> None:
    selection = runtime.selection
    print(
        f"Behavior cloning complete: best_step={runtime.best_step}, "
        f"control_score={selection.checkpoint_score:.5f}, "
        f"checkpoint_loss={selection.checkpoint_loss:.5f}, "
        f"minimum_loss={selection.minimum_loss:.5f}, "
        f"lr={runtime.run.learner.current_learning_rate():.2e}, "
        f"checkpoint={runtime.checkpoints.best}"
    )


def _serialize_behavior_cloning_selection(
    selection: _BehaviorCloningSelection,
) -> dict[str, Any]:
    return {
        "minimum_loss": selection.minimum_loss,
        "checkpoint_score": selection.checkpoint_score,
        "checkpoint_loss": selection.checkpoint_loss,
        "checkpoint_state": selection.checkpoint_state,
        "stale_validations": selection.stale_validations,
    }


def _restore_behavior_cloning_selection(state: Any) -> _BehaviorCloningSelection:
    if not isinstance(state, Mapping):
        raise ValueError("BC resume checkpoint has invalid selection state")
    return _BehaviorCloningSelection(
        minimum_loss=float(state["minimum_loss"]),
        checkpoint_score=float(state["checkpoint_score"]),
        checkpoint_loss=float(state["checkpoint_loss"]),
        checkpoint_state=state["checkpoint_state"],
        stale_validations=int(state["stale_validations"]),
    )
