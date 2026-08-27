"""Behavior-cloning validation metrics and checkpoint selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp
from typing import Any

import torch

from trackmaniarl.commands.behavior_types import (
    _BehaviorCloningSelection,
    _BehaviorRuntime,
)
from trackmaniarl.trackmania.imitation_learning import clone_state


@dataclass(slots=True)
class _ValidationTotals:
    per_action_correct: torch.Tensor
    per_action_count: torch.Tensor
    loss_numerator: float = 0.0
    loss_denominator: float = 0.0
    correct: int = 0
    total: int = 0
    transition_correct: int = 0
    transition_total: int = 0
    steering_correct: int = 0
    steering_total: int = 0
    steering_transition_correct: int = 0
    steering_transition_total: int = 0
    weighted_correct: float = 0.0
    sample_weight_total: float = 0.0
    intervention_correct: int = 0
    intervention_total: int = 0
    disagreement_correct: int = 0
    disagreement_total: int = 0

    def add(self, batch: Any) -> None:
        self._add_loss(batch)
        self._add_actions(batch)
        self._add_steering(batch)
        self._add_recovery(batch)

    def _add_loss(self, batch: Any) -> None:
        self.loss_numerator += batch.loss_numerator
        self.loss_denominator += batch.loss_denominator
        self.correct += batch.correct
        self.total += batch.total
        self.weighted_correct += batch.weighted_correct
        self.sample_weight_total += batch.sample_weight_total

    def _add_actions(self, batch: Any) -> None:
        self.per_action_correct += batch.per_action_correct
        self.per_action_count += batch.per_action_count
        self.transition_correct += batch.transition_correct
        self.transition_total += batch.transition_count

    def _add_steering(self, batch: Any) -> None:
        self.steering_correct += batch.steering_correct
        self.steering_total += batch.steering_count
        self.steering_transition_correct += batch.steering_transition_correct
        self.steering_transition_total += batch.steering_transition_count

    def _add_recovery(self, batch: Any) -> None:
        self.intervention_correct += batch.intervention_correct
        self.intervention_total += batch.intervention_count
        self.disagreement_correct += batch.student_disagreement_correct
        self.disagreement_total += batch.student_disagreement_count


@dataclass(frozen=True, slots=True)
class _SelectionCandidate:
    loss: float
    score: float
    improved: bool


def _behavior_cloning_control_score(metrics: Mapping[str, float]) -> float:
    if metrics.get("intervention_count", 0.0) > 0.0:
        return (
            0.25 * metrics["steering_transition_accuracy"]
            + 0.20 * metrics["transition_accuracy"]
            + 0.15 * metrics["steering_accuracy"]
            + 0.15 * metrics["balanced_accuracy"]
            + 0.15 * metrics["intervention_accuracy"]
            + 0.05 * metrics["weighted_accuracy"]
            + 0.05 * metrics["accuracy"]
        )
    return (
        0.35 * metrics["steering_transition_accuracy"]
        + 0.20 * metrics["transition_accuracy"]
        + 0.15 * metrics["steering_accuracy"]
        + 0.15 * metrics["balanced_accuracy"]
        + 0.10 * metrics["accuracy"]
        + 0.05 * exp(-metrics["loss"])
    )


def _behavior_cloning_checkpoint_improved(
    selection: _BehaviorCloningSelection, loss: float, score: float
) -> bool:
    minimum_loss = min(selection.minimum_loss, loss)
    eligible = loss <= 1.10 * minimum_loss
    selected_eligible = selection.checkpoint_loss <= 1.10 * minimum_loss
    return eligible and (not selected_eligible or score > selection.checkpoint_score + 1.0e-4)


def _validate_behavior_cloning(runtime: _BehaviorRuntime, step: int) -> _SelectionCandidate:
    totals = _validation_totals(runtime)
    metrics = _validation_metrics(runtime, totals)
    score = _behavior_cloning_control_score(metrics)
    improved = _behavior_cloning_checkpoint_improved(runtime.selection, metrics["loss"], score)
    candidate = _SelectionCandidate(metrics["loss"], score, improved)
    runtime.selection = _updated_selection(runtime, candidate)
    metrics["control_score"] = score
    metrics["checkpoint_loss_eligible"] = float(
        metrics["loss"] <= 1.10 * runtime.selection.minimum_loss
    )
    metrics["best"] = float(candidate.improved)
    _add_action_metrics(runtime, totals, metrics)
    runtime.run.logger.log("bc/validation", metrics, step=step)
    _print_validation(step, metrics, candidate)
    return candidate


def _validation_totals(runtime: _BehaviorRuntime) -> _ValidationTotals:
    action_count = runtime.run.learner.model.action_count
    totals = _ValidationTotals(
        torch.zeros(action_count, dtype=torch.long),
        torch.zeros(action_count, dtype=torch.long),
    )
    data = runtime.data
    batch_size = runtime.run.spec.training.batch_size
    for start in range(0, len(data.validation_labels), batch_size):
        end = start + batch_size
        observations = {
            key: value[start:end] for key, value in data.validation_observations.items()
        }
        batch = runtime.run.learner.evaluate_batch(
            observations, data.validation_labels[start:end], data.weights
        )
        totals.add(batch)
    return totals


def _validation_metrics(runtime: _BehaviorRuntime, totals: _ValidationTotals) -> dict[str, float]:
    loss = totals.loss_numerator / totals.loss_denominator
    learning_rate = runtime.run.learner.step_scheduler(loss)
    recall = totals.per_action_correct.float() / totals.per_action_count.clamp_min(1)
    observed = totals.per_action_count > 0
    metrics = _classification_metrics(totals, loss, float(recall[observed].mean()))
    metrics.update(_recovery_metrics(totals))
    metrics["learning_rate"] = learning_rate
    return metrics


def _classification_metrics(
    totals: _ValidationTotals, loss: float, balanced_accuracy: float
) -> dict[str, float]:
    return {
        "loss": loss,
        "accuracy": totals.correct / totals.total,
        "balanced_accuracy": balanced_accuracy,
        "transition_accuracy": totals.transition_correct / max(totals.transition_total, 1),
        "transition_count": float(totals.transition_total),
        "steering_accuracy": totals.steering_correct / max(totals.steering_total, 1),
        "steering_transition_accuracy": totals.steering_transition_correct
        / max(totals.steering_transition_total, 1),
        "steering_transition_count": float(totals.steering_transition_total),
    }


def _recovery_metrics(totals: _ValidationTotals) -> dict[str, float]:
    return {
        "weighted_accuracy": totals.weighted_correct / max(totals.sample_weight_total, 1.0e-8),
        "intervention_accuracy": totals.intervention_correct / max(totals.intervention_total, 1),
        "intervention_count": float(totals.intervention_total),
        "student_disagreement_accuracy": totals.disagreement_correct
        / max(totals.disagreement_total, 1),
        "student_disagreement_count": float(totals.disagreement_total),
    }


def _updated_selection(
    runtime: _BehaviorRuntime, candidate: _SelectionCandidate
) -> _BehaviorCloningSelection:
    previous = runtime.selection
    minimum_loss = min(previous.minimum_loss, candidate.loss)
    if candidate.improved:
        return _BehaviorCloningSelection(
            minimum_loss,
            candidate.score,
            candidate.loss,
            clone_state(runtime.run.learner.state_dict()),
        )
    return _BehaviorCloningSelection(
        minimum_loss,
        previous.checkpoint_score,
        previous.checkpoint_loss,
        previous.checkpoint_state,
        previous.stale_validations + 1,
    )


def _add_action_metrics(
    runtime: _BehaviorRuntime, totals: _ValidationTotals, metrics: dict[str, float]
) -> None:
    recall = totals.per_action_correct.float() / totals.per_action_count.clamp_min(1)
    values = zip(
        runtime.run.learner.model.action_ids,
        recall.tolist(),
        totals.per_action_count.tolist(),
        strict=True,
    )
    for action_id, action_recall, count in values:
        metrics[f"action_recall/{action_id}"] = action_recall
        metrics[f"action_count/{action_id}"] = count


def _print_validation(
    step: int, metrics: Mapping[str, float], candidate: _SelectionCandidate
) -> None:
    print(
        f"BC validation step={step}: loss={metrics['loss']:.5f}, "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
        f"transition_accuracy={metrics['transition_accuracy']:.4f}, "
        f"steering_accuracy={metrics['steering_accuracy']:.4f}, "
        f"steering_transition_accuracy={metrics['steering_transition_accuracy']:.4f}, "
        f"intervention_accuracy={metrics['intervention_accuracy']:.4f}, "
        f"control_score={metrics['control_score']:.5f}, "
        f"lr={metrics['learning_rate']:.2e}, best={candidate.improved}"
    )
