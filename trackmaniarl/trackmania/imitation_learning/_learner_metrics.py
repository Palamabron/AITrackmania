from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.imitation_learning._data_types import (
    INTERVENTION_KEY,
    SAMPLE_WEIGHT_KEY,
    STUDENT_ACTION_KEY,
)


@dataclass(frozen=True, slots=True)
class BehaviorCloningValidationBatch:
    loss: float
    loss_numerator: float
    loss_denominator: float
    correct: int
    total: int
    per_action_correct: torch.Tensor
    per_action_count: torch.Tensor
    transition_correct: int
    transition_count: int
    steering_correct: int
    steering_count: int
    steering_transition_correct: int
    steering_transition_count: int
    weighted_correct: float
    sample_weight_total: float
    intervention_correct: int
    intervention_count: int
    student_disagreement_correct: int
    student_disagreement_count: int


@dataclass(frozen=True, slots=True)
class LossConfiguration:
    label_smoothing: float
    steering_auxiliary_loss_weight: float
    action_transition_weight: float
    focal_gamma: float


@dataclass(frozen=True, slots=True)
class ClassificationBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    class_weights: torch.Tensor
    observations: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class RecoveryMetricInputs:
    logits: torch.Tensor
    targets: torch.Tensor
    observations: Mapping[str, torch.Tensor]
    action_count: int


@dataclass(frozen=True, slots=True)
class ValidationInputs:
    batch: ClassificationBatch
    numerator: torch.Tensor
    denominator: torch.Tensor
    steering: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ValidationSummary:
    correct: torch.Tensor
    per_action_correct: torch.Tensor
    per_action_count: torch.Tensor
    transitions: torch.Tensor
    steering_correct: torch.Tensor
    steering_transitions: torch.Tensor
    weights: torch.Tensor
    intervention: tuple[int, int]
    disagreement: tuple[int, int]


def to_device(
    observations: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in observations.items()}


def sample_weights(observations: Mapping[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    weights = observations.get(SAMPLE_WEIGHT_KEY)
    if weights is None:
        return torch.ones_like(targets, dtype=torch.float32)
    values = weights.to(targets.device, dtype=torch.float32).reshape(targets.shape)
    if not bool(torch.isfinite(values).all()) or bool((values <= 0.0).any()):
        raise ValueError("behavior-cloning sample weights must be finite and positive")
    return values


def transition_mask(
    observations: Mapping[str, torch.Tensor], targets: torch.Tensor, action_count: int
) -> torch.Tensor:
    previous = observations.get("expert_previous_action")
    if previous is None:
        previous = observations.get("previous_action")
    if previous is None:
        return torch.zeros_like(targets, dtype=torch.bool)
    previous = previous.to(targets.device).long()
    return (previous < action_count) & (previous != targets)


def steering_classes(action_ids: tuple[int, ...], device: torch.device) -> torch.Tensor:
    _, actions = select_brake_tap_actions(action_ids)
    action_array = np.asarray(actions, dtype=np.float32)
    steering_bins = np.rint((action_array[:, 2] + 1.0) * 6.0).astype(np.int64)
    return torch.from_numpy(steering_bins).to(device)


def steering_transition_mask(
    observations: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    steering: torch.Tensor,
) -> torch.Tensor:
    previous = observations.get("expert_previous_action")
    if previous is None:
        return torch.zeros_like(targets, dtype=torch.bool)
    previous = previous.to(targets.device).long()
    action_count = len(steering)
    valid = previous < action_count
    safe_previous = previous.clamp_max(action_count - 1)
    return valid & (steering[safe_previous] != steering[targets])


def steering_loss(
    logits: torch.Tensor, targets: torch.Tensor, steering: torch.Tensor
) -> torch.Tensor:
    steering_bins = torch.unique(steering, sorted=True)
    grouped = []
    for steering_bin in steering_bins:
        selected = logits[:, steering == steering_bin]
        grouped.append(torch.logsumexp(selected, dim=-1) - np.log(selected.shape[-1]))
    steering_targets = torch.searchsorted(steering_bins, steering[targets])
    return F.cross_entropy(torch.stack(grouped, dim=-1), steering_targets, reduction="none")


def classification_loss_terms(
    batch: ClassificationBatch,
    configuration: LossConfiguration,
    steering: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = _classification_losses(batch, configuration, steering)
    multipliers = _loss_multipliers(batch, configuration)
    denominator = (batch.class_weights[batch.targets] * multipliers).sum().clamp_min(1e-8)
    return (losses * multipliers).sum(), denominator


def _classification_losses(
    batch: ClassificationBatch, configuration: LossConfiguration, steering: torch.Tensor
) -> torch.Tensor:
    losses = F.cross_entropy(
        batch.logits,
        batch.targets,
        weight=batch.class_weights,
        label_smoothing=configuration.label_smoothing,
        reduction="none",
    )
    if configuration.steering_auxiliary_loss_weight:
        losses += configuration.steering_auxiliary_loss_weight * steering_loss(
            batch.logits, batch.targets, steering
        )
    return losses


def _loss_multipliers(batch: ClassificationBatch, configuration: LossConfiguration) -> torch.Tensor:
    multipliers = sample_weights(batch.observations, batch.targets)
    transitions = transition_mask(batch.observations, batch.targets, batch.logits.shape[-1])
    multipliers = multipliers * torch.where(
        transitions,
        torch.full_like(multipliers, configuration.action_transition_weight),
        torch.ones_like(multipliers),
    )
    if configuration.focal_gamma:
        probabilities = batch.logits.softmax(dim=-1)
        target_probability = probabilities.gather(1, batch.targets[:, None]).squeeze(1)
        multipliers *= (1.0 - target_probability).pow(configuration.focal_gamma)
    return multipliers


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    predictions = logits.argmax(dim=-1) if logits.ndim > 1 else logits
    accuracy = (predictions[mask] == targets[mask]).float().mean()
    return float(accuracy.detach())


def recovery_metrics(inputs: RecoveryMetricInputs) -> dict[str, float]:
    predictions = inputs.logits.argmax(dim=-1)
    weights = sample_weights(inputs.observations, inputs.targets)
    weighted = (predictions == inputs.targets).float() * weights
    metrics = {
        "weighted_accuracy": float(weighted.sum() / weights.sum()),
        "sample_weight_mean": float(weights.mean()),
    }
    metrics.update(_subset_metrics(inputs, predictions))
    return metrics


def _subset_metrics(inputs: RecoveryMetricInputs, predictions: torch.Tensor) -> dict[str, float]:
    intervention_mask = _optional_mask(inputs.observations.get(INTERVENTION_KEY), inputs.targets)
    disagreement_mask = _disagreement_mask(inputs)
    return {
        "intervention_accuracy": masked_accuracy(predictions, inputs.targets, intervention_mask),
        "intervention_count": float(intervention_mask.sum()),
        "student_disagreement_accuracy": masked_accuracy(
            predictions, inputs.targets, disagreement_mask
        ),
        "student_disagreement_count": float(disagreement_mask.sum()),
    }


def _optional_mask(value: torch.Tensor | None, targets: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(targets, dtype=torch.bool)
    return value.to(targets.device).bool()


def _disagreement_mask(inputs: RecoveryMetricInputs) -> torch.Tensor:
    student = inputs.observations.get(STUDENT_ACTION_KEY)
    if student is None:
        return torch.zeros_like(inputs.targets, dtype=torch.bool)
    actions = student.to(inputs.targets.device).long()
    return (actions < inputs.action_count) & (actions != inputs.targets)


def validation_batch(inputs: ValidationInputs) -> BehaviorCloningValidationBatch:
    summary = _validation_summary(inputs)
    return BehaviorCloningValidationBatch(
        *_validation_loss_values(inputs),
        int(summary.correct.sum().item()),
        int(inputs.batch.targets.numel()),
        summary.per_action_correct.cpu(),
        summary.per_action_count.cpu(),
        int((summary.correct & summary.transitions).sum().item()),
        int(summary.transitions.sum()),
        int(summary.steering_correct.sum()),
        int(inputs.batch.targets.numel()),
        int((summary.steering_correct & summary.steering_transitions).sum()),
        int(summary.steering_transitions.sum()),
        float((summary.correct.float() * summary.weights).sum()),
        float(summary.weights.sum()),
        *summary.intervention,
        *summary.disagreement,
    )


def _validation_summary(inputs: ValidationInputs) -> _ValidationSummary:
    batch = inputs.batch
    predicted = batch.logits.argmax(dim=-1)
    correct = predicted == batch.targets
    action_count = batch.logits.shape[-1]
    intervention, disagreement = _validation_subsets(inputs, correct)
    return _ValidationSummary(
        correct,
        torch.bincount(batch.targets[correct], minlength=action_count),
        torch.bincount(batch.targets, minlength=action_count),
        transition_mask(batch.observations, batch.targets, action_count),
        inputs.steering[predicted] == inputs.steering[batch.targets],
        steering_transition_mask(batch.observations, batch.targets, inputs.steering),
        sample_weights(batch.observations, batch.targets),
        intervention,
        disagreement,
    )


def _validation_loss_values(inputs: ValidationInputs) -> tuple[float, float, float]:
    return (
        float(inputs.numerator / inputs.denominator),
        float(inputs.numerator),
        float(inputs.denominator),
    )


def _validation_subsets(
    inputs: ValidationInputs, correct: torch.Tensor
) -> tuple[tuple[int, int], tuple[int, int]]:
    batch = inputs.batch
    intervention = batch.observations.get(INTERVENTION_KEY)
    counts = _validation_subset_counts(correct, batch.targets, intervention)
    disagreement = _disagreement_mask(_recovery_inputs(batch))
    return counts, _validation_subset_counts(correct, batch.targets, disagreement)


def _recovery_inputs(batch: ClassificationBatch) -> RecoveryMetricInputs:
    return RecoveryMetricInputs(
        batch.logits, batch.targets, batch.observations, batch.logits.shape[-1]
    )


def _validation_subset_counts(
    correct: torch.Tensor, targets: torch.Tensor, subset: torch.Tensor | None
) -> tuple[int, int]:
    mask = (
        torch.zeros_like(targets, dtype=torch.bool)
        if subset is None
        else subset.to(targets.device).bool()
    )
    return int((correct & mask).sum().item()), int(mask.sum().item())
