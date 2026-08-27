"""Loss, optimization, priority, and diagnostic helpers for value learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from trackmaniarl.algorithms.optimization import GradientClipStats
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.algorithms.value_based.objectives import ValueObjectiveContext
from trackmaniarl.models.backbones import project_hyperspherical_weights
from trackmaniarl.models.composite import BatchLayout, CompositeValueModel
from trackmaniarl.models.contracts import FractionLossContext

if TYPE_CHECKING:
    from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner

_SEQUENCE_PRIORITY_MAX_WEIGHT = 0.9


@dataclass(frozen=True, slots=True)
class FeatureInputs:
    view: ValueBatchView
    positions: list[int]


@dataclass(frozen=True, slots=True)
class FractionInputs:
    features: torch.Tensor
    actions: torch.Tensor
    valid: torch.Tensor
    support: Any
    predictions: torch.Tensor


@dataclass(frozen=True, slots=True)
class ObjectiveInputs:
    expected: torch.Tensor | None
    actions: torch.Tensor
    valid: torch.Tensor
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OptimizationInputs:
    loss: torch.Tensor
    fraction: Any


@dataclass(frozen=True, slots=True)
class PriorityInputs:
    predictions: torch.Tensor
    current_support: Any
    targets: torch.Tensor
    target_support: Any
    valid: torch.Tensor


@dataclass(frozen=True, slots=True)
class DiagnosticInputs:
    priority: PriorityInputs
    rewards: torch.Tensor
    discounts: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True, slots=True)
class _EncodedFeatures:
    online: torch.Tensor
    target: torch.Tensor
    final_online: torch.Tensor
    final_target: torch.Tensor


def features(
    learner: DiscreteValueLearner, inputs: FeatureInputs
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = _encoded_features(learner, inputs.view)
    indices = torch.tensor(
        [position - learner.burn_in for position in inputs.positions], device=learner.device
    )
    current = encoded.online[:, indices]
    online_next, target_next = _next_features(learner, inputs, encoded)
    return current, torch.stack(online_next, dim=1), torch.stack(target_next, dim=1)


def _encoded_features(learner: DiscreteValueLearner, view: ValueBatchView) -> _EncodedFeatures:
    assert isinstance(learner.model, CompositeValueModel)
    layout = BatchLayout.SEQUENCE if view.sequence else BatchLayout.FRAMES
    online = learner.model.encode_sequence(view.batch.observations, layout, learner.burn_in)
    target = learner.target_model.encode_sequence(view.batch.observations, layout, learner.burn_in)
    final_online = learner.model.encode_sequence(
        view.batch.next_observations, layout, learner.burn_in
    )[:, -1]
    final_target = learner.target_model.encode_sequence(
        view.batch.next_observations, layout, learner.burn_in
    )[:, -1]
    return _EncodedFeatures(online, target, final_online, final_target)


def _next_features(
    learner: DiscreteValueLearner, inputs: FeatureInputs, encoded: _EncodedFeatures
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    online_next: list[torch.Tensor] = []
    target_next: list[torch.Tensor] = []
    view = inputs.view
    for position in inputs.positions:
        if position == view.time_steps - 1 or not view.sequence:
            online_next.append(encoded.final_online)
            target_next.append(encoded.final_target)
        else:
            index = position + view.n_step - learner.burn_in
            online_next.append(encoded.online[:, index].detach())
            target_next.append(encoded.target[:, index])
    return online_next, target_next


def fraction_loss(learner: DiscreteValueLearner, inputs: FractionInputs) -> Any:
    assert isinstance(learner.model, CompositeValueModel)
    if learner.fraction_optimizer is None:
        return None
    boundaries = learner.model.values_at_internal_boundaries(
        inputs.features, inputs.support, inputs.actions
    )
    context = FractionLossContext(inputs.support, boundaries, inputs.predictions, inputs.valid)
    return learner.model.strategy.auxiliary_loss(context)


def objective_values(
    learner: DiscreteValueLearner, features: torch.Tensor, support: Any
) -> torch.Tensor | None:
    assert isinstance(learner.model, CompositeValueModel)
    if not any(objective.requires_all_actions for objective in learner.objectives):
        return None
    return learner.model.expected_all_actions(features, support.detached(), learner.neutral_risk)


def objective_loss(learner: DiscreteValueLearner, inputs: ObjectiveInputs) -> torch.Tensor:
    loss = torch.zeros((), device=learner.device)
    if not learner.objectives:
        return loss
    if inputs.expected is None:
        raise RuntimeError("configured objectives require all-action values")
    context = ValueObjectiveContext(
        inputs.expected,
        inputs.actions,
        inputs.valid,
        dict(inputs.metadata),
        _action_mask(learner, inputs.expected),
    )
    for objective in learner.objectives:
        value = objective.loss(context)
        if value is not None:
            loss = loss + value
    return loss


def _action_mask(learner: DiscreteValueLearner, expected: torch.Tensor) -> torch.Tensor | None:
    if learner.policy_action_ids is None:
        return None
    mask = torch.zeros(expected.shape[-1], dtype=torch.bool, device=expected.device)
    mask[list(learner.policy_action_ids)] = True
    return mask


def optimize_update(
    learner: DiscreteValueLearner, inputs: OptimizationInputs
) -> tuple[torch.Tensor, torch.Tensor, GradientClipStats | None]:
    assert learner.scaler is not None
    _backward_losses(learner, inputs)
    main_norm, adaptive_stats = _main_gradient_norm(learner)
    fraction_norm = _fraction_gradient_norm(learner)
    _step_optimizers(learner)
    return main_norm, fraction_norm, adaptive_stats


def _backward_losses(learner: DiscreteValueLearner, inputs: OptimizationInputs) -> None:
    assert learner.scaler is not None
    learner.optimizer.zero_grad(set_to_none=True)
    if learner.fraction_optimizer is not None:
        learner.fraction_optimizer.zero_grad(set_to_none=True)
    learner.scaler.scale(inputs.loss).backward()
    if inputs.fraction is not None:
        learner.scaler.scale(inputs.fraction.loss).backward()
    learner.scaler.unscale_(learner.optimizer)


def _main_gradient_norm(
    learner: DiscreteValueLearner,
) -> tuple[torch.Tensor, GradientClipStats | None]:
    main_parameters = [
        parameter for group in learner.optimizer.param_groups for parameter in group["params"]
    ]
    adaptive_stats = (
        learner.adaptive_gradient_clipper(main_parameters)
        if learner.adaptive_gradient_clipper is not None
        else None
    )
    hard_clip_norm = torch.nn.utils.clip_grad_norm_(main_parameters, learner.gradient_clip_norm)
    main_norm = (
        hard_clip_norm if adaptive_stats is None else hard_clip_norm.new_tensor(adaptive_stats.norm)
    )
    return main_norm, adaptive_stats


def _fraction_gradient_norm(learner: DiscreteValueLearner) -> torch.Tensor:
    if learner.fraction_optimizer is None:
        return torch.zeros((), device=learner.device)
    assert learner.scaler is not None
    learner.scaler.unscale_(learner.fraction_optimizer)
    parameters = [
        parameter
        for group in learner.fraction_optimizer.param_groups
        for parameter in group["params"]
    ]
    return torch.nn.utils.clip_grad_norm_(parameters, learner.fraction_gradient_clip_norm)


def _step_optimizers(learner: DiscreteValueLearner) -> None:
    assert learner.scaler is not None
    learner.scaler.step(learner.optimizer)
    project_hyperspherical_weights(learner.model)
    if learner.fraction_optimizer is not None:
        learner.scaler.step(learner.fraction_optimizer)
    learner.scaler.update()


def priorities(learner: DiscreteValueLearner, inputs: PriorityInputs) -> list[float]:
    predicted = learner.model.strategy.expectation(
        inputs.predictions.float().unsqueeze(-1), inputs.current_support, learner.neutral_risk
    ).squeeze(-1)
    target = learner.target_model.strategy.expectation(
        inputs.targets.float().unsqueeze(-1), inputs.target_support, learner.neutral_risk
    ).squeeze(-1)
    errors = (predicted - target).detach().abs() * inputs.valid
    maximum = errors.max(dim=1).values
    mean = errors.sum(dim=1) / inputs.valid.sum(dim=1).clamp_min(1)
    priority = (
        _SEQUENCE_PRIORITY_MAX_WEIGHT * maximum + (1.0 - _SEQUENCE_PRIORITY_MAX_WEIGHT) * mean
    )
    return [float(value) for value in priority.cpu().tolist()]


@torch.no_grad()
def value_diagnostics(learner: DiscreteValueLearner, inputs: DiagnosticInputs) -> dict[str, float]:
    assert isinstance(learner.model, CompositeValueModel)
    priority = inputs.priority
    selected = learner.model.strategy.expectation(
        priority.predictions.detach().float().unsqueeze(-1),
        priority.current_support,
        learner.neutral_risk,
    ).squeeze(-1)
    target = learner.target_model.strategy.expectation(
        priority.targets.detach().float().unsqueeze(-1),
        priority.target_support,
        learner.neutral_risk,
    ).squeeze(-1)
    return _diagnostic_metrics(learner, inputs, (selected, target))


def _diagnostic_metrics(
    learner: DiscreteValueLearner,
    inputs: DiagnosticInputs,
    values: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    selected, target = values
    valid = inputs.priority.valid
    selected_valid = selected[valid]
    target_valid = target[valid]
    return {
        **_q_diagnostics(selected_valid, target_valid),
        **_batch_diagnostics(learner, inputs),
    }


def _q_diagnostics(selected: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    td_abs = (selected - target).abs()
    return {
        "debug/q_selected_mean": float(selected.mean().item()),
        "debug/q_selected_max": float(selected.max().item()),
        "debug/q_target_mean": float(target.mean().item()),
        "debug/q_target_max": float(target.max().item()),
        "debug/td_abs_mean": float(td_abs.mean().item()),
        "debug/td_abs_max": float(td_abs.max().item()),
    }


def _batch_diagnostics(learner: DiscreteValueLearner, inputs: DiagnosticInputs) -> dict[str, float]:
    valid = inputs.priority.valid
    valid_actions = inputs.actions[valid]
    counts = torch.bincount(valid_actions, minlength=learner.model.action_count).float()
    entropy = _action_entropy(counts)
    return {
        "debug/n_step_return_mean": float(inputs.rewards[valid].mean().item()),
        "debug/bootstrap_discount_mean": float(inputs.discounts[valid].mean().item()),
        "debug/bootstrap_zero_fraction": float(
            (inputs.discounts[valid] == 0.0).float().mean().item()
        ),
        "debug/action_batch_unique_fraction": float((counts > 0.0).float().mean().item()),
        "debug/action_batch_entropy": float(entropy.item()),
    }


def _action_entropy(counts: torch.Tensor) -> torch.Tensor:
    probabilities = counts / counts.sum().clamp_min(1.0)
    positive = probabilities[probabilities > 0.0]
    entropy = -(positive * positive.log()).sum()
    if len(counts) > 1:
        entropy = entropy / entropy.new_tensor(float(len(counts))).log()
    return entropy
