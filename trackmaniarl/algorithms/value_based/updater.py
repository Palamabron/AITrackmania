from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

import trackmaniarl.algorithms.value_based.update_helpers as value_updates
from trackmaniarl.algorithms._torch import weighted_mean
from trackmaniarl.algorithms.optimization import GradientClipStats
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.algorithms.value_based.targets import BootstrapInputs, bootstrap_target
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import ValuePhase

if TYPE_CHECKING:
    from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner


@dataclass(frozen=True, slots=True)
class _BatchStep:
    batch: TrainingBatch
    view: ValueBatchView
    positions: list[int]
    valid: torch.Tensor
    actions: torch.Tensor


@dataclass(frozen=True, slots=True)
class _ForwardStep:
    batch: _BatchStep
    features: torch.Tensor
    current_support: Any
    predictions: torch.Tensor
    target_support: Any
    targets: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor


@dataclass(frozen=True, slots=True)
class _LossStep:
    value: torch.Tensor
    objective: torch.Tensor
    total: torch.Tensor
    fraction: Any
    priority: value_updates.PriorityInputs


@dataclass(frozen=True, slots=True)
class _GradientStep:
    main: torch.Tensor
    fraction: torch.Tensor
    adaptive: GradientClipStats | None


@dataclass(frozen=True, slots=True)
class _MetricStep:
    forward: _ForwardStep
    losses: _LossStep
    gradients: _GradientStep
    target_state: tuple[bool, float]


class ValueUpdater:
    def __init__(self, learner: DiscreteValueLearner) -> None:
        self.learner = learner

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        started = perf_counter()
        prepared = self._batch_step(batch)
        forward = self._forward_step(prepared)
        losses = self._loss_step(forward)
        priorities = value_updates.priorities(self.learner, losses.priority)
        priority_update = PriorityUpdate(prepared.view.priority_transition_ids(), priorities)
        diagnostics = self._diagnostics(forward, losses.priority)
        gradients = self._optimize(losses)
        self.learner.update_count += 1
        synced = self.learner._sync_target()
        metric_step = _MetricStep(forward, losses, gradients, (synced, perf_counter() - started))
        metrics = self._metrics(metric_step)
        metrics.update(self._replay_metrics(prepared.batch))
        metrics.update(diagnostics)
        self._optional_metrics(metrics, losses, gradients)
        return metrics, priority_update

    def _batch_step(self, batch: TrainingBatch) -> _BatchStep:
        batch = self.learner._batch(batch)
        view = ValueBatchView.from_batch(batch)
        positions = view.training_positions(self.learner.burn_in)
        return _BatchStep(
            batch,
            view,
            positions,
            view.position_masks(positions),
            view.position_actions(positions),
        )

    def _forward_step(self, batch: _BatchStep) -> _ForwardStep:
        learner = self.learner
        assert isinstance(learner.model, CompositeValueModel)
        with learner.autocast():
            feature_inputs = value_updates.FeatureInputs(batch.view, batch.positions)
            features, online_next, target_next = value_updates.features(learner, feature_inputs)
            support = learner.model.support(features, ValuePhase.TRAIN)
            predictions = learner.model.distribution_for_actions(
                features, support.detached_points(), batch.actions
            )
            target_support, targets, rewards, discounts = self._targets(
                batch, online_next, target_next
            )
        return _ForwardStep(
            batch, features, support, predictions, target_support, targets, rewards, discounts
        )

    def _targets(
        self, batch: _BatchStep, online_next: torch.Tensor, target_next: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]:
        learner = self.learner
        assert isinstance(learner.model, CompositeValueModel)
        with torch.no_grad():
            next_actions = self._next_actions(learner, online_next)
            target_support = learner.target_model.support(target_next, ValuePhase.TARGET)
            target_values = learner.target_model.distribution_for_actions(
                target_next, target_support, next_actions
            )
            rewards, discounts = batch.view.returns_and_discounts(batch.positions)
            inputs = BootstrapInputs(rewards, discounts, target_values, learner.value_rescaling)
            return target_support, bootstrap_target(inputs), rewards, discounts

    @staticmethod
    def _next_actions(learner: DiscreteValueLearner, online_next: torch.Tensor) -> torch.Tensor:
        support = learner.model.support(online_next, ValuePhase.EVALUATE)
        values = learner.model.expected_all_actions(online_next, support, learner.neutral_risk)
        return learner._masked(values).argmax(dim=-1)

    def _loss_step(self, forward: _ForwardStep) -> _LossStep:
        value_loss = self._value_loss(forward)
        objective_loss = self._objective_loss(forward)
        fraction = self._fraction_loss(forward)
        priority = self._priority_inputs(forward)
        return _LossStep(
            value_loss, objective_loss, value_loss + objective_loss, fraction, priority
        )

    def _value_loss(self, forward: _ForwardStep) -> torch.Tensor:
        learner = self.learner
        batch = forward.batch
        losses = learner.model.strategy.regression_loss(
            forward.predictions.float(), forward.targets.float(), forward.current_support
        )
        per_sample = (losses * batch.valid).sum(dim=1) / batch.valid.sum(dim=1).clamp_min(1)
        importance = self._importance_weights(batch.batch)
        return weighted_mean(per_sample, importance)

    def _objective_loss(self, forward: _ForwardStep) -> torch.Tensor:
        learner = self.learner
        expected = value_updates.objective_values(
            learner, forward.features, forward.current_support
        )
        objective_inputs = self._objective_inputs(forward, expected)
        return value_updates.objective_loss(learner, objective_inputs)

    @staticmethod
    def _objective_inputs(
        forward: _ForwardStep, expected: torch.Tensor | None
    ) -> value_updates.ObjectiveInputs:
        batch = forward.batch
        features = value_updates.FeatureInputs(batch.view, batch.positions)
        metadata = value_updates.objective_metadata(features, batch.batch.metadata)
        return value_updates.ObjectiveInputs(expected, batch.actions, batch.valid, metadata)

    def _fraction_loss(self, forward: _ForwardStep) -> Any:
        fraction_inputs = value_updates.FractionInputs(
            forward.features,
            forward.batch.actions,
            forward.batch.valid,
            forward.current_support,
            forward.predictions,
        )
        return value_updates.fraction_loss(self.learner, fraction_inputs)

    @staticmethod
    def _importance_weights(batch: TrainingBatch) -> torch.Tensor | None:
        if not isinstance(batch.importance_weights, torch.Tensor):
            return None
        return batch.importance_weights.float().reshape(-1)

    @staticmethod
    def _replay_metrics(batch: TrainingBatch) -> dict[str, float]:
        keys = (
            "replay/demo_sample_fraction",
            "replay/expert_demo_active_fraction",
            "replay/expert_demo_sample_fraction",
            "replay/expert_demo_target_fraction",
        )
        return {key: float(batch.metadata[key]) for key in keys if key in batch.metadata}

    @staticmethod
    def _priority_inputs(forward: _ForwardStep) -> value_updates.PriorityInputs:
        return value_updates.PriorityInputs(
            forward.predictions,
            forward.current_support,
            forward.targets,
            forward.target_support,
            forward.batch.valid,
        )

    def _diagnostics(
        self, forward: _ForwardStep, priority: value_updates.PriorityInputs
    ) -> dict[str, float]:
        next_update = self.learner.update_count + 1
        if next_update % self.learner.diagnostics_interval_updates:
            return {}
        inputs = value_updates.DiagnosticInputs(
            priority, forward.rewards, forward.discounts, forward.batch.actions
        )
        metrics = value_updates.value_diagnostics(self.learner, inputs)
        with torch.no_grad():
            expected = value_updates.objective_values(
                self.learner, forward.features, forward.current_support
            )
        objective = self._objective_inputs(forward, expected)
        metrics.update(value_updates.demonstration_diagnostics(self.learner, objective))
        return metrics

    def _optimize(self, losses: _LossStep) -> _GradientStep:
        inputs = value_updates.OptimizationInputs(losses.total, losses.fraction)
        main, fraction, adaptive = value_updates.optimize_update(self.learner, inputs)
        return _GradientStep(main, fraction, adaptive)

    @staticmethod
    def _metrics(step: _MetricStep) -> dict[str, float]:
        synced, elapsed = step.target_state
        return {
            "loss/value": float(step.losses.value.detach().item()),
            "loss/total": float(step.losses.total.detach().item()),
            "loss/objectives": float(step.losses.objective.detach().item()),
            "gradients/norm": float(step.gradients.main.detach().item()),
            "gradients/fraction_norm": float(step.gradients.fraction.detach().item()),
            "debug/trained_positions": float(len(step.forward.batch.positions)),
            "debug/target_synced_fraction": float(synced),
            "timing/update_s": elapsed,
        }

    @staticmethod
    def _optional_metrics(
        metrics: dict[str, float], losses: _LossStep, gradients: _GradientStep
    ) -> None:
        if gradients.adaptive is not None:
            metrics.update(
                {
                    "gradients/adaptive_ema_norm": gradients.adaptive.ema_norm,
                    "gradients/adaptive_coefficient": gradients.adaptive.coefficient,
                    "gradients/adaptive_clipped": float(gradients.adaptive.clipped),
                }
            )
        if losses.fraction is not None:
            metrics.update(
                {key: float(value.item()) for key, value in losses.fraction.metrics.items()}
            )
