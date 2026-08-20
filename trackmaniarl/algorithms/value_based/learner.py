"""Unified scalar, QR-DQN, IQN and FQF discrete value learner."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from trackmaniarl.algorithms._torch import TorchLearnerBase, polyak_update, weighted_mean
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.value_based.batches import ValueBatchView
from trackmaniarl.algorithms.value_based.objectives import (
    ValueObjective,
    ValueObjectiveContext,
)
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.algorithms.value_based.targets import bootstrap_target
from trackmaniarl.core.checkpoints import CHECKPOINT_SCHEMA_VERSION, validate_checkpoint_v2
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import (
    FractionLossContext,
    RiskDistortion,
    RiskSpec,
    ValuePhase,
)

_SEQUENCE_PRIORITY_MAX_WEIGHT = 0.9


class DiscreteValueLearner(TorchLearnerBase):
    accepted_model_contracts = frozenset({ModelContract.DISCRETE_VALUE})

    def __init__(
        self,
        model: CompositeValueModel | None = None,
        *,
        model_factory: Any | None = None,
        learning_rate: float = 1e-4,
        fraction_learning_rate: float = 1e-7,
        target_update_interval: int = 1_000,
        target_tau: float = 0.0,
        gradient_clip_norm: float = 10.0,
        fraction_gradient_clip_norm: float = 10.0,
        burn_in: int = 0,
        exploration_epsilon: float = 0.1,
        policy_action_ids: tuple[int, ...] | None = None,
        online_quantile_distortion: str = "neutral",
        evaluation_quantile_distortion: str = "neutral",
        upper_cvar_alpha: float = 0.25,
        value_rescaling: bool = False,
        objectives: Sequence[ValueObjective] = (),
        action_selector: Any | None = None,
        model_initialization_checkpoint: str | Path | None = None,
        warm_start_submodules: tuple[str, ...] = ("encoder", "temporal", "head"),
        warm_start_prefix_map: Mapping[str, str] | None = None,
        warm_start_shape_policy: str = "exact",
        warm_start_required_tensors: tuple[str, ...] = (),
        base_dir: str | Path = ".",
        execution: TorchExecutionConfig | Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__(model, model_factory=model_factory, execution=execution, seed=seed)
        if learning_rate <= 0.0 or fraction_learning_rate <= 0.0:
            raise ValueError("optimizer learning rates must be positive")
        if target_update_interval < 1 or not 0.0 <= target_tau <= 1.0:
            raise ValueError("target update configuration is invalid")
        if min(gradient_clip_norm, fraction_gradient_clip_norm) <= 0.0 or burn_in < 0:
            raise ValueError("gradient clips must be positive and burn_in non-negative")
        if not 0.0 <= exploration_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between zero and one")
        self.learning_rate = learning_rate
        self.fraction_learning_rate = fraction_learning_rate
        self.target_update_interval = target_update_interval
        self.target_tau = target_tau
        self.gradient_clip_norm = gradient_clip_norm
        self.fraction_gradient_clip_norm = fraction_gradient_clip_norm
        self.burn_in = burn_in
        self.exploration_epsilon = exploration_epsilon
        self.policy_action_ids = policy_action_ids
        self.online_risk = RiskSpec(RiskDistortion(online_quantile_distortion), upper_cvar_alpha)
        self.evaluation_risk = RiskSpec(
            RiskDistortion(evaluation_quantile_distortion), upper_cvar_alpha
        )
        self.neutral_risk = RiskSpec()
        self.value_rescaling = value_rescaling
        self.objectives = tuple(self._configured(value) for value in objectives)
        self.action_selector = self._configured(action_selector)
        initialization = (
            None
            if model_initialization_checkpoint is None
            else (Path(base_dir) / model_initialization_checkpoint).resolve()
        )
        self.model_initialization_checkpoint = initialization
        self.warm_start_submodules = warm_start_submodules
        self.warm_start_prefix_map = dict(warm_start_prefix_map or {})
        self.warm_start_shape_policy = warm_start_shape_policy
        self.warm_start_required_tensors = warm_start_required_tensors
        self.update_count = 0

    def _setup_model(self) -> None:
        if not isinstance(self.model, CompositeValueModel):
            raise TypeError("DiscreteValueLearner requires CompositeValueModel")
        resolver = getattr(self.model.temporal, "resolve_backend", None)
        if callable(resolver):
            resolver(self.device)
        self._load_warm_start()
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        auxiliary = self.model.auxiliary_parameters()
        auxiliary_ids = {id(parameter) for parameter in auxiliary}
        main = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in auxiliary_ids
        ]
        self.optimizer = torch.optim.Adam(main, lr=self.learning_rate)
        self.fraction_optimizer = (
            torch.optim.Adam(auxiliary, lr=self.fraction_learning_rate) if auxiliary else None
        )

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        assert isinstance(self.model, CompositeValueModel)
        started = perf_counter()
        batch = self._batch(batch)
        view = ValueBatchView.from_batch(batch)
        positions = view.training_positions(self.burn_in)
        valid = view.position_masks(positions)
        actions = view.position_actions(positions)
        with self.autocast():
            features, online_next, target_next = self._features(view, positions)
            current_support = self.model.support(features, ValuePhase.TRAIN)
            predictions = self.model.distribution_for_actions(
                features, current_support.detached_points(), actions
            )
            with torch.no_grad():
                online_support = self.model.support(online_next, ValuePhase.EVALUATE)
                online_q = self._masked(
                    self.model.expected_all_actions(online_next, online_support, self.neutral_risk)
                )
                next_actions = online_q.argmax(dim=-1)
                target_support = self.target_model.support(target_next, ValuePhase.TARGET)
                target_values = self.target_model.distribution_for_actions(
                    target_next, target_support, next_actions
                )
                rewards, discounts = view.returns_and_discounts(positions)
                targets = bootstrap_target(
                    rewards, discounts, target_values, rescale=self.value_rescaling
                )
        losses = self.model.strategy.regression_loss(
            predictions.float(), targets.float(), current_support
        )
        per_sample = (losses * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        importance = (
            batch.importance_weights.float().reshape(-1)
            if isinstance(batch.importance_weights, torch.Tensor)
            else None
        )
        value_loss = weighted_mean(per_sample, importance)
        expected_all = self._objective_values(features, current_support)
        objective_loss = self._objective_loss(expected_all, actions, valid, batch.metadata)
        total_loss = value_loss + objective_loss
        fraction = self._fraction_loss(features, actions, valid, current_support, predictions)
        gradient_norm, fraction_gradient_norm = self._optimize_update(total_loss, fraction)
        self.update_count += 1
        target_synced = self._sync_target()
        priorities = self._priorities(predictions, current_support, targets, target_support, valid)
        elapsed = perf_counter() - started
        metrics: dict[str, float] = {
            "loss/value": float(value_loss.detach().item()),
            "loss/total": float(total_loss.detach().item()),
            "loss/objectives": float(objective_loss.detach().item()),
            "gradients/norm": float(gradient_norm.detach().item()),
            "gradients/fraction_norm": float(fraction_gradient_norm.detach().item()),
            "debug/trained_positions": float(len(positions)),
            "debug/target_synced_fraction": float(target_synced),
            "timing/update_s": elapsed,
        }
        if fraction is not None:
            metrics.update({key: float(value.item()) for key, value in fraction.metrics.items()})
        return metrics, PriorityUpdate(view.priority_transition_ids(), priorities)

    def _features(
        self, view: ValueBatchView, positions: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert isinstance(self.model, CompositeValueModel)
        all_online = self.model.encode_sequence(
            view.batch.observations, sequence=view.sequence, burn_in=self.burn_in
        )
        all_target = self.target_model.encode_sequence(
            view.batch.observations, sequence=view.sequence, burn_in=self.burn_in
        )
        current_indices = torch.tensor(
            [position - self.burn_in for position in positions], device=self.device
        )
        current = all_online[:, current_indices]
        online_next: list[torch.Tensor] = []
        target_next: list[torch.Tensor] = []
        final_online = self.model.encode_sequence(
            view.batch.next_observations, sequence=view.sequence, burn_in=self.burn_in
        )[:, -1]
        final_target = self.target_model.encode_sequence(
            view.batch.next_observations, sequence=view.sequence, burn_in=self.burn_in
        )[:, -1]
        for position in positions:
            if position == view.time_steps - 1 or not view.sequence:
                online_next.append(final_online)
                target_next.append(final_target)
            else:
                index = position + view.n_step - self.burn_in
                online_next.append(all_online[:, index].detach())
                target_next.append(all_target[:, index])
        return current, torch.stack(online_next, dim=1), torch.stack(target_next, dim=1)

    def _fraction_loss(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        valid: torch.Tensor,
        support: Any,
        predictions: torch.Tensor,
    ) -> Any:
        assert isinstance(self.model, CompositeValueModel)
        if self.fraction_optimizer is None:
            return None
        boundaries = self.model.values_at_internal_boundaries(features, support, actions)
        context = FractionLossContext(support, boundaries, predictions, valid)
        return self.model.strategy.auxiliary_loss(context)

    def _objective_values(self, features: torch.Tensor, support: Any) -> torch.Tensor | None:
        assert isinstance(self.model, CompositeValueModel)
        if not any(objective.requires_all_actions for objective in self.objectives):
            return None
        return self.model.expected_all_actions(features, support.detached(), self.neutral_risk)

    def _objective_loss(
        self,
        expected: torch.Tensor | None,
        actions: torch.Tensor,
        valid: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> torch.Tensor:
        loss = torch.zeros((), device=self.device)
        if not self.objectives:
            return loss
        if expected is None:
            raise RuntimeError("configured objectives require all-action values")
        context = ValueObjectiveContext(expected, actions, valid, dict(metadata))
        for objective in self.objectives:
            value = objective.loss(context)
            if value is not None:
                loss = loss + value
        return loss

    def _optimize_update(
        self, loss: torch.Tensor, fraction: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.scaler is not None
        self.optimizer.zero_grad(set_to_none=True)
        if self.fraction_optimizer is not None:
            self.fraction_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        if fraction is not None:
            self.scaler.scale(fraction.loss).backward()
        self.scaler.unscale_(self.optimizer)
        main_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for group in self.optimizer.param_groups for parameter in group["params"]],
            self.gradient_clip_norm,
        )
        fraction_norm = torch.zeros((), device=self.device)
        if self.fraction_optimizer is not None:
            self.scaler.unscale_(self.fraction_optimizer)
            fraction_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in self.fraction_optimizer.param_groups
                    for parameter in group["params"]
                ],
                self.fraction_gradient_clip_norm,
            )
        self.scaler.step(self.optimizer)
        if self.fraction_optimizer is not None:
            self.scaler.step(self.fraction_optimizer)
        self.scaler.update()
        return main_norm, fraction_norm

    def _priorities(
        self,
        predictions: torch.Tensor,
        current_support: Any,
        targets: torch.Tensor,
        target_support: Any,
        valid: torch.Tensor,
    ) -> list[float]:
        predicted = self.model.strategy.expectation(
            predictions.float(), current_support, self.neutral_risk
        )
        target = self.target_model.strategy.expectation(
            targets.float(), target_support, self.neutral_risk
        )
        errors = (predicted - target).detach().abs() * valid
        maximum = errors.max(dim=1).values
        mean = errors.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        priority = (
            _SEQUENCE_PRIORITY_MAX_WEIGHT * maximum + (1.0 - _SEQUENCE_PRIORITY_MAX_WEIGHT) * mean
        )
        return [float(value) for value in priority.cpu().tolist()]

    def _masked(self, values: torch.Tensor) -> torch.Tensor:
        if self.policy_action_ids is None:
            return values
        mask = torch.zeros(values.shape[-1], dtype=torch.bool, device=values.device)
        mask[list(self.policy_action_ids)] = True
        return values.masked_fill(~mask, -torch.inf)

    def _sync_target(self) -> bool:
        assert isinstance(self.model, CompositeValueModel)
        if self.target_tau:
            polyak_update(self.model, self.target_model, self.target_tau)
            return True
        if self.update_count % self.target_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict(), strict=True)
            return True
        return False

    def policy(self) -> DiscreteValuePolicy:
        assert isinstance(self.model, CompositeValueModel)
        return DiscreteValuePolicy(
            self.model,
            self.device,
            exploration_epsilon=self.exploration_epsilon,
            policy_action_ids=self.policy_action_ids,
            online_risk=self.online_risk,
            evaluation_risk=self.evaluation_risk,
            action_selector=self.action_selector,
        )

    def execution_manifest(self) -> Mapping[str, object]:
        manifest = dict(super().execution_manifest())
        if isinstance(self.model, CompositeValueModel):
            manifest["value_model"] = self.model.execution_manifest()
        return manifest

    def state_dict(self) -> Mapping[str, Any]:
        assert isinstance(self.model, CompositeValueModel)
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "architecture_fingerprint": self.model.architecture_fingerprint(),
            "online": self._module_state(self.model),
            "target": self._module_state(self.target_model),
            "optimizers": {
                "main": self.optimizer.state_dict(),
                "strategy": (
                    self.fraction_optimizer.state_dict()
                    if self.fraction_optimizer is not None
                    else None
                ),
            },
            "objectives": self._objective_state(),
            "training": {
                "update_count": self.update_count,
                "rng": self._rng_state(),
                "schedules": {},
            },
            "runtime": self.model.execution_manifest(),
        }

    def state_dict_for_policy(self, policy_state: Mapping[str, Any]) -> Mapping[str, Any]:
        assert isinstance(self.model, CompositeValueModel)
        expected = set(self.model.state_dict())
        if set(policy_state) != expected:
            raise ValueError("evaluated policy state does not match composite value model")
        state = dict(self.state_dict())
        modules = self._module_state_from_flat(policy_state)
        state["online"] = modules
        state["target"] = deepcopy(modules)
        auxiliary = self.model.auxiliary_parameters()
        auxiliary_ids = {id(parameter) for parameter in auxiliary}
        main = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in auxiliary_ids
        ]
        fresh = torch.optim.Adam(main, lr=self.learning_rate)
        optimizers = dict(cast(Mapping[str, Any], state["optimizers"]))
        optimizers["main"] = fresh.state_dict()
        optimizers["strategy"] = (
            torch.optim.Adam(auxiliary, lr=self.fraction_learning_rate).state_dict()
            if auxiliary
            else None
        )
        state["optimizers"] = optimizers
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        validate_checkpoint_v2(state)
        assert isinstance(self.model, CompositeValueModel)
        expected = self.model.architecture_fingerprint()
        if state.get("architecture_fingerprint") != expected:
            raise ValueError("checkpoint architecture fingerprint does not match the model")
        self._load_modules(self.model, cast(Mapping[str, Any], state["online"]))
        self._load_modules(self.target_model, cast(Mapping[str, Any], state["target"]))
        optimizers = cast(Mapping[str, Any], state["optimizers"])
        self.optimizer.load_state_dict(optimizers["main"])
        strategy_state = optimizers.get("strategy")
        if self.fraction_optimizer is None and strategy_state is not None:
            raise ValueError("checkpoint has a strategy optimizer but model does not")
        if self.fraction_optimizer is not None:
            if strategy_state is None:
                raise ValueError("checkpoint is missing strategy optimizer state")
            self.fraction_optimizer.load_state_dict(strategy_state)
        training = cast(Mapping[str, Any], state["training"])
        self.update_count = int(training["update_count"])
        self._restore_rng(cast(Mapping[str, Any], training["rng"]))
        self._load_objective_state(cast(Sequence[Any], state["objectives"]))

    @staticmethod
    def _module_state(model: CompositeValueModel) -> dict[str, Mapping[str, Any]]:
        return {
            "encoder": model.encoder.state_dict(),
            "temporal": model.temporal.state_dict(),
            "head": model.head.state_dict(),
            "strategy": model.strategy.state_dict(),
        }

    @staticmethod
    def _load_modules(model: CompositeValueModel, state: Mapping[str, Any]) -> None:
        for name in ("encoder", "temporal", "head", "strategy"):
            getattr(model, name).load_state_dict(state[name], strict=True)

    @staticmethod
    def _module_state_from_flat(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        modules: dict[str, dict[str, Any]] = {
            name: {} for name in ("encoder", "temporal", "head", "strategy")
        }
        for name, value in state.items():
            prefix, separator, parameter = name.partition(".")
            if not separator or prefix not in modules:
                raise ValueError(f"unknown composite policy tensor {name!r}")
            modules[prefix][parameter] = value
        return modules

    def _objective_state(self) -> list[Mapping[str, Any] | None]:
        result: list[Mapping[str, Any] | None] = []
        for objective in self.objectives:
            state_dict = getattr(objective, "state_dict", None)
            result.append(dict(state_dict()) if callable(state_dict) else None)
        return result

    def _load_objective_state(self, states: Sequence[Any]) -> None:
        if len(states) != len(self.objectives):
            raise ValueError("checkpoint objective count does not match configuration")
        for objective, state in zip(self.objectives, states, strict=True):
            if state is None:
                continue
            loader = getattr(objective, "load_state_dict", None)
            if not callable(loader) or not isinstance(state, Mapping):
                raise ValueError("checkpoint objective state is incompatible")
            loader(state)

    @staticmethod
    def _configured(value: Any) -> Any:
        if value is None or not isinstance(value, Mapping):
            return value
        class_path = value.get("class_path")
        kwargs = value.get("kwargs", {})
        if not isinstance(class_path, str) or not isinstance(kwargs, Mapping):
            raise TypeError("nested components require class_path and kwargs")
        module_name, separator, symbol_name = class_path.partition(":")
        if not separator:
            raise ValueError("nested component class_path must use module:attribute")
        factory = getattr(importlib.import_module(module_name), symbol_name)
        return factory(**dict(kwargs))

    def _load_warm_start(self) -> None:
        if self.model_initialization_checkpoint is None:
            return
        from trackmaniarl.models.loading import warm_start_composite_model

        assert isinstance(self.model, CompositeValueModel)
        report = warm_start_composite_model(
            self.model,
            self.model_initialization_checkpoint,
            submodules=self.warm_start_submodules,
            prefix_map=self.warm_start_prefix_map,
            shape_policy=self.warm_start_shape_policy,
            required_tensors=self.warm_start_required_tensors,
        )
        if self.run_dir is not None:
            report.write(self.run_dir / "warm-start.json")
            manifest_path = self.run_dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["warm_start"] = {
                    "source": report.source,
                    "matched": list(report.matched),
                    "missing": list(report.missing),
                    "unexpected": list(report.unexpected),
                    "shape_mismatch": list(report.shape_mismatch),
                }
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
                )
