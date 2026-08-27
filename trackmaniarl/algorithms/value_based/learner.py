"""Unified scalar, QR-DQN, IQN and FQF discrete value learner."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Unpack

import torch

import trackmaniarl.algorithms.value_based.runtime_helpers as value_runtime
import trackmaniarl.algorithms.value_based.state_helpers as value_state
import trackmaniarl.algorithms.value_based.update_helpers as value_updates
from trackmaniarl.algorithms._torch import TorchLearnerBase
from trackmaniarl.algorithms.optimization import AdaptiveGradientClipper
from trackmaniarl.algorithms.value_based.config import DiscreteValueConfig, DiscreteValueOptions
from trackmaniarl.algorithms.value_based.policy import DiscreteValuePolicy
from trackmaniarl.algorithms.value_based.updater import ValueUpdater
from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.data import PriorityUpdate, TrainingBatch
from trackmaniarl.models.composite import CompositeValueModel
from trackmaniarl.models.contracts import RiskDistortion, RiskSpec


class DiscreteValueLearner(TorchLearnerBase):
    accepted_model_contracts = frozenset({ModelContract.DISCRETE_VALUE})

    def __init__(
        self,
        model: CompositeValueModel | None = None,
        **options: Unpack[DiscreteValueOptions],
    ) -> None:
        config = DiscreteValueConfig(**options)
        super().__init__(
            model, model_factory=config.model_factory, execution=config.execution, seed=config.seed
        )
        config.validate()
        self._configure_training(config)
        self._configure_policy(config)
        self._configure_components(config)
        self._configure_warm_start(config)
        self._offline_warm_start_requires_grad: (
            tuple[tuple[torch.nn.Parameter, bool], ...] | None
        ) = None
        self.update_count = 0

    def _configure_training(self, config: DiscreteValueConfig) -> None:
        self.learning_rate = config.learning_rate
        self.fraction_learning_rate = config.fraction_learning_rate
        self.target_update_interval = config.target_update_interval
        self.target_tau = config.target_tau
        self.diagnostics_interval_updates = config.diagnostics_interval_updates
        self.gradient_clip_norm = config.gradient_clip_norm
        self.fraction_gradient_clip_norm = config.fraction_gradient_clip_norm
        self.burn_in = config.burn_in
        self.value_rescaling = config.value_rescaling

    def _configure_policy(self, config: DiscreteValueConfig) -> None:
        self.exploration_epsilon = config.exploration_epsilon
        self.policy_action_ids = config.policy_action_ids
        self.online_risk = RiskSpec(
            RiskDistortion(config.online_quantile_distortion), config.upper_cvar_alpha
        )
        self.evaluation_risk = RiskSpec(
            RiskDistortion(config.evaluation_quantile_distortion), config.upper_cvar_alpha
        )
        self.neutral_risk = RiskSpec()

    def _configure_components(self, config: DiscreteValueConfig) -> None:
        configured_clipper = self._configured(config.adaptive_gradient_clipper)
        if configured_clipper is not None and not isinstance(
            configured_clipper, AdaptiveGradientClipper
        ):
            raise TypeError("adaptive_gradient_clipper must be an AdaptiveGradientClipper")
        self.adaptive_gradient_clipper = configured_clipper
        self.objectives = tuple(self._configured(value) for value in config.objectives)
        self.action_selector = self._configured(config.action_selector)

    def _configure_warm_start(self, config: DiscreteValueConfig) -> None:
        initialization = (
            None
            if config.model_initialization_checkpoint is None
            else (Path(config.base_dir) / config.model_initialization_checkpoint).resolve()
        )
        self.model_initialization_checkpoint = initialization
        self.warm_start_submodules = config.warm_start_submodules
        self.warm_start_required_tensors = config.warm_start_required_tensors
        self.freeze_warm_start_during_offline_pretraining = (
            config.freeze_warm_start_during_offline_pretraining
        )

    def _setup_model(self) -> None:
        self._validate_model()
        assert isinstance(self.model, CompositeValueModel)
        self._prepare_model()
        self.target_model = deepcopy(self.model).to(self.device).eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._setup_optimizers()

    def _validate_model(self) -> None:
        if not isinstance(self.model, CompositeValueModel):
            raise TypeError("DiscreteValueLearner requires CompositeValueModel")
        if (
            self.policy_action_ids is not None
            and max(self.policy_action_ids) >= self.model.action_count
        ):
            raise ValueError("policy_action_ids must be below the model action_count")

    def _prepare_model(self) -> None:
        assert isinstance(self.model, CompositeValueModel)
        resolver = getattr(self.model.temporal, "resolve_backend", None)
        if callable(resolver):
            resolver(self.device)
        if self.adaptive_gradient_clipper is not None:
            self.adaptive_gradient_clipper.to(self.device)
        self._load_warm_start()

    def _setup_optimizers(self) -> None:
        assert isinstance(self.model, CompositeValueModel)
        auxiliary = self.model.auxiliary_parameters()
        auxiliary_ids = {id(parameter) for parameter in auxiliary}
        main_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad and id(parameter) not in auxiliary_ids
        ]
        self.optimizer = torch.optim.Adam(main_parameters, lr=self.learning_rate)
        self.fraction_optimizer = (
            torch.optim.Adam(auxiliary, lr=self.fraction_learning_rate) if auxiliary else None
        )

    def update(self, batch: TrainingBatch) -> tuple[Mapping[str, float], PriorityUpdate]:
        return ValueUpdater(self).update(batch)

    def _priorities(self, inputs: value_updates.PriorityInputs) -> list[float]:
        return value_updates.priorities(self, inputs)

    def _masked(self, values: torch.Tensor) -> torch.Tensor:
        return value_runtime.masked(self, values)

    def _sync_target(self) -> bool:
        return value_runtime.sync_target(self)

    def policy(self) -> DiscreteValuePolicy:
        return value_runtime.policy(self)

    def begin_offline_pretraining(self) -> None:
        value_runtime.begin_offline_pretraining(self)

    def end_offline_pretraining(self) -> None:
        value_runtime.end_offline_pretraining(self)

    def _warm_start_parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return value_runtime.warm_start_parameters(self)

    def execution_manifest(self) -> Mapping[str, object]:
        manifest = dict(super().execution_manifest())
        if isinstance(self.model, CompositeValueModel):
            manifest["value_model"] = self.model.execution_manifest()
        return manifest

    def state_dict(self) -> Mapping[str, Any]:
        return value_state.state_dict(self)

    def state_dict_for_policy(self, policy_state: Mapping[str, Any]) -> Mapping[str, Any]:
        return value_state.state_dict_for_policy(self, policy_state)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        value_state.load_state_dict(self, state)

    def load_policy_state_dict(self, state: Mapping[str, Any]) -> None:
        value_state.load_policy_state_dict(self, state)

    @staticmethod
    def _module_state(model: CompositeValueModel) -> dict[str, Mapping[str, Any]]:
        return value_state.module_state(model)

    @staticmethod
    def _load_modules(model: CompositeValueModel, state: Mapping[str, Any]) -> None:
        value_state.load_modules(model, state)

    @staticmethod
    def _module_state_from_flat(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        return value_state.module_state_from_flat(state)

    def _objective_state(self) -> list[Mapping[str, Any] | None]:
        return value_state.objective_state(self)

    def _load_objective_state(self, states: Sequence[Any]) -> None:
        value_state.load_objective_state(self, states)

    @staticmethod
    def _configured(value: Any) -> Any:
        return value_state.configured(value)

    def _load_warm_start(self) -> None:
        value_state.load_warm_start(self)
