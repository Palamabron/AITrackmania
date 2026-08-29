from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tests.unit._composite_value_fixtures import (
    StatefulTestScaler,
    _batch,
    _scalar_model,
    _sequence_batch,
    _value_model,
)
from trackmaniarl.algorithms import AdaptiveGradientClipper
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.algorithms.value_based.update_helpers import PriorityInputs
from trackmaniarl.models.backbones import HypersphericalLinear, SimbaV2Backbone
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.contracts import (
    RiskDistortion,
    RiskSpec,
    ValuePhase,
    ValueSupport,
)
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import (
    ImplicitQuantileHead,
    ImplicitQuantileHeadConfig,
    ScalarQHead,
    ScalarQMode,
)
from trackmaniarl.models.strategies import (
    LearnedFractionStrategy,
    RandomQuantileStrategy,
    ScalarValueStrategy,
)
from trackmaniarl.models.temporal import IdentityTemporalCore


@dataclass(frozen=True)
class _PriorityCase:
    predictions: torch.Tensor
    current_support: ValueSupport
    targets: torch.Tensor
    target_support: ValueSupport
    valid: torch.Tensor


def _priority_case(learner: DiscreteValueLearner) -> _PriorityCase:
    features = torch.zeros(2, 3, 6, device=learner.device)
    current = learner.model.support(features, ValuePhase.TRAIN)
    target = learner.target_model.support(features, ValuePhase.TARGET)
    predictions = torch.arange(
        current.points.numel(), dtype=torch.float32, device=learner.device
    ).reshape_as(current.points)
    targets = 0.5 * torch.arange(
        target.points.numel(), dtype=torch.float32, device=learner.device
    ).reshape_as(target.points)
    valid = torch.tensor([[True, True, False], [True, False, True]], device=learner.device)
    return _PriorityCase(predictions, current, targets, target, valid)


def _expected_priorities(case: _PriorityCase) -> list[float]:
    predictions = (case.predictions * case.current_support.weights).sum(dim=-1)
    targets = (case.targets * case.target_support.weights).sum(dim=-1)
    errors = (predictions - targets).abs() * case.valid
    averages = errors.sum(dim=1) / case.valid.sum(dim=1)
    return (0.9 * errors.max(dim=1).values + 0.1 * averages).tolist()


@dataclass(frozen=True)
class _LearnerSnapshot:
    online: Mapping[str, torch.Tensor]
    target: Mapping[str, torch.Tensor]


def _learner_snapshot(learner: DiscreteValueLearner) -> _LearnerSnapshot:
    return _LearnerSnapshot(
        deepcopy(learner.model.state_dict()),
        deepcopy(learner.target_model.state_dict()),
    )


def _assert_learner_unchanged(learner: DiscreteValueLearner, snapshot: _LearnerSnapshot) -> None:
    assert learner.update_count == 0
    assert not learner.optimizer.state
    assert all(parameter.grad is None for parameter in learner.model.parameters())
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, snapshot.online[name])
    for name, value in learner.target_model.state_dict().items():
        torch.testing.assert_close(value, snapshot.target[name])


def _clipper_learner() -> DiscreteValueLearner:
    learner = DiscreteValueLearner(
        _scalar_model(),
        adaptive_gradient_clipper=AdaptiveGradientClipper(
            decay=0.5, warmup_steps=0, clip_factor=1.0
        ),
    )
    learner.setup({"seed": 7})
    return learner


def _assert_clipper_state_equal(
    source: DiscreteValueLearner, restored: DiscreteValueLearner
) -> None:
    assert source.adaptive_gradient_clipper is not None
    assert restored.adaptive_gradient_clipper is not None
    source_state = source.adaptive_gradient_clipper.state_dict()
    restored_state = restored.adaptive_gradient_clipper.state_dict()
    assert restored_state.keys() == source_state.keys()
    for name, value in source_state.items():
        torch.testing.assert_close(restored_state[name], value)


def _assert_model_state_equal(model: torch.nn.Module, expected: Mapping[str, torch.Tensor]) -> None:
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0.0, atol=0.0)


def _simba_learner() -> tuple[SimbaV2Backbone, DiscreteValueLearner]:
    backbone = SimbaV2Backbone(4, 6, block_count=1, expansion=2)
    model = CompositeValueModel(
        CompositeModules(
            backbone,
            IdentityTemporalCore(6),
            ScalarQHead(6, 3, ScalarQMode.DUELING),
            ScalarValueStrategy(),
        )
    )
    learner = DiscreteValueLearner(model)
    learner.setup({"seed": 7})
    return backbone, learner


def _restore_scaler_learner(checkpoint: Mapping[str, Any]) -> DiscreteValueLearner:
    restored = DiscreteValueLearner(_scalar_model())
    restored.setup({"seed": 99})
    restored.scaler = StatefulTestScaler(scale=99.0)
    restored.load_state_dict(checkpoint)
    return restored


def _assert_selected_action_priorities_use_quantile_axis(kind: str) -> None:
    learner = DiscreteValueLearner(_value_model(kind))
    learner.setup({"seed": 7})
    case = _priority_case(learner)

    priorities = learner._priorities(
        PriorityInputs(
            case.predictions,
            case.current_support,
            case.targets,
            case.target_support,
            case.valid,
        )
    )

    assert priorities == pytest.approx(_expected_priorities(case))


def test_selected_action_priorities_use_the_quantile_axis_for_every_strategy() -> None:
    for kind in ("scalar", "qr", "iqn", "fqf"):
        _assert_selected_action_priorities_use_quantile_axis(kind)


def test_invalid_priority_ids_abort_before_backward_or_optimizer_step() -> None:
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    batch = _sequence_batch()
    invalid = replace(
        batch,
        metadata={**batch.metadata, "priority_transition_ids": (batch.transition_ids[-1],)},
    )
    snapshot = _learner_snapshot(learner)

    with pytest.raises(ValueError, match="equal length"):
        learner.update(invalid)

    _assert_learner_unchanged(learner, snapshot)


def test_fqf_uses_dedicated_fraction_optimizer() -> None:
    model = CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(4, 6, 8),
            IdentityTemporalCore(6),
            ImplicitQuantileHead(ImplicitQuantileHeadConfig(6, 3, 8, True)),
            LearnedFractionStrategy(6, fraction_count=4),
        )
    )
    learner = DiscreteValueLearner(model, fraction_learning_rate=1e-4)
    learner.setup({"seed": 7})
    proposal_before = model.strategy.proposal.weight.detach().clone()
    encoder_before = model.encoder.network[0].weight.detach().clone()
    learner.update(_batch())
    assert learner.fraction_optimizer is not None
    assert not torch.equal(model.strategy.proposal.weight, proposal_before)
    assert not torch.equal(model.encoder.network[0].weight, encoder_before)


def test_random_quantile_upper_cvar_uses_partial_interval_mass() -> None:
    strategy = RandomQuantileStrategy(
        train_quantile_count=4,
        target_quantile_count=4,
        evaluation_quantile_count=4,
    )
    support = strategy.support(torch.zeros(1, 6), ValuePhase.EVALUATE, None)
    values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    result = strategy.expectation(
        values,
        support,
        RiskSpec(RiskDistortion.UPPER_CVAR, alpha=0.1),
    )

    torch.testing.assert_close(result, torch.tensor([[4.0]]))


def _assert_quantile_counts_change_architecture_fingerprint() -> None:
    first = _value_model("iqn")
    second = CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(4, 6, 8),
            IdentityTemporalCore(6),
            ImplicitQuantileHead(ImplicitQuantileHeadConfig(6, 3, 8, True)),
            RandomQuantileStrategy(
                train_quantile_count=8,
                target_quantile_count=7,
                evaluation_quantile_count=6,
            ),
        )
    )

    assert first.architecture_fingerprint() != second.architecture_fingerprint()


def test_value_learner_projects_simba_weights_after_update() -> None:
    backbone, learner = _simba_learner()

    learner.update(_batch())

    layers = [module for module in backbone.modules() if isinstance(module, HypersphericalLinear)]
    for layer in layers:
        torch.testing.assert_close(
            layer.weight.norm(dim=1), layer.weight.new_ones(layer.weight.shape[0])
        )


def test_value_learner_persists_adaptive_gradient_clipper_state() -> None:
    learner = _clipper_learner()

    metrics, _ = learner.update(_batch())
    state = deepcopy(learner.state_dict())
    restored = _clipper_learner()
    restored.load_state_dict(state)

    assert metrics["gradients/adaptive_ema_norm"] > 0.0
    assert metrics["gradients/adaptive_coefficient"] <= 1.0
    _assert_clipper_state_equal(learner, restored)


@dataclass(frozen=True)
class _EvaluatedCheckpointCase:
    learner: DiscreteValueLearner
    evaluated: dict[str, torch.Tensor]
    checkpoint: dict[str, Any]


def _evaluated_checkpoint_case() -> _EvaluatedCheckpointCase:
    learner = DiscreteValueLearner(_value_model("iqn"))
    learner.setup({"seed": 7})
    learner.update(_batch())
    exported = learner.policy().export_state()
    evaluated = {name: value.detach().clone() for name, value in exported.items()}
    changed_name = next(name for name, value in evaluated.items() if value.is_floating_point())
    evaluated[changed_name].add_(0.25)
    return _EvaluatedCheckpointCase(learner, evaluated, learner.state_dict_for_policy(evaluated))


def _assert_checkpoint_models(case: _EvaluatedCheckpointCase) -> None:
    expected_modules = case.learner._module_state_from_flat(case.evaluated)
    for module_name, module_state in expected_modules.items():
        for name, expected in module_state.items():
            torch.testing.assert_close(
                case.checkpoint["online"][module_name][name], expected, rtol=0.0, atol=0.0
            )
            torch.testing.assert_close(
                case.checkpoint["target"][module_name][name], expected, rtol=0.0, atol=0.0
            )


def _assert_restored_policy(
    restored: DiscreteValueLearner, expected: Mapping[str, torch.Tensor]
) -> None:
    restored_policy = restored.policy().export_state()
    assert restored_policy.keys() == expected.keys()
    for name, value in expected.items():
        torch.testing.assert_close(restored_policy[name], value, rtol=0.0, atol=0.0)
    _assert_model_state_equal(restored.target_model, expected)


def test_iqn_builds_resumable_exact_evaluated_policy_checkpoint() -> None:
    case = _evaluated_checkpoint_case()
    assert case.learner.optimizer.state
    assert case.checkpoint["optimizers"]["main"]["state"] == {}
    assert case.checkpoint["optimizers"]["strategy"] is None
    _assert_checkpoint_models(case)

    restored = DiscreteValueLearner(_value_model("iqn"))
    restored.setup({"seed": 11})
    restored.load_state_dict(case.checkpoint)

    assert not restored.optimizer.state
    _assert_restored_policy(restored, case.evaluated)


def test_evaluated_checkpoint_preserves_temporarily_frozen_optimizer_membership() -> None:
    learner = DiscreteValueLearner(_value_model("iqn"))
    learner.setup({"seed": 7})
    next(learner.model.encoder.parameters()).requires_grad_(False)
    evaluated = learner.policy().export_state()

    checkpoint = learner.state_dict_for_policy(evaluated)
    restored = DiscreteValueLearner(_value_model("iqn"))
    restored.setup({"seed": 11})
    restored.load_state_dict(checkpoint)

    assert not restored.optimizer.state
    _assert_restored_policy(restored, evaluated)


def test_value_learner_resume_restores_scaler_before_deterministic_continuation() -> None:
    batch = _batch()
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    learner.scaler = StatefulTestScaler()
    learner.update(batch)
    checkpoint = deepcopy(learner.state_dict())

    learner.update(batch)
    expected = deepcopy(learner.model.state_dict())

    restored = _restore_scaler_learner(checkpoint)
    assert isinstance(restored.scaler, StatefulTestScaler)
    assert restored.scaler.current_scale == 2.0

    restored.update(batch)

    assert restored.scaler.current_scale == 3.0
    _assert_model_state_equal(restored.model, expected)


def _assert_checkpoint_rejects_architecture_change() -> None:
    learner = DiscreteValueLearner(_scalar_model())
    learner.setup({"seed": 7})
    state = deepcopy(learner.state_dict())
    changed = DiscreteValueLearner(
        CompositeValueModel(
            CompositeModules(
                MlpSensorEncoder(4, 7, 8),
                IdentityTemporalCore(7),
                ScalarQHead(7, 3),
                ScalarValueStrategy(),
            )
        )
    )
    changed.setup({"seed": 7})
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(state)


def _assert_policy_loading_rejects_outdated_fingerprint() -> None:
    source = DiscreteValueLearner(_value_model("iqn"))
    source.setup({"seed": 7})
    source.update(_batch())
    checkpoint = deepcopy(source.state_dict())
    checkpoint["architecture_fingerprint"] = "outdated"

    restored = DiscreteValueLearner(_value_model("iqn"))
    restored.setup({"seed": 11})
    with pytest.raises(ValueError, match="fingerprint"):
        restored.load_policy_state_dict(checkpoint)


def test_value_model_architecture_fingerprint_contracts() -> None:
    _assert_quantile_counts_change_architecture_fingerprint()
    _assert_checkpoint_rejects_architecture_change()
    _assert_policy_loading_rejects_outdated_fingerprint()
