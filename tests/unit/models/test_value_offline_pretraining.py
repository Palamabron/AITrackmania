from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.unit._composite_value_fixtures import (
    _batch,
    _scalar_model,
)
from trackmaniarl.algorithms.value_based import DiscreteValueLearner
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.models.composite import CompositeValueModel


@dataclass(frozen=True)
class _PretrainingCase:
    model: CompositeValueModel
    learner: DiscreteValueLearner
    initially_frozen: nn.Parameter
    optimizer: torch.optim.Optimizer
    optimizer_parameter_ids: tuple[int, ...]
    encoder_before: Mapping[str, torch.Tensor]
    head_before: Mapping[str, torch.Tensor]


def _warm_start_checkpoint(tmp_path: Path) -> Path:
    source = _scalar_model()
    learner = DiscreteValueLearner(source, execution={"device": "cpu"})
    learner.setup({"seed": 0})
    checkpoint = tmp_path / "warm-start.pt"
    TorchCheckpointCodec().save({"learner": learner.state_dict()}, checkpoint)
    return checkpoint


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> tuple[int, ...]:
    return tuple(id(parameter) for group in optimizer.param_groups for parameter in group["params"])


def _prepared_model() -> tuple[CompositeValueModel, nn.Parameter]:
    model = _scalar_model()
    initially_frozen = model.encoder.network[0].bias
    initially_frozen.requires_grad_(False)
    return model, initially_frozen


@pytest.fixture
def pretraining_case(tmp_path: Path) -> _PretrainingCase:
    model, initially_frozen = _prepared_model()
    learner = DiscreteValueLearner(
        model,
        model_initialization_checkpoint=_warm_start_checkpoint(tmp_path),
        warm_start_submodules=("encoder",),
        freeze_warm_start_during_offline_pretraining=True,
    )
    learner.setup({"seed": 7})
    return _PretrainingCase(
        model,
        learner,
        initially_frozen,
        learner.optimizer,
        _optimizer_parameter_ids(learner.optimizer),
        deepcopy(model.encoder.state_dict()),
        deepcopy(model.head.state_dict()),
    )


def _run_pretraining(case: _PretrainingCase) -> None:
    case.learner.begin_offline_pretraining()
    case.learner.update(_batch())


def _assert_only_warm_started_submodules_are_frozen(
    pretraining_case: _PretrainingCase,
) -> None:
    model = pretraining_case.model
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert all(
        torch.equal(value, pretraining_case.encoder_before[name])
        for name, value in model.encoder.state_dict().items()
    )
    assert any(
        not torch.equal(value, pretraining_case.head_before[name])
        for name, value in model.head.state_dict().items()
    )


def _assert_optimizer_membership_is_preserved(
    pretraining_case: _PretrainingCase,
) -> None:
    assert pretraining_case.learner.optimizer is pretraining_case.optimizer
    assert _optimizer_parameter_ids(pretraining_case.learner.optimizer) == (
        pretraining_case.optimizer_parameter_ids
    )


def test_offline_pretraining_freeze_contracts(pretraining_case: _PretrainingCase) -> None:
    _run_pretraining(pretraining_case)
    _assert_only_warm_started_submodules_are_frozen(pretraining_case)
    _assert_optimizer_membership_is_preserved(pretraining_case)


def test_offline_pretraining_restores_original_gradient_state(
    pretraining_case: _PretrainingCase,
) -> None:
    _run_pretraining(pretraining_case)

    pretraining_case.learner.end_offline_pretraining()

    assert pretraining_case.model.encoder.network[0].weight.requires_grad
    assert not pretraining_case.initially_frozen.requires_grad
    assert all(parameter.requires_grad for parameter in pretraining_case.model.head.parameters())


def test_offline_pretraining_freeze_requires_a_warm_start_checkpoint() -> None:
    model = _scalar_model()
    learner = DiscreteValueLearner(
        model,
        freeze_warm_start_during_offline_pretraining=True,
    )
    learner.setup({"seed": 7})

    learner.begin_offline_pretraining()

    assert all(parameter.requires_grad for parameter in model.parameters())
