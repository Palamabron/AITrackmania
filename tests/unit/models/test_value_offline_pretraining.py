from __future__ import annotations

import json
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
    run_dir: Path
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


def _pretraining_learner(model: CompositeValueModel, tmp_path: Path) -> DiscreteValueLearner:
    return DiscreteValueLearner(
        model,
        model_initialization_checkpoint=_warm_start_checkpoint(tmp_path),
        warm_start_submodules=("encoder",),
        freeze_warm_start_during_offline_pretraining=True,
    )


@pytest.fixture
def pretraining_case(tmp_path: Path) -> _PretrainingCase:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"run_id": "warm-start"}', encoding="utf-8")
    model, initially_frozen = _prepared_model()
    learner = _pretraining_learner(model, tmp_path)
    learner.setup({"seed": 7, "run_dir": run_dir})
    return _PretrainingCase(
        run_dir,
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
    report_path = pretraining_case.run_dir / "warm-start.json"
    manifest_path = pretraining_case.run_dir / "manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["warm_start"] == report


def test_offline_pretraining_restores_original_gradient_state(
    pretraining_case: _PretrainingCase,
) -> None:
    _run_pretraining(pretraining_case)

    pretraining_case.learner.end_offline_pretraining()

    assert pretraining_case.model.encoder.network[0].weight.requires_grad
    assert not pretraining_case.initially_frozen.requires_grad
    assert all(parameter.requires_grad for parameter in pretraining_case.model.head.parameters())


def _online_and_target_differ(learner: DiscreteValueLearner) -> bool:
    pairs = zip(
        learner.model.state_dict().values(),
        learner.target_model.state_dict().values(),
        strict=True,
    )
    return any(not torch.equal(online, target) for online, target in pairs)


def _assert_exact_target_sync(learner: DiscreteValueLearner) -> None:
    pairs = zip(
        learner.model.state_dict().values(),
        learner.target_model.state_dict().values(),
        strict=True,
    )
    for online, target in pairs:
        torch.testing.assert_close(online, target, rtol=0.0, atol=0.0)


def test_offline_pretraining_ends_with_an_exact_target_sync() -> None:
    model = _scalar_model()
    learner = DiscreteValueLearner(model, target_update_interval=1_000)
    learner.setup({"seed": 7})
    learner.begin_offline_pretraining()
    learner.update(_batch())
    assert _online_and_target_differ(learner)
    learner.end_offline_pretraining()
    _assert_exact_target_sync(learner)


def test_offline_pretraining_freeze_requires_a_warm_start_checkpoint() -> None:
    model = _scalar_model()
    learner = DiscreteValueLearner(
        model,
        freeze_warm_start_during_offline_pretraining=True,
    )
    learner.setup({"seed": 7})

    learner.begin_offline_pretraining()

    assert all(parameter.requires_grad for parameter in model.parameters())


def _assert_unprepared_warm_start_fails(run_dir: Path, checkpoint: Path) -> None:
    learner = DiscreteValueLearner(
        _scalar_model(),
        model_initialization_checkpoint=checkpoint,
    )
    with pytest.raises(FileNotFoundError, match="run manifest must exist"):
        learner.setup({"seed": 7, "run_dir": run_dir})
    assert not (run_dir / "warm-start.json").exists()


def _resume_without_warm_start_source(
    checkpoint: Path, saved: Mapping[str, object]
) -> Mapping[str, object]:
    checkpoint.unlink()
    resumed = DiscreteValueLearner(
        _scalar_model(),
        model_initialization_checkpoint=checkpoint,
        execution={"device": "cpu"},
    )
    resumed.setup({"seed": 7, "restoring_checkpoint": True})
    resumed.load_state_dict(saved)
    return resumed.state_dict()


def test_warm_start_lifecycle_requires_a_manifest_but_exact_resume_skips_source(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "unprepared"
    run_dir.mkdir()
    checkpoint = _warm_start_checkpoint(tmp_path)
    _assert_unprepared_warm_start_fails(run_dir, checkpoint)
    saved = TorchCheckpointCodec().load(checkpoint)["learner"]
    restored = _resume_without_warm_start_source(checkpoint, saved)
    assert restored["architecture_fingerprint"] == saved["architecture_fingerprint"]
    torch.testing.assert_close(restored["online"], saved["online"], rtol=0.0, atol=0.0)
