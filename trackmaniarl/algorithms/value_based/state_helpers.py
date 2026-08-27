"""Checkpoint and warm-start helpers for the discrete value learner."""

from __future__ import annotations

import importlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from trackmaniarl.core.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    validate_checkpoint_v2,
    validate_policy_checkpoint_v2,
)
from trackmaniarl.models.composite import CompositeValueModel

if TYPE_CHECKING:
    from trackmaniarl.algorithms.value_based.learner import DiscreteValueLearner
    from trackmaniarl.models.loading import WarmStartOptions, WarmStartReport


def state_dict(learner: DiscreteValueLearner) -> Mapping[str, Any]:
    assert isinstance(learner.model, CompositeValueModel)
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "architecture_fingerprint": learner.model.architecture_fingerprint(),
        "online": learner._module_state(learner.model),
        "target": learner._module_state(learner.target_model),
        "optimizers": _optimizer_state(learner),
        "objectives": learner._objective_state(),
        "training": _training_state(learner),
        "runtime": learner.model.execution_manifest(),
    }


def _optimizer_state(learner: DiscreteValueLearner) -> Mapping[str, Any]:
    strategy = (
        learner.fraction_optimizer.state_dict() if learner.fraction_optimizer is not None else None
    )
    return {"main": learner.optimizer.state_dict(), "strategy": strategy}


def _training_state(learner: DiscreteValueLearner) -> Mapping[str, Any]:
    clipper = (
        learner.adaptive_gradient_clipper.state_dict()
        if learner.adaptive_gradient_clipper is not None
        else None
    )
    return {
        "update_count": learner.update_count,
        "scaler": learner._scaler_state(),
        "rng": learner._rng_state(),
        "schedules": {"adaptive_gradient_clipper": clipper},
    }


def state_dict_for_policy(
    learner: DiscreteValueLearner, policy_state: Mapping[str, Any]
) -> Mapping[str, Any]:
    assert isinstance(learner.model, CompositeValueModel)
    expected = set(learner.model.state_dict())
    if set(policy_state) != expected:
        raise ValueError("evaluated policy state does not match composite value model")
    state = dict(learner.state_dict())
    modules = learner._module_state_from_flat(policy_state)
    state["online"] = modules
    state["target"] = deepcopy(modules)
    state["optimizers"] = _fresh_policy_optimizers(learner, state)
    return state


def _fresh_policy_optimizers(
    learner: DiscreteValueLearner, state: Mapping[str, Any]
) -> Mapping[str, Any]:
    assert isinstance(learner.model, CompositeValueModel)
    auxiliary = learner.model.auxiliary_parameters()
    auxiliary_ids = {id(parameter) for parameter in auxiliary}
    main = [
        parameter
        for parameter in learner.model.parameters()
        if parameter.requires_grad and id(parameter) not in auxiliary_ids
    ]
    optimizers = dict(cast(Mapping[str, Any], state["optimizers"]))
    optimizers["main"] = _fresh_optimizer_state(main, learner.learning_rate)
    optimizers["strategy"] = _fresh_optimizer_state(auxiliary, learner.fraction_learning_rate)
    return optimizers


def _fresh_optimizer_state(
    parameters: Sequence[torch.nn.Parameter], learning_rate: float
) -> Mapping[str, Any] | None:
    if not parameters:
        return None
    return cast(
        Mapping[str, Any],
        torch.optim.Adam(parameters, lr=learning_rate).state_dict(),
    )


def load_state_dict(learner: DiscreteValueLearner, state: Mapping[str, Any]) -> None:
    validate_checkpoint_v2(state)
    assert isinstance(learner.model, CompositeValueModel)
    expected = learner.model.architecture_fingerprint()
    if state["architecture_fingerprint"] != expected:
        raise ValueError("checkpoint architecture fingerprint does not match the model")
    learner._load_modules(learner.model, cast(Mapping[str, Any], state["online"]))
    learner._load_modules(learner.target_model, cast(Mapping[str, Any], state["target"]))
    _restore_optimizers(learner, cast(Mapping[str, Any], state["optimizers"]))
    _restore_training(learner, cast(Mapping[str, Any], state["training"]))
    learner._load_objective_state(cast(Sequence[Any], state["objectives"]))


def _restore_optimizers(learner: DiscreteValueLearner, optimizers: Mapping[str, Any]) -> None:
    learner.optimizer.load_state_dict(optimizers["main"])
    strategy_state = optimizers["strategy"]
    if learner.fraction_optimizer is None and strategy_state is not None:
        raise ValueError("checkpoint has a strategy optimizer but model does not")
    if learner.fraction_optimizer is not None:
        if strategy_state is None:
            raise ValueError("checkpoint is missing strategy optimizer state")
        learner.fraction_optimizer.load_state_dict(strategy_state)


def _restore_training(learner: DiscreteValueLearner, training: Mapping[str, Any]) -> None:
    learner.update_count = int(training["update_count"])
    learner._restore_scaler(training["scaler"])
    learner._restore_rng(cast(Mapping[str, Any], training["rng"]))
    schedules = cast(Mapping[str, Any], training["schedules"])
    clipper_state = schedules["adaptive_gradient_clipper"]
    if learner.adaptive_gradient_clipper is None:
        if clipper_state is not None:
            raise ValueError("checkpoint gradient clipper does not match configuration")
        return
    if not isinstance(clipper_state, Mapping):
        raise ValueError("checkpoint is missing adaptive gradient clipper state")
    learner.adaptive_gradient_clipper.load_state_dict(clipper_state, strict=True)


def load_policy_state_dict(learner: DiscreteValueLearner, state: Mapping[str, Any]) -> None:
    validate_policy_checkpoint_v2(state)
    assert isinstance(learner.model, CompositeValueModel)
    if state["architecture_fingerprint"] != learner.model.architecture_fingerprint():
        raise ValueError("checkpoint architecture fingerprint does not match the model")
    learner._load_modules(learner.model, cast(Mapping[str, Any], state["online"]))
    learner.target_model.load_state_dict(learner.model.state_dict(), strict=True)


def module_state(model: CompositeValueModel) -> dict[str, Mapping[str, Any]]:
    return {
        "encoder": model.encoder.state_dict(),
        "temporal": model.temporal.state_dict(),
        "head": model.head.state_dict(),
        "strategy": model.strategy.state_dict(),
    }


def load_modules(model: CompositeValueModel, state: Mapping[str, Any]) -> None:
    for name in ("encoder", "temporal", "head", "strategy"):
        getattr(model, name).load_state_dict(state[name], strict=True)


def module_state_from_flat(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    modules: dict[str, dict[str, Any]] = {
        name: {} for name in ("encoder", "temporal", "head", "strategy")
    }
    for name, value in state.items():
        prefix, separator, parameter = name.partition(".")
        if not separator or prefix not in modules:
            raise ValueError(f"unknown composite policy tensor {name!r}")
        modules[prefix][parameter] = value
    return modules


def objective_state(learner: DiscreteValueLearner) -> list[Mapping[str, Any] | None]:
    result: list[Mapping[str, Any] | None] = []
    for objective in learner.objectives:
        state_dict = getattr(objective, "state_dict", None)
        result.append(dict(state_dict()) if callable(state_dict) else None)
    return result


def load_objective_state(learner: DiscreteValueLearner, states: Sequence[Any]) -> None:
    if len(states) != len(learner.objectives):
        raise ValueError("checkpoint objective count does not match configuration")
    for objective, state in zip(learner.objectives, states, strict=True):
        if state is None:
            continue
        loader = getattr(objective, "load_state_dict", None)
        if not callable(loader) or not isinstance(state, Mapping):
            raise ValueError("checkpoint objective state is incompatible")
        loader(state)


def configured(value: Any) -> Any:
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


def load_warm_start(learner: DiscreteValueLearner) -> None:
    if learner.model_initialization_checkpoint is None:
        return
    from trackmaniarl.models.loading import warm_start_composite_model

    assert isinstance(learner.model, CompositeValueModel)
    report = warm_start_composite_model(
        learner.model,
        learner.model_initialization_checkpoint,
        _warm_start_options(learner),
    )
    if learner.run_dir is not None:
        _write_warm_start_report(learner.run_dir, report)


def _warm_start_options(learner: DiscreteValueLearner) -> WarmStartOptions:
    from trackmaniarl.models.loading import WarmStartOptions

    return WarmStartOptions(
        submodules=learner.warm_start_submodules,
        required_tensors=learner.warm_start_required_tensors,
    )


def _write_warm_start_report(run_dir: Path, report: WarmStartReport) -> None:
    report.write(run_dir / "warm-start.json")
    _record_warm_start_manifest(run_dir / "manifest.json", report)


def _record_warm_start_manifest(path: Path, report: WarmStartReport) -> None:
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["warm_start"] = {
        "source": report.source,
        "matched": list(report.matched),
        "missing": list(report.missing),
        "unexpected": list(report.unexpected),
        "shape_mismatch": list(report.shape_mismatch),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
