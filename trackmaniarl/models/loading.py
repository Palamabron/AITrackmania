"""Safe module-level warm starts for composite value models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.checkpoints import validate_policy_checkpoint_v2
from trackmaniarl.models.composite import CompositeValueModel


@dataclass(frozen=True, slots=True)
class WarmStartReport:
    source: str
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]

    def write(self, path: Path) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


@dataclass(frozen=True, slots=True)
class WarmStartOptions:
    submodules: tuple[str, ...] = ("encoder", "temporal", "head", "strategy")
    required_tensors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _WarmStartState:
    checkpoint: Path
    target: Mapping[str, torch.Tensor]
    mapped: Mapping[str, torch.Tensor]
    options: WarmStartOptions


@dataclass(frozen=True, slots=True)
class _TensorMatches:
    matched: list[str]
    mismatch: list[str]


def warm_start_composite_model(
    model: CompositeValueModel,
    checkpoint: Path,
    options: WarmStartOptions,
) -> WarmStartReport:
    _validate_options(options, checkpoint)
    source = _source_state(TorchCheckpointCodec().load(checkpoint))
    target = model.state_dict()
    matches = _apply_tensors(target, source, options)
    if not matches.matched:
        raise ValueError("warm-start checkpoint has no compatible model tensors")
    model.load_state_dict(target, strict=True)
    return _warm_start_report(_WarmStartState(checkpoint, target, source, options), matches)


def _validate_options(options: WarmStartOptions, checkpoint: Path) -> None:
    if not options.submodules or any(
        name not in {"encoder", "temporal", "head", "strategy"} for name in options.submodules
    ):
        raise ValueError("warm-start submodules must select composite model components")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {checkpoint}")


def _apply_tensors(
    target: Mapping[str, torch.Tensor],
    mapped: Mapping[str, torch.Tensor],
    options: WarmStartOptions,
) -> _TensorMatches:
    matched: list[str] = []
    mismatch: list[str] = []
    for name, value in mapped.items():
        expected = target.get(name)
        if expected is None or name.partition(".")[0] not in options.submodules:
            continue
        if expected.dtype != value.dtype or expected.shape != value.shape:
            mismatch.append(name)
            continue
        expected.copy_(value.to(device=expected.device))
        matched.append(name)
    return _TensorMatches(matched, mismatch)


def _warm_start_report(state: _WarmStartState, matches: _TensorMatches) -> WarmStartReport:
    missing = sorted(
        name
        for name in state.target
        if name.partition(".")[0] in state.options.submodules and name not in matches.matched
    )
    absent_required = sorted(set(state.options.required_tensors) - set(matches.matched))
    if absent_required:
        raise ValueError(f"warm-start is missing required tensors: {absent_required}")
    unexpected = sorted(set(state.mapped) - set(state.target))
    return WarmStartReport(
        str(state.checkpoint),
        tuple(sorted(matches.matched)),
        tuple(missing),
        tuple(unexpected),
        tuple(sorted(matches.mismatch)),
    )


def _source_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    learner = checkpoint["learner"]
    if not isinstance(learner, Mapping):
        raise ValueError("warm-start checkpoint has no learner mapping")
    validate_policy_checkpoint_v2(learner)
    online = learner["online"]
    if not isinstance(online, Mapping):
        raise ValueError("warm-start checkpoint has no online composite model state")
    return _flatten_modules(online)


def _flatten_modules(online: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    flattened: dict[str, torch.Tensor] = {}
    for module, values in online.items():
        if not isinstance(values, Mapping):
            raise ValueError(f"warm-start module {module!r} is not a tensor mapping")
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"warm-start tensor {module}.{name} is not a tensor")
            flattened[f"{module}.{name}"] = value
    return flattened
