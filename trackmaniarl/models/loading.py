"""Safe module-level warm starts for composite value models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.models.composite import CompositeValueModel


@dataclass(frozen=True, slots=True)
class WarmStartReport:
    source: str
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[str, ...]

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True), encoding="utf-8")


def warm_start_composite_model(
    model: CompositeValueModel,
    checkpoint: Path,
    *,
    submodules: tuple[str, ...] = ("encoder", "temporal", "head", "strategy"),
    prefix_map: Mapping[str, str] | None = None,
    shape_policy: str = "exact",
    required_tensors: tuple[str, ...] = (),
) -> WarmStartReport:
    if shape_policy not in {"exact", "overlap"}:
        raise ValueError("shape_policy must be exact or overlap")
    if not submodules or any(
        name not in {"encoder", "temporal", "head", "strategy"} for name in submodules
    ):
        raise ValueError("warm-start submodules must select composite model components")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {checkpoint}")
    loaded = TorchCheckpointCodec().load(checkpoint)
    source = _source_state(loaded)
    target = model.state_dict()
    mapped = _map_legacy_iqn(source, target, prefix_map or {})
    matched: list[str] = []
    mismatch: list[str] = []
    for name, value in mapped.items():
        expected = target.get(name)
        if expected is None:
            continue
        if name.partition(".")[0] not in submodules:
            continue
        if expected.dtype != value.dtype:
            mismatch.append(name)
            continue
        if expected.shape != value.shape and (
            shape_policy == "exact" or expected.ndim != value.ndim
        ):
            mismatch.append(name)
            continue
        _copy_tensor(expected, value, overlap=shape_policy == "overlap")
        matched.append(name)
    if not matched:
        raise ValueError("warm-start checkpoint has no compatible model tensors")
    model.load_state_dict(target, strict=True)
    missing = sorted(
        name for name in target if name.partition(".")[0] in submodules and name not in matched
    )
    absent_required = sorted(set(required_tensors) - set(matched))
    if absent_required:
        raise ValueError(f"warm-start is missing required tensors: {absent_required}")
    unexpected = sorted(set(mapped) - set(target))
    return WarmStartReport(
        str(checkpoint),
        tuple(sorted(matched)),
        tuple(missing),
        tuple(unexpected),
        tuple(sorted(mismatch)),
    )


def _source_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    learner = checkpoint.get("learner", checkpoint)
    if not isinstance(learner, Mapping):
        raise ValueError("warm-start checkpoint has no learner mapping")
    online = learner.get("online")
    if isinstance(online, Mapping):
        flattened: dict[str, torch.Tensor] = {}
        for module, values in online.items():
            if isinstance(values, Mapping):
                for name, value in values.items():
                    if isinstance(value, torch.Tensor):
                        flattened[f"{module}.{name}"] = value
        return flattened
    model = learner.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("warm-start checkpoint has no model state")
    return {str(name): value for name, value in model.items() if isinstance(value, torch.Tensor)}


def _map_legacy_iqn(
    source: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    prefix_map: Mapping[str, str],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, value in source.items():
        candidates = [name]
        candidates.extend(
            new + name.removeprefix(old) for old, new in prefix_map.items() if name.startswith(old)
        )
        replacements = (
            ("encoder.encoder.frame.", "encoder.frame."),
            ("encoder.encoder.recurrent.", "temporal.recurrent."),
            ("encoder.encoder.normalization.", "temporal.normalization."),
            ("encoder.encoder.", "encoder.frame."),
            ("encoder.auxiliary.", "encoder.auxiliary."),
            ("quantile_embedding.", "head.quantile_embedding."),
            ("frequencies", "head.frequencies"),
            ("head.", "head.advantage."),
            ("value.", "head.value."),
        )
        for old, new in replacements:
            if name.startswith(old):
                candidates.append(new + name.removeprefix(old))
        destination = next((item for item in candidates if item in target), candidates[-1])
        if destination in result:
            raise ValueError(f"ambiguous warm-start mapping for {destination!r}")
        result[destination] = value
    return result


def _copy_tensor(target: torch.Tensor, source: torch.Tensor, *, overlap: bool) -> None:
    if target.shape == source.shape:
        target.copy_(source.to(device=target.device))
        return
    if not overlap or target.ndim != source.ndim:
        raise ValueError("overlap loading requires tensors with equal rank")
    slices = tuple(
        slice(0, min(left, right)) for left, right in zip(target.shape, source.shape, strict=True)
    )
    target[slices].copy_(source[slices].to(device=target.device))
