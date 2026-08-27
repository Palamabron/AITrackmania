"""Composable discrete value model and frame-batch shape adapter."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, cast

import torch
from torch import nn

from trackmaniarl.core.pytree import PyTree, tree_map
from trackmaniarl.models.contracts import (
    RiskSpec,
    ValuePhase,
    ValueRepresentation,
    ValueSupport,
)


def _tensor_leaves(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _tensor_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _tensor_leaves(item)]
    return []


@dataclass(frozen=True, slots=True)
class FrameBatch:
    frames: PyTree
    batch_size: int
    time_steps: int

    def restore(self, features: torch.Tensor) -> torch.Tensor:
        expected = self.batch_size * self.time_steps
        if features.ndim != 2 or features.shape[0] != expected:
            raise ValueError("sensor encoder must return (batch*time, feature_dim)")
        return features.reshape(self.batch_size, self.time_steps, features.shape[-1])


class BatchLayout(Enum):
    FRAMES = "frames"
    SEQUENCE = "sequence"


def _flatten_sequence(observation: PyTree, batch_size: int, time_steps: int) -> PyTree:
    def flatten_leaf(leaf: Any) -> Any:
        if not isinstance(leaf, torch.Tensor):
            return leaf
        return leaf.reshape(batch_size * time_steps, *leaf.shape[2:])

    return cast(PyTree, tree_map(flatten_leaf, observation))


class FrameBatchAdapter:
    @staticmethod
    def flatten(observation: PyTree, layout: BatchLayout) -> FrameBatch:
        leaves = _tensor_leaves(observation)
        if not leaves:
            raise TypeError("observation must contain at least one tensor")
        sequence = layout is BatchLayout.SEQUENCE
        required_rank = 2 if sequence else 1
        if any(leaf.ndim < required_rank for leaf in leaves):
            raise ValueError("observation tensor rank is too small for its batch layout")
        batch_size = int(leaves[0].shape[0])
        time_steps = int(leaves[0].shape[1]) if sequence else 1
        leading = (batch_size, time_steps) if sequence else (batch_size,)
        if any(tuple(leaf.shape[:required_rank]) != leading for leaf in leaves):
            raise ValueError("all observation tensors must share batch and time axes")
        frames = _flatten_sequence(observation, batch_size, time_steps) if sequence else observation
        return FrameBatch(frames, batch_size, time_steps)


@dataclass(frozen=True, slots=True)
class CompositeModules:
    encoder: nn.Module
    temporal: nn.Module
    head: nn.Module
    strategy: nn.Module


class CompositeValueModel(nn.Module):
    """An encoder, temporal core, value head and distribution strategy."""

    def __init__(self, modules: CompositeModules) -> None:
        super().__init__()
        representation = self._validate_modules(modules)
        self.encoder: Any = modules.encoder
        self.temporal: Any = modules.temporal
        self.head: Any = modules.head
        self.strategy: Any = modules.strategy
        self.action_count = int(cast(Any, modules.head).action_count)
        self.representation = ValueRepresentation(cast(str, representation))

    @staticmethod
    def _validate_modules(modules: CompositeModules) -> object:
        encoder_dim = getattr(modules.encoder, "output_dim", None)
        temporal_input = getattr(modules.temporal, "input_dim", None)
        temporal_output = getattr(modules.temporal, "output_dim", None)
        head_dim = getattr(modules.head, "feature_dim", None)
        representation = getattr(modules.head, "representation", None)
        required = getattr(modules.strategy, "required_representation", None)
        if encoder_dim != temporal_input:
            raise ValueError("encoder output_dim must match temporal input_dim")
        if temporal_output != head_dim:
            raise ValueError("temporal output_dim must match head feature_dim")
        if representation != required:
            raise ValueError(
                f"head representation {representation!r} does not match strategy {required!r}"
            )
        return representation

    def encode_frames(self, frames: PyTree) -> torch.Tensor:
        values = self.encoder(frames)
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            raise TypeError("sensor encoder must return a rank-two tensor")
        return values

    def encode_sequence(
        self, observation: PyTree, layout: BatchLayout, burn_in: int
    ) -> torch.Tensor:
        batch = FrameBatchAdapter.flatten(observation, layout)
        frame_features = batch.restore(self.encode_frames(batch.frames))
        return cast(torch.Tensor, self.temporal.unroll(frame_features, burn_in))

    def support(
        self,
        features: torch.Tensor,
        phase: ValuePhase,
        generator: torch.Generator | None = None,
    ) -> ValueSupport:
        return cast(ValueSupport, self.strategy.support(features, phase, generator))

    def expected_all_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        risk: RiskSpec,
    ) -> torch.Tensor:
        values = self.head.evaluate_all(features, support)
        return cast(torch.Tensor, self.strategy.expectation(values, support, risk))

    def distribution_for_actions(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.head.evaluate_actions(features, support, actions))

    def values_at_internal_boundaries(
        self,
        features: torch.Tensor,
        support: ValueSupport,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        if support.boundaries is None:
            raise ValueError("strategy has no internal quantile boundaries")
        points = support.boundaries[..., 1:-1]
        boundary_support = ValueSupport(points, torch.zeros_like(points))
        return self.distribution_for_actions(features, boundary_support, actions)

    def auxiliary_parameters(self) -> tuple[nn.Parameter, ...]:
        return cast(tuple[nn.Parameter, ...], self.strategy.auxiliary_parameters())

    def initial_policy_state(self, batch_size: int, device: torch.device) -> PyTree:
        return cast(PyTree, self.temporal.initial_state(batch_size, device))

    def policy_step(self, observation: PyTree, state: PyTree) -> tuple[torch.Tensor, PyTree]:
        batch = FrameBatchAdapter.flatten(observation, BatchLayout.FRAMES)
        features = self.encode_frames(batch.frames)
        return cast(tuple[torch.Tensor, PyTree], self.temporal.step(features, state))

    def execution_manifest(self) -> dict[str, object]:
        manifest = getattr(self.temporal, "execution_manifest", None)
        return dict(manifest()) if callable(manifest) else {}

    def architecture_fingerprint(self) -> str:
        return self._architecture_fingerprint(
            (
                "input_dim",
                "output_dim",
                "feature_dim",
                "action_count",
                "quantile_count",
                "train_quantile_count",
                "target_quantile_count",
                "evaluation_quantile_count",
                "fraction_count",
                "entropy_coefficient",
                "d_state",
                "d_conv",
                "inner_dim",
            )
        )

    def _architecture_fingerprint(self, dimensions: tuple[str, ...]) -> str:
        modules = ("encoder", "temporal", "head", "strategy")
        architecture: dict[str, object] = {}
        for name in modules:
            module = cast(nn.Module, getattr(self, name))
            architecture[name] = {
                "class": f"{type(module).__module__}:{type(module).__qualname__}",
                "state": {
                    key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for key, value in module.state_dict().items()
                },
                "dimensions": {
                    key: getattr(module, key) for key in dimensions if hasattr(module, key)
                },
            }
        encoded = json.dumps(architecture, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode("utf-8")).hexdigest()
