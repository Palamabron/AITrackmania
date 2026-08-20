"""Factory for recursively configured composite models."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.core.spec import ComponentSpec
from trackmaniarl.models.composite import CompositeValueModel


def _component(value: ComponentSpec | Mapping[str, Any]) -> nn.Module:
    spec = value if isinstance(value, ComponentSpec) else ComponentSpec.model_validate(value)
    module_name, _, symbol_name = spec.class_path.partition(":")
    component = getattr(importlib.import_module(module_name), symbol_name)(**spec.kwargs)
    if not isinstance(component, nn.Module):
        raise TypeError(f"{spec.class_path} must construct a torch module")
    return component


class CompositeValueModelFactory:
    model_contract = ModelContract.DISCRETE_VALUE

    def __init__(
        self,
        encoder: ComponentSpec | Mapping[str, Any],
        temporal: ComponentSpec | Mapping[str, Any],
        head: ComponentSpec | Mapping[str, Any],
        strategy: ComponentSpec | Mapping[str, Any],
    ) -> None:
        self.encoder = ComponentSpec.model_validate(encoder)
        self.temporal = ComponentSpec.model_validate(temporal)
        self.head = ComponentSpec.model_validate(head)
        self.strategy = ComponentSpec.model_validate(strategy)

    def build(self) -> CompositeValueModel:
        return CompositeValueModel(
            _component(self.encoder),
            _component(self.temporal),
            _component(self.head),
            _component(self.strategy),
        )
