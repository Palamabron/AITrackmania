from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn

from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, SequenceSampler
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.contracts import (
    ValueSupport,
)
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import (
    FixedQuantileHead,
    FixedQuantileHeadConfig,
    ImplicitQuantileHead,
    ImplicitQuantileHeadConfig,
    ScalarQHead,
    ScalarQMode,
)
from trackmaniarl.models.strategies import (
    FixedQuantileStrategy,
    LearnedFractionStrategy,
    RandomQuantileStrategy,
    ScalarValueStrategy,
)
from trackmaniarl.models.temporal import GruTemporalCore, IdentityTemporalCore, MambaTemporalCore
from trackmaniarl.models.temporal.selective_scan import SelectiveScanInput


class CountingScalarHead(ScalarQHead):
    def __init__(self, feature_dim: int, action_count: int) -> None:
        super().__init__(feature_dim, action_count, ScalarQMode.DUELING)
        self.all_calls = 0
        self.selected_calls = 0

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        self.all_calls += 1
        return super().evaluate_all(features, support)

    def evaluate_actions(
        self, features: torch.Tensor, support: ValueSupport, actions: torch.Tensor
    ) -> torch.Tensor:
        self.selected_calls += 1
        return super().evaluate_actions(features, support, actions)


class StatefulTestScaler:
    def __init__(self, scale: float = 1.0) -> None:
        self.current_scale = scale

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        return outputs * self.current_scale

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        self.current_scale += 1.0

    def state_dict(self) -> dict[str, float]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.current_scale = float(state["current_scale"])


class FailingNativeMamba(MambaTemporalCore):
    @staticmethod
    def _native_scan() -> object:
        raise ImportError("native kernel unavailable in test")


def _differentiable_native_scan(scan: SelectiveScanInput) -> torch.Tensor:
    state_values = torch.einsum(
        "btn,in->bti", scan.input_matrix * scan.output_matrix, scan.state_matrix
    )
    return scan.inputs * scan.deltas + state_values + scan.inputs * scan.skip


class FunctionalNativeMamba(MambaTemporalCore):
    @staticmethod
    def _native_scan() -> Any:
        return _differentiable_native_scan


class NonFiniteOutputMamba(MambaTemporalCore):
    @staticmethod
    def _native_scan() -> Any:
        def scan(operands: SelectiveScanInput) -> torch.Tensor:
            return _differentiable_native_scan(operands) * float("nan")

        return scan


class NonFiniteGradientMamba(MambaTemporalCore):
    @staticmethod
    def _native_scan() -> Any:
        def scan(operands: SelectiveScanInput) -> torch.Tensor:
            operands.inputs.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
            return _differentiable_native_scan(operands)

        return scan


def _batch(batch_size: int = 3) -> TrainingBatch:
    observations = torch.randn(batch_size, 4)
    return TrainingBatch(
        data={},
        observations=observations,
        actions=torch.randint(0, 3, (batch_size,)),
        rewards=torch.randn(batch_size),
        next_observations=torch.randn(batch_size, 4),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        bootstrap_discounts=torch.full((batch_size,), 0.99),
        transition_ids=list(range(batch_size)),
    )


def _composite_model(head: nn.Module, strategy: nn.Module) -> CompositeValueModel:
    return CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(4, 6, 8),
            IdentityTemporalCore(6),
            head,
            strategy,
        )
    )


def _scalar_model(head: nn.Module | None = None) -> CompositeValueModel:
    value_head = head or ScalarQHead(6, 3, ScalarQMode.DUELING)
    return _composite_model(value_head, ScalarValueStrategy())


def _recurrent_core(
    kind: str, input_dim: int, output_dim: int
) -> GruTemporalCore | MambaTemporalCore:
    if kind == "gru":
        return GruTemporalCore(input_dim, output_dim)
    if kind == "mamba-torch":
        return MambaTemporalCore(
            input_dim,
            hidden_dim=output_dim,
            d_state=3,
            d_conv=2,
            expand=1,
            backend="torch",
        )
    raise ValueError(f"unsupported recurrent core: {kind}")


def _recurrent_value_model(kind: str) -> CompositeValueModel:
    return CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(4, 6, 8),
            _recurrent_core(kind, 6, 6),
            ScalarQHead(6, 3, ScalarQMode.DUELING),
            ScalarValueStrategy(),
        )
    )


def _assert_nested_state_equal(actual: Any, expected: Any) -> None:
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    elif isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, Mapping) and isinstance(expected, Mapping):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)) and isinstance(expected, type(actual)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def _fixed_quantile_model() -> CompositeValueModel:
    return _composite_model(
        FixedQuantileHead(FixedQuantileHeadConfig(6, 3, quantile_count=4, dueling=True)),
        FixedQuantileStrategy(quantile_count=4),
    )


def _random_quantile_model() -> CompositeValueModel:
    strategy = RandomQuantileStrategy(
        train_quantile_count=4,
        target_quantile_count=5,
        evaluation_quantile_count=6,
    )
    config = ImplicitQuantileHeadConfig(6, 3, cosine_count=8, dueling=True)
    return _composite_model(ImplicitQuantileHead(config), strategy)


def _learned_fraction_model() -> CompositeValueModel:
    head = ImplicitQuantileHead(ImplicitQuantileHeadConfig(6, 3, 8, True))
    return _composite_model(head, LearnedFractionStrategy(6, fraction_count=4))


def _value_model(kind: str) -> CompositeValueModel:
    factories: dict[str, Callable[[], CompositeValueModel]] = {
        "scalar": _scalar_model,
        "qr": _fixed_quantile_model,
        "iqn": _random_quantile_model,
        "fqf": _learned_fraction_model,
    }
    try:
        return factories[kind]()
    except KeyError as error:
        raise ValueError(f"unsupported value model: {kind}") from error


def _sequence_batch() -> TrainingBatch:
    store = InMemoryReplayStore()
    for step in range(8):
        store.append(
            Transition(
                observation=torch.tensor([float(step), 1.0, -1.0, 0.5]),
                action=step % 3,
                reward=float(step + 1) / 10.0,
                next_observation=torch.tensor([float(step + 1), 1.0, -1.0, 0.5]),
                terminated=step == 7,
                truncated=False,
                episode_id="episode-0",
                step=step,
            )
        )
    return SequenceSampler(IdentityFeaturePipeline(), sequence_length=4, seed=2).sample(
        store,
        BatchRequest(batch_size=2, sequence_length=4, n_step=2, gamma=0.9),
    )
