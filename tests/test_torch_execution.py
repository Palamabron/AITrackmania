from __future__ import annotations

import pytest
import torch
from torch import nn

from tmrl.algorithms import ImplicitQuantileQLearning
from tmrl.algorithms.execution import (
    TorchExecutionConfig,
    TorchExecutionError,
    _supported_precisions,
    resolve_torch_execution,
)
from tmrl.core.data import TrainingBatch
from tmrl.models.critics import DiscreteQuantileNetwork


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(observation))


def _batch() -> TrainingBatch:
    return TrainingBatch(
        data={},
        observations=torch.randn(4, 4),
        actions=torch.zeros(4, dtype=torch.int64),
        rewards=torch.ones(4),
        next_observations=torch.randn(4, 4),
        terminated=torch.zeros(4, dtype=torch.bool),
        truncated=torch.zeros(4, dtype=torch.bool),
        bootstrap_discounts=torch.full((4,), 0.99),
        transition_ids=list(range(4)),
    )


def test_auto_execution_resolves_cpu_when_no_accelerator_is_visible(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(
        "tmrl.algorithms.execution.visible_accelerators",
        lambda: set(),
    )

    resolved = resolve_torch_execution(TorchExecutionConfig())

    assert resolved.backend == "cpu"
    assert resolved.precision == "float32"
    assert not resolved.scaler_enabled


def test_auto_execution_rejects_accelerator_build_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(
        "tmrl.algorithms.execution.visible_accelerators",
        lambda: {"cuda"},
    )

    with pytest.raises(TorchExecutionError, match="matching accelerator-enabled"):
        resolve_torch_execution(TorchExecutionConfig())


def test_explicit_unavailable_accelerator_fails_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(TorchExecutionError, match="unavailable"):
        resolve_torch_execution(TorchExecutionConfig(device="cuda"))


def test_modern_cuda_prefers_bfloat16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr("tmrl.algorithms.execution._precision_probe", lambda device, dtype: True)

    supported = _supported_precisions("cuda", torch.device("cuda"))

    assert supported == {"bfloat16", "float16", "float32"}


def test_legacy_cuda_uses_float16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr("tmrl.algorithms.execution._precision_probe", lambda device, dtype: True)

    supported = _supported_precisions("cuda", torch.device("cuda"))

    assert supported == {"float16", "float32"}


def test_rocm_supports_native_bfloat16_and_float16(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr("tmrl.algorithms.execution._precision_probe", lambda device, dtype: True)

    supported = _supported_precisions("rocm", torch.device("cuda"))

    assert supported == {"bfloat16", "float16", "float32"}


def test_failed_mps_probe_falls_back_to_float32(monkeypatch) -> None:
    monkeypatch.setattr("tmrl.algorithms.execution._precision_probe", lambda device, dtype: False)

    supported = _supported_precisions("mps", torch.device("mps"))

    assert supported == {"float32"}


def test_compile_failure_retries_iqn_update_eagerly(monkeypatch) -> None:
    class BrokenCompiled:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("compiler unavailable")

        def q_values(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("compiler unavailable")

    monkeypatch.setattr(torch, "compile", lambda model, mode: BrokenCompiled())
    model = DiscreteQuantileNetwork(_Encoder(), 8, 2, cosine_count=4)
    learner = ImplicitQuantileQLearning(
        model,
        train_quantile_count=4,
        target_quantile_count=4,
        evaluation_quantile_count=4,
        execution={
            "device": "cpu",
            "precision": "float32",
            "compile": True,
        },
    )
    learner.setup({"seed": 0})

    metrics, priorities = learner.update(_batch())

    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()
    assert len(priorities.priorities) == 4
    assert learner.resolved_execution is not None
    assert not learner.resolved_execution.compile_effective
    assert "compiler unavailable" in str(learner.resolved_execution.fallback_reason)
