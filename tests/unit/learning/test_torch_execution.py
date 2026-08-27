from __future__ import annotations

from typing import Any

import pytest
import torch

from tests.unit.learning._algorithm_fixtures import (
    ContinuousModel,
    ContinuousPpoModel,
    DiscreteSacModel,
    RedqModel,
)
from trackmaniarl.algorithms import (
    ProximalPolicyOptimization,
    RandomizedEnsembleSAC,
    SoftActorCritic,
    StableDiscreteSoftActorCritic,
    TruncatedQuantileCritic,
)
from trackmaniarl.algorithms.execution import (
    TorchExecutionConfig,
    TorchExecutionError,
    _supported_precisions,
    resolve_torch_execution,
)


def _new_checkpoint_learner(name: str) -> Any:
    match name:
        case "ppo":
            return ProximalPolicyOptimization(ContinuousPpoModel())
        case "redq":
            return RandomizedEnsembleSAC(RedqModel())
        case "sac":
            return SoftActorCritic(ContinuousModel())
        case "sd_sac":
            return StableDiscreteSoftActorCritic(DiscreteSacModel())
        case "tqc":
            return TruncatedQuantileCritic(ContinuousModel(quantiles=5))
        case _:
            raise ValueError(f"unknown checkpoint learner: {name}")


def _checkpoint_learner(name: str) -> Any:
    learner = _new_checkpoint_learner(name)
    learner.execution = TorchExecutionConfig(device="cpu", precision="float32")
    learner.setup({"seed": 0})
    return learner


def test_auto_execution_resolves_cpu_when_no_accelerator_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution.visible_accelerators",
        lambda: set(),
    )

    resolved = resolve_torch_execution(TorchExecutionConfig())

    assert resolved.backend == "cpu"
    assert resolved.precision == "float32"
    assert not resolved.scaler_enabled


def test_auto_execution_rejects_accelerator_build_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution.visible_accelerators",
        lambda: {"cuda"},
    )

    with pytest.raises(TorchExecutionError, match="matching accelerator-enabled"):
        resolve_torch_execution(TorchExecutionConfig())


def test_explicit_unavailable_accelerator_fails_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(TorchExecutionError, match="unavailable"):
        resolve_torch_execution(TorchExecutionConfig(device="cuda"))


def test_modern_cuda_prefers_bfloat16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution._precision_probe", lambda device, dtype: True
    )

    supported = _supported_precisions("cuda", torch.device("cuda"))

    assert supported == {"bfloat16", "float16", "float32"}


def test_pre_ampere_cuda_uses_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution._precision_probe", lambda device, dtype: True
    )

    supported = _supported_precisions("cuda", torch.device("cuda"))

    assert supported == {"float16", "float32"}


def test_rocm_supports_native_bfloat16_and_float16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution._precision_probe", lambda device, dtype: True
    )

    supported = _supported_precisions("rocm", torch.device("cuda"))

    assert supported == {"bfloat16", "float16", "float32"}


def test_failed_mps_probe_falls_back_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "trackmaniarl.algorithms.execution._precision_probe", lambda device, dtype: False
    )

    supported = _supported_precisions("mps", torch.device("mps"))

    assert supported == {"float32"}


def _assert_checkpoint_requires_execution_field(name: str, field: str) -> None:
    learner = _checkpoint_learner(name)
    incomplete = dict(learner.state_dict())
    incomplete.pop(field)

    with pytest.raises((KeyError, ValueError), match=field):
        learner.load_state_dict(incomplete)


def test_torch_checkpoint_requires_execution_state() -> None:
    for name in ("ppo", "redq", "sac", "sd_sac", "tqc"):
        for field in ("scaler", "rng"):
            _assert_checkpoint_requires_execution_field(name, field)


def _assert_ppo_checkpoint_requires_runtime_field(field: str) -> None:
    learner = _checkpoint_learner("ppo")
    incomplete = dict(learner.state_dict())
    incomplete.pop(field)

    with pytest.raises(KeyError, match=field):
        learner.load_state_dict(incomplete)


def test_ppo_checkpoint_requires_runtime_state() -> None:
    for field in ("observation_normalizer", "reward_normalizer", "processed_transitions"):
        _assert_ppo_checkpoint_requires_runtime_field(field)


def _assert_entropy_checkpoint_requires_field(name: str, field: str) -> None:
    learner = _checkpoint_learner(name)
    incomplete = dict(learner.state_dict())
    incomplete.pop(field)

    with pytest.raises(KeyError, match=field):
        learner.load_state_dict(incomplete)


def test_entropy_checkpoint_requires_nullable_fields() -> None:
    for name in ("sac", "sd_sac", "tqc"):
        for field in ("log_alpha", "alpha_optimizer"):
            _assert_entropy_checkpoint_requires_field(name, field)


def test_redq_checkpoint_requires_target_rng_state() -> None:
    learner = _checkpoint_learner("redq")
    incomplete = dict(learner.state_dict())
    incomplete.pop("target_rng")

    with pytest.raises(KeyError, match="target_rng"):
        learner.load_state_dict(incomplete)


def _assert_checkpoint_requires_rng_field(field: str) -> None:
    learner = _checkpoint_learner("sac")
    incomplete = dict(learner.state_dict())
    rng = dict(incomplete["rng"])
    rng.pop(field)
    incomplete["rng"] = rng

    with pytest.raises(ValueError, match=field):
        learner.load_state_dict(incomplete)


def test_torch_checkpoint_requires_complete_rng_state() -> None:
    for field in ("python", "numpy", "torch"):
        _assert_checkpoint_requires_rng_field(field)


def test_fixed_entropy_checkpoint_preserves_nullable_state() -> None:
    learner = SoftActorCritic(ContinuousModel(), learn_entropy_coefficient=False)
    learner.execution = TorchExecutionConfig(device="cpu", precision="float32")
    learner.setup({"seed": 0})
    state = learner.state_dict()

    learner.load_state_dict(state)

    assert state["log_alpha"] is None
    assert state["alpha_optimizer"] is None
