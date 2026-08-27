from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pytest
import torch

from tests.unit._composite_value_fixtures import (
    FailingNativeMamba,
    FunctionalNativeMamba,
    NonFiniteGradientMamba,
    NonFiniteOutputMamba,
)
from trackmaniarl.models.composite import CompositeModules, CompositeValueModel
from trackmaniarl.models.encoders import MlpSensorEncoder
from trackmaniarl.models.heads import ScalarQHead
from trackmaniarl.models.strategies import (
    ScalarValueStrategy,
)
from trackmaniarl.models.temporal import MambaTemporalCore


def _cuda_core(
    device: torch.device, backend: Literal["auto", "native", "torch"]
) -> MambaTemporalCore:
    return MambaTemporalCore(8, hidden_dim=8, d_state=4, d_conv=2, expand=1, backend=backend).to(
        device
    )


def _mamba_gradients(
    core: MambaTemporalCore, inputs: torch.Tensor
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    output = core.unroll(inputs, burn_in=0)
    gradients = torch.autograd.grad(output.square().mean(), (inputs, *core.parameters()))
    return output, gradients


def _assert_gradients_match(
    native: Sequence[torch.Tensor], portable: Sequence[torch.Tensor]
) -> None:
    for native_gradient, portable_gradient in zip(native, portable, strict=True):
        torch.testing.assert_close(native_gradient, portable_gradient, rtol=3e-3, atol=3e-4)


def _restore_native(
    checkpoint: Mapping[str, torch.Tensor], device: torch.device
) -> MambaTemporalCore:
    restored = _cuda_core(device, "native")
    restored.load_state_dict(checkpoint)
    restored.resolve_backend(device)
    return restored


def _value_model(core: MambaTemporalCore) -> CompositeValueModel:
    return CompositeValueModel(
        CompositeModules(
            MlpSensorEncoder(4, 4, 6),
            core,
            ScalarQHead(4, 2),
            ScalarValueStrategy(),
        )
    )


def _assert_torch_unroll_matches_streaming_step() -> None:
    core = MambaTemporalCore(4, d_state=3, d_conv=2, expand=1, backend="torch").eval()
    features = torch.randn(2, 5, 4)
    unrolled = core.unroll(features, burn_in=0)
    state = core.initial_state(2, torch.device("cpu"))
    outputs = []
    for step in features.unbind(dim=1):
        output, state = core.step(step, state)
        outputs.append(output)
    torch.testing.assert_close(torch.stack(outputs, dim=1), unrolled)


def _assert_native_probe_preserves_rng_and_parameter_gradients() -> None:
    core = FunctionalNativeMamba(4, d_state=3, expand=1, backend="auto")
    torch.manual_seed(19)
    rng_state = torch.random.get_rng_state().clone()

    core.resolve_backend(torch.device("cpu"))

    assert core.resolved_backend == "native"
    assert core.fallback_reason is None
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert all(parameter.grad is None for parameter in core.parameters())


def test_mamba_execution_matches_streaming_without_probe_side_effects() -> None:
    _assert_torch_unroll_matches_streaming_step()
    _assert_native_probe_preserves_rng_and_parameter_gradients()


def _assert_auto_falls_back_from_non_finite_native_probe(
    core_type: type[MambaTemporalCore], reason: str
) -> None:
    core = core_type(4, d_state=3, expand=1, backend="auto")
    torch.manual_seed(23)
    rng_state = torch.random.get_rng_state().clone()

    core.resolve_backend(torch.device("cpu"))

    assert core.resolved_backend == "torch"
    assert core.fallback_reason is not None
    assert reason in core.fallback_reason
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert all(parameter.grad is None for parameter in core.parameters())


def _assert_non_finite_native_probes_fall_back() -> None:
    cases = (
        (NonFiniteOutputMamba, "non-finite probe output"),
        (NonFiniteGradientMamba, "non-finite probe gradients"),
    )
    for core_type, reason in cases:
        _assert_auto_falls_back_from_non_finite_native_probe(core_type, reason)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native Mamba requires CUDA")
def test_mamba_native_cuda_matches_torch_gradients_and_checkpoint() -> None:
    pytest.importorskip("mamba_ssm.ops.selective_scan_interface")
    device = torch.device("cuda")
    portable = _cuda_core(device, "torch")
    native = _cuda_core(device, "native")
    native.load_state_dict(portable.state_dict())
    native.resolve_backend(device)
    portable_inputs = torch.linspace(-1.0, 1.0, 80, device=device).reshape(2, 5, 8)
    native_inputs = portable_inputs.detach().clone().requires_grad_(True)
    portable_inputs = portable_inputs.requires_grad_(True)

    native_output, native_gradients = _mamba_gradients(native, native_inputs)
    portable_output, portable_gradients = _mamba_gradients(portable, portable_inputs)

    torch.testing.assert_close(native_output, portable_output, rtol=2e-4, atol=2e-5)
    _assert_gradients_match(native_gradients, portable_gradients)
    checkpoint = {name: value.detach().cpu() for name, value in native.state_dict().items()}
    restored = _restore_native(checkpoint, device)
    torch.testing.assert_close(restored.unroll(native_inputs.detach(), 0), native_output)


def _assert_auto_records_torch_fallback_without_fingerprint_change() -> None:
    automatic = FailingNativeMamba(4, d_state=3, expand=1, backend="auto")
    automatic.resolve_backend(torch.device("cpu"))
    pure = FailingNativeMamba(4, d_state=3, expand=1, backend="torch")
    pure.load_state_dict(automatic.state_dict())
    assert automatic.resolved_backend == "torch"
    assert automatic.fallback_reason == "ImportError: native kernel unavailable in test"
    assert automatic.state_dict().keys() == pure.state_dict().keys()
    automatic_model = _value_model(automatic)
    pure_model = _value_model(pure)
    assert automatic_model.architecture_fingerprint() == pure_model.architecture_fingerprint()


def test_mamba_auto_fallback_contracts() -> None:
    _assert_non_finite_native_probes_fall_back()
    _assert_auto_records_torch_fallback_without_fingerprint_change()
