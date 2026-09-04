"""Mamba-1 temporal core with native and portable selective-scan backends."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, TypedDict, Unpack, cast

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.core.pytree import PyTree
from trackmaniarl.models.temporal.selective_scan import SelectiveScanInput, selective_scan_torch

MambaBackend = Literal["auto", "native", "torch"]
ScanFunction = Callable[[SelectiveScanInput], torch.Tensor]


class _MambaKwargs(TypedDict, total=False):
    hidden_dim: int | None
    d_state: int
    d_conv: int
    expand: int
    backend: MambaBackend


@dataclass(frozen=True, slots=True)
class MambaOptions:
    hidden_dim: int | None = None
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    backend: MambaBackend = "auto"


def _validate_options(input_dim: int, hidden_dim: int, options: MambaOptions) -> None:
    dimensions = (input_dim, hidden_dim, options.d_state, options.d_conv, options.expand)
    if min(dimensions) < 1:
        raise ValueError("Mamba dimensions must be positive")
    if options.backend not in {"auto", "native", "torch"}:
        raise ValueError("Mamba backend must be auto, native, or torch")


def _scan_tensors(scan: SelectiveScanInput) -> tuple[torch.Tensor, ...]:
    return (
        scan.inputs,
        scan.deltas,
        scan.state_matrix,
        scan.input_matrix,
        scan.output_matrix,
        scan.skip,
    )


def _native_selective_scan(native: Any, scan: SelectiveScanInput) -> torch.Tensor:
    result = native(
        scan.inputs.transpose(1, 2),
        scan.deltas.transpose(1, 2),
        scan.state_matrix,
        scan.input_matrix.transpose(1, 2),
        scan.output_matrix.transpose(1, 2),
        scan.skip,
        delta_softplus=False,
    )
    return cast(torch.Tensor, result).transpose(1, 2)


class MambaTemporalCore(nn.Module):
    """A parameter-portable Mamba block whose scan backend is runtime-selectable."""

    fingerprint_ignored_parameters = frozenset({"backend"})

    def __init__(
        self,
        input_dim: int,
        **kwargs: Unpack[_MambaKwargs],
    ) -> None:
        super().__init__()
        options = MambaOptions(**kwargs)
        hidden_dim = input_dim if options.hidden_dim is None else options.hidden_dim
        _validate_options(input_dim, hidden_dim, options)
        self._configure_dimensions(input_dim, hidden_dim, options)
        self._initialize_layers(input_dim, hidden_dim, max(1, math.ceil(input_dim / 16)))

    def _configure_dimensions(self, input_dim: int, hidden_dim: int, options: MambaOptions) -> None:
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.d_state = options.d_state
        self.d_conv = options.d_conv
        self.inner_dim = hidden_dim * options.expand
        self.requested_backend = options.backend
        self.resolved_backend = "torch"
        self.fallback_reason: str | None = None

    def _initialize_layers(self, input_dim: int, hidden_dim: int, rank: int) -> None:
        self.input_projection = nn.Linear(input_dim, 2 * self.inner_dim)
        self.convolution = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            self.d_conv,
            padding=self.d_conv - 1,
            groups=self.inner_dim,
        )
        self.parameter_projection = nn.Linear(self.inner_dim, rank + 2 * self.d_state, bias=False)
        self.delta_projection = nn.Linear(rank, self.inner_dim)
        self.log_state_matrix = nn.Parameter(
            torch.log(torch.arange(1, self.d_state + 1).float()).repeat(self.inner_dim, 1)
        )
        self.skip = nn.Parameter(torch.ones(self.inner_dim))
        self.output_projection = nn.Linear(self.inner_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)

    def resolve_backend(self, device: torch.device) -> None:
        if self.requested_backend == "torch":
            self.resolved_backend = "torch"
            self.fallback_reason = None
            return
        try:
            self._probe_native_backend(self._native_scan(), device)
        except (ImportError, RuntimeError, TypeError, AttributeError) as exc:
            if self.requested_backend == "native":
                raise RuntimeError(f"native Mamba backend is unavailable: {exc}") from exc
            self.resolved_backend = "torch"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return
        self.resolved_backend = "native"
        self.fallback_reason = None

    def _probe_native_backend(self, native: ScanFunction, device: torch.device) -> None:
        scan = self._probe_input(device)
        result = native(scan)
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("native Mamba backend produced non-finite probe output")
        gradients = torch.autograd.grad(result.sum(), _scan_tensors(scan))
        if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("native Mamba backend produced non-finite probe gradients")

    def _probe_input(self, device: torch.device) -> SelectiveScanInput:
        dtype = self.skip.dtype
        inputs = torch.linspace(-0.5, 0.5, 2 * self.inner_dim, device=device, dtype=dtype).reshape(
            1, 2, self.inner_dim
        )
        inputs.requires_grad_(True)
        deltas = torch.full_like(inputs, 0.5, requires_grad=True)
        matrices = self._probe_matrices(device, dtype)
        return SelectiveScanInput(inputs, deltas, *matrices)

    def _probe_matrices(
        self, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_matrix = self.log_state_matrix.detach().clone().to(device=device)
        state_matrix = (-state_matrix.exp()).requires_grad_(True)
        input_matrix = torch.full(
            (1, 2, self.d_state),
            0.25,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        output_matrix = torch.full_like(input_matrix, 0.5, requires_grad=True)
        skip = self.skip.detach().clone().to(device=device).requires_grad_(True)
        return state_matrix, input_matrix, output_matrix, skip

    def unroll(self, features: torch.Tensor, burn_in: int) -> torch.Tensor:
        self._validate(features, burn_in)
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        if burn_in:
            with torch.no_grad():
                _, state = self._forward(features[:, :burn_in], None)
            state = (state[0].detach(), state[1].detach())
        values, _ = self._forward(features[:, burn_in:], state)
        return cast(torch.Tensor, self.normalization(values))

    def initial_state(self, batch_size: int, device: torch.device) -> PyTree:
        convolution = torch.zeros(batch_size, self.inner_dim, self.d_conv - 1, device=device)
        ssm = torch.zeros(batch_size, self.inner_dim, self.d_state, device=device)
        return convolution, ssm

    def step(self, feature: torch.Tensor, state: PyTree) -> tuple[torch.Tensor, PyTree]:
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError("Mamba state must contain convolution and SSM tensors")
        convolution, ssm = state
        if not isinstance(convolution, torch.Tensor) or not isinstance(ssm, torch.Tensor):
            raise TypeError("Mamba state entries must be tensors")
        value, next_state = self._forward(feature.unsqueeze(1), (convolution, ssm))
        return cast(torch.Tensor, self.normalization(value[:, 0])), next_state

    def execution_manifest(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "fallback_reason": self.fallback_reason,
        }

    def _forward(
        self,
        features: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        projected = self.input_projection(features)
        inputs, gate = projected.chunk(2, dim=-1)
        convolution_state = None if state is None else state[0]
        convolved, next_convolution = self._causal_convolution(inputs, convolution_state)
        scan = self._scan_input(convolved, None if state is None else state[1])
        scanned, next_ssm = self._scan(scan)
        output = self.output_projection(scanned * F.silu(gate))
        return output, (next_convolution, next_ssm)

    def _scan_input(
        self, convolved: torch.Tensor, initial_state: torch.Tensor | None
    ) -> SelectiveScanInput:
        parameters = self.parameter_projection(F.silu(convolved))
        rank = self.delta_projection.in_features
        delta_raw, input_matrix, output_matrix = torch.split(
            parameters, [rank, self.d_state, self.d_state], dim=-1
        )
        deltas = F.softplus(self.delta_projection(delta_raw))
        state_matrix = -self.log_state_matrix.exp()
        return SelectiveScanInput(
            F.silu(convolved),
            deltas,
            state_matrix,
            input_matrix,
            output_matrix,
            self.skip,
            initial_state,
        )

    def _causal_convolution(
        self, inputs: torch.Tensor, state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = (
            torch.zeros(
                inputs.shape[0],
                self.inner_dim,
                self.d_conv - 1,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            if state is None
            else state
        )
        sequence = torch.cat([prefix, inputs.transpose(1, 2)], dim=-1)
        weight = self.convolution.weight
        values = F.conv1d(sequence, weight, self.convolution.bias, groups=self.inner_dim)
        next_state = sequence[..., -self.d_conv + 1 :] if self.d_conv > 1 else sequence[..., :0]
        return values.transpose(1, 2), next_state

    def _scan(self, scan: SelectiveScanInput) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resolved_backend == "native" and scan.initial_state is None:
            values = self._native_scan()(scan)
            _, final_state = selective_scan_torch(scan)
            return values, final_state
        return selective_scan_torch(scan)

    @staticmethod
    def _native_scan() -> ScanFunction:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        return cast(ScanFunction, partial(_native_selective_scan, selective_scan_fn))

    def _validate(self, features: torch.Tensor, burn_in: int) -> None:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("Mamba features must have shape (batch, time, input_dim)")
        if not 0 <= burn_in < features.shape[1]:
            raise ValueError("burn_in must be in [0, time)")
