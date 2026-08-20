"""Mamba-1 temporal core with native and portable selective-scan backends."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

import torch
from torch import nn
from torch.nn import functional as F

from trackmaniarl.core.pytree import PyTree
from trackmaniarl.models.temporal.selective_scan import selective_scan_torch

MambaBackend = Literal["auto", "native", "torch"]


class MambaTemporalCore(nn.Module):
    """A parameter-portable Mamba block whose scan backend is runtime-selectable."""

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        backend: MambaBackend = "auto",
    ) -> None:
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        if min(input_dim, hidden_dim, d_state, d_conv, expand) < 1:
            raise ValueError("Mamba dimensions must be positive")
        if backend not in {"auto", "native", "torch"}:
            raise ValueError("Mamba backend must be auto, native, or torch")
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.inner_dim = hidden_dim * expand
        self.requested_backend = backend
        self.resolved_backend = "torch"
        self.fallback_reason: str | None = None
        rank = max(1, math.ceil(input_dim / 16))
        self.input_projection = nn.Linear(input_dim, 2 * self.inner_dim)
        self.convolution = nn.Conv1d(
            self.inner_dim,
            self.inner_dim,
            d_conv,
            padding=d_conv - 1,
            groups=self.inner_dim,
        )
        self.parameter_projection = nn.Linear(self.inner_dim, rank + 2 * d_state, bias=False)
        self.delta_projection = nn.Linear(rank, self.inner_dim)
        self.log_state_matrix = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float()).repeat(self.inner_dim, 1)
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
            native = self._native_scan()
            probe = torch.randn(1, 2, self.inner_dim, device=device, requires_grad=True)
            delta = torch.ones_like(probe)
            matrix = -self.log_state_matrix.exp().to(device)
            bc = torch.ones(1, 2, self.d_state, device=device)
            result = native(probe, delta, matrix, bc, bc, self.skip.to(device))
            result.sum().backward()
        except (ImportError, RuntimeError, TypeError, AttributeError) as exc:
            if self.requested_backend == "native":
                raise RuntimeError(f"native Mamba backend is unavailable: {exc}") from exc
            self.resolved_backend = "torch"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return
        self.resolved_backend = "native"
        self.fallback_reason = None

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
        parameters = self.parameter_projection(F.silu(convolved))
        rank = self.delta_projection.in_features
        delta_raw, input_matrix, output_matrix = torch.split(
            parameters, [rank, self.d_state, self.d_state], dim=-1
        )
        deltas = F.softplus(self.delta_projection(delta_raw))
        state_matrix = -self.log_state_matrix.exp()
        initial_ssm = None if state is None else state[1]
        scanned, next_ssm = self._scan(
            F.silu(convolved),
            deltas,
            state_matrix,
            input_matrix,
            output_matrix,
            initial_ssm,
        )
        output = self.output_projection(scanned * F.silu(gate))
        return output, (next_convolution, next_ssm)

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

    def _scan(
        self,
        inputs: torch.Tensor,
        deltas: torch.Tensor,
        state_matrix: torch.Tensor,
        input_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resolved_backend == "native" and initial_state is None:
            native = self._native_scan()
            values = native(inputs, deltas, state_matrix, input_matrix, output_matrix, self.skip)
            _, final_state = selective_scan_torch(
                inputs,
                deltas,
                state_matrix,
                input_matrix,
                output_matrix,
                self.skip,
            )
            return values, final_state
        return selective_scan_torch(
            inputs,
            deltas,
            state_matrix,
            input_matrix,
            output_matrix,
            self.skip,
            initial_state=initial_state,
        )

    @staticmethod
    def _native_scan() -> Any:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

        def scan(
            inputs: torch.Tensor,
            deltas: torch.Tensor,
            state_matrix: torch.Tensor,
            input_matrix: torch.Tensor,
            output_matrix: torch.Tensor,
            skip: torch.Tensor,
        ) -> torch.Tensor:
            result = selective_scan_fn(
                inputs.transpose(1, 2),
                deltas.transpose(1, 2),
                state_matrix,
                input_matrix.transpose(1, 2),
                output_matrix.transpose(1, 2),
                skip,
                delta_softplus=False,
            )
            return cast(torch.Tensor, result).transpose(1, 2)

        return scan

    def _validate(self, features: torch.Tensor, burn_in: int) -> None:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError("Mamba features must have shape (batch, time, input_dim)")
        if not 0 <= burn_in < features.shape[1]:
            raise ValueError("burn_in must be in [0, time)")
