"""Portable selective state-space scan implemented with PyTorch operations."""

from __future__ import annotations

import torch


def selective_scan_torch(
    inputs: torch.Tensor,
    deltas: torch.Tensor,
    state_matrix: torch.Tensor,
    input_matrix: torch.Tensor,
    output_matrix: torch.Tensor,
    skip: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a Mamba-1 selective scan over ``(batch, time, channels)`` inputs."""

    if inputs.shape != deltas.shape or inputs.ndim != 3:
        raise ValueError("selective scan inputs and deltas must share (batch, time, channels)")
    batch, time, channels = inputs.shape
    state_count = state_matrix.shape[-1]
    expected_bc = (batch, time, state_count)
    if input_matrix.shape != expected_bc or output_matrix.shape != expected_bc:
        raise ValueError("selective scan B and C must have shape (batch, time, state)")
    state = (
        torch.zeros(
            batch,
            channels,
            state_count,
            device=inputs.device,
            dtype=inputs.dtype,
        )
        if initial_state is None
        else initial_state
    )
    outputs: list[torch.Tensor] = []
    matrix = state_matrix.to(device=inputs.device, dtype=inputs.dtype)
    skip = skip.to(device=inputs.device, dtype=inputs.dtype)
    for index in range(time):
        delta = deltas[:, index].unsqueeze(-1)
        transition = torch.exp(delta * matrix)
        injected = delta * input_matrix[:, index].unsqueeze(1) * inputs[:, index].unsqueeze(-1)
        state = transition * state + injected
        value = (state * output_matrix[:, index].unsqueeze(1)).sum(dim=-1)
        outputs.append(value + skip * inputs[:, index])
    return torch.stack(outputs, dim=1), state
