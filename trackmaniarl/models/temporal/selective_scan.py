"""Portable selective state-space scan implemented with PyTorch operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SelectiveScanInput:
    inputs: torch.Tensor
    deltas: torch.Tensor
    state_matrix: torch.Tensor
    input_matrix: torch.Tensor
    output_matrix: torch.Tensor
    skip: torch.Tensor
    initial_state: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class _ScanExecution:
    scan: SelectiveScanInput
    state_matrix: torch.Tensor
    skip: torch.Tensor


def _scan_shape(scan: SelectiveScanInput) -> tuple[int, int, int, int]:
    if scan.inputs.shape != scan.deltas.shape or scan.inputs.ndim != 3:
        raise ValueError("selective scan inputs and deltas must share (batch, time, channels)")
    batch, time, channels = scan.inputs.shape
    state_count = scan.state_matrix.shape[-1]
    expected = (batch, time, state_count)
    if scan.input_matrix.shape != expected or scan.output_matrix.shape != expected:
        raise ValueError("selective scan B and C must have shape (batch, time, state)")
    return batch, time, channels, state_count


def _initial_state(scan: SelectiveScanInput, shape: tuple[int, int, int, int]) -> torch.Tensor:
    if scan.initial_state is not None:
        return scan.initial_state
    batch, _, channels, state_count = shape
    return torch.zeros(
        batch,
        channels,
        state_count,
        device=scan.inputs.device,
        dtype=scan.inputs.dtype,
    )


def _scan_step(
    execution: _ScanExecution, index: int, state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scan = execution.scan
    delta = scan.deltas[:, index].unsqueeze(-1)
    transition = torch.exp(delta * execution.state_matrix)
    injected = delta * scan.input_matrix[:, index].unsqueeze(1)
    injected = injected * scan.inputs[:, index].unsqueeze(-1)
    next_state = transition * state + injected
    value = (next_state * scan.output_matrix[:, index].unsqueeze(1)).sum(dim=-1)
    return next_state, value + execution.skip * scan.inputs[:, index]


def selective_scan_torch(scan: SelectiveScanInput) -> tuple[torch.Tensor, torch.Tensor]:
    shape = _scan_shape(scan)
    state = _initial_state(scan, shape)
    execution = _ScanExecution(
        scan,
        scan.state_matrix.to(device=scan.inputs.device, dtype=scan.inputs.dtype),
        scan.skip.to(device=scan.inputs.device, dtype=scan.inputs.dtype),
    )
    outputs: list[torch.Tensor] = []
    for index in range(shape[1]):
        state, value = _scan_step(execution, index, state)
        outputs.append(value)
    return torch.stack(outputs, dim=1), state
