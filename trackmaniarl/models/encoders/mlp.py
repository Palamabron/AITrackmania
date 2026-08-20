"""Generic MLP sensor encoder."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class MlpSensorEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 2 or frames.shape[-1] != self.input_dim:
            raise ValueError("MLP frames must have shape (frames, input_dim)")
        return cast(torch.Tensor, self.network(frames))
