"""Generic image-like convolutional sensor encoder."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ConvolutionalSensorEncoder(nn.Module):
    def __init__(self, channels: int, output_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if min(channels, output_dim, hidden_dim) < 1:
            raise ValueError("encoder dimensions must be positive")
        self.channels = channels
        self.output_dim = output_dim
        self.convolution = nn.Sequential(
            nn.Conv2d(channels, hidden_dim // 2, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[1] != self.channels:
            raise ValueError("convolutional frames must have shape (frames, channels, H, W)")
        encoded = self.convolution(frames.float()).flatten(1)
        return cast(torch.Tensor, self.projection(encoded))
