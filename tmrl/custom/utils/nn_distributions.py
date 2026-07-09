"""TanhNormal distribution and associated utilities for continuous SAC-style actors."""

from typing import cast

import torch
from torch.distributions import Distribution, Normal


class TanhNormal(Distribution):
    """Distribution of X ~ tanh(Z) where Z ~ N(mean, std).

    Reference:
        Adapted from https://github.com/vitchyr/rlkit
    """

    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        super().__init__(self.normal.batch_shape, self.normal.event_shape)

    def log_prob(self, x):
        """Calculate log probability of a value.

        Args:
            x: Value in tanh-transformed space.

        Returns:
            Log probability.
        """
        if hasattr(x, "pre_tanh_value"):
            pre_tanh_value = cast(torch.Tensor, x.pre_tanh_value)
        else:
            pre_tanh_value = (torch.log(1 + x + self.epsilon) - torch.log(1 - x + self.epsilon)) / 2
        assert x.dim() == 2, "x must be 2D"
        assert pre_tanh_value.dim() == 2, "pre_tanh_value must be 2D"
        return self.normal.log_prob(pre_tanh_value) - torch.log(1 - x * x + self.epsilon)

    def sample(self, sample_shape=None):
        """Sample from the distribution.

        Args:
            sample_shape: Shape of samples to draw.

        Returns:
            Sampled values.
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        z = self.normal.sample(sample_shape)
        out = torch.tanh(z)
        out.pre_tanh_value = z  # type: ignore[attr-defined]
        return out

    def rsample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size()
        z = self.normal.rsample(sample_shape)
        out = torch.tanh(z)
        out.pre_tanh_value = z  # type: ignore[attr-defined]
        return out


class Independent(torch.distributions.Independent):
    def sample_test(self):
        return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()

        self.lin_mean = torch.nn.Linear(n, m)

        self.lin_std = torch.nn.Linear(n, m)
        self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
        self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        mean = self.lin_mean(x)
        log_std = self.lin_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        a = Independent(TanhNormal(mean, std), 1)
        return a
