"""Linear layer variants, conv factory functions, and gSDE exploration module."""

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.init import kaiming_uniform_

from tmrl.util import partial


class RlkitLinear(torch.nn.Linear):
    """Linear layer with RLKit-style initialization.

    Note:
        This implementation follows the original RLKit weight initialization, which
        uses fan_out (weight.shape[0]) instead of the more conventional fan_in for
        computing the uniform bound. PyTorch Linear stores weights as (out_features, in_features),
        so weight.shape[0] = out_features = fan_out.

        Reference: https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/pytorch_util.py
    """

    def __init__(self, *args):
        super().__init__(*args)
        fan_in = self.weight.shape[0]
        bound = 1.0 / np.sqrt(fan_in)
        self.weight.data.uniform_(-bound, bound)
        self.bias.data.fill_(0.1)


class SacLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            self.weight.uniform_(-0.06, 0.06)  # 0.06 == 1 / sqrt(256)
            self.bias.fill_(0.1)


class BasicReLU(torch.nn.Linear):
    def forward(self, x):
        x = super().forward(x)
        return torch.relu(x)


class AffineReLU(BasicReLU):
    def __init__(
        self, in_features, out_features, init_weight_bound: float = 1.0, init_bias: float = 0.0
    ):
        super().__init__(in_features, out_features)
        bound = init_weight_bound / np.sqrt(in_features)
        self.weight.data.uniform_(-bound, bound)
        self.bias.data.fill_(init_bias)


class NormalizedReLU(torch.nn.Sequential):
    def __init__(self, in_features, out_features, prenorm_bias=True):
        super().__init__(
            torch.nn.Linear(in_features, out_features, bias=prenorm_bias),
            torch.nn.LayerNorm(out_features),
            torch.nn.ReLU(),
        )


class KaimingReLU(torch.nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        with torch.no_grad():
            kaiming_uniform_(self.weight)
            self.bias.fill_(0.0)

    def forward(self, x):
        x = super().forward(x)
        return torch.relu(x)


Linear10 = partial(AffineReLU, init_bias=1.0)
Linear04 = partial(AffineReLU, init_bias=0.4)
LinearConstBias = partial(AffineReLU, init_bias=0.1)
LinearZeroBias = partial(AffineReLU, init_bias=0.0)
AffineSimon = partial(AffineReLU, init_weight_bound=0.01, init_bias=1.0)


def dqn_conv(n):
    """Create a DQN-style convolutional network.

    Args:
        n: Number of input channels.

    Returns:
        Sequential CNN module.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(n, 32, kernel_size=8, stride=4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
        torch.nn.ReLU(),
    )


def big_conv(n):
    """Create a larger convolutional network.

    Args:
        n: Number of input channels.

    Returns:
        Sequential CNN module (e.g., 64x256 input -> 2x26 output).
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(n, 64, 8, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(128, 128, 4, stride=1),
        torch.nn.LeakyReLU(),
    )


def hd_conv(n):
    """Create a deeper convolutional network for high-resolution inputs.

    Args:
        n: Number of input channels.

    Returns:
        Sequential CNN module with additional downsampling layers.
    """
    return torch.nn.Sequential(
        torch.nn.Conv2d(n, 32, 8, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(32, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 64, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(64, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
        torch.nn.Conv2d(128, 128, 4, stride=2),
        torch.nn.LeakyReLU(),
    )


GSDE_LOG_STD_MIN = -3.0
GSDE_LOG_STD_MAX = 2.0


class GSDEModule(nn.Module):
    """Generalized State-Dependent Exploration (gSDE) noise module.

    Implements the noise exploration matrix from the gSDE paper
    (https://arxiv.org/abs/2005.05719). Noise is correlated with
    latent features so similar states get similar exploration patterns.

    Args:
        latent_dim: Dimension of the latent features (backbone output).
        action_dim: Dimension of the continuous action space.
        log_std_init: Initial value for the log standard deviation parameter.
        full_std: If True, use (latent_dim x action_dim) parameters for std.
            If False, use (latent_dim x 1) and broadcast.

    Note:
        Log-std is clamped to [GSDE_LOG_STD_MIN, GSDE_LOG_STD_MAX] = [-3.0, 2.0]
        to prevent entropy collapse. The lower bound of -3 ensures std >= exp(-3) ~ 0.05,
        maintaining meaningful exploration. See model_constants.LOG_STD_MIN for context.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        log_std_init: float = -3.0,
        full_std: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.full_std = full_std
        self.epsilon = 1e-6

        if full_std:
            self.log_std = nn.Parameter(torch.ones(latent_dim, action_dim) * log_std_init)
        else:
            self.log_std = nn.Parameter(torch.ones(latent_dim, 1) * log_std_init)

        self.register_buffer(
            "exploration_mat", torch.zeros(latent_dim, action_dim), persistent=False
        )
        self.register_buffer(
            "exploration_matrices", torch.zeros(1, latent_dim, action_dim), persistent=False
        )
        self.reset_noise()

    def get_std(self) -> torch.Tensor:
        log_std = torch.clamp(self.log_std, GSDE_LOG_STD_MIN, GSDE_LOG_STD_MAX)
        std = torch.exp(log_std)
        if self.full_std:
            return std
        return torch.ones(self.latent_dim, self.action_dim, device=std.device) * std

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new exploration matrix weights from N(0, std).

        Args:
            batch_size: Number of exploration matrices to sample for batch processing.

        Note:
            Samples are detached to make them graph leaves (deepcopy-safe).
            Log_std gradients flow through get_variance(), not through noise samples.
        """
        std = self.get_std()
        weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = weights_dist.rsample().detach()
        self.exploration_matrices = weights_dist.rsample((batch_size,)).detach()

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent noise: latent @ exploration_matrix.

        Args:
            latent_sde: Latent features (batch, latent_dim).

        Returns:
            State-dependent noise (batch, action_dim).
        """
        if len(latent_sde) == 1 or len(latent_sde) != len(self.exploration_matrices):
            return latent_sde @ self.exploration_mat
        latent_sde_3d = latent_sde.unsqueeze(1)
        noise = torch.bmm(latent_sde_3d, self.exploration_matrices)
        return cast(torch.Tensor, noise.squeeze(1))

    def get_variance(self, latent_sde: torch.Tensor) -> torch.Tensor:
        """Compute action variance: latent^2 @ std^2.

        Args:
            latent_sde: Latent features (batch, latent_dim).

        Returns:
            Action variance (batch, action_dim).
        """
        std = self.get_std()
        return cast(torch.Tensor, (latent_sde**2) @ (std**2))
