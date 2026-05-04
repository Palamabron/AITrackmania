from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Distribution, Normal
from torch.nn import Module
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

from tmrl.util import partial


def detach(x):
    """Detach tensor from computation graph, or recursively detach elements.

    Args:
        x: A torch.Tensor or iterable of tensors/objects.

    Returns:
        Detached tensor if input is a tensor, otherwise a list of detached elements.
    """
    if isinstance(x, torch.Tensor):
        return x.detach()
    else:
        return [detach(elem) for elem in x]


def no_grad(model):
    """Set requires_grad=False for all parameters in the model.

    Args:
        model: A torch.nn.Module.

    Returns:
        The same model instance with all parameters frozen (requires_grad=False).
    """
    for p in model.parameters():
        p.requires_grad = False
    return model


def exponential_moving_average(averages, values, factor):
    """Update averages in-place using exponential moving average.

    Args:
        averages: Iterable of tensors to update (modified in-place).
        values: Iterable of current values.
        factor: EMA blending factor (0-1). Higher values weight new values more.
    """
    with torch.no_grad():
        for a, v in zip(averages, values, strict=True):
            a += factor * (v - a)


def copy_shared(model_a):
    """Create a deepcopy of a model with shared parameter storage.

    The copied model shares the underlying parameter tensors with the original,
    so updates to one affect the other. Useful for creating no-grad evaluation
    copies that stay in sync with a training model.

    Args:
        model_a: Model to copy.

    Returns:
        A deepcopy of the model with state_dict storage shared with the original.

    Note:
        torch.cuda.Stream cannot be pickled/deepcopied, so streams are replaced
        with None during copy and will be recreated on first use if needed.
    """
    import copy as copy_module

    stream_type = getattr(getattr(torch, "cuda", None), "Stream", None)
    dispatch = cast(
        MutableMapping[type, Any] | None, getattr(copy_module, "_deepcopy_dispatch", None)
    )
    old_dispatch = None
    if stream_type is not None and dispatch is not None:
        old_dispatch = dispatch.get(stream_type)
        dispatch[stream_type] = lambda obj, memo: None
    try:
        model_b = deepcopy(model_a)
    finally:
        if stream_type is not None and dispatch is not None:
            if old_dispatch is not None:
                dispatch[stream_type] = old_dispatch
            else:
                dispatch.pop(stream_type, None)
    sda = model_a.state_dict(keep_vars=True)
    sdb = model_b.state_dict(keep_vars=True)
    for key in sda:
        a, b = sda[key], sdb[key]
        b.data = a.data  # a.data and b.data differ but underlying data_ptr is shared
        assert b.untyped_storage().data_ptr() == a.untyped_storage().data_ptr()
    return model_b


class PopArt(Module):
    """PopArt normalization for value functions.

    Adaptively normalizes targets and rescales output layer weights to handle
    rewards spanning many orders of magnitude.

    Reference:
        http://papers.nips.cc/paper/6076-learning-values-across-many-orders-of-magnitude

    Args:
        output_layer: Linear layer(s) to adapt. Can be a single layer or tuple/list.
        beta: EMA update rate for statistics.
        zero_debias: If True, uses bias-corrected EMA (recommended).
        start_pop: Delay weight rescaling for this many updates to accumulate stable stats.
    """

    def __init__(
        self, output_layer, beta: float = 0.0003, zero_debias: bool = True, start_pop: int = 8
    ):
        super().__init__()
        self.start_pop = start_pop
        self.beta = beta
        self.zero_debias = zero_debias
        self.output_layers = (
            output_layer
            if isinstance(output_layer, (tuple, list, torch.nn.ModuleList))
            else (output_layer,)
        )
        layer0 = self.output_layers[0]
        shape = tuple(layer0.bias.shape)  # type: ignore[arg-type,union-attr]
        device = layer0.bias.device  # type: ignore[union-attr]
        assert all(shape == tuple(x.bias.shape) for x in self.output_layers)  # type: ignore[arg-type]
        self.mean = Parameter(torch.zeros(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.mean_square = Parameter(torch.ones(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.std = Parameter(torch.ones(shape, device=device), requires_grad=False)  # type: ignore[arg-type]
        self.updates = 0

    @torch.no_grad()
    def update(self, targets):
        """Update statistics and rescale output layer, then return normalized targets.

        Args:
            targets: Target values to incorporate into statistics.

        Returns:
            Normalized targets.
        """
        beta = max(1 / (self.updates + 1), self.beta) if self.zero_debias else self.beta

        new_mean = (1 - beta) * self.mean + beta * targets.mean(0)
        new_mean_square = (1 - beta) * self.mean_square + beta * (targets * targets).mean(0)
        new_std = (new_mean_square - new_mean * new_mean).sqrt().clamp(0.0001, 1e6)

        if self.updates >= self.start_pop:
            for layer in self.output_layers:
                layer.weight *= (self.std / new_std)[:, None]  # type: ignore[operator]
                layer.bias *= self.std  # type: ignore[operator]
                layer.bias += self.mean - new_mean
                layer.bias /= new_std

        self.mean.copy_(new_mean)
        self.mean_square.copy_(new_mean_square)
        self.std.copy_(new_std)
        self.updates += 1
        return self.normalize(targets)

    def normalize(self, x):
        """Normalize input using current statistics.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return (x - self.mean) / self.std

    def unnormalize(self, x):
        """Inverse of normalize - convert from normalized to original scale.

        Args:
            x: Normalized tensor.

        Returns:
            Unnormalized tensor.
        """
        return x * self.std + self.mean

    def normalize_sum(self, s):
        """Normalize sum while preserving relative weightings.

        Args:
            s: Sum of values.

        Returns:
            Normalized sum.
        """
        return (s - self.mean.sum()) / self.std.norm()


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
        # a = TanhTransformedDist(Independent(Normal(m, std), 1))
        a = Independent(TanhNormal(mean, std), 1)
        return a


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
