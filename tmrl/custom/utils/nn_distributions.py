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
        """Initialize TanhNormal distribution.

        Args:
            normal_mean: Mean of the underlying Normal; shape (batch, action_dim).
            normal_std: Standard deviation of the underlying Normal; shape
                (batch, action_dim). Must be positive.
            epsilon: Small constant added inside log() calls for numerical
                stability near the tanh saturation boundaries (|x| -> 1).
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        super().__init__(self.normal.batch_shape, self.normal.event_shape)

    def log_prob(self, x):
        """Compute element-wise log probability under the TanhNormal distribution.

        Applies the change-of-variables correction for the tanh bijection::

            log p(x) = log p_Normal(atanh(x)) - log(1 - x^2)

        where ``-log(1 - x^2)`` is the log absolute Jacobian determinant of the
        tanh transform.  ``epsilon`` is added inside every log() call to keep the
        computation finite when x is close to +-1 (tanh saturation).

        When x carries a ``pre_tanh_value`` attribute (attached by ``sample`` and
        ``rsample``), that stored pre-activation value is reused directly to avoid
        computing atanh numerically.  Otherwise atanh is recovered in closed form::

            atanh(x) = (log(1 + x) - log(1 - x)) / 2

        Args:
            x: Sampled actions in tanh space; shape (batch, action_dim).

        Returns:
            Element-wise log probabilities; shape (batch, action_dim).
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
        """Reparameterized sample from the distribution (gradients flow through the sample).

        Args:
            sample_shape: Shape of samples to draw; defaults to ``torch.Size()``.

        Returns:
            Sampled actions in tanh space with a ``pre_tanh_value`` attribute
            holding the pre-activation sample, reusable in ``log_prob`` to
            avoid a redundant atanh computation.
        """
        if sample_shape is None:
            sample_shape = torch.Size()
        z = self.normal.rsample(sample_shape)
        out = torch.tanh(z)
        out.pre_tanh_value = z  # type: ignore[attr-defined]
        return out


class Independent(torch.distributions.Independent):
    """Independent wrapper that adds a deterministic test-time action.

    Extends ``torch.distributions.Independent`` with ``sample_test``, which
    returns the tanh of the underlying TanhNormal's mean — the mode of the
    distribution — for use during evaluation (no stochastic exploration noise).
    """

    def sample_test(self):
        """Return the deterministic test-time action: tanh of the distribution mean.

        Returns:
            Action tensor equal to ``tanh(base_dist.normal_mean)``; shape
            matches the action dimension of the underlying TanhNormal.
        """
        return torch.tanh(self.base_dist.normal_mean)


class TanhNormalLayer(torch.nn.Module):
    """Output head that maps latent features to a TanhNormal action distribution.

    Two linear projections produce the mean and log-std of an underlying Normal.
    The log-std output is clamped to [-20, 2] before exponentiation:

    - Lower bound -20 prevents std from collapsing to zero (exp(-20) ≈ 2e-9).
    - Upper bound 2 prevents unbounded variance (exp(2) ≈ 7.4).

    The std head's weights and biases are initialized near zero (-1e-3, 1e-3)
    so that the initial policy has approximately unit variance (log_std ≈ 0
    → std ≈ 1) regardless of input state.

    Args:
        n: Input feature dimension.
        m: Action dimension (output dimension for both mean and std projections).
    """

    def __init__(self, n, m):
        super().__init__()

        self.lin_mean = torch.nn.Linear(n, m)

        self.lin_std = torch.nn.Linear(n, m)
        self.lin_std.weight.data.uniform_(-1e-3, 1e-3)
        self.lin_std.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, x):
        """Map latent features to an action distribution.

        Args:
            x: Input tensor; shape (batch, n).

        Returns:
            An ``Independent(TanhNormal(...), 1)`` distribution over actions of
            dimension m.  Call ``.rsample()`` for a reparameterized sample or
            ``.sample_test()`` for the deterministic (mean) action.
        """
        mean = self.lin_mean(x)
        log_std = self.lin_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        a = Independent(TanhNormal(mean, std), 1)
        return a
