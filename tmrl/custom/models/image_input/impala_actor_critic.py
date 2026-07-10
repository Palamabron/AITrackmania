"""IMPALA QR-CNN actor-critic registered as ``impala_ac``.

Combines the IMPALA CNN encoder (``CNNModule``) with a two-branch LSTM architecture:
one branch processes flattened scalar observations via MLP + LSTM, the other fuses the
CNN embedding and optionally an action vector through a second LSTM, and finally
projects to quantile Q-values or a squashed Gaussian policy.
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.autograd import Variable
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.image_input._impala_utils import gru, init_kaiming, lstm, mlp  # noqa: F401
from tmrl.custom.models.image_input.impala_encoder import CNNModule
from tmrl.custom.models.shared.blocks import LOG_STD_MAX, LOG_STD_MIN
from tmrl.registry import MODELS


@MODELS.register("impala_qr_critic")
class QRCNNQFunction(nn.Module):
    """Quantile-regression Q-function built on an IMPALA CNN encoder.

    Architecture: CNN branch processes game frames; an MLP + LSTM branch processes
    flattened scalar observations; both are fused through a second LSTM before a
    linear head outputs ``quantiles_number`` quantile estimates per sample.

    Pairs with ``SquashedActorQRCNN`` and ``QRCNNActorCritic`` in the registry.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
        quantiles_number: int = 1,
        api_layernorm: bool = False,
        mlp_layernorm: bool = False,
        rnn_dropout: float = 0.0,
        noisy_linear_critic: bool = False,
        output_dropout: float = 0.0,
        grayscale: bool = False,
    ):
        """Construct the QR-CNN Q-function.

        The image observation is assumed to sit at index -3 of the observation tuple.
        Its flattened dimension is subtracted from ``dim_obs`` before the scalar branch.

        Args:
            observation_space: Gymnasium observation space; iterable of sub-spaces.
            action_space: Gymnasium action space; its shape determines the action dim.
            rnn_sizes: Hidden sizes for the two LSTM layers. Defaults to [64, 64].
            rnn_lens: Number of stacked layers for each LSTM. Defaults to [1, 1].
            mlp_branch_sizes: Hidden widths of the scalar MLP branch. Defaults to [256].
            activation: Activation class used in the MLP branch.
            seed: Random seed for reproducibility (also sets cuDNN flags).
            quantiles_number: Number of quantile outputs per sample.
            api_layernorm: Apply LayerNorm to concatenated scalar observations.
            mlp_layernorm: Apply LayerNorm after the scalar MLP branch.
            rnn_dropout: Dropout between LSTM layers in the fusion LSTM.
            noisy_linear_critic: Replace the output linear with NoisyLinear.
            output_dropout: Dropout rate applied to the final output.
            grayscale: If True, skip the (N,H,W,C) -> (N,C,H,W) permute.
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.grayscale = grayscale
        self.api_layernorm = api_layernorm
        self.mlp_layernorm = mlp_layernorm
        self.output_dropout_rate = output_dropout

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        self.num_quantiles = quantiles_number

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0] + dim_act,
            rnn_sizes[1],
            rnn_lens[1],
            dropout=rnn_dropout,
        )

        if noisy_linear_critic:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], self.num_quantiles)

        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)
        self.img_index = -3

    def forward(self, observation, act: torch.Tensor, save_hidden: bool = False) -> torch.Tensor:
        """Compute quantile Q-values for a batch of (observation, action) pairs.

        Args:
            observation: Observation tuple; index -3 must be the image tensor of shape
                (N, H, W, C) for colour or (N, C, H, W) when grayscale=True.
            act: Action tensor of shape (N, dim_act).
            save_hidden: If True, persist LSTM hidden states across calls for
                sequential rollout (worker side). If False, use zero initial states.

        Returns:
            Q-value tensor of shape (N,) or (N, quantiles_number) when
            quantiles_number > 1.
        """
        self.rnn_block_api.flatten_parameters()
        self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[-3].float()
        if batch_size == 1:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)
            conv_branch_out = self.cnn_module(cnn_branch_input)

        if (
            not save_hidden
            or self.h0 is None
            or self.c0 is None
            or self.h1 is None
            or self.c1 is None
        ):
            device = observation[0].device
            h0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            c0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            h1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
            c1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
        else:
            h0 = self.h0
            c0 = self.c0
            h1 = self.h1
            c1 = self.c1

        observation[-3] = conv_branch_out

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        observation_except_img = observation[: self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1) :]

        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self.api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self.mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out, act], dim=-1)

        _rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if self.output_dropout_rate > 0.0:
            model_out = self.dropout(model_out)

        if save_hidden:
            self.h0 = h0
            self.c0 = c0
            self.h1 = h1
            self.c1 = c1

        return torch.squeeze(model_out, -1)


@MODELS.register("impala_qr_actor")
class SquashedActorQRCNN(TorchActorModule):
    """Squashed Gaussian actor built on an IMPALA CNN encoder.

    Mirrors the two-branch LSTM architecture of ``QRCNNQFunction`` but outputs
    mean and log-std for a tanh-squashed Normal distribution instead of quantile values.
    Stateful hidden-state caching is supported for sequential rollout via ``act()``.

    Pairs with ``QRCNNQFunction`` inside ``QRCNNActorCritic``.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
        api_layernorm: bool = False,
        mlp_layernorm: bool = False,
        rnn_dropout: float = 0.0,
        noisy_linear_actor: bool = False,
        output_dropout: float = 0.0,
        grayscale: bool = False,
    ):
        """Construct the QRCNN actor.

        Args:
            observation_space: Gymnasium observation space; iterable of sub-spaces.
            action_space: Gymnasium action space; shape determines output dimension.
            rnn_sizes: Hidden sizes for the two LSTM layers. Defaults to [64, 64].
            rnn_lens: Number of stacked layers for each LSTM. Defaults to [1, 1].
            mlp_branch_sizes: Hidden widths of the scalar MLP branch. Defaults to [256].
            activation: Activation class used in the MLP branch.
            seed: Random seed for reproducibility (also sets cuDNN flags).
            api_layernorm: Apply LayerNorm to concatenated scalar observations.
            mlp_layernorm: Apply LayerNorm after the scalar MLP branch.
            rnn_dropout: Dropout between layers in the fusion LSTM.
            noisy_linear_actor: Replace the output linear with NoisyLinear.
            output_dropout: Dropout rate applied before the policy head.
            grayscale: If True, skip the (N,H,W,C) -> (N,C,H,W) permute.
        """
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.grayscale = grayscale
        self.api_layernorm = api_layernorm
        self.mlp_layernorm = mlp_layernorm
        self.output_dropout_rate = output_dropout

        self.cnn_module = CNNModule()
        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_obs -= math.prod(s for s in observation_space[-3].shape)
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.rnn_block_api = lstm(mlp_branch_sizes[-1], rnn_sizes[0], rnn_lens[0])

        self.rnn_block_cat = lstm(
            self.cnn_module.mlp_out_size + rnn_sizes[0],
            rnn_sizes[1],
            rnn_lens[1],
            dropout=rnn_dropout,
        )

        if noisy_linear_actor:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(rnn_sizes[0], mlp_out_size)

        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self.h0 = None
        self.h1 = None
        self.c0 = None
        self.c1 = None
        self.rnn_sizes = list(rnn_sizes)
        self.rnn_lens = list(rnn_lens)
        self.img_index = -3

    def forward(
        self, observation, test: bool = False, with_logprob: bool = True, save_hidden: bool = False
    ):
        """Sample an action from the squashed Gaussian policy.

        Args:
            observation: Observation tuple; index -3 must be the image tensor of shape
                (N, H, W, C) for colour or (N, C, H, W) when grayscale=True.
            test: If True, return the deterministic mean action instead of sampling.
            with_logprob: If True, compute and return the tanh-corrected log-probability.
            save_hidden: If True, persist LSTM hidden states for sequential rollout.

        Returns:
            Tuple of (action, logp_pi) where action has shape (dim_act,) after squeeze
            and logp_pi is a scalar tensor or None when with_logprob=False.
        """
        self.rnn_block_api.flatten_parameters()
        self.rnn_block_cat.flatten_parameters()

        batch_size = observation[0].shape[0]
        if type(observation) is tuple:
            observation = list(observation)

        cnn_branch_input = observation[-3].float()
        if batch_size == 1:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)

            conv_branch_out = self.cnn_module(cnn_branch_input)
        else:
            if not self.grayscale:
                cnn_branch_input = cnn_branch_input.permute(0, 3, 1, 2)

            conv_branch_out = self.cnn_module(cnn_branch_input)

        if (
            not save_hidden
            or self.h0 is None
            or self.c0 is None
            or self.h1 is None
            or self.c1 is None
        ):
            device = observation[0].device
            h0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            c0 = Variable(torch.zeros((self.rnn_lens[0], self.rnn_sizes[0]), device=device))
            h1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
            c1 = Variable(torch.zeros((self.rnn_lens[1], self.rnn_sizes[1]), device=device))
        else:
            h0 = self.h0
            c0 = self.c0
            h1 = self.h1
            c1 = self.c1

        observation[-3] = conv_branch_out

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        observation_except_img = observation[: self.img_index]
        if len(observation) > self.img_index + 1:
            observation_except_img += observation[(self.img_index + 1) :]

        obs_seq_cat = torch.cat(observation_except_img, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self.api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self.mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        rnn_block_api_out, (h0, c0) = self.rnn_block_api(mlp_api_out, (h0, c0))

        img_api_out = torch.cat([rnn_block_api_out, conv_branch_out], dim=-1)

        _rnn_block_cat_out, (h1, c1) = self.rnn_block_cat(img_api_out, (h1, c1))

        model_out = self.model_out(rnn_block_api_out)

        if self.output_dropout_rate > 0.0:
            model_out = self.dropout(model_out)

        mu = self.mu_layer(model_out)
        log_std = self.log_std_layer(model_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)

        pi_action = mu if test else pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - functional.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        pi_action = pi_action.squeeze()

        if save_hidden:
            self.h0 = h0
            self.c0 = c0
            self.h1 = h1
            self.c1 = c1

        return pi_action, logp_pi

    def act(self, obs: tuple, test: bool = False):
        """Return a numpy action for the rollout worker (no-grad, stateful hidden states).

        Args:
            obs: Observation tuple as expected by :meth:`forward`.
            test: If True, use the deterministic mean action.

        Returns:
            numpy.ndarray of shape (dim_act,).
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(
                observation=obs_seq, test=test, with_logprob=False, save_hidden=True
            )
            return a.cpu().numpy()


@MODELS.register("impala_ac")
class QRCNNActorCritic(nn.Module):
    """QRCNN actor-critic; default constructor args must match config."""

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_sizes: list | None = None,
        rnn_lens: list | None = None,
        mlp_branch_sizes: list | None = None,
        activation=nn.ReLU,
        seed: int = 42,
    ):
        """Construct the QRCNN actor-critic with one actor and two independent critics.

        Args:
            observation_space: Gymnasium observation space passed to actor and critics.
            action_space: Gymnasium action space passed to actor and critics.
            rnn_sizes: Hidden sizes for the LSTM layers (default [64, 64]).
            rnn_lens: Number of stacked layers per LSTM (default [1, 1]).
            mlp_branch_sizes: Hidden widths of the scalar MLP branch (default [256]).
            activation: Activation class used throughout.
            seed: Random seed applied to all three sub-networks (critics use seed+1/+2).
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [256]

        self.actor = SquashedActorQRCNN(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q1 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
        self.q2 = QRCNNQFunction(
            observation_space, action_space, rnn_sizes, rnn_lens, mlp_branch_sizes, activation
        )
