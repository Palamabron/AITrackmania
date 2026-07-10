"""Legacy Sophy actor-critic classes (QRCNNSophy, SquashedActorSophy, SophyActorCritic)."""

import math

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
)
from tmrl.custom.models.shared.track_encoders import (
    TRACK_CHANNELS_DEFAULT,
    TRACK_CHANNELS_GTN,
    is_gtn_encoder,
)
from tmrl.registry import MODELS

_TRACK_CHANNELS_DEFAULT = TRACK_CHANNELS_DEFAULT
_TRACK_CHANNELS_GTN = TRACK_CHANNELS_GTN
_is_gtn_encoder = is_gtn_encoder


def mlp(sizes, dim_obs, activation=nn.ReLU):
    """
    Builds a multi-layer perceptron (MLP).

    Args:
        sizes (list[int]): List of layer sizes.
        dim_obs (int): Input dimension.
        activation (torch.nn.Module): Activation function class. Defaults to nn.ReLU.

    Returns:
        torch.nn.Sequential: The constructed MLP.
    """

    layers = [nn.Linear(dim_obs, sizes[0]), activation()]

    for i in range(1, len(sizes)):
        layers.append(nn.Linear(sizes[i - 1], sizes[i]))
        layers.append(activation())

    return nn.Sequential(*layers)


@MODELS.register("sophy_critic")
class QRCNNSophy(nn.Module):
    """
    Quantile Regression Critic for Sophy architecture.
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
        noisy_linear_critic: bool = False,
        output_dropout: float = 0.0,
    ):
        """
        Initializes the QRCNNSophy.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to [64].
            rnn_lens: List of RNN lengths. Defaults to [1].
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to [64].
            activation: Activation function class. Defaults to nn.ReLU.
            seed: Random seed.
            quantiles_number: Number of quantiles for TQC.
            api_layernorm: Whether to apply LayerNorm to the API input.
            mlp_layernorm: Whether to apply LayerNorm after the MLP branch.
            noisy_linear_critic: Whether to use NoisyLinear for the output.
            output_dropout: Dropout rate for the output.
        """
        super().__init__()
        torch.manual_seed(seed)

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [64]

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        self.num_quantiles = quantiles_number

        self.mlp_api = mlp(mlp_branch_sizes[:-1], dim_obs, activation)

        self._api_layernorm = api_layernorm
        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        self._mlp_layernorm = mlp_layernorm
        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-2])

        self.mlp_act = mlp([mlp_branch_sizes[-1]], mlp_branch_sizes[-2] + dim_act, activation)

        self.head_proj = nn.Sequential(
            nn.Linear(mlp_branch_sizes[-1], mlp_branch_sizes[-1]),
            nn.SiLU(),
        )

        if noisy_linear_critic:
            self.model_out = NoisyLinear(
                rnn_sizes[0], self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], self.num_quantiles)

        self._output_dropout = output_dropout
        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

    def forward(self, observation, act):
        """
        Forward pass for the critic.

        Args:
            observation: Input observation.
            act: Input action.

        Returns:
            torch.Tensor: Quantile values.
        """
        batch_size = observation[0].shape[0]
        if isinstance(observation, tuple):
            observation = list(observation)

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self._api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self._mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        cat_mlp_api_act_out = torch.cat([mlp_api_out, act], dim=-1)

        mlp_api_out = self.mlp_act(cat_mlp_api_act_out)

        head_out = self.head_proj(mlp_api_out)

        model_out = self.model_out(head_out)

        if self._output_dropout > 0.0:
            model_out = self.dropout(model_out)

        return torch.squeeze(model_out, -1)


@MODELS.register("sophy_actor")
class SquashedActorSophy(TorchActorModule):
    """
    Actor network for Sophy architecture with squashed Gaussian distribution.
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
        noisy_linear_actor: bool = False,
        output_dropout: float = 0.0,
        init_gas_bias: float = 0.0,
    ):
        """
        Initializes the SquashedActorSophy.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to [64].
            rnn_lens: List of RNN lengths. Defaults to [1].
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to [64].
            activation: Activation function class. Defaults to nn.ReLU.
            seed: Random seed.
            api_layernorm: Whether to apply LayerNorm to the API input.
            mlp_layernorm: Whether to apply LayerNorm after the MLP branch.
            noisy_linear_actor: Whether to use NoisyLinear for the output.
            output_dropout: Dropout rate for the output.
            init_gas_bias: Initial bias for the gas (throttle) output.
        """
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)

        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [64]

        self.activation = activation()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        mlp_out_size = 1

        self.mlp_api = mlp(mlp_branch_sizes, dim_obs, activation)

        self._api_layernorm = api_layernorm
        if api_layernorm:
            self.layernorm_api = nn.LayerNorm(dim_obs)

        self._mlp_layernorm = mlp_layernorm
        if mlp_layernorm:
            self.layernorm_mlp = nn.LayerNorm(mlp_branch_sizes[-1])

        self.head_proj = nn.Sequential(
            nn.Linear(mlp_branch_sizes[-1], mlp_branch_sizes[-1]),
            nn.SiLU(),
        )

        if noisy_linear_actor:
            self.model_out = NoisyLinear(
                rnn_sizes[0], mlp_out_size, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(mlp_branch_sizes[-1], mlp_out_size)

        self._output_dropout = output_dropout
        if output_dropout > 0.0:
            self.dropout = nn.Dropout(output_dropout)

        self.mu_layer = nn.Linear(mlp_out_size, dim_act)
        self.log_std_layer = nn.Linear(mlp_out_size, dim_act)
        if dim_act > 0 and init_gas_bias != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = init_gas_bias
        self.act_limit = action_space.high[0]
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX

    def forward(self, observation, test=False, with_logprob=True, **kwargs):
        """
        Forward pass for the actor.

        Args:
            observation: Input observation.
            test (bool): Whether in test mode. Defaults to False.
            with_logprob (bool): Whether to return log probability. Defaults to True.
            **kwargs: Optional return_pre_tanh_mean (bool). If True, return (action, logp_pi, mu).

        Returns:
            tuple[torch.Tensor, torch.Tensor | None] or tuple[..., torch.Tensor]: Action, log prob,
            and optionally pre-tanh mean (for L2 regularization in the trainer).
        """
        batch_size = observation[0].shape[0]
        if isinstance(observation, tuple):
            observation = list(observation)

        for index, _ in enumerate(observation):
            observation[index] = observation[index].view(batch_size, -1)

        obs_seq_cat = torch.cat(observation, -1)
        obs_seq_cat = obs_seq_cat.view(batch_size, -1).float()

        if self._api_layernorm:
            obs_seq_cat = self.layernorm_api(obs_seq_cat)

        mlp_api_out = self.activation(self.mlp_api(obs_seq_cat))

        if self._mlp_layernorm:
            mlp_api_out = self.layernorm_mlp(mlp_api_out)

        head_out = self.head_proj(mlp_api_out)

        model_out = self.model_out(head_out)

        if self._output_dropout > 0.0:
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

        if kwargs.get("return_pre_tanh_mean", False):
            return pi_action, logp_pi, mu
        return pi_action, logp_pi

    def act(self, obs: tuple, test=False):
        """
        Predicts an action from an observation.

        Args:
            obs (tuple): Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False)
            return a.cpu().numpy()


@MODELS.register("sophy_ac")
class SophyActorCritic(nn.Module):
    """
    Actor-critic architecture for Sophy.
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
    ):
        """
        Initializes the SophyActorCritic.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            rnn_sizes: List of RNN layer sizes. Defaults to [64].
            rnn_lens: List of RNN lengths. Defaults to [1].
            mlp_branch_sizes: List of sizes for MLP branches. Defaults to [64].
            activation: Activation function class. Defaults to nn.ReLU.
            seed: Random seed.
        """
        super().__init__()
        rnn_sizes = rnn_sizes or [64]
        rnn_lens = rnn_lens or [1]
        mlp_branch_sizes = mlp_branch_sizes or [64]
        self.actor = SquashedActorSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed,
        )
        self.q1 = QRCNNSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophy(
            observation_space,
            action_space,
            rnn_sizes,
            rnn_lens,
            mlp_branch_sizes,
            activation,
            seed=seed + 2,
        )
