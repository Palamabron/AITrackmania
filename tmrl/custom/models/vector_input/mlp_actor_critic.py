"""MLP actor-critic for vector observations (SAC / REDQ).

Supports both tuple and single Box observation spaces.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn import ModuleList

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    mlp,
    squashed_logprob,
)
from tmrl.util import prod


def _act_numpy(a: torch.Tensor) -> np.ndarray:
    res = a.squeeze().cpu().numpy()
    return np.expand_dims(res, 0) if not len(res.shape) else res


class MLPActor(TorchActorModule):
    """Squashed Gaussian actor with plain MLP trunk.

    Supports tuple or single Box observation spaces.
    """

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self._tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self._tuple_obs = False
        dim_act = action_space.shape[0]
        self.net = mlp([dim_obs, *list(hidden_sizes)], activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = action_space.high[0]

    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self._tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()
        logp_pi = squashed_logprob(pi_dist, pi_action) if with_logprob else None
        pi_action = self.act_limit * torch.tanh(pi_action)
        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            return _act_numpy(a)


class MLPQFunction(nn.Module):
    """Q-function with plain MLP trunk.

    Supports tuple or single Box observation spaces.
    """

    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self._tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self._tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim, *list(hidden_sizes), 1], activation)

    def forward(self, obs, act):
        x = (
            torch.cat((*obs, act), -1)
            if self._tuple_obs
            else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        )
        return torch.squeeze(self.q(x), -1)


class MLPActorCritic(nn.Module):
    """MLP actor-critic: one actor and two Q-networks."""

    def __init__(
        self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU
    ):
        super().__init__()
        self.actor = MLPActor(observation_space, action_space, hidden_sizes, activation)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return _act_numpy(a)


class REDQMLPActorCritic(nn.Module):
    """REDQ agent: one actor, n Q-networks."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU,
        n: int = 10,
    ):
        super().__init__()
        self.actor = MLPActor(observation_space, action_space, hidden_sizes, activation)
        self.n = n
        self.qs = ModuleList(
            [
                MLPQFunction(observation_space, action_space, hidden_sizes, activation)
                for _ in range(n)
            ]
        )

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return _act_numpy(a)
