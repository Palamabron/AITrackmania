"""Residual MLP actor-critic for vector observations (SAC / REDQ)."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.nn import ModuleList

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    cat_obs,
    obs_dim,
    residual_mlp_backbone,
    squashed_logprob,
)


def _is_tuple_obs(obs_space) -> bool:
    try:
        sum(s for s in obs_space[0].shape for _ in obs_space)
        return True
    except TypeError:
        return False


def _act_numpy(model: nn.Module, obs, test: bool = False) -> np.ndarray:
    with torch.no_grad():
        a, _ = model.forward(obs, test, False)
        res = a.squeeze().cpu().numpy()
        return np.expand_dims(res, 0) if not len(res.shape) else np.asarray(res)


class ResidualMLPActor(TorchActorModule):
    """Actor with residual MLP backbone (LayerNorm + SiLU, configurable depth)."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 6,
    ):
        super().__init__(observation_space, action_space)
        self._tuple_obs = _is_tuple_obs(observation_space)
        dim_obs = obs_dim(observation_space)
        dim_act = action_space.shape[0]
        self.backbone = residual_mlp_backbone(dim_obs, hidden_dim, num_blocks)
        self.mu_layer = nn.Linear(hidden_dim, dim_act)
        self.log_std_layer = nn.Linear(hidden_dim, dim_act)
        self.act_limit = action_space.high[0]

    def forward(self, obs, test=False, with_logprob=True):
        x = cat_obs(obs, self._tuple_obs)
        net_out = self.backbone(x)
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()
        logp_pi = squashed_logprob(pi_dist, pi_action) if with_logprob else None
        pi_action = self.act_limit * torch.tanh(pi_action)
        return pi_action, logp_pi

    def act(self, obs, test=False):
        return _act_numpy(self, obs, test)


class ResidualMLPQFunction(nn.Module):
    """Q-function with residual MLP backbone."""

    def __init__(self, obs_space, act_space, hidden_dim: int = 256, num_blocks: int = 6):
        super().__init__()
        self._tuple_obs = _is_tuple_obs(obs_space)
        dim_obs = obs_dim(obs_space)
        act_dim = act_space.shape[0]
        self.backbone = residual_mlp_backbone(dim_obs + act_dim, hidden_dim, num_blocks)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, act):
        x = (
            torch.cat((*obs, act), -1)
            if self._tuple_obs
            else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        )
        return torch.squeeze(self.q_head(self.backbone(x)), -1)


class ResidualMLPActorCritic(nn.Module):
    """Residual MLP actor-critic (depth 4-8, width 256)."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 6,
    ):
        super().__init__()
        self.actor = ResidualMLPActor(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.q1 = ResidualMLPQFunction(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.q2 = ResidualMLPQFunction(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )

    def act(self, obs, test=False):
        return _act_numpy(self.actor, obs, test)


class REDQResidualMLPActorCritic(nn.Module):
    """REDQ with residual MLP backbone."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 6,
        n: int = 10,
    ):
        super().__init__()
        self.actor = ResidualMLPActor(
            observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
        )
        self.n = n
        self.qs = ModuleList(
            [
                ResidualMLPQFunction(
                    observation_space, action_space, hidden_dim=hidden_dim, num_blocks=num_blocks
                )
                for _ in range(self.n)
            ]
        )

    def act(self, obs, test=False):
        return _act_numpy(self.actor, obs, test)
