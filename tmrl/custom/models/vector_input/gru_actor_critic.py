"""GRU-based actor-critic for sequential vector observations (SAC)."""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from tmrl.custom.models.shared.blocks import LOG_STD_MAX, LOG_STD_MIN, mlp, squashed_logprob
from tmrl.util import prod


def build_stacked_gru(input_size: int, rnn_size: int, rnn_len: int) -> nn.GRU:
    """Return a stacked GRU with ``rnn_len`` layers and ``batch_first=True``.

    Args:
        input_size: Input feature dimension at each time step.
        rnn_size: Number of hidden units per GRU layer.
        rnn_len: Number of stacked GRU layers (must be ≥ 1).

    Returns:
        A configured ``nn.GRU`` with ``batch_first=True``, no dropout, and
        unidirectional layers.

    Raises:
        AssertionError: If ``rnn_len < 1``.
    """
    assert rnn_len >= 1
    return nn.GRU(
        input_size=input_size,
        hidden_size=rnn_size,
        num_layers=rnn_len,
        bias=True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
    )


class GRUActor(nn.Module):
    """GRU-based actor with squashed Gaussian policy.

    Expects sequence observations; action is produced from the last time step.
    """

    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size: int = 100,
        rnn_len: int = 2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        """Initialize the GRU actor.

        Args:
            obs_space: Gym observation space (sequence of sub-spaces with shapes).
            act_space: Gym continuous action space (Box).
            rnn_size: Number of hidden units per GRU layer.
            rnn_len: Number of stacked GRU layers.
            mlp_sizes: Hidden layer widths of the post-GRU MLP policy head.
            activation: Activation class applied to MLP hidden layers.
        """
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = build_stacked_gru(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size, *list(mlp_sizes)], activation, activation)
        self.mu_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(mlp_sizes[-1], dim_act)
        self.act_limit = act_space.high[0]
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.h = None

    def forward(self, obs_seq, test=False, with_logprob=True, save_hidden=False):
        """Compute action and optional log-probability from a sequence observation.

        Args:
            obs_seq: Tuple of observation tensors, each of shape ``(B, T, ...)``.
                Concatenated along the last dim before passing to the GRU.
            test: When True, return the deterministic mean action.
            with_logprob: When True, compute and return the squashed log-prob.
            save_hidden: When True, persist the GRU hidden state in ``self.h``
                for sequential (online) inference across episode timesteps.

        Returns:
            Tuple ``(action, logp)`` where ``action`` has shape ``(B, act_dim)``
            (from the last timestep) and ``logp`` has shape ``(B,)`` or is None
            when ``with_logprob=False``.
        """
        self.rnn.flatten_parameters()
        batch_size = obs_seq[0].shape[0]
        if not save_hidden or self.h is None:
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=obs_seq[0].device)
        else:
            h = self.h

        net_out, h = self.rnn(torch.cat(obs_seq, -1), h)
        net_out = self.mlp(net_out[:, -1])
        mu = self.mu_layer(net_out)
        log_std = torch.clamp(self.log_std_layer(net_out), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_dist = Normal(mu, std)
        pi_action = mu if test else pi_dist.rsample()
        logp_pi = squashed_logprob(pi_dist, pi_action) if with_logprob else None
        pi_action = self.act_limit * torch.tanh(pi_action)
        if save_hidden:
            self.h = h
        return pi_action, logp_pi

    def act(self, obs, test=False):
        """Produce a numpy action using saved hidden state (sequential inference).

        Wraps each observation component in a length-1 time dimension and calls
        ``forward`` with ``save_hidden=True`` to maintain GRU state across steps.

        Args:
            obs: Tuple of unbatched observation tensors.
            test: When True, use the deterministic mean action.

        Returns:
            np.ndarray of shape ``(act_dim,)`` — the selected action.
        """
        obs_seq = tuple(o.view(1, *o.shape) for o in obs)
        with torch.no_grad():
            a, _ = self.forward(obs_seq=obs_seq, test=test, with_logprob=False, save_hidden=True)
            return a.squeeze().cpu().numpy()


class GRUQFunction(nn.Module):
    """GRU-based Q-function; action is merged into the latent space after the RNN."""

    def __init__(
        self,
        obs_space,
        act_space,
        rnn_size: int = 100,
        rnn_len: int = 2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        """Initialize the GRU Q-function.

        The action vector is merged with the GRU output at the last time step
        before passing through the MLP critic head.

        Args:
            obs_space: Gym observation space (sequence of sub-spaces).
            act_space: Gym continuous action space (Box).
            rnn_size: Number of hidden units per GRU layer.
            rnn_len: Number of stacked GRU layers.
            mlp_sizes: Hidden layer widths of the post-GRU MLP critic head.
            activation: Activation class applied to MLP hidden layers.
        """
        super().__init__()
        dim_obs = sum(prod(s for s in space.shape) for space in obs_space)
        dim_act = act_space.shape[0]
        self.rnn = build_stacked_gru(dim_obs, rnn_size, rnn_len)
        self.mlp = mlp([rnn_size + dim_act, *list(mlp_sizes), 1], activation)
        self.rnn_size = rnn_size
        self.rnn_len = rnn_len
        self.h = None

    def forward(self, obs_seq, act, save_hidden=False):
        """Compute Q(s, a) from a sequence observation and an action.

        Args:
            obs_seq: Tuple of observation tensors, each of shape ``(B, T, ...)``.
            act: Action tensor of shape ``(B, act_dim)``, merged at the last
                GRU time step.
            save_hidden: When True, persist the GRU hidden state in ``self.h``.

        Returns:
            Tensor of shape ``(B,)`` — scalar Q-values.
        """
        self.rnn.flatten_parameters()
        batch_size = obs_seq[0].shape[0]
        if not save_hidden or self.h is None:
            h = torch.zeros((self.rnn_len, batch_size, self.rnn_size), device=obs_seq[0].device)
        else:
            h = self.h

        net_out, h = self.rnn(torch.cat(obs_seq, -1), h)
        net_out = torch.cat((net_out[:, -1], act), -1)
        q = self.mlp(net_out)
        if save_hidden:
            self.h = h
        return torch.squeeze(q, -1)


class GRUActorCritic(nn.Module):
    """Actor-critic using stacked GRU for sequential observation processing."""

    def __init__(
        self,
        observation_space,
        action_space,
        rnn_size: int = 100,
        rnn_len: int = 2,
        mlp_sizes=(100, 100),
        activation=nn.ReLU,
    ):
        """Initialize the actor and twin GRU Q-networks.

        Args:
            observation_space: Gym observation space.
            action_space: Gym continuous action space (Box).
            rnn_size: Number of GRU hidden units per layer.
            rnn_len: Number of stacked GRU layers.
            mlp_sizes: Hidden layer widths for both actor and critic MLP heads.
            activation: Activation class applied to MLP hidden layers.
        """
        super().__init__()
        self.actor = GRUActor(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q1 = GRUQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )
        self.q2 = GRUQFunction(
            observation_space, action_space, rnn_size, rnn_len, mlp_sizes, activation
        )

    def act(self, obs, test=False):
        """Produce a numpy action from the actor (no gradient, no log-prob).

        Args:
            obs: Observation tuple.
            test: When True, use the deterministic mean action.

        Returns:
            np.ndarray of shape ``(act_dim,)`` — the selected action.
        """
        with torch.no_grad():
            a, _ = self.actor(obs, test, False)
            return a.squeeze().cpu().numpy()
