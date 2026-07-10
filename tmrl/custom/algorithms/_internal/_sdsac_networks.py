"""Network architectures for SD-SAC: discrete Q-heads and rollout actor."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import DQNActor


class DiscreteQHead(nn.Module):
    """Simple Q-value head: maps features to per-action Q-values."""

    def __init__(self, hidden_dim: int, n_actions: int):
        """Build the Q-value head layers.

        Args:
            hidden_dim: Width of the input feature layer and the hidden layer.
            n_actions: Number of discrete actions (output dimension).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map feature vector to per-action Q-values.

        Args:
            features: Feature tensor, shape ``(batch, hidden_dim)``.

        Returns:
            Q-value tensor, shape ``(batch, n_actions)``.
        """
        return cast(torch.Tensor, self.net(features))


class DiscreteSACNetwork(nn.Module):
    """Actor-critic network for discrete SAC.

    The actor produces a Categorical distribution over actions.
    Each critic maps observation features to Q-values for every action.
    """

    def __init__(
        self,
        observation_space,
        n_actions: int,
        hidden_dim: int = 256,
        num_blocks_actor: int = 2,
        num_blocks_critic: int = 4,
        n_cos: int = 64,
    ):
        """Build actor and two independent critic backbones with separate heads.

        Args:
            observation_space: Observation space passed to ``IQNFeatureBackbone``.
            n_actions: Number of discrete actions.
            hidden_dim: Width of all feature layers.
            num_blocks_actor: Number of residual blocks in the actor backbone.
            num_blocks_critic: Number of residual blocks in each critic backbone.
            n_cos: Number of cosine features in ``IQNFeatureBackbone``.
        """
        super().__init__()
        from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import IQNFeatureBackbone

        self.actor_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_actor, n_cos=n_cos
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self.q1_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_critic, n_cos=n_cos
        )
        self.q1_head = DiscreteQHead(hidden_dim, n_actions)

        self.q2_backbone = IQNFeatureBackbone(
            observation_space, hidden_dim=hidden_dim, num_blocks=num_blocks_critic, n_cos=n_cos
        )
        self.q2_head = DiscreteQHead(hidden_dim, n_actions)

        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

    def _dummy_tau(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Single quantile fraction = 0.5 to reuse IQNFeatureBackbone without IQN."""
        return torch.full((batch_size, 1), 0.5, device=device)

    def actor_logits(self, obs) -> torch.Tensor:
        """Return raw logits over actions.  Shape: (batch, n_actions)."""
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.actor_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.actor_head(features))

    def q1(self, obs) -> torch.Tensor:
        """Compute Q1 values for all actions.

        Args:
            obs: Observation tuple; ``obs[0].shape[0]`` gives batch size.

        Returns:
            Q-value tensor, shape ``(batch, n_actions)``.
        """
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.q1_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.q1_head(features))

    def q2(self, obs) -> torch.Tensor:
        """Compute Q2 values for all actions.

        Args:
            obs: Observation tuple; ``obs[0].shape[0]`` gives batch size.

        Returns:
            Q-value tensor, shape ``(batch, n_actions)``.
        """
        batch_size = obs[0].shape[0]
        tau = self._dummy_tau(batch_size, obs[0].device)
        features = self.q2_backbone(obs, tau).squeeze(1)
        return cast(torch.Tensor, self.q2_head(features))


class DiscreteSACActor(DQNActor):
    """Rollout actor for SD-SAC: samples from the softmax policy."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        n_cos: int = 64,
        n_actions: int = 78,
        epsilon: float = 0.01,
    ):
        """Build actor with shared backbone and a separate actor head.

        Extends ``DQNActor`` by adding ``actor_head`` (a 2-layer MLP) that
        produces the soft-policy logits used at rollout time, independent of
        the Q-network head inherited from ``DQNActor``.

        Args:
            observation_space: Observation space for the backbone.
            action_space: Action space (passed to ``DQNActor``).
            hidden_dim: Feature width for all linear layers.
            num_blocks: Number of residual blocks in the backbone.
            n_cos: Number of cosine features.
            n_actions: Number of discrete actions.
            epsilon: Epsilon-greedy noise level (0.0 = pure softmax policy).
        """
        super().__init__(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
            dueling=False,
            n_actions=n_actions,
            epsilon=epsilon,
            n_quantiles_eval=1,
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def act(self, obs, test=False):
        """Sample an action from the soft policy (or take the argmax in test mode).

        Args:
            obs: Observation tuple; ``obs[0].shape[0]`` gives batch size.
            test: If True, return the greedy argmax action (deterministic);
                if False, sample from the Categorical distribution.

        Returns:
            Integer action index as ``np.int64`` scalar or array.
        """
        with torch.no_grad():
            batch_size = obs[0].shape[0]
            tau = torch.full((batch_size, 1), 0.5, device=obs[0].device)
            features = self.q_net.backbone(obs, tau).squeeze(1)
            logits = self.actor_head(features)
            if test:
                return logits.argmax(dim=-1).squeeze().cpu().numpy().astype(np.int64)
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action.squeeze().cpu().numpy().astype(np.int64)
