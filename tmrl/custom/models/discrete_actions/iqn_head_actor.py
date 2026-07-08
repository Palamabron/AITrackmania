"""IQN Dueling head, Q-network, and DQN actor."""

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.discrete_actions.iqn_backbone import (
    _IQN_BACKBONE_KWARGS,
    _IQN_OUTPUT_INIT_GAIN,
    IQNFeatureBackbone,
    _init_cosine_embedding,
    _init_linear_small,
    _init_noisy_linear_small,
)
from tmrl.registry import MODELS


def _init_dueling_output_layers(head: "DuelingHead", gain: float = _IQN_OUTPUT_INIT_GAIN) -> None:
    for stream in (head.value_stream, head.advantage_stream):
        out = stream[-1]
        if isinstance(out, nn.Linear):
            _init_linear_small(out, gain=gain)
        elif isinstance(out, NoisyLinear):
            _init_noisy_linear_small(out, gain=gain)


def _init_iqn_q_head(head: nn.Module, gain: float = _IQN_OUTPUT_INIT_GAIN) -> None:
    if isinstance(head, DuelingHead):
        _init_dueling_output_layers(head, gain=gain)
    elif isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        _init_linear_small(head[-1], gain=gain)


class DuelingHead(nn.Module):
    """Dueling DQN head: Q(s,a) = V(s) + A(s,a) - mean(A).

    When ``noisy=True``, the output linear layers use factorized Gaussian
    NoisyLinear (NoisyNet paper) instead of ``nn.Linear``.  Call
    ``reset_noise()`` every training step and ``set_noise_scale(s)`` to
    anneal the exploration noise over time without interfering with the
    learned sigma parameters.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_actions: int,
        noisy: bool = False,
        noisy_std_init: float = 0.5,
    ):
        super().__init__()
        self._noisy = noisy

        out_linear_v: nn.Module
        out_linear_a: nn.Module
        if noisy:
            out_linear_v = NoisyLinear(hidden_dim, 1, std_init=noisy_std_init)
            out_linear_a = NoisyLinear(hidden_dim, n_actions, std_init=noisy_std_init)
        else:
            out_linear_v = nn.Linear(hidden_dim, 1)
            out_linear_a = nn.Linear(hidden_dim, n_actions)

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            out_linear_v,
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            out_linear_a,
        )
        self._noise_scale = 1.0

    # ------------------------------------------------------------------
    # NoisyLinear helpers
    # ------------------------------------------------------------------

    def _noisy_layers(self) -> list[NoisyLinear]:
        layers: list[NoisyLinear] = []
        for stream in (self.value_stream, self.advantage_stream):
            for m in stream.modules():
                if isinstance(m, NoisyLinear):
                    layers.append(m)
        return layers

    def reset_noise(self) -> None:
        """Resample factorized noise, then scale epsilon buffers."""
        for layer in self._noisy_layers():
            layer.reset_noise()
            if self._noise_scale < 1.0:
                layer.weight_epsilon.mul_(self._noise_scale)
                if layer.bias_epsilon is not None:
                    layer.bias_epsilon.mul_(self._noise_scale)

    def set_noise_scale(self, scale: float) -> None:
        """Set a multiplier applied to epsilon noise buffers after each reset."""
        self._noise_scale = max(0.0, min(1.0, scale))

    def forward(
        self, features: torch.Tensor, return_components: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map features to Q-values via dueling decomposition.

        Args:
            features: Tensor of shape ``(..., hidden_dim)``.

        Returns:
            If ``return_components`` is False:
                Q-values of shape ``(..., n_actions)``.
            If ``return_components`` is True:
                Tuple ``(q_values, value, advantage, centered_advantage)`` where
                ``value`` has shape ``(..., 1)`` and ``advantage`` / ``centered_advantage``
                have shape ``(..., n_actions)``.
        """
        v = self.value_stream(features)
        a = self.advantage_stream(features)
        centered_a = a - a.mean(dim=-1, keepdim=True)
        result: torch.Tensor = v + centered_a
        if return_components:
            return result, v, a, centered_a
        return result


class IQNQNetwork(nn.Module):
    """Full IQN Q-network with optional Dueling architecture.

    Forward pass samples n_quantiles tau values, embeds them, and produces
    per-action quantile values of shape (batch, n_quantiles, n_actions).
    The mean over quantiles gives the expected Q-values.
    """

    def __init__(
        self,
        observation_space,
        n_actions: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
        dueling: bool = True,
        noisy: bool = False,
        noisy_std_init: float = 0.5,
        **backbone_kwargs,
    ):
        super().__init__()
        self.n_actions = n_actions
        bb_kw = {k: v for k, v in backbone_kwargs.items() if k in _IQN_BACKBONE_KWARGS}
        self.backbone = IQNFeatureBackbone(
            observation_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
            **bb_kw,
        )
        self.head: nn.Module
        if dueling:
            self.head = DuelingHead(
                hidden_dim,
                n_actions,
                noisy=noisy,
                noisy_std_init=noisy_std_init,
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, n_actions),
            )

        _init_cosine_embedding(self.backbone.cos_embed)
        _init_iqn_q_head(self.head)

    def forward(
        self,
        observation,
        tau: torch.Tensor | None = None,
        n_quantiles: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            observation: env observation tuple.
            tau: (batch, n_quantiles) fractions. Sampled if None.
            n_quantiles: how many quantiles to sample when tau is None.

        Returns:
            quantile_values: (batch, n_quantiles, n_actions)
            tau: the quantile fractions used.
        """
        batch_size = observation[0].shape[0]
        device = observation[0].device

        if tau is None:
            tau = torch.rand(batch_size, n_quantiles, device=device)

        features = self.backbone(observation, tau)
        quantile_values: torch.Tensor = self.head(features)
        return quantile_values, tau

    def forward_with_head_stats(
        self,
        observation,
        tau: torch.Tensor | None = None,
        n_quantiles: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Forward pass plus optional dueling-head internals for diagnostics.

        Returns:
            quantile_values: (batch, n_quantiles, n_actions)
            tau: quantile fractions used
            head_stats: dict with dueling streams when dueling is enabled, else None
        """
        batch_size = observation[0].shape[0]
        device = observation[0].device
        if tau is None:
            tau = torch.rand(batch_size, n_quantiles, device=device)

        features = self.backbone(observation, tau)
        if isinstance(self.head, DuelingHead):
            quantile_values, value, advantage, centered_advantage = self.head(
                features, return_components=True
            )
            head_stats = {
                "value": value,
                "advantage": advantage,
                "centered_advantage": centered_advantage,
            }
            return quantile_values, tau, head_stats

        quantile_values = self.head(features)
        return quantile_values, tau, None

    def q_values(self, observation, n_quantiles: int = 32) -> torch.Tensor:
        """Expected Q-values: mean over quantile dimension.

        Returns:
            (batch, n_actions)
        """
        qv, _ = self.forward(observation, n_quantiles=n_quantiles)
        return qv.mean(dim=1)


@MODELS.register("dqn_actor")
class DQNActor(TorchActorModule):
    """Actor for DQN rollout workers: wraps IQNQNetwork with epsilon-greedy.

    The trainer broadcasts updated Q-network weights to this actor.
    At inference, it computes Q-values and selects actions epsilon-greedily.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
        dueling: bool = True,
        n_actions: int = 78,
        epsilon: float = 0.00005,
        n_quantiles_eval: int = 32,
        explore_repeat_steps: int = 1,
        noisy: bool = False,
        noisy_std_init: float = 0.5,
        noisy_eval_std: float = 0.01,
        **backbone_kwargs,
    ):
        super().__init__(observation_space, action_space)
        self.q_net = IQNQNetwork(
            observation_space,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            n_cos=n_cos,
            dueling=dueling,
            noisy=noisy,
            noisy_std_init=noisy_std_init,
            **backbone_kwargs,
        )
        # Store epsilon as a buffer so it is included in state_dict and
        # survives save_to_bytes/load_from_bytes serialization.
        self.register_buffer("_epsilon_buf", torch.tensor(epsilon, dtype=torch.float32))
        self.register_buffer("_noise_scale_buf", torch.tensor(1.0, dtype=torch.float32))
        self._noisy = bool(noisy)
        self._noisy_eval_std = float(noisy_eval_std)
        self.n_actions = n_actions
        self.n_quantiles_eval = n_quantiles_eval
        self.explore_repeat_steps = max(1, int(explore_repeat_steps))
        self._current_explore_count = 0
        self._last_explore_action: np.ndarray | None = None

    @property
    def noise_scale(self) -> float:
        b = cast(Any, self._noise_scale_buf)
        scalars = b.detach().cpu().reshape(-1).tolist()
        return float(scalars[0])

    def set_noise_scale(self, value: float | int) -> None:
        """Sync NoisyNet exploration scale from trainer (buffer + DuelingHead)."""
        b = cast(Any, self._noise_scale_buf)
        scale = float(value)
        with torch.no_grad():
            b.copy_(torch.tensor(scale, dtype=b.dtype, device=b.device))
        head = getattr(self.q_net, "head", None)
        if head is not None and hasattr(head, "set_noise_scale"):
            head.set_noise_scale(scale)

    def reset_noise(self, batch_size: int = 1) -> None:
        """Resample NoisyLinear noise (worker rollout / episode reset)."""
        del batch_size  # API compat with other actors; IQN uses batch_size=1
        if not self._noisy:
            return
        head = getattr(self.q_net, "head", None)
        if head is not None and hasattr(head, "set_noise_scale"):
            head.set_noise_scale(self.noise_scale)
        if head is not None and hasattr(head, "reset_noise"):
            head.reset_noise()

    def reset_explore_state(self) -> None:
        """Clear the explore-repeat hold so a held random action never leaks
        from the previous episode into the first steps of a new one."""
        self._current_explore_count = 0
        self._last_explore_action = None

    @property
    def epsilon(self) -> float:
        b = cast(Any, self._epsilon_buf)
        scalars = b.detach().cpu().reshape(-1).tolist()
        return float(scalars[0])

    def set_epsilon(self, value: float | int) -> None:
        """Update exploration epsilon (buffer scalar; same semantics as former property setter)."""
        b = cast(Any, self._epsilon_buf)
        with torch.no_grad():
            b.copy_(torch.tensor(float(value), dtype=b.dtype, device=b.device))

    def forward(self, observation, **kwargs):
        return self.q_net.q_values(observation, n_quantiles=self.n_quantiles_eval)

    def act_(self, obs, test=False):
        """Override base act_ to skip the float np.clip that breaks integer actions."""
        from tmrl.util import collate_torch

        obs = collate_torch([obs], device=self.device)
        with torch.no_grad():
            action = self.act(obs, test=test)
        return action

    def act(self, obs, test=False):
        """Epsilon-greedy action selection.

        Args:
            obs: batched observation tuple (from act_()).
            test: if True, use greedy (epsilon=0).

        Returns:
            np.ndarray: scalar action index.
        """
        if test:
            self._current_explore_count = 0
            self._last_explore_action = None

        if not test and self._current_explore_count > 0 and self._last_explore_action is not None:
            self._current_explore_count -= 1
            return self._last_explore_action

        # torchrl NoisyLinear only injects noise when module.training is True.
        # Scope train/eval to the dueling head only so backbone layers with
        # dropout/batchnorm are not accidentally left in train mode.
        head = getattr(self.q_net, "head", None) if self._noisy else None
        if self._noisy and head is not None:
            if test:
                if self._noisy_eval_std > 0.0:
                    head.train()
                    if hasattr(head, "set_noise_scale"):
                        head.set_noise_scale(self._noisy_eval_std)
                    if hasattr(head, "reset_noise"):
                        head.reset_noise()
                else:
                    head.eval()
            else:
                head.train()
                self.reset_noise()

        with torch.no_grad():
            q_vals = self.forward(obs)  # (1, n_actions)

        if not test and np.random.random() < self.epsilon:
            action = np.array(np.random.randint(self.n_actions), dtype=np.int64)
            self._last_explore_action = action
            self._current_explore_count = self.explore_repeat_steps - 1
            return action

        self._current_explore_count = 0
        self._last_explore_action = None
        return q_vals.argmax(dim=-1).squeeze().cpu().numpy().astype(np.int64)
