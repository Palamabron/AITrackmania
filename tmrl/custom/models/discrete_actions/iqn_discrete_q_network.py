"""Discrete-action IQN: ``DQNActor`` (worker) and ``IQNQNetwork`` (trainer).

IQN Q-network with Dueling heads for discrete-action DQN.

Reuses the same track encoder (GNN / Conv1d / spline-MLP) and residual-MLP
backbone that the continuous-action Sophy models use, but replaces the
actor/critic heads with:
  - Implicit Quantile Network (IQN) cosine embedding for distributional RL
  - Dueling architecture (V + A streams)

The network outputs Q(s, a) for every discrete action in a single forward pass.
"""

import math
import warnings
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.hybrid_input.sophy import (
    _build_track_conv1d_branch,
    _build_track_spline_mlp_branch,
    _obs_to_flat_tensor,
)
from tmrl.custom.models.shared.blocks import (
    residual_mlp_backbone,
    simba_v2_backbone,
)
from tmrl.custom.models.shared.track_encoders import (
    TRACK_CHANNELS_GTN,
)
from tmrl.custom.models.shared.track_encoders import (
    build_track_gtn_branch as _build_track_gnn_branch,
)
from tmrl.custom.models.shared.track_encoders import (
    is_gtn_encoder as _is_gtn_encoder,
)
from tmrl.registry import MODELS

_IQN_OUTPUT_INIT_GAIN = 0.01


def _init_linear_small(linear: nn.Linear, gain: float = _IQN_OUTPUT_INIT_GAIN) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_noisy_linear_small(layer: NoisyLinear, gain: float = _IQN_OUTPUT_INIT_GAIN) -> None:
    """Init learned mu weights only; leave factorized noise buffers untouched."""
    if hasattr(layer, "weight_mu"):
        nn.init.orthogonal_(layer.weight_mu, gain=gain)
    if getattr(layer, "bias_mu", None) is not None:
        nn.init.zeros_(layer.bias_mu)


def _init_cosine_embedding(
    cos_embed: "CosineEmbedding", gain: float = _IQN_OUTPUT_INIT_GAIN
) -> None:
    _init_linear_small(cos_embed.linear, gain=gain)


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


class CosineEmbedding(nn.Module):
    """Cosine basis embedding for IQN quantile fractions.

    Maps tau in [0, 1] to a fixed-size vector using:
        phi(tau)_j = ReLU(sum_i cos(pi * i * tau) * w_ij + b_j)
    """

    def __init__(self, n_cos: int = 64, embed_dim: int = 64):
        super().__init__()
        self.n_cos = n_cos
        self.linear = nn.Linear(n_cos, embed_dim)
        i_pi = torch.arange(1, n_cos + 1, dtype=torch.float32) * math.pi
        self.register_buffer("i_pi", i_pi)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """Map quantile fractions to embeddings.

        Args:
            tau: Quantile fractions of shape ``(batch, n_quantiles)``.

        Returns:
            Embeddings of shape ``(batch, n_quantiles, embed_dim)``.
        """
        from einops import rearrange

        i_pi: torch.Tensor = self.i_pi  # type: ignore[assignment]
        cos_input = rearrange(tau, "b n -> b n 1") * i_pi
        cos_features = torch.cos(cos_input)
        return F.relu(self.linear(cos_features))


_IQN_BACKBONE_KWARGS = frozenset(
    {
        "split_track_observation",
        "track_encoder",
        "use_rnn",
        "rnn_hidden_size",
        "api_layernorm",
        "use_simbav2",
        "r2d2_sequence_length",
        "r2d2_burn_in",
        "gnn_hidden",
        "gnn_layers",
    }
)


class IQNFeatureBackbone(nn.Module):
    """Shared feature extractor for the IQN Q-network.

    Encodes the observation (track + physics telemetry) into a fixed-dim
    feature vector, identical to the Sophy actor backbone, then multiplies
    element-wise with the cosine-embedded quantile fractions.
    """

    def __init__(
        self,
        observation_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        n_cos: int = 64,
        split_track_observation: bool = True,
        track_encoder: str = "conv1d",
        use_rnn: bool = False,
        rnn_hidden_size: int | None = None,
        api_layernorm: bool = False,
        use_simbav2: bool = False,
        r2d2_sequence_length: int = 0,
        r2d2_burn_in: int = 0,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
    ):
        super().__init__()
        self._track_channels = TRACK_CHANNELS_GTN if _is_gtn_encoder(track_encoder) else 4
        self._r2d2_sequence_length = r2d2_sequence_length
        self._r2d2_burn_in = r2d2_burn_in
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        self._use_track_conv = split_track_observation and len(observation_space) > 1
        if self._use_track_conv:
            dim_track_first = math.prod(observation_space[0].shape)
            if dim_track_first % self._track_channels != 0:
                from loguru import logger

                logger.warning(
                    "split_track_observation requested but track dim {} is not divisible by "
                    "{} channels (encoder={!r}); falling back to a flat MLP over the whole "
                    "observation. Use a matching track_encoder (e.g. 'gtn' for 7-channel "
                    "world-telemetry track) if you want the split branch.",
                    dim_track_first,
                    self._track_channels,
                    track_encoder,
                )
                self._use_track_conv = False

        if self._use_track_conv:
            dim_track = math.prod(observation_space[0].shape)
            dim_physics = dim_obs - dim_track
            self._dim_track = dim_track
            if track_encoder == "spline_mlp":
                self.track_conv = _build_track_spline_mlp_branch(dim_track, hidden_dim)
            elif _is_gtn_encoder(track_encoder):
                self.track_conv = _build_track_gnn_branch(
                    dim_track, hidden_dim, gnn_hidden=gnn_hidden, gnn_layers=gnn_layers
                )
            else:
                self.track_conv = _build_track_conv1d_branch(dim_track, hidden_dim)
            self.physics_proj = nn.Sequential(
                nn.Linear(dim_physics, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            joint_dim = 2 * hidden_dim
            self._use_rnn = bool(use_rnn)
            rnn_hidden = int(rnn_hidden_size if rnn_hidden_size is not None else hidden_dim)
            self.rnn: nn.GRU | None
            if self._use_rnn:
                self.rnn = nn.GRU(joint_dim, rnn_hidden, num_layers=1, batch_first=True)
                backbone_input_dim = rnn_hidden
            else:
                self.rnn = None
                backbone_input_dim = joint_dim
            self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
            self.layernorm_api = None
        else:
            self._dim_track = 0
            self.rnn = None
            self._use_rnn = False
            self._api_input_dim = dim_obs
            self._warned_api_dim_mismatch = False
            self.layernorm_api = nn.LayerNorm(dim_obs) if api_layernorm else None
            backbone_input_dim = dim_obs

        self.backbone: nn.Module
        if use_simbav2:
            self.backbone = simba_v2_backbone(backbone_input_dim, hidden_dim, num_blocks)
        else:
            self.backbone = residual_mlp_backbone(backbone_input_dim, hidden_dim, num_blocks)
        self.cos_embed = CosineEmbedding(n_cos=n_cos, embed_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        track = observation[0].view(batch_size, -1).float()
        track = track.view(
            batch_size, self._track_channels, self._dim_track // self._track_channels
        )
        track_embed = self.track_conv(track)
        physics = _obs_to_flat_tensor(observation[1:], batch_size)
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, physics_embed], dim=-1)

    def _gru_joint(self, joint: torch.Tensor) -> torch.Tensor:
        """Single-layer GRU on track+physics joint (matches SophyResidual)."""
        if self.rnn is None:
            return joint
        batch_size = joint.shape[0]
        seq_len = self._r2d2_sequence_length
        burn_in_len = self._r2d2_burn_in

        if seq_len > 0 and batch_size % seq_len == 0 and burn_in_len > 0 and burn_in_len < seq_len:
            num_seq = batch_size // seq_len
            joint_seq = joint.view(num_seq, seq_len, -1)
            with torch.no_grad():
                joint_burn = joint_seq[:, :burn_in_len, :]
                self.rnn.flatten_parameters()
                _, h_burn = self.rnn(joint_burn)
            joint_active = joint_seq[:, burn_in_len:, :]
            self.rnn.flatten_parameters()
            out_active, _ = self.rnn(joint_active, h_burn)
            with torch.no_grad():
                out_burn, _ = self.rnn(joint_burn)
            joint_out = torch.cat([out_burn, out_active], dim=1).reshape(batch_size, -1)
            return joint_out
        if seq_len > 0 and batch_size % seq_len == 0:
            num_seq = batch_size // seq_len
            joint_seq = joint.view(num_seq, seq_len, -1)
            self.rnn.flatten_parameters()
            joint_out, _ = self.rnn(joint_seq)
            joint_out_seq: torch.Tensor = joint_out
            return joint_out_seq.reshape(batch_size, -1)
        joint_seq = joint.unsqueeze(1)
        self.rnn.flatten_parameters()
        joint_out, _ = self.rnn(joint_seq)
        joint_out_single: torch.Tensor = joint_out
        return joint_out_single.squeeze(1)

    def _align_api_obs_dim(self, obs_flat: torch.Tensor) -> torch.Tensor:
        """Ensure API-only observation width matches constructor-time model width.

        In real-time setups, observation vectors can occasionally miss or include a few
        tail scalars (e.g. action-history buffering race around reset). We pad/truncate
        to the expected width so LayerNorm/backbone stay shape-safe.
        """
        expected = int(getattr(self, "_api_input_dim", obs_flat.shape[-1]))
        current = int(obs_flat.shape[-1])
        if current == expected:
            return obs_flat

        if not getattr(self, "_warned_api_dim_mismatch", False):
            warnings.warn(
                f"IQNFeatureBackbone API obs dim mismatch (expected {expected}, got {current}); "
                "auto-aligning by zero-padding/truncation.",
                stacklevel=2,
            )
            self._warned_api_dim_mismatch = True

        if current < expected:
            pad = obs_flat.new_zeros(obs_flat.shape[0], expected - current)
            return torch.cat([obs_flat, pad], dim=-1)

        return obs_flat[:, :expected]

    def forward(self, observation, tau: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: tuple of tensors from the env.
            tau: (batch, n_quantiles) sampled quantile fractions.

        Returns:
            (batch, n_quantiles, hidden_dim) features ready for the heads.
        """
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]

        if self._use_track_conv:
            joint = self._joint_features(observation, batch_size)
            joint = self._gru_joint(joint)
            joint = self.layernorm_joint(joint)
            features = self.backbone(joint)
        else:
            obs_flat = _obs_to_flat_tensor(observation, batch_size)
            obs_flat = self._align_api_obs_dim(obs_flat)
            if self.layernorm_api is not None:
                obs_flat = self.layernorm_api(obs_flat)
            features = self.backbone(obs_flat)

        from einops import rearrange

        tau_embed = self.cos_embed(tau)
        combined: torch.Tensor = rearrange(features, "b h -> b 1 h") * tau_embed
        return combined


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
