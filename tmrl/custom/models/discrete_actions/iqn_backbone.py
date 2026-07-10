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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchrl.modules import NoisyLinear

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

_IQN_OUTPUT_INIT_GAIN = 0.01


def _init_linear_small(linear: nn.Linear, gain: float = _IQN_OUTPUT_INIT_GAIN) -> None:
    """Initialize a linear layer with orthogonal weights scaled by ``gain``.

    Small gain keeps initial Q-value outputs near zero, preventing premature
    policy commitment at the start of training.

    Args:
        linear: The ``nn.Linear`` layer to initialize.
        gain: Orthogonal init gain applied to ``linear.weight``.
    """
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
    """Initialize the CosineEmbedding linear layer with a small orthogonal gain.

    Args:
        cos_embed: The ``CosineEmbedding`` module to initialize.
        gain: Orthogonal init gain; kept consistent with the head output init.
    """
    _init_linear_small(cos_embed.linear, gain=gain)


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
        "init_gas_bias",
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
        init_gas_bias: float = 0.0,
    ):
        """Initialize the IQN feature backbone.

        Builds either a split track/physics encoder (when
        ``split_track_observation`` is True and the observation space has
        multiple sub-spaces) or a flat MLP encoder over the whole concatenated
        observation.

        When the split path is active, the track sub-space is encoded by the
        selected ``track_encoder`` and the remaining physics scalars are
        projected by a separate MLP, then joined and optionally processed by a
        GRU (for R2D2-style recurrence).  Both paths share the same
        residual-MLP or SimbaV2 backbone.

        Args:
            observation_space: Sequence of sub-spaces; the first sub-space is
                treated as the track observation when ``split_track_observation``
                is True.
            hidden_dim: Width of the backbone hidden layers and the output.
            num_blocks: Number of residual blocks in the backbone.
            n_cos: Number of cosine basis functions for the IQN quantile embedding.
            split_track_observation: Enable the track/physics split encoder.
            track_encoder: Track encoder variant — ``"conv1d"``, ``"spline_mlp"``,
                or ``"gtn"`` (graph transformer, requires 7-channel track obs).
            use_rnn: Wrap the joint track+physics embedding in a single-layer GRU.
            rnn_hidden_size: GRU hidden size; defaults to ``hidden_dim`` when None.
            api_layernorm: Apply LayerNorm to the flat observation in the
                non-split path.
            use_simbav2: Use SimbaV2 backbone instead of ResidualMLP.
            r2d2_sequence_length: Sequence length for R2D2 burn-in GRU stepping
                (0 = disabled).
            r2d2_burn_in: Number of burn-in steps computed without gradient.
            gnn_hidden: Hidden dimension for the GTN encoder.
            gnn_layers: Number of transformer layers in the GTN encoder.
            init_gas_bias: Unused by the backbone; accepted for API compatibility.
        """
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
        """Encode track and physics sub-observations into a joint embedding.

        Args:
            observation: Sequence of observation tensors; ``observation[0]`` is
                the flat track tensor reshaped to ``(B, track_channels, N)``
                before encoding, and ``observation[1:]`` are the physics scalars.
            batch_size: Batch size B.

        Returns:
            Concatenated tensor of shape ``(B, 2 * hidden_dim)`` — track
            embedding followed by physics embedding.
        """
        track = observation[0].view(batch_size, -1).float()
        track = track.view(
            batch_size, self._track_channels, self._dim_track // self._track_channels
        )
        track_embed = self.track_conv(track)
        physics = _obs_to_flat_tensor(observation[1:], batch_size)
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, physics_embed], dim=-1)

    def _gru_joint(self, joint: torch.Tensor) -> torch.Tensor:
        """Apply a single-layer GRU to the joint track+physics embedding.

        Mirrors the GRU step logic in SophyResidual.  Handles three execution
        modes based on batch shape vs. ``r2d2_sequence_length``:

        1. R2D2 burn-in: batch is a multiple of ``r2d2_sequence_length``,
           ``r2d2_burn_in > 0``, and burn-in < seq_len.  Burn-in steps run
           under ``torch.no_grad()`` to seed the hidden state; active steps
           are trained normally.
        2. Plain sequence mode: batch is a multiple of ``r2d2_sequence_length``
           without burn-in.  The entire sequence is rolled through the GRU.
        3. Single-step mode: batch does not align with ``r2d2_sequence_length``;
           the input is treated as an individual time step.

        Args:
            joint: Concatenated track+physics embedding of shape
                ``(B, joint_dim)``.

        Returns:
            GRU output tensor of shape ``(B, rnn_hidden_size)``.
        """
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
