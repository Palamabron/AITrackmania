"""Sophy residual critic: QRCNNSophyResidual."""

import math

import torch
from torch import nn
from torchrl.modules import NoisyLinear

from tmrl.custom.models.hybrid_input.sophy_residual_actor import (
    _build_track_conv1d_branch,
    _build_track_spline_mlp_branch,
    _make_backbone,
    _obs_to_flat_tensor,
)
from tmrl.custom.models.shared.track_encoders import (
    TRACK_CHANNELS_DEFAULT,
    TRACK_CHANNELS_GTN,
    build_track_gtn_branch,
    is_gtn_encoder,
)
from tmrl.registry import MODELS

_TRACK_CHANNELS_DEFAULT = TRACK_CHANNELS_DEFAULT
_TRACK_CHANNELS_GTN = TRACK_CHANNELS_GTN
_is_gtn_encoder = is_gtn_encoder
_build_track_gnn_branch = build_track_gtn_branch


@MODELS.register("sophy_residual_critic")
class QRCNNSophyResidual(nn.Module):
    """
    Sophy critic (TQC quantiles) with residual MLP backbone and optional
    Conv1d track branch and RNN.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        seed: int = 42,
        quantiles_number: int = 1,
        split_track_observation: bool = True,
        use_rnn: bool = False,
        rnn_hidden_size: int | None = None,
        track_encoder: str = "conv1d",
        api_layernorm: bool = False,
        noisy_linear_critic: bool = False,
        output_dropout: float = 0.0,
        r2d2_sequence_length: int = 0,
        r2d2_burn_in: int = 0,
        use_simbav2: bool = False,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
    ):
        """
        Initializes the QRCNNSophyResidual.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            hidden_dim: Hidden dimension for MLP.
            num_blocks: Number of residual blocks.
            seed: Random seed.
            quantiles_number: Number of quantiles for TQC.
            split_track_observation: Whether to split track observation into a Conv1d branch.
            use_rnn: Whether to use an RNN layer.
            rnn_hidden_size: Hidden size for the RNN (defaults to hidden_dim if None).
            track_encoder: Type of track encoder ("conv1d", "spline_mlp", or "gtn").
            api_layernorm: Whether to apply LayerNorm to the API input.
            noisy_linear_critic: Whether to use NoisyLinear for the output.
            output_dropout: Dropout rate for the output.
            r2d2_sequence_length: Sequence length for R2D2 protocol.
            r2d2_burn_in: Burn-in length for R2D2 protocol.
            use_simbav2: Whether to use SimbaV2 backbone instead of residual MLP.
            gnn_hidden: Hidden dim for GNN track encoder.
            gnn_layers: Number of GNN layers for track encoder.
        """
        super().__init__()
        torch.manual_seed(seed)

        rnn_hidden = rnn_hidden_size if rnn_hidden_size is not None else hidden_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        self.num_quantiles = quantiles_number

        self._use_track_conv = split_track_observation and len(observation_space) > 1
        self._track_channels = (
            _TRACK_CHANNELS_GTN if _is_gtn_encoder(track_encoder) else _TRACK_CHANNELS_DEFAULT
        )
        if self._use_track_conv:
            dim_track_first = math.prod(observation_space[0].shape)
            if dim_track_first % self._track_channels != 0:
                self._use_track_conv = False
        self._use_rnn = use_rnn
        self._r2d2_sequence_length = r2d2_sequence_length
        self._r2d2_burn_in = r2d2_burn_in

        if self._use_track_conv:
            dim_track = math.prod(observation_space[0].shape)
            dim_physics = dim_obs - dim_track
            assert dim_track % self._track_channels == 0, (
                f"track_info should be {self._track_channels}*N for selected track encoder"
            )
            self._dim_track = dim_track
            self._dim_physics = dim_physics
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
            self.rnn: nn.GRU | None = None
            if self._use_rnn:
                self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
                backbone_input_dim = rnn_hidden
            else:
                backbone_input_dim = joint_dim
            self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
            self.backbone = _make_backbone(
                backbone_input_dim, hidden_dim, num_blocks, use_simbav2=use_simbav2
            )
            self.layernorm_api = None
        else:
            self.layernorm_api = nn.LayerNorm(dim_obs) if api_layernorm else None
            self.backbone = _make_backbone(dim_obs, hidden_dim, num_blocks, use_simbav2=use_simbav2)

        self.mlp_act = nn.Linear(hidden_dim + dim_act, hidden_dim)
        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        if noisy_linear_critic:
            self.model_out = NoisyLinear(
                hidden_dim, self.num_quantiles, device=self.device, std_init=0.01
            )
        else:
            self.model_out = nn.Linear(hidden_dim, self.num_quantiles)
        self.dropout = nn.Dropout(output_dropout) if output_dropout > 0.0 else None

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        """
        Combines track features from Conv1d and physics features from MLP.

        Args:
            observation: Input observation.
            batch_size (int): Current batch size.

        Returns:
            torch.Tensor: Combined features.
        """
        if self._use_track_conv:
            track = observation[0].view(batch_size, -1).float()
            track = track.view(
                batch_size, self._track_channels, self._dim_track // self._track_channels
            )
            track_embed = self.track_conv(track)
            physics = _obs_to_flat_tensor(observation[1:], batch_size)
            physics_embed = self.physics_proj(physics)
            return torch.cat([track_embed, physics_embed], dim=-1)
        result: torch.Tensor = self._obs_to_tensor(observation)
        return result

    def _obs_to_tensor(self, observation) -> torch.Tensor:
        """
        Converts observation to a single tensor.

        Args:
            observation: Input observation.

        Returns:
            torch.Tensor: Combined observation tensor.
        """
        if isinstance(observation, torch.Tensor):
            return observation.view(observation.shape[0], -1).float()
        if isinstance(observation, (tuple, list)) and len(observation) > 0:
            batch_size = observation[0].shape[0]
            return _obs_to_flat_tensor(observation, batch_size)
        raise ValueError(
            "SophyResidual critic expected observation to be a non-empty tuple/list of tensors "
            "or a single tensor (batch, dim). Got empty or unsupported type."
        )

    def forward(self, observation, act):
        """
        Forward pass for the critic.

        Args:
            observation: Input observation.
            act: Input action.

        Returns:
            torch.Tensor: Quantile values.
        """
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]

        if self._use_track_conv:
            joint = self._joint_features(observation, batch_size)
            if self._use_rnn and self.rnn is not None:
                seq_len = int(self._r2d2_sequence_length)

                burn_in_len = int(self._r2d2_burn_in)

                if (
                    seq_len > 0
                    and batch_size % seq_len == 0
                    and burn_in_len > 0
                    and burn_in_len < seq_len
                ):
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

                    joint = torch.cat([out_burn, out_active], dim=1).reshape(batch_size, -1)
                elif seq_len > 0 and batch_size % seq_len == 0:
                    num_seq = batch_size // seq_len
                    joint = joint.view(num_seq, seq_len, -1)
                    self.rnn.flatten_parameters()
                    joint, _ = self.rnn(joint)
                    joint = joint.reshape(batch_size, -1)
                else:
                    joint = joint.unsqueeze(1)
                    self.rnn.flatten_parameters()
                    joint, _ = self.rnn(joint)
                    joint = joint.squeeze(1)
            joint = self.layernorm_joint(joint)
            backbone_out = self.backbone(joint)
        else:
            obs_seq_cat = self._obs_to_tensor(observation)
            if self.layernorm_api is not None:
                obs_seq_cat = self.layernorm_api(obs_seq_cat)
            backbone_out = self.backbone(obs_seq_cat)

        cat_act = torch.cat([backbone_out, act], dim=-1)
        mlp_out = torch.nn.functional.silu(self.mlp_act(cat_act))
        out = self.head_proj(mlp_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.squeeze(self.model_out(out), -1)
