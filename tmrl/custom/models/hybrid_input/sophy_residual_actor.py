"""Sophy residual actor: backbone helpers + SquashedActorSophyResidual."""

import math

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Normal

from tmrl.actor import TorchActorModule
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    residual_mlp_backbone,
    simba_v2_backbone,
)
from tmrl.custom.models.shared.track_encoders import (
    TRACK_CHANNELS_DEFAULT,
    TRACK_CHANNELS_GTN,
    build_track_gtn_branch,
    is_gtn_encoder,
)
from tmrl.custom.utils.nn_conv import GSDEModule
from tmrl.registry import MODELS

_TRACK_CHANNELS_DEFAULT = TRACK_CHANNELS_DEFAULT
_TRACK_CHANNELS_GTN = TRACK_CHANNELS_GTN
_is_gtn_encoder = is_gtn_encoder


def _build_track_conv1d_branch(dim_track: int, hidden_dim: int) -> nn.Module:
    """
    Builds a Conv1d extractor for track_info.

    Args:
        dim_track (int): Dimension of the track information.
        hidden_dim (int): Dimension of the output hidden layer.

    Returns:
        torch.nn.Module: The constructed Conv1d branch.
    """
    assert dim_track >= 4, "track dim must be at least 4"
    assert dim_track % 4 == 0, "track dim must be 4*N (left_x, left_y, right_x, right_y)"
    return nn.Sequential(
        nn.Conv1d(4, 32, kernel_size=5, padding=2),
        nn.BatchNorm1d(32),
        nn.ReLU(inplace=True),
        nn.Conv1d(32, 64, kernel_size=5, padding=2),
        nn.BatchNorm1d(64),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(64, hidden_dim),
    )


def _build_track_spline_mlp_branch(dim_track: int, hidden_dim: int) -> nn.Module:
    """
    Parametric track encoder: compresses track (left/center/right) into a low-dim
    representation via pooling + MLP (B-spline / Frenet-Serret style compact encoding).
    Input (B, 3, N) -> pool to fixed size -> MLP -> (B, hidden_dim).
    """
    assert dim_track >= 4, "track dim must be at least 4"
    assert dim_track % 4 == 0, "track dim must be 4*N (4 channels)"
    n_pts = dim_track // 4
    pool_size = min(16, max(4, n_pts // 4))
    return nn.Sequential(
        nn.AdaptiveAvgPool1d(pool_size),
        nn.Flatten(),
        nn.Linear(4 * pool_size, 128),
        nn.LayerNorm(128),
        nn.SiLU(),
        nn.Linear(128, hidden_dim),
    )


_build_track_gnn_branch = build_track_gtn_branch


def _make_backbone(
    input_dim: int, hidden_dim: int, num_blocks: int, use_simbav2: bool = False
) -> nn.Module:
    """Build residual MLP or SimbaV2 backbone depending on config."""
    if use_simbav2:
        return simba_v2_backbone(input_dim, hidden_dim, num_blocks)
    return residual_mlp_backbone(input_dim, hidden_dim, num_blocks)


def _obs_to_flat_tensor(observation, batch_size: int) -> torch.Tensor:
    """
    Combines a list of observation tensors into a single flat tensor.

    Args:
        observation: Observation list or tuple of tensors.
        batch_size (int): Current batch size.

    Returns:
        torch.Tensor: Flattened and concatenated observation tensor.
    """
    if isinstance(observation, torch.Tensor):
        return observation.view(batch_size, -1).float()
    obs_list = list(observation)
    for i in range(len(obs_list)):
        obs_list[i] = obs_list[i].view(batch_size, -1)
    return torch.cat(obs_list, -1).float()


@MODELS.register("sophy_residual_actor")
class SquashedActorSophyResidual(TorchActorModule):
    """
    Sophy actor with residual MLP backbone and optional Conv1d track branch and RNN.
    Supports gSDE (generalized State-Dependent Exploration) when use_sde=True.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        seed: int = 42,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
        split_track_observation: bool = True,
        use_rnn: bool = False,
        rnn_hidden_size: int | None = None,
        track_encoder: str = "conv1d",
        api_layernorm: bool = False,
        binary_brake: bool = False,
        init_gas_bias: float = 0.0,
        output_dropout: float = 0.0,
        r2d2_sequence_length: int = 0,
        r2d2_burn_in: int = 0,
        use_simbav2: bool = False,
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
    ):
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)

        rnn_hidden = rnn_hidden_size if rnn_hidden_size is not None else hidden_dim

        dim_obs = sum(math.prod(s for s in space.shape) for space in observation_space)
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self._dim_obs = dim_obs
        self.use_sde = use_sde
        self._sde_clip_mean = sde_clip_mean
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

        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self._binary_brake = binary_brake and dim_act >= 3
        self.brake_logits_layer: nn.Linear | None
        if self._binary_brake:
            self._cont_dim = 2
            self.brake_logits_layer = nn.Linear(hidden_dim, 2)
        else:
            self._cont_dim = dim_act
            self.brake_logits_layer = None

        self.mu_layer = nn.Linear(hidden_dim, self._cont_dim)
        self.sde: GSDEModule | None = None
        self.log_std_layer: nn.Linear | None = None

        if self.use_sde:
            self.sde = GSDEModule(hidden_dim, self._cont_dim, log_std_init=log_std_init)
        else:
            self.log_std_layer = nn.Linear(hidden_dim, self._cont_dim)

        if dim_act > 0 and init_gas_bias != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = init_gas_bias
        self.act_limit = act_limit
        self.log_std_min = LOG_STD_MIN
        self.log_std_max = LOG_STD_MAX
        self._dim_act = dim_act
        self.dropout = nn.Dropout(output_dropout) if output_dropout > 0.0 else None

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE exploration matrix. No-op if gSDE is disabled."""
        if self.sde is not None:
            self.sde.reset_noise(batch_size)

    def load_from_bytes(self, payload: bytes, device) -> bool:
        ok = super().load_from_bytes(payload, device)
        if ok:
            self.reset_noise()
        return ok

    def _joint_features(self, observation, batch_size: int) -> torch.Tensor:
        """
        Combines track features from Conv1d and physics features from MLP.

        Args:
            observation: Input observation.
            batch_size (int): Current batch size.

        Returns:
            torch.Tensor: Combined features.
        """
        track = observation[0].view(batch_size, -1).float()
        track = track.view(
            batch_size, self._track_channels, self._dim_track // self._track_channels
        )
        track_embed = self.track_conv(track)
        physics = _obs_to_flat_tensor(observation[1:], batch_size)
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, physics_embed], dim=-1)

    def _obs_to_tensor(self, observation):
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
            "SophyResidual actor expected observation to be a non-empty tuple/list of tensors "
            "or a single tensor (batch, dim). Got empty or unsupported type."
        )

    def _compute_features(self, observation):
        """Extract backbone features from observation."""
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
            if obs_seq_cat.shape[-1] != self._dim_obs:
                raise ValueError(
                    f"SophyResidual actor expected observation dimension {self._dim_obs}, "
                    f"but got {obs_seq_cat.shape[-1]}."
                )
            if self.layernorm_api is not None:
                obs_seq_cat = self.layernorm_api(obs_seq_cat)
            backbone_out = self.backbone(obs_seq_cat)

        out = self.head_proj(backbone_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

    @staticmethod
    def _squash_log_prob(logp: torch.Tensor, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Apply tanh squash correction to Gaussian log prob (SAC appendix C)."""
        corr = 2 * (np.log(2) - pre_tanh_action - functional.softplus(-2 * pre_tanh_action))
        logp -= corr.sum(axis=1)
        return logp

    def _policy_head_standard(self, out, mu, test, with_logprob):
        """Standard Gaussian + tanh policy head."""
        log_std_layer = self.log_std_layer
        assert log_std_layer is not None
        log_std = log_std_layer(out)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi = self._squash_log_prob(logp_pi, pi_action)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def _policy_head_sde(self, out, mu, test, with_logprob):
        """gSDE policy head: state-dependent noise through exploration matrix.
        Clamp pre-tanh mean to [-sde_clip_mean, sde_clip_mean] so actions never saturate
        (tanh(±2) ≈ ±0.96). Forward still returns raw mu for mean_penalty gradient.
        """
        sde = self.sde
        assert sde is not None
        latent_sde = out.float()
        mu = mu.float()
        mu_clipped = torch.clamp(mu, -self._sde_clip_mean, self._sde_clip_mean)
        variance = sde.get_variance(latent_sde)
        pi_distribution = Normal(mu_clipped, torch.sqrt(variance + sde.epsilon))
        if test:
            pi_action = mu_clipped
        else:
            noise = sde.get_noise(latent_sde)
            pi_action = mu_clipped + noise
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi = self._squash_log_prob(logp_pi, pi_action)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def forward(self, observation, test=False, with_logprob=True, **kwargs):
        out = self._compute_features(observation)
        mu = self.mu_layer(out)

        if self._binary_brake:
            brake_lin = self.brake_logits_layer
            assert brake_lin is not None
            brake_logits = brake_lin(out).float()
            if self.use_sde and self.sde is not None:
                pi_cont, logp_cont, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_cont, logp_cont, _pre_tanh = self._policy_head_standard(
                    out, mu, test, with_logprob
                )
            if test:
                brake_onehot = functional.one_hot(
                    brake_logits.argmax(dim=-1), num_classes=2
                ).float()
            else:
                brake_onehot = functional.gumbel_softmax(brake_logits, tau=1.0, hard=True)
            brake_val = brake_onehot[:, 1:2]
            # pi_cont is [gas, steer], we want [gas, brake, steer]
            pi_action = torch.cat([pi_cont[..., 0:1], brake_val, pi_cont[..., 1:2]], dim=-1)
            logp_pi = None
            if with_logprob and logp_cont is not None:
                logp_brake = functional.log_softmax(brake_logits, dim=-1)
                brake_idx = (brake_val > 0.5).long().squeeze(-1)
                logp_pi = logp_cont + logp_brake.gather(1, brake_idx.unsqueeze(-1)).squeeze(-1)
            if kwargs.get("return_pre_tanh_mean", False):
                return pi_action.squeeze(), logp_pi, mu
            return pi_action.squeeze(), logp_pi
        else:
            if self.use_sde and self.sde is not None:
                pi_action, logp_pi, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_action, logp_pi, _ = self._policy_head_standard(out, mu, test, with_logprob)
            if kwargs.get("return_pre_tanh_mean", False):
                return pi_action.squeeze(), logp_pi, mu
            return pi_action.squeeze(), logp_pi

    def act(self, obs, test=False):
        """
        Predicts an action from an observation.

        Args:
            obs: Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(observation=obs_seq, test=test, with_logprob=False)
            return a.cpu().numpy()
