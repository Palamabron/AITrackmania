"""GNN + EfficientNet + Sophy hybrid actor-critic models.

This module contains actor-critic implementations combining GNN track encoders,
EfficientNet image encoders, and Sophy-style residual MLP backbones.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchrl.modules import NoisyLinear

from tmrl.actor import TorchActorModule
from tmrl.custom.models.hybrid_input.sophy import _obs_to_flat_tensor
from tmrl.custom.models.image_input.efficientnet import (
    _gnn_effnet_image_index,
    _gnn_effnet_physics_dims,
)
from tmrl.custom.models.shared.blocks import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    FrozenEfficientNetEncoder,
    ResidualMLPBlock,
    ensure_float,
    obs_spaces_list,
)
from tmrl.custom.models.shared.track_encoders import (
    TRACK_CHANNELS_GTN,
    build_track_gtn_branch,
)
from tmrl.custom.utils.nn_conv import GSDEModule
from tmrl.registry import MODELS
from tmrl.util import prod

_LOG2 = float(np.log(2.0))
_GNN_TRACK_FEATURES_PER_POINT = TRACK_CHANNELS_GTN


def _gnn_effnet_sophy_residual_backbone(
    input_dim: int, hidden_dim: int, num_blocks: int
) -> nn.Module:
    """Residual MLP with LayerNorm after the input projection and each residual sum.

    Scale factor ``1 / sqrt(num_blocks)`` keeps residual branch magnitudes bounded
    as depth increases (following T-Fixup / μP intuition).

    Args:
        input_dim: Input feature dimension fed into the first linear layer.
        hidden_dim: Width of all hidden layers and residual blocks.
        num_blocks: Number of ``ResidualMLPBlock`` stages appended after the projection.

    Returns:
        Sequential module mapping (N, input_dim) -> (N, hidden_dim).
    """
    scale = 1.0 / max(1, num_blocks) ** 0.5
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    ]
    for _ in range(num_blocks):
        layers.append(ResidualMLPBlock(hidden_dim, scale=scale))
        layers.append(nn.LayerNorm(hidden_dim))
    return nn.Sequential(*layers)


_build_track_gnn_branch = build_track_gtn_branch


def _ensure_image_4d(imgs: torch.Tensor, image_index: int) -> None:
    """Validate that the image tensor has the expected 4D shape (N,C,H,W)."""
    if imgs.dim() == 2 and imgs.shape[-1] == 3:
        raise RuntimeError(
            f"Obs at index {image_index} has shape {tuple(imgs.shape)} (action, not image). "
            "Env may append action buffer after image. Use TQCGRAB_IMAGES + USE_IMAGES=true; "
            "same config on trainer and worker."
        )
    if imgs.dim() != 4:
        raise RuntimeError(
            f"Image at index {image_index} must be 4D (N,C,H,W), got {tuple(imgs.shape)}."
        )


class _GnnEffNetJointFeaturesMixin:
    """Shared feature extraction for GNN + EfficientNet + Sophy models.

    Provides :meth:`_joint_features` and :meth:`_apply_rnn` for both the actor
    and critic to use without code duplication.  Concrete sub-classes must populate
    the declared class attributes during ``__init__`` before calling these methods.
    """

    _image_index: int
    _dim_track: int
    _use_rnn: bool
    _r2d2_sequence_length: int
    track_gnn: nn.Module
    image_encoder: nn.Module
    img_proj: nn.Module
    physics_proj: nn.Module
    rnn: nn.GRU | None
    layernorm_joint: nn.Module

    def _joint_features(self, obs, batch_size: int) -> torch.Tensor:
        """Encode track (GNN), image (EffNet + proj), and physics (linear) and concatenate.

        Observation layout: obs[0]=track, obs[1:image_index]=physics, obs[image_index]=image.
        Track is reshaped to (N, _GNN_TRACK_FEATURES_PER_POINT, n_points) before the GNN.

        Args:
            obs: List of observation tensors for the current batch.
            batch_size: Number of samples in the batch.

        Returns:
            Joint embedding of shape (N, track_hidden + embed_dim + physics_hidden).
        """
        track = ensure_float(obs[0].view(batch_size, -1)).view(
            batch_size,
            _GNN_TRACK_FEATURES_PER_POINT,
            self._dim_track // _GNN_TRACK_FEATURES_PER_POINT,
        )
        physics = _obs_to_flat_tensor(obs[1 : self._image_index], batch_size)
        track_embed = self.track_gnn(track)
        imgs = ensure_float(obs[self._image_index])
        _ensure_image_4d(imgs, self._image_index)
        img_embed = self.img_proj(self.image_encoder(imgs))
        physics_embed = self.physics_proj(physics)
        return torch.cat([track_embed, img_embed, physics_embed], dim=-1)

    def _apply_rnn(self, joint: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Optionally pass the joint embedding through the GRU.

        When ``r2d2_sequence_length > 0`` and ``batch_size`` is divisible by it,
        the batch is reshaped to (num_sequences, seq_len, dim) for sequential GRU
        processing.  Otherwise a single-step unsqueeze/squeeze is used.

        Args:
            joint: Joint embedding of shape (N, joint_dim).
            batch_size: Number of samples (equals N).

        Returns:
            GRU output of shape (N, rnn_hidden) or the input unchanged if use_rnn=False.
        """
        if not self._use_rnn or self.rnn is None:
            return joint
        seq_len = int(self._r2d2_sequence_length)
        if seq_len > 0 and batch_size % seq_len == 0:
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
        return joint


@MODELS.register("gnn_effnet_sophy_residual_actor")
class SquashedActorGnnEffNetSophyResidual(_GnnEffNetJointFeaturesMixin, TorchActorModule):
    """Sophy-style actor with GNN track encoder, EfficientNet image encoder, and residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = 42,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
        use_rnn: bool = False,
        rnn_hidden_size: int | None = None,
        binary_brake: bool = False,
        output_dropout: float = 0.0,
        init_gas_bias: float = 0.0,
        r2d2_sequence_length: int = 0,
    ):
        """Construct the GNN + EffNet + Sophy residual actor.

        Observation layout is inferred from the observation space:
        index 0 is the track geometry (GNN input), the image is at ``image_index``
        (auto-detected), and everything else is physics.

        Args:
            observation_space: Full observation space; iterable of sub-spaces.
            action_space: Gymnasium action space.
            hidden_dim: Hidden width for track GNN output, physics projection,
                and residual backbone.
            num_blocks: Number of residual blocks in the backbone.
            embed_dim: EfficientNet encoder output dimension.
            width_mult: EfficientNet channel multiplier.
            variant: EfficientNet-V2 variant identifier.
            use_dw_stem: Depthwise stem in EffNet.
            use_frozen_effnet: If True, freeze EffNet weights.
            seed: Random seed.
            use_sde: Enable gSDE.
            log_std_init: gSDE initial log-std.
            sde_clip_mean: Pre-tanh mean clipping for gSDE.
            use_rnn: Whether to add a GRU over the joint embedding.
            rnn_hidden_size: GRU hidden size (defaults to hidden_dim).
            binary_brake: Discrete brake head via Gumbel-softmax.
            output_dropout: Dropout after head projection.
            init_gas_bias: Initial bias for the gas (throttle) output.
            r2d2_sequence_length: Sequence length for R2D2 GRU unrolling (0=off).
        """
        super().__init__(observation_space, action_space)
        torch.manual_seed(seed)
        self.use_sde = use_sde
        self._sde_clip_mean = sde_clip_mean
        self._r2d2_sequence_length = r2d2_sequence_length
        spaces = obs_spaces_list(observation_space)
        image_index = _gnn_effnet_image_index(observation_space)
        self._image_index = image_index
        dim_track, dim_physics = _gnn_effnet_physics_dims(observation_space, image_index)
        self._dim_track = dim_track
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]

        self.track_gnn = _build_track_gnn_branch(dim_track, hidden_dim)
        nb_channels_in = (
            int(spaces[image_index].shape[0])
            if len(spaces[image_index].shape) >= 2
            else int(prod(spaces[image_index].shape))
        )
        self.image_encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            frozen=use_frozen_effnet,
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(dim_physics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        joint_dim = hidden_dim + embed_dim + hidden_dim
        self._use_rnn = use_rnn
        rnn_hidden = rnn_hidden_size if rnn_hidden_size is not None else hidden_dim
        self.rnn: nn.GRU | None
        if self._use_rnn:
            self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
            backbone_input_dim = rnn_hidden
        else:
            self.rnn = None
            backbone_input_dim = joint_dim
        self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
        self.backbone = _gnn_effnet_sophy_residual_backbone(
            backbone_input_dim, hidden_dim, num_blocks
        )
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

        self.act_limit = act_limit
        self.dropout = nn.Dropout(output_dropout) if output_dropout > 0.0 else None
        if dim_act > 0 and init_gas_bias != 0.0:
            with torch.no_grad():
                self.mu_layer.bias.data[0] = init_gas_bias

    def reset_noise(self, batch_size: int = 1) -> None:
        """Sample new gSDE exploration matrix. No-op if gSDE is disabled."""
        if self.sde is not None:
            self.sde.reset_noise(batch_size)

    def load_from_bytes(self, payload: bytes, device) -> bool:
        ok = super().load_from_bytes(payload, device)
        if ok:
            self.reset_noise()
        return ok

    @staticmethod
    def _squash_log_prob(logp: torch.Tensor, pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Apply tanh squash correction to Gaussian log prob (SAC appendix C).

        Args:
            logp: Un-corrected log-probability tensor of shape (N,).
            pre_tanh_action: Pre-tanh action sample of shape (N, dim_act),
                already clamped to ``_SQUASH_LOGPROB_CLAMP`` by the caller.

        Returns:
            Corrected log-probability tensor of shape (N,).
        """
        corr = 2 * (_LOG2 - pre_tanh_action - F.softplus(-2 * pre_tanh_action))
        logp -= corr.sum(dim=1)  # type: ignore[call-overload]
        return logp

    # Clamp pre-tanh to avoid -inf/NaN in squash log-prob correction when action is extreme
    _SQUASH_LOGPROB_CLAMP = 20.0

    def _policy_head_standard(
        self, out: torch.Tensor, mu: torch.Tensor, test: bool, with_logprob: bool
    ):
        """Standard Gaussian + tanh policy head.

        Args:
            out: Backbone feature vector of shape (N, hidden_dim).
            mu: Pre-tanh mean from mu_layer, shape (N, _cont_dim).
            test: If True, use mu directly (no sampling).
            with_logprob: If True, compute tanh-corrected log-probability with
                pre-tanh action clamped to ``_SQUASH_LOGPROB_CLAMP`` for numerical
                stability.

        Returns:
            Tuple of (squashed_action, logp_pi, pre_tanh_action).
        """
        log_std_layer = self.log_std_layer
        assert log_std_layer is not None
        log_std = log_std_layer(out).float()
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        pi_action = mu if test else pi_distribution.rsample()
        logp_pi = None
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            pi_action_for_corr = pi_action.clamp(
                -self._SQUASH_LOGPROB_CLAMP, self._SQUASH_LOGPROB_CLAMP
            )
            logp_pi = self._squash_log_prob(logp_pi, pi_action_for_corr)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def _policy_head_sde(self, out, mu, test, with_logprob):
        """Clamp pre-tanh mean to [-sde_clip_mean, sde_clip_mean] so actions never saturate.
        Forward still returns raw mu for mean_penalty gradient."""
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
            pi_action_for_corr = pi_action.clamp(
                -self._SQUASH_LOGPROB_CLAMP, self._SQUASH_LOGPROB_CLAMP
            )
            logp_pi = self._squash_log_prob(logp_pi, pi_action_for_corr)
        return torch.tanh(pi_action) * self.act_limit, logp_pi, pi_action

    def forward(self, obs, test: bool = False, with_logprob: bool = True, **kwargs):
        """Sample an action from the squashed Gaussian policy.

        Extracts joint features via :meth:`_joint_features`, optionally passes them
        through the GRU, normalises, runs the residual backbone, then routes through
        the standard or gSDE policy head.  When ``binary_brake=True`` the brake
        dimension is discrete.

        Args:
            obs: Observation tuple with track at [0], physics at [1:image_index], and
                image at [image_index].
            test: If True, return the deterministic mean action.
            with_logprob: If True, compute and return the log-probability.
            **kwargs: Optional ``return_pre_tanh_mean`` (bool).

        Returns:
            (action, logp_pi) or (action, logp_pi, mu). action shape: (dim_act,).
        """
        if isinstance(obs, (tuple, list)):
            obs = list(obs)
        batch_size = obs[0].shape[0]
        joint = self._joint_features(obs, batch_size)
        joint = self._apply_rnn(joint, batch_size)
        joint = self.layernorm_joint(joint)
        out = self.backbone(joint)
        out = self.head_proj(out)
        if self.dropout is not None:
            out = self.dropout(out)

        mu = self.mu_layer(out).float()

        if self._binary_brake:
            brake_lin = self.brake_logits_layer
            assert brake_lin is not None
            brake_logits = brake_lin(out).float()
            if self.use_sde and self.sde is not None:
                pi_cont, logp_cont, _ = self._policy_head_sde(out, mu, test, with_logprob)
            else:
                pi_cont, logp_cont, _ = self._policy_head_standard(out, mu, test, with_logprob)
            if test:
                brake_onehot = F.one_hot(brake_logits.argmax(dim=-1), num_classes=2).float()
            else:
                brake_onehot = F.gumbel_softmax(brake_logits, tau=1.0, hard=True)
            brake_val = brake_onehot[:, 1:2]
            # pi_cont is [gas, steer], we want [gas, brake, steer]
            pi_action = torch.cat([pi_cont[..., 0:1], brake_val, pi_cont[..., 1:2]], dim=-1)
            logp_pi = None
            if with_logprob and logp_cont is not None:
                logp_brake = F.log_softmax(brake_logits, dim=-1)
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

    def act(self, obs, test: bool = False):
        """Return a numpy action for the rollout worker (no-grad, CPU).

        Args:
            obs: Observation tuple as expected by :meth:`forward`.
            test: If True, use the deterministic mean action.

        Returns:
            numpy.ndarray of shape (dim_act,).
        """
        obs_seq = list(obs)
        with torch.no_grad():
            a, _ = self.forward(obs_seq, test=test, with_logprob=False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


@MODELS.register("gnn_effnet_sophy_residual_critic")
class QRCNNGnnEffNetSophyResidual(_GnnEffNetJointFeaturesMixin, nn.Module):
    """Sophy-style quantile regression critic with GNN, EfficientNet, and residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks: int = 16,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = 42,
        quantiles_number: int = 1,
        use_rnn: bool = False,
        rnn_hidden_size: int | None = None,
        noisy_linear_critic: bool = False,
        output_dropout: float = 0.0,
        r2d2_sequence_length: int = 0,
    ):
        """Construct the GNN + EffNet + Sophy residual critic.

        Args:
            observation_space: Full observation space; iterable of sub-spaces.
            action_space: Gymnasium action space.
            hidden_dim: Hidden width for track GNN, physics projection, and backbone.
            num_blocks: Number of residual blocks in the backbone.
            embed_dim: EfficientNet encoder output dimension.
            width_mult: EfficientNet channel multiplier.
            variant: EfficientNet-V2 variant identifier.
            use_dw_stem: Depthwise stem in EffNet.
            use_frozen_effnet: If True, freeze EffNet weights.
            seed: Random seed.
            quantiles_number: Number of quantile outputs per sample.
            use_rnn: Add a GRU over the joint embedding.
            rnn_hidden_size: GRU hidden size (defaults to hidden_dim).
            noisy_linear_critic: Replace the output linear with NoisyLinear.
            output_dropout: Dropout after head projection.
            r2d2_sequence_length: Sequence length for R2D2 GRU unrolling (0=off).
        """
        super().__init__()
        torch.manual_seed(seed)
        self._r2d2_sequence_length = r2d2_sequence_length
        spaces = obs_spaces_list(observation_space)
        image_index = _gnn_effnet_image_index(observation_space)
        self._image_index = image_index
        dim_track, dim_physics = _gnn_effnet_physics_dims(observation_space, image_index)
        self._dim_track = dim_track
        dim_act = action_space.shape[0]
        self.num_quantiles = quantiles_number

        self.track_gnn = _build_track_gnn_branch(dim_track, hidden_dim)
        nb_channels_in = (
            int(spaces[image_index].shape[0])
            if len(spaces[image_index].shape) >= 2
            else int(prod(spaces[image_index].shape))
        )
        self.image_encoder = FrozenEfficientNetEncoder(
            nb_channels_in=nb_channels_in,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            frozen=use_frozen_effnet,
        )
        self.img_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(dim_physics, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        joint_dim = hidden_dim + embed_dim + hidden_dim
        self._use_rnn = use_rnn
        rnn_hidden = rnn_hidden_size if rnn_hidden_size is not None else hidden_dim
        self.rnn: nn.GRU | None
        if self._use_rnn:
            self.rnn = nn.GRU(joint_dim, rnn_hidden, batch_first=True)
            backbone_input_dim = rnn_hidden
        else:
            self.rnn = None
            backbone_input_dim = joint_dim
        self.layernorm_joint = nn.LayerNorm(backbone_input_dim)
        self.backbone = _gnn_effnet_sophy_residual_backbone(
            backbone_input_dim, hidden_dim, num_blocks
        )
        self.mlp_act = nn.Linear(hidden_dim + dim_act, hidden_dim)
        self.head_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        if noisy_linear_critic:
            self.model_out = NoisyLinear(
                hidden_dim,
                self.num_quantiles,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                std_init=0.01,
            )
        else:
            self.model_out = nn.Linear(hidden_dim, self.num_quantiles)
        self.dropout = nn.Dropout(output_dropout) if output_dropout > 0.0 else None

    def forward(self, observation, act: torch.Tensor) -> torch.Tensor:
        """Compute quantile Q-values for an (observation, action) pair.

        Encodes joint track + image + physics features, passes them through the
        residual backbone, concatenates the action, and projects to quantile values.

        Args:
            observation: Observation tuple with track at [0], physics at [1:image_index],
                and image at [image_index].
            act: Action tensor of shape (N, dim_act).

        Returns:
            Q-value tensor of shape (N,) or (N, quantiles_number).
        """
        if isinstance(observation, (tuple, list)):
            observation = list(observation)
        batch_size = observation[0].shape[0]
        joint = self._joint_features(observation, batch_size)
        joint = self._apply_rnn(joint, batch_size)
        joint = self.layernorm_joint(joint)
        backbone_out = self.backbone(joint)
        cat_act = torch.cat([backbone_out, act], dim=-1)
        mlp_out = F.silu(self.mlp_act(cat_act))
        out = self.head_proj(mlp_out)
        if self.dropout is not None:
            out = self.dropout(out)
        return torch.squeeze(self.model_out(out), -1)


@MODELS.register("gnn_effnet_sophy_residual_ac")
class GnnEffNetSophyResidualActorCritic(nn.Module):
    """Complete actor-critic with GNN track + EffNet image + Sophy residual MLP."""

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks_actor: int = 8,
        num_blocks_critic: int = 16,
        embed_dim: int = 256,
        width_mult: float = 0.5,
        variant: str = "xs",
        use_dw_stem: bool = False,
        use_frozen_effnet: bool = True,
        seed: int = 42,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        """Construct the actor-critic with one actor and two independent critics.

        Args:
            observation_space: Passed through to actor and both critics.
            action_space: Passed through to actor and both critics.
            hidden_dim: Residual backbone hidden width.
            num_blocks_actor: Residual blocks in the actor backbone.
            num_blocks_critic: Residual blocks in each critic backbone.
            embed_dim: EfficientNet output dimension.
            width_mult: EfficientNet channel multiplier.
            variant: EfficientNet-V2 variant identifier.
            use_dw_stem: Depthwise stem in EffNet.
            use_frozen_effnet: Freeze EffNet weights.
            seed: Base random seed (critics use seed+1 and seed+2).
            use_sde: Enable gSDE in the actor.
            log_std_init: gSDE initial log-std.
            sde_clip_mean: Pre-tanh mean clipping for gSDE.
        """
        super().__init__()
        self.actor = SquashedActorGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        self.q1 = QRCNNGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed + 1,
        )
        self.q2 = QRCNNGnnEffNetSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            embed_dim=embed_dim,
            width_mult=width_mult,
            variant=variant,
            use_dw_stem=use_dw_stem,
            use_frozen_effnet=use_frozen_effnet,
            seed=seed + 2,
        )

    def act(self, obs, test: bool = False):
        """Return a numpy action using the actor (no-grad, CPU).

        Args:
            obs: Observation tuple as expected by the actor.
            test: If True, use the deterministic mean action.

        Returns:
            numpy.ndarray of shape (dim_act,).
        """
        with torch.no_grad():
            return self.actor.act(obs, test=test)
