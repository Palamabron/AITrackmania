"""Sophy residual actor-critic compositions."""

from io import BytesIO

import torch
from torch import nn

from tmrl.actor import TorchActorModule
from tmrl.custom.models.hybrid_input.sophy_residual_actor import SquashedActorSophyResidual
from tmrl.custom.models.hybrid_input.sophy_residual_critic import QRCNNSophyResidual
from tmrl.registry import MODELS


@MODELS.register("sophy_residual_ac")
class SophyResidualActorCritic(nn.Module):
    """
    Actor-critic for TQC with residual MLP backbone (LayerNorm + SiLU).
    Asymmetric: critic can have more blocks than actor.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks_actor: int = 3,
        num_blocks_critic: int = 3,
        seed: int = 42,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        """
        Initializes the SophyResidualActorCritic.

        Args:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            hidden_dim: Hidden dimension for MLP.
            num_blocks_actor: Number of residual blocks for actor.
            num_blocks_critic: Number of residual blocks for critic.
            seed: Random seed.
            use_sde: Enable generalized State-Dependent Exploration.
            log_std_init: Initial log-std for gSDE.
            sde_clip_mean: Clip pre-tanh mean when using gSDE.
        """
        super().__init__()
        self.actor = SquashedActorSophyResidual(
            observation_space,
            action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        # Critic receives full privileged observation (Track + Telemetry)
        self.q1 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 2,
        )

    def act(self, obs, test=False):
        """
        Predicts an action from an observation.

        Args:
            obs: Input observation.
            test (bool): Whether in test mode. Defaults to False.

        Returns:
            np.ndarray: Predicted action.
        """
        with torch.no_grad():
            return self.actor.act(obs, test=test)


class _AsymmetricActorAdapter(TorchActorModule):
    """Adapter so actor can consume full obs and internally keep ego-only inputs.

    Wraps a ``SquashedActorSophyResidual`` that was built on ego-only observations
    and strips the privileged track observation (index 0) before forwarding.
    Serialisation writes the inner actor's state dict directly so workers can load
    without an ``'actor.'`` key prefix.
    """

    def __init__(self, actor: SquashedActorSophyResidual, full_obs_len: int):
        """Wrap an ego-only actor for use in a full-obs actor-critic.

        Args:
            actor: Pre-built SquashedActorSophyResidual trained on ego-only observations.
            full_obs_len: Expected length of the full (privileged) observation tuple.
                Used to detect whether stripping is needed.
        """
        super().__init__(actor.observation_space, actor.action_space)
        self.actor = actor
        self.full_obs_len = full_obs_len

    def _to_ego_obs(self, obs):
        """Strip the privileged track slot (index 0) from a full observation tuple.

        If ``obs`` does not look like a full observation (wrong length or not a
        sequence), it is returned unchanged.

        Args:
            obs: Full or ego-only observation tuple.

        Returns:
            Ego-only slice obs[1:] when obs has exactly ``full_obs_len`` elements,
            otherwise obs unchanged.
        """
        if (
            isinstance(obs, (tuple, list))
            and self.full_obs_len > 1
            and len(obs) == self.full_obs_len
        ):
            return obs[1:]
        return obs

    def forward(self, observation, test: bool = False, with_logprob: bool = True, **kwargs):
        """Forward call stripping privileged obs before delegating to the inner actor.

        Args:
            observation: Full observation tuple of length ``full_obs_len``.
            test: Passed through to the inner actor.
            with_logprob: Passed through to the inner actor.
            **kwargs: Forwarded to the inner actor (e.g. return_pre_tanh_mean).

        Returns:
            Same as ``SquashedActorSophyResidual.forward``.
        """
        return self.actor(
            self._to_ego_obs(observation), test=test, with_logprob=with_logprob, **kwargs
        )

    def act(self, obs, test: bool = False):
        """Return a numpy action for the rollout worker.

        Args:
            obs: Full or ego-only observation tuple.
            test: If True, use the deterministic mean action.

        Returns:
            numpy.ndarray of shape (dim_act,).
        """
        return self.actor.act(self._to_ego_obs(obs), test=test)

    def save_to_bytes(self) -> bytes:
        """Serialize only the inner actor so the worker (a bare SquashedActorSophyResidual)
        can load the state_dict without an 'actor.' key prefix."""
        buffer = BytesIO()
        torch.save(self.actor.state_dict(), buffer)
        return buffer.getvalue()

    def load_from_bytes(self, payload: bytes, device) -> bool:
        """Load weights into the inner actor, tolerating shape mismatches gracefully.

        Args:
            payload: Serialised state dict bytes produced by :meth:`save_to_bytes`.
            device: Target device for weight loading.

        Returns:
            True on success, False if weights were skipped due to a shape mismatch.

        Raises:
            RuntimeError: For errors unrelated to shape or key mismatches.
        """
        self.device = device
        buffer = BytesIO(payload)
        try:
            state = torch.load(buffer, map_location=self.device, weights_only=True)
            self.actor.load_state_dict(state)
        except RuntimeError as e:
            err = str(e)
            if "size mismatch" in err or "Missing key" in err or "shape" in err.lower():
                from loguru import logger

                logger.warning(
                    "Ignoring incompatible asymmetric actor weights (shape mismatch): {}",
                    err.split("\n", 1)[0].strip(),
                )
                return False
            raise
        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise()
        return True

    def reset_noise(self, batch_size: int = 1) -> None:
        """Re-sample the gSDE exploration matrix if the inner actor supports it.

        Args:
            batch_size: Batch size hint passed to the inner actor's reset_noise.
        """
        if hasattr(self.actor, "reset_noise"):
            self.actor.reset_noise(batch_size)


@MODELS.register("sophy_asymmetric_ac")
class AsymmetricSophyResidualActorCritic(nn.Module):
    """Implements the Blueprint from GT Sophy:
    Actor: Restricted to ego-centric telemetry (velocity, inputs, local rays).
    Critic: Privileged access to global track geometry lookahead.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_dim: int = 256,
        num_blocks_actor: int = 3,
        num_blocks_critic: int = 3,
        seed: int = 42,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        sde_clip_mean: float = 2.0,
    ):
        """Construct the asymmetric actor-critic.

        The actor receives only ego-centric telemetry (observation_space[1:]);
        the privileged track geometry at index 0 is stripped at both training and
        rollout time via ``_AsymmetricActorAdapter``.  Both critics receive the
        full observation.

        Args:
            observation_space: Full (privileged) observation space including track.
            action_space: Gymnasium action space.
            hidden_dim: Residual MLP hidden width for actor and critics.
            num_blocks_actor: Number of residual blocks in the actor backbone.
            num_blocks_critic: Number of residual blocks in each critic backbone.
            seed: Base random seed (critics use seed+1 and seed+2).
            use_sde: Enable generalized State-Dependent Exploration in the actor.
            log_std_init: Initial log-std for gSDE.
            sde_clip_mean: Pre-tanh mean clipping bound when using gSDE.
        """
        super().__init__()
        # Actor only receives ego-centric telemetry (drop privileged track slot 0).
        if hasattr(observation_space, "spaces"):
            ego_space = tuple(observation_space.spaces[1:])
        elif hasattr(observation_space, "__getitem__"):
            ego_space = observation_space[1:]
        else:
            ego_space = observation_space

        base_actor = SquashedActorSophyResidual(
            observation_space=ego_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_actor,
            seed=seed,
            use_sde=use_sde,
            log_std_init=log_std_init,
            sde_clip_mean=sde_clip_mean,
        )
        full_obs_len = len(observation_space) if hasattr(observation_space, "__len__") else 0
        self.actor = _AsymmetricActorAdapter(base_actor, full_obs_len)

        # Critic receives full privileged observation (Track + Telemetry)
        self.q1 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 1,
        )
        self.q2 = QRCNNSophyResidual(
            observation_space=observation_space,
            action_space=action_space,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks_critic,
            seed=seed + 2,
        )

    def act(self, obs, test: bool = False):
        """Return a numpy action for the rollout worker.

        Args:
            obs: Full observation tuple (with privileged track at index 0).
            test: If True, use the deterministic mean action.

        Returns:
            numpy.ndarray of shape (dim_act,).
        """
        # Slice privileged data off for the actor (indices 1:15 to be safe with tuple spaces)
        if (isinstance(obs, tuple) and len(obs) > 1) or (isinstance(obs, list) and len(obs) > 1):
            ego_obs = obs[1:]
        else:
            ego_obs = obs
        return self.actor.act(ego_obs, test=test)
