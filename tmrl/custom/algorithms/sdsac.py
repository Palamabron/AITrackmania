"""Stable Discrete SAC (SD-SAC) agent for discrete action spaces.

Implements the three stabilisation tricks from
"Revisiting Discrete Soft Actor-Critic" (Zhou et al., TMLR 2024):
  1. Double Average Q-learning  (--avg-q)
  2. Q-clip                     (--clip-q)
  3. Entropy Penalty             (--entropy-penalty)

References:
  - coldsummerday/SD-SAC (official implementation)
  - arXiv / OpenReview: https://openreview.net/forum?id=EUF2R6VBeU
"""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from torch.distributions import Categorical
from torch.optim import Adam

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from tmrl.custom.algorithms._common import (
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    amp_setup,
    autocast_context,
    polyak_update,
    project_simbav2_weights,
    sanitize_obs,
    set_seed,
)
from tmrl.custom.algorithms._internal._sdsac_networks import (
    DiscreteSACActor,
    DiscreteSACNetwork,
)
from tmrl.custom.utils.nn_utils import copy_shared, no_grad
from tmrl.custom.utils.optim import GradientStabilizer
from tmrl.registry import ALGORITHMS
from tmrl.training import TrainingAgent
from tmrl.util import cached_property, wandb_monotonic_step


@ALGORITHMS.register("SDSAC")
@dataclass(eq=False)
class SDSACAgent(TrainingAgent):
    """Stable Discrete SAC agent with Double Avg Q, Q-clip, and Entropy Penalty.

    Designed for Trackmania with the same observation encoding as IQN but
    using a soft actor-critic framework for discrete actions.

    All hyperparameters are required constructor arguments — values must be
    supplied explicitly by the config pipeline (no hidden numeric defaults).
    """

    # --- Required: spaces ---
    observation_space: Any
    action_space: Any

    # --- Required: architecture ---
    hidden_dim: int
    num_blocks_actor: int
    num_blocks_critic: int
    n_cos: int
    n_actions: int

    # --- Required: SAC hyper-parameters ---
    gamma: float
    lr_actor: float
    lr_critic: float
    lr_alpha: float
    polyak: float
    n_steps: int
    auto_alpha: bool
    alpha_init: float

    # --- Required: SD-SAC tricks ---
    use_avg_q: bool
    use_clip_q: bool
    clip_q_epsilon: float
    use_entropy_penalty: bool
    entropy_penalty_beta: float

    # --- Required: optimizer ---
    weight_decay: float

    # --- Required: previously hidden globals ---
    reward_normalize_scale: float
    r2d2_burn_in: int
    r2d2_sequence_length: int

    # --- Required: mixed precision ---
    mixed_precision: bool
    mixed_precision_dtype: str

    # --- Required: reproducibility ---
    seed: int

    # --- Structural defaults (None = auto-detect / optional) ---
    device: str | None = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        set_seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DiscreteSACNetwork(
            self.observation_space,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            num_blocks_actor=self.num_blocks_actor,
            num_blocks_critic=self.num_blocks_critic,
            n_cos=self.n_cos,
        ).to(device)

        self.model_target = no_grad(deepcopy(self.model))

        self.actor_optimizer = Adam(
            itertools.chain(
                self.model.actor_backbone.parameters(), self.model.actor_head.parameters()
            ),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )
        self.critic_optimizer = Adam(
            itertools.chain(
                self.model.q1_backbone.parameters(),
                self.model.q1_head.parameters(),
                self.model.q2_backbone.parameters(),
                self.model.q2_head.parameters(),
            ),
            lr=self.lr_critic,
            weight_decay=self.weight_decay,
        )

        self.use_mixed_precision, self.amp_dtype, self.grad_scaler = amp_setup(
            device, self.mixed_precision, self.mixed_precision_dtype
        )

        self.log_alpha: torch.Tensor | None
        self.alpha_optimizer: Adam | None
        if self.auto_alpha:
            self.target_entropy = 0.98 * math.log(self.n_actions)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_alpha)
            self._alpha: float | torch.Tensor = self.log_alpha.detach().exp()
        else:
            self.target_entropy = 0.0
            self.log_alpha = None
            self.alpha_optimizer = None
            self._alpha = self.alpha_init

        self._grad_stabilizer_critic = GradientStabilizer(ema_decay=0.995)
        self._grad_stabilizer_actor = GradientStabilizer(ema_decay=0.995)
        self._training_step = 0
        logger.info(
            "SDSACAgent: n_actions={}, avg_q={}, clip_q={}, entropy_penalty={}",
            self.n_actions,
            self.use_avg_q,
            self.use_clip_q,
            self.use_entropy_penalty,
        )

        if self.reward_normalize_scale != 1.0:
            logger.warning(
                "SDSACAgent: reward_normalize_scale={:.4g} — rewards are MULTIPLIED by this "
                "factor. Previous versions divided; if you are loading an old config that "
                "used a large scale (e.g. 200) to shrink rewards, use the reciprocal "
                "(1/200 ≈ 0.005) to preserve the original effect.",
                self.reward_normalize_scale,
            )

    def get_actor(self) -> DiscreteSACActor:
        actor = DiscreteSACActor(
            self.observation_space,
            self.action_space,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks_actor,
            n_cos=self.n_cos,
            n_actions=self.n_actions,
            epsilon=0.0,
        )
        actor.q_net.backbone.load_state_dict(self.model.actor_backbone.state_dict())
        actor.actor_head.load_state_dict(self.model.actor_head.state_dict())
        return actor

    _sanitize_obs = staticmethod(sanitize_obs)

    def train(
        self,
        batch: tuple,
        epoch: int | None = None,
        batch_index: int | None = None,
        iters: int | None = None,
    ) -> dict[str, float]:
        self._training_step += 1

        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        o = self._sanitize_obs(o)
        o2 = self._sanitize_obs(o2)

        batch_size = r.shape[0]
        if self.n_steps > 1 and self.n_steps >= batch_size:
            raise ValueError(
                f"Invalid n-step config: n_steps ({self.n_steps}) must be smaller than "
                f"batch_size ({batch_size})."
            )
        actions = a.long().squeeze(-1)

        reward_scale = float(self.reward_normalize_scale)
        if reward_scale != 1.0 and reward_scale > 0:
            r = r * reward_scale

        # -- Sequence-aware n-step returns --
        burn_in_len = int(self.r2d2_burn_in)
        seq_len = int(self.r2d2_sequence_length)

        if self.n_steps > 1:
            truncated_batch_size = batch_size - self.n_steps
            n_step_return, bootstrap_mask = _compute_n_step_return_and_bootstrap_mask(
                r, d, self.gamma, self.n_steps
            )
            n_step_return = n_step_return[:truncated_batch_size].squeeze(-1)
            bootstrap_mask = bootstrap_mask[:truncated_batch_size].squeeze(-1)
            gamma_n = self.gamma**self.n_steps
            if seq_len > 0:
                step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
                valid_n_step = (step_in_seq + self.n_steps <= seq_len) & (
                    step_in_seq >= burn_in_len
                )
            else:
                valid_n_step = None
        else:
            truncated_batch_size = batch_size
            if seq_len > 0:
                step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
                valid_n_step = step_in_seq >= burn_in_len
            else:
                valid_n_step = None
            n_step_return = r[:truncated_batch_size].squeeze(-1)
            bootstrap_mask = (1.0 - d[:truncated_batch_size]).squeeze(-1)
            gamma_n = self.gamma

        o_t = tuple(t[:truncated_batch_size] for t in o)
        actions_t = actions[:truncated_batch_size]

        alpha_t = float(self._alpha) if isinstance(self._alpha, (int, float)) else self._alpha

        def autocast_ctx():
            return autocast_context(self.use_mixed_precision, self.amp_dtype)

        # -- Target Q --
        with torch.no_grad():
            if self.n_steps > 1:
                target_indices = (
                    torch.arange(truncated_batch_size, device=r.device) + self.n_steps - 1
                )
                o2_target = tuple(t[target_indices] for t in o2)
            else:
                o2_target = tuple(t[:truncated_batch_size] for t in o2)

            with autocast_ctx():
                logits_next = self.model.actor_logits(o2_target)
                dist_next = Categorical(logits=logits_next)
                probs_next = dist_next.probs

                if self.use_avg_q:
                    q_target = (
                        self.model_target.q1(o2_target) + self.model_target.q2(o2_target)
                    ) * 0.5
                else:
                    q_target = torch.min(
                        self.model_target.q1(o2_target), self.model_target.q2(o2_target)
                    )

            v_next = (probs_next * q_target).sum(dim=-1) + alpha_t * dist_next.entropy()
            target = n_step_return + gamma_n * bootstrap_mask * v_next

        # -- Critic losses --
        with autocast_ctx():
            current_q1_all = self.model.q1(o_t)
            current_q2_all = self.model.q2(o_t)
        current_q1 = current_q1_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        current_q2 = current_q2_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        if self.use_clip_q:
            with torch.no_grad(), autocast_ctx():
                q1_old = self.model_target.q1(o_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                q2_old = self.model_target.q2(o_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
            clipped_q1 = q1_old + (current_q1 - q1_old).clamp(
                -self.clip_q_epsilon, self.clip_q_epsilon
            )
            clipped_q2 = q2_old + (current_q2 - q2_old).clamp(
                -self.clip_q_epsilon, self.clip_q_epsilon
            )
            loss_q1 = torch.maximum((current_q1 - target) ** 2, (clipped_q1 - target) ** 2)
            loss_q2 = torch.maximum((current_q2 - target) ** 2, (clipped_q2 - target) ** 2)
            if valid_n_step is not None:
                denom = valid_n_step.float().sum().clamp(min=1.0)
                critic_loss = (loss_q1 * valid_n_step.float()).sum() / denom + (
                    loss_q2 * valid_n_step.float()
                ).sum() / denom
            else:
                critic_loss = loss_q1.mean() + loss_q2.mean()
        else:
            if valid_n_step is not None:
                denom = valid_n_step.float().sum().clamp(min=1.0)
                c1_loss = ((current_q1 - target) ** 2 * valid_n_step.float()).sum() / denom
                c2_loss = ((current_q2 - target) ** 2 * valid_n_step.float()).sum() / denom
                critic_loss = c1_loss + c2_loss
            else:
                critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(critic_loss).backward()
        else:
            critic_loss.backward()
        critic_grad_norm = self._grad_stabilizer_critic.step(
            itertools.chain(
                self.model.q1_backbone.parameters(),
                self.model.q1_head.parameters(),
                self.model.q2_backbone.parameters(),
                self.model.q2_head.parameters(),
            )
        )
        if self.use_mixed_precision:
            self.grad_scaler.step(self.critic_optimizer)
        else:
            self.critic_optimizer.step()

        # -- Actor loss --
        with autocast_ctx():
            logits = self.model.actor_logits(o_t)
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        with torch.no_grad(), autocast_ctx():
            if self.use_avg_q:
                q_for_actor = (self.model.q1(o_t) + self.model.q2(o_t)) * 0.5
            else:
                q_for_actor = torch.min(self.model.q1(o_t), self.model.q2(o_t))

        actor_loss_unmasked = -(alpha_t * entropy + (dist.probs * q_for_actor).sum(dim=-1))
        if valid_n_step is not None:
            denom = valid_n_step.float().sum().clamp(min=1.0)
            actor_loss = (actor_loss_unmasked * valid_n_step.float()).sum() / denom
        else:
            actor_loss = actor_loss_unmasked.mean()

        if self.use_entropy_penalty:
            with torch.no_grad(), autocast_ctx():
                target_logits = self.model_target.actor_logits(o_t)
                target_dist = Categorical(logits=target_logits)
                target_entropy = target_dist.entropy()
            entropy_penalty = F.mse_loss(entropy, target_entropy)
            actor_loss = actor_loss + self.entropy_penalty_beta * entropy_penalty

        self.actor_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(actor_loss).backward()
        else:
            actor_loss.backward()
        actor_grad_norm = self._grad_stabilizer_actor.step(
            itertools.chain(
                self.model.actor_backbone.parameters(), self.model.actor_head.parameters()
            )
        )
        if self.use_mixed_precision:
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            self.actor_optimizer.step()

        # -- Alpha update --
        alpha_loss_val = 0.0
        if self.auto_alpha and self.log_alpha is not None:
            log_prob = -entropy.detach() + self.target_entropy
            alpha_loss = -(self.log_alpha * log_prob).mean()
            assert self.alpha_optimizer is not None
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self._alpha = self.log_alpha.detach().exp()
            alpha_loss_val = float(alpha_loss.item())

        # -- Target network Polyak update --
        project_simbav2_weights(self.model)
        polyak_update(self.model, self.model_target, self.polyak)

        # -- Logging --
        ret: dict[str, float] = {
            "loss/actor": _tensor_to_scalar(actor_loss),
            "loss/critic": _tensor_to_scalar(critic_loss),
            "state/entropy": _tensor_to_scalar(entropy.mean()),
            "state/q1": _tensor_to_scalar(current_q1.mean()),
            "state/q2": _tensor_to_scalar(current_q2.mean()),
            "state/q_target": _tensor_to_scalar(target.mean()),
            "debug/critic_grad_norm": critic_grad_norm,
            "debug/actor_grad_norm": actor_grad_norm,
            "debug/critic_grad_ema": self._grad_stabilizer_critic.ema_norm,
            "debug/actor_grad_ema": self._grad_stabilizer_actor.ema_norm,
            "alpha": (
                float(self._alpha)
                if isinstance(self._alpha, (int, float))
                else float(self._alpha.item())
            ),
            "train/step": self._training_step,
        }
        if self.auto_alpha:
            ret["loss/alpha"] = alpha_loss_val

        if wandb is not None and wandb.run is not None:
            wandb.log(ret, step=wandb_monotonic_step(self._training_step, wandb.run))

        return ret
