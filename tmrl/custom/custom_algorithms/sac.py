"""Soft Actor-Critic (SAC) agent."""

import itertools
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam

from tmrl.custom.custom_algorithms._common import (
    _make_optimizer,
    amp_setup,
    autocast_context,
    polyak_update,
    set_seed,
)
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.registry import ALGORITHMS
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@ALGORITHMS.register("SAC")
@dataclass(eq=False)
class SpinupSacAgent(TrainingAgent):
    """Soft Actor-Critic (SAC) agent with optional learnable entropy coefficient.

    Adapted from Spinning Up in Deep RL. Supports SAC v1 (fixed alpha) and
    SAC v2 (learnable alpha with target entropy). Uses two Q-networks and
    min-Q target for value estimation.

    All hyperparameters are required constructor arguments — values must be
    supplied explicitly by the config pipeline (no hidden numeric defaults).
    """

    observation_space: type[Any]
    action_space: type[Any]
    model_cls: type[Any]
    gamma: float
    polyak: float
    alpha: float
    lr_actor: float
    lr_critic: float
    lr_entropy: float
    learn_entropy_coef: bool
    optimizer_actor: str
    optimizer_critic: str
    mixed_precision: bool
    mixed_precision_dtype: str
    seed: int
    device: str | None = None
    target_entropy: float | None = None
    betas_actor: tuple[float, ...] | None = None
    betas_critic: tuple[float, ...] | None = None
    l2_actor: float | None = None
    l2_critic: float | None = None
    debug_mode: bool = False

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        """Build model, target, optimizers, and entropy coefficient (if learned)."""
        set_seed(self.seed)
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logger.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.pi_optimizer = _make_optimizer(
            self.model.actor.parameters(),
            self.optimizer_actor,
            self.lr_actor,
            weight_decay=self.l2_actor,
            betas=self.betas_actor,
        )
        self.q_optimizer = _make_optimizer(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            self.optimizer_critic,
            self.lr_critic,
            weight_decay=self.l2_critic,
            betas=self.betas_critic,
        )
        self.use_mixed_precision, self.amp_dtype, self.grad_scaler = amp_setup(
            device, self.mixed_precision, self.mixed_precision_dtype
        )
        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * self.alpha
            ).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """Return the current actor (policy) module for rollout workers."""
        return self.model_nograd.actor

    def train(self, batch, epoch=None, batch_index=None, iters=None):
        """Perform one SAC training step on the given batch.

        Args:
            batch: Tuple (obs, actions, rewards, next_obs, dones, ...).
            epoch: Current epoch (unused, for API compat with training loop).
            batch_index: Current batch index (unused).
            iters: Total iterations (unused).

        Returns:
            Dict with losses/actor, losses/critic, and optionally loss_entropy_coef,
            entropy_coef and debug metrics when debug_mode is True.
        """
        obs, actions, rewards, next_obs, dones = batch[:5]

        def autocast_ctx():
            return autocast_context(self.use_mixed_precision, self.amp_dtype)

        with autocast_ctx():
            policy_actions, log_prob_pi = self.model.actor(obs)
        alpha_t, loss_alpha = self._sac_update_entropy_coef(policy_actions, log_prob_pi)
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        with autocast_ctx():
            q1_pred = self.model.q1(obs, actions)
            q2_pred = self.model.q2(obs, actions)
        td_target = self._sac_compute_td_target(next_obs, rewards, dones, alpha_t)
        loss_q1 = ((q1_pred - td_target) ** 2).mean()
        loss_q2 = ((q2_pred - td_target) ** 2).mean()
        loss_q = (loss_q1 + loss_q2) / 2

        self.q_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_q).backward()
            self.grad_scaler.step(self.q_optimizer)
        else:
            loss_q.backward()
            self.q_optimizer.step()
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)

        with autocast_ctx():
            q1_pi = self.model.q1(obs, policy_actions)
            q2_pi = self.model.q2(obs, policy_actions)
            q_pi = torch.min(q1_pi, q2_pi)
        loss_pi = (alpha_t * log_prob_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_pi).backward()
            self.grad_scaler.step(self.pi_optimizer)
            self.grad_scaler.update()
        else:
            loss_pi.backward()
            self.pi_optimizer.step()
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)

        polyak_update(self.model, self.model_target, self.polyak)
        ret_dict = self._sac_build_return_dict(
            {
                "loss_pi": loss_pi,
                "loss_q": loss_q,
                "alpha_t": alpha_t,
                "loss_alpha": loss_alpha,
                "obs": obs,
                "actions": actions,
                "next_obs": next_obs,
                "dones": dones,
                "rewards": rewards,
                "policy_actions": policy_actions,
                "log_prob_pi": log_prob_pi,
                "q1_pred": q1_pred,
                "q2_pred": q2_pred,
                "q1_pi": q1_pi,
                "q2_pi": q2_pi,
                "q_pi": q_pi,
                "td_target": td_target,
            }
        )
        return ret_dict

    def _sac_update_entropy_coef(self, policy_actions, log_prob_pi):
        """Compute current alpha and optional entropy loss for SAC v2.

        Args:
            policy_actions: Unused; kept for signature compatibility.
            log_prob_pi: Log probability of policy actions.

        Returns:
            Tuple (alpha_t, loss_alpha or None).
        """
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
            return alpha_t, loss_alpha
        return self.alpha_t, None

    def _sac_compute_td_target(self, next_obs, rewards, dones, alpha_t):
        """Compute Bellman backup (TD target) for Q-learning."""
        with torch.no_grad():
            with autocast_context(self.use_mixed_precision, self.amp_dtype):
                next_actions, log_prob_next = self.model.actor(next_obs)
                q1_next = self.model_target.q1(next_obs, next_actions)
                q2_next = self.model_target.q2(next_obs, next_actions)
            min_q_next = torch.min(q1_next, q2_next)
            return rewards + self.gamma * (1 - dones) * (min_q_next - alpha_t * log_prob_next)

    def _sac_build_return_dict(self, ctx: dict) -> dict:
        """Build the dict of scalars to log (and optionally debug metrics)."""
        loss_pi = ctx["loss_pi"]
        loss_q = ctx["loss_q"]
        alpha_t = ctx["alpha_t"]
        loss_alpha = ctx["loss_alpha"]
        with torch.no_grad():
            if not self.debug_mode:
                ret_dict = {
                    "losses/actor": loss_pi.detach().item(),
                    "losses/critic": loss_q.detach().item(),
                }
            else:
                obs = ctx["obs"]
                actions = ctx["actions"]
                next_obs = ctx["next_obs"]
                dones = ctx["dones"]
                rewards = ctx["rewards"]
                policy_actions = ctx["policy_actions"]
                log_prob_pi = ctx["log_prob_pi"]
                q1_pred = ctx["q1_pred"]
                q2_pred = ctx["q2_pred"]
                q1_pi = ctx["q1_pi"]
                q2_pi = ctx["q2_pi"]
                q_pi = ctx["q_pi"]
                td_target = ctx["td_target"]
                next_actions, log_prob_next = self.model.actor(next_obs)
                q1_next_obs_next_a = self.model.q1(next_obs, next_actions)
                q2_next_obs_next_a = self.model.q2(next_obs, next_actions)
                q1_targ_pi = self.model_target.q1(obs, policy_actions)
                q2_targ_pi = self.model_target.q2(obs, policy_actions)
                q1_targ_a = self.model_target.q1(obs, actions)
                q2_targ_a = self.model_target.q2(obs, actions)
                q1_pi_targ = self.model_target.q1(next_obs, next_actions)
                q2_pi_targ = self.model_target.q2(next_obs, next_actions)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

                ret_dict = {
                    "losses/actor": loss_pi.detach().item(),
                    "losses/critic": loss_q.detach().item(),
                    "debug_log_pi": log_prob_pi.detach().mean().item(),
                    "debug_log_pi_std": log_prob_pi.detach().std().item(),
                    "debug_logp_a2": log_prob_next.detach().mean().item(),
                    "debug_logp_a2_std": log_prob_next.detach().std().item(),
                    "debug_q_a1": q_pi.detach().mean().item(),
                    "debug_q_a1_std": q_pi.detach().std().item(),
                    "debug_q_a1_targ": q_pi_targ.detach().mean().item(),
                    "debug_q_a1_targ_std": q_pi_targ.detach().std().item(),
                    "debug_backup": td_target.detach().mean().item(),
                    "debug_backup_std": td_target.detach().std().item(),
                    "debug_q1": q1_pred.detach().mean().item(),
                    "debug_q1_std": q1_pred.detach().std().item(),
                    "debug_q2": q2_pred.detach().mean().item(),
                    "debug_q2_std": q2_pred.detach().std().item(),
                    "debug_diff_q1": (q1_pred - td_target).detach().mean().item(),
                    "debug_diff_q1_std": (q1_pred - td_target).detach().std().item(),
                    "debug_diff_q2": (q2_pred - td_target).detach().mean().item(),
                    "debug_diff_q2_std": (q2_pred - td_target).detach().std().item(),
                    "debug_diff_r_q1": (q1_pred - td_target + rewards).detach().mean().item(),
                    "debug_diff_r_q1_std": (q1_pred - td_target + rewards).detach().std().item(),
                    "debug_diff_r_q2": (q2_pred - td_target + rewards).detach().mean().item(),
                    "debug_diff_r_q2_std": (q2_pred - td_target + rewards).detach().std().item(),
                    "debug_diff_q1pt_qpt": (q1_pi_targ - q_pi_targ).detach().mean().item(),
                    "debug_diff_q2pt_qpt": (q2_pi_targ - q_pi_targ).detach().mean().item(),
                    "debug_diff_q1_q1t_a2": (q1_next_obs_next_a - q1_pi_targ)
                    .detach()
                    .mean()
                    .item(),
                    "debug_diff_q2_q2t_a2": (q2_next_obs_next_a - q2_pi_targ)
                    .detach()
                    .mean()
                    .item(),
                    "debug_diff_q1_q1t_pi": (q1_pi - q1_targ_pi).detach().mean().item(),
                    "debug_diff_q2_q2t_pi": (q2_pi - q2_targ_pi).detach().mean().item(),
                    "debug_diff_q1_q1t_a": (q1_pred - q1_targ_a).detach().mean().item(),
                    "debug_diff_q2_q2t_a": (q2_pred - q2_targ_a).detach().mean().item(),
                    "debug_diff_q1pt_qpt_std": (q1_pi_targ - q_pi_targ).detach().std().item(),
                    "debug_diff_q2pt_qpt_std": (q2_pi_targ - q_pi_targ).detach().std().item(),
                    "debug_diff_q1_q1t_a2_std": (q1_next_obs_next_a - q1_pi_targ)
                    .detach()
                    .std()
                    .item(),
                    "debug_diff_q2_q2t_a2_std": (q2_next_obs_next_a - q2_pi_targ)
                    .detach()
                    .std()
                    .item(),
                    "debug_diff_q1_q1t_pi_std": (q1_pi - q1_targ_pi).detach().std().item(),
                    "debug_diff_q2_q2t_pi_std": (q2_pi - q2_targ_pi).detach().std().item(),
                    "debug_diff_q1_q1t_a_std": (q1_pred - q1_targ_a).detach().std().item(),
                    "debug_diff_q2_q2t_a_std": (q2_pred - q2_targ_a).detach().std().item(),
                    "debug_r": rewards.detach().mean().item(),
                    "debug_r_std": rewards.detach().std().item(),
                    "debug_d": dones.detach().mean().item(),
                    "debug_d_std": dones.detach().std().item(),
                    "debug_a_0": actions[:, 0].detach().mean().item(),
                    "debug_a_0_std": actions[:, 0].detach().std().item(),
                    "debug_a_1": actions[:, 1].detach().mean().item(),
                    "debug_a_1_std": actions[:, 1].detach().std().item(),
                    "debug_a_2": actions[:, 2].detach().mean().item(),
                    "debug_a_2_std": actions[:, 2].detach().std().item(),
                    "debug_a1_0": policy_actions[:, 0].detach().mean().item(),
                    "debug_a1_0_std": policy_actions[:, 0].detach().std().item(),
                    "debug_a1_1": policy_actions[:, 1].detach().mean().item(),
                    "debug_a1_1_std": policy_actions[:, 1].detach().std().item(),
                    "debug_a1_2": policy_actions[:, 2].detach().mean().item(),
                    "debug_a1_2_std": policy_actions[:, 2].detach().std().item(),
                    "debug_a2_0": next_actions[:, 0].detach().mean().item(),
                    "debug_a2_0_std": next_actions[:, 0].detach().std().item(),
                    "debug_a2_1": next_actions[:, 1].detach().mean().item(),
                    "debug_a2_1_std": next_actions[:, 1].detach().std().item(),
                    "debug_a2_2": next_actions[:, 2].detach().mean().item(),
                    "debug_a2_2_std": next_actions[:, 2].detach().std().item(),
                }

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
