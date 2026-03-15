"""SAC agent with hyperparameters from config (ALG_CONFIG)."""

import itertools
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

import tmrl.config as cfg
from tmrl.custom.custom_algorithms._common import (
    _amp_dtype,
    _amp_enabled,
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    set_seed,
)
from tmrl.custom.models import MLPActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@dataclass(eq=False)
class SpinupSacAgentConfig(TrainingAgent):
    """SAC agent with hyperparameters read from config (ALG_CONFIG).

    Same as SpinupSacAgent but lr_actor, lr_critic, lr_entropy, n_steps
    default to cfg.ALG_CONFIG values for config-driven training.
    """

    observation_space: type[Any]
    action_space: type[Any]
    device: str | None = None
    model_cls: type[Any] = MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = cfg.ALG_CONFIG["LR_ACTOR"]
    lr_critic: float = cfg.ALG_CONFIG["LR_CRITIC"]
    lr_entropy: float = cfg.ALG_CONFIG["LR_ENTROPY"]
    learn_entropy_coef: bool = True
    target_entropy: float | None = None
    n_steps: int = cfg.ALG_CONFIG["N_STEPS"]

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        set_seed()
        if self.n_steps == 1:
            self.n_steps = 0
        observation_space, action_space = self.observation_space, self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.model_cls(observation_space, action_space)
        logger.debug(f" device SAC: {device}")
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.actor_optimizer = Adam(
            self.model.actor.parameters(),
            lr=self.lr_actor,
            weight_decay=cfg.ACTOR_WEIGHT_DECAY,
            eps=cfg.ADAM_EPS,
        )
        self.critic_optimizer = Adam(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            lr=self.lr_critic,
            weight_decay=cfg.CRITIC_WEIGHT_DECAY,
            eps=cfg.ADAM_EPS,
        )
        self.use_mixed_precision = _amp_enabled(device)
        self.amp_dtype = _amp_dtype()
        use_scaler = self.use_mixed_precision and (self.amp_dtype != torch.bfloat16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        if len(cfg.SCHEDULER_CONFIG["NAME"]) > 0:
            self.actor_scheduler = CosineAnnealingWarmRestarts(
                self.actor_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"],
            )

            self.critic_scheduler = CosineAnnealingWarmRestarts(
                self.critic_optimizer,
                cfg.SCHEDULER_CONFIG["T_0"],
                cfg.SCHEDULER_CONFIG["T_mult"],
                cfg.SCHEDULER_CONFIG["eta_min"],
                cfg.SCHEDULER_CONFIG["last_epoch"],
            )

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(
                torch.ones(1, device=self.device) * self.alpha
            ).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

        if cfg.WANDB_GRADIENTS and wandb is not None:
            wandb.watch(self.model, log_freq=10)

    def get_actor(self):
        return self.model_nograd.actor

    @staticmethod
    def clip_weights(model, max_value=0.98):
        for param in model.parameters():
            param.data.clamp_(-max_value, max_value)

    def train(self, batch, epoch, batch_index, iters):
        if cfg.DEBUG_MODE:
            torch.autograd.set_detect_anomaly(True)
        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]

        def autocast_ctx():
            return (
                torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True)
                if self.use_mixed_precision
                else nullcontext()
            )

        batch_size = r.shape[0]
        if self.n_steps <= 1:
            truncated_batch_size = batch_size
        else:
            truncated_batch_size = batch_size - self.n_steps

        with autocast_ctx():
            pi, logp_pi = self.model.actor(o)
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
        with autocast_ctx():
            q1 = self.model.q1(o, a)[:truncated_batch_size]
            q2 = self.model.q2(o, a)[:truncated_batch_size]

        n_step_not_done = None
        if self.n_steps > 1:
            n_step_return, n_step_not_done = _compute_n_step_return_and_bootstrap_mask(
                r, d, self.gamma, self.n_steps
            )
            r = n_step_return[:truncated_batch_size].squeeze(-1)
            # Do NOT slice o/o2: they are tuples; slicing would change tuple length, not batch dim.
            n_step_not_done = n_step_not_done[:truncated_batch_size].squeeze(-1)
        with torch.no_grad():
            with autocast_ctx():
                a2, logp_a2 = self.model.actor(o2)
            if self.n_steps > 1:
                logp_a2 = logp_a2[:truncated_batch_size]
                with autocast_ctx():
                    q1_pi_targ = self.model_target.q1(o2, a2)[:truncated_batch_size]
                    q2_pi_targ = self.model_target.q2(o2, a2)[:truncated_batch_size]
            else:
                with autocast_ctx():
                    q1_pi_targ = self.model_target.q1(o2, a2)
                    q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            if self.n_steps > 1:
                backup = r + (self.gamma**self.n_steps) * n_step_not_done * (
                    q_pi_targ.sub_(alpha_t * logp_a2)
                )
            else:
                backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)
        with autocast_ctx():
            loss_q1 = q1.sub_(backup).pow_(2).mean()
            loss_q2 = q2.sub_(backup).pow_(2).mean()
            loss_critic = loss_q1.add_(loss_q2).div_(2)

        self.critic_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_critic).backward()
            self.grad_scaler.step(self.critic_optimizer)
        else:
            loss_critic.backward()
            self.critic_optimizer.step()
        self.model.q1.requires_grad_(False)
        self.model.q2.requires_grad_(False)
        if cfg.WEIGHT_CLIPPING_ENABLED:
            self.clip_weights(self.model.q1)
            self.clip_weights(self.model.q2)
        with autocast_ctx():
            q1_pi = self.model.q1(o, pi)[:truncated_batch_size]
            q2_pi = self.model.q2(o, pi)[:truncated_batch_size]
            q_pi = torch.min(q1_pi, q2_pi)
        with autocast_ctx():
            loss_actor = (alpha_t * logp_pi[:truncated_batch_size] - q_pi).mean()
        self.actor_optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss_actor).backward()
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            loss_actor.backward()
            self.actor_optimizer.step()

        if len(cfg.SCHEDULER_CONFIG["NAME"]) > 0:
            self.actor_scheduler.step(epoch + batch_index / iters)
            self.critic_scheduler.step(epoch + batch_index / iters)
        if cfg.WEIGHT_CLIPPING_ENABLED:
            self.clip_weights(self.model.actor)
        self.model.q1.requires_grad_(True)
        self.model.q2.requires_grad_(True)
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak).add_(p.data, alpha=(1 - self.polyak))
        with torch.no_grad():
            ret_dict = dict()
            ret_dict["losses/actor"] = _tensor_to_scalar(loss_actor.detach())
            ret_dict["losses/critic"] = _tensor_to_scalar(loss_critic.detach())
            ret_dict["lrs/actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
            ret_dict["lrs/critic_lr"] = self.critic_optimizer.param_groups[0]["lr"]
            if cfg.WANDB_DEBUG:
                q1_o2_a2 = self.model.q1(o2, a2)[:truncated_batch_size]
                q2_o2_a2 = self.model.q2(o2, a2)[:truncated_batch_size]
                q1_targ_pi = self.model_target.q1(o, pi)[:truncated_batch_size]
                q2_targ_pi = self.model_target.q2(o, pi)[:truncated_batch_size]
                q1_targ_a = self.model_target.q1(o, a)[:truncated_batch_size]
                q2_targ_a = self.model_target.q2(o, a)[:truncated_batch_size]

                diff_q1pt_qpt = (q1_pi_targ - q_pi_targ).detach()
                diff_q2pt_qpt = (q2_pi_targ - q_pi_targ).detach()
                diff_q1_q1t_a2 = (q1_o2_a2 - q1_pi_targ).detach()
                diff_q2_q2t_a2 = (q2_o2_a2 - q2_pi_targ).detach()
                diff_q1_q1t_pi = (q1_pi - q1_targ_pi).detach()
                diff_q2_q2t_pi = (q2_pi - q2_targ_pi).detach()
                diff_q1_q1t_a = (q1 - q1_targ_a).detach()
                diff_q2_q2t_a = (q2 - q2_targ_a).detach()
                diff_q1_backup = (q1 - backup).detach()
                diff_q2_backup = (q2 - backup).detach()
                diff_q1_backup_r = (q1 - backup + r).detach()
                diff_q2_backup_r = (q2 - backup + r).detach()
                ret_dict["debug/log_pi"] = _tensor_to_scalar(logp_pi.detach().mean())
                ret_dict["debug/log_pi_std"] = _tensor_to_scalar(logp_pi.detach().std())
                ret_dict["debug/logp_a2"] = _tensor_to_scalar(logp_a2.detach().mean())
                ret_dict["debug/logp_a2_std"] = _tensor_to_scalar(logp_a2.detach().std())
                ret_dict["debug/q_a1"] = _tensor_to_scalar(q_pi.detach().mean())
                ret_dict["debug/q_a1_std"] = _tensor_to_scalar(q_pi.detach().std())
                ret_dict["debug/q_a1_targ"] = _tensor_to_scalar(q_pi_targ.detach().mean())
                ret_dict["debug/q_a1_targ_std"] = _tensor_to_scalar(q_pi_targ.detach().std())
                ret_dict["debug/backup"] = _tensor_to_scalar(backup.detach().mean())
                ret_dict["debug/backup_std"] = _tensor_to_scalar(backup.detach().std())
                ret_dict["debug/q1"] = _tensor_to_scalar(q1.detach().mean())
                ret_dict["debug/q1_std"] = _tensor_to_scalar(q1.detach().std())
                ret_dict["debug/q2"] = _tensor_to_scalar(q2.detach().mean())
                ret_dict["debug/q2_std"] = _tensor_to_scalar(q2.detach().std())
                ret_dict["debug/diff_q1"] = _tensor_to_scalar(diff_q1_backup.mean())
                ret_dict["debug/diff_q1_std"] = _tensor_to_scalar(diff_q1_backup.std())
                ret_dict["debug/diff_q2"] = _tensor_to_scalar(diff_q2_backup.mean())
                ret_dict["debug/diff_q2_std"] = _tensor_to_scalar(diff_q2_backup.std())
                ret_dict["debug/diff_r_q1"] = _tensor_to_scalar(diff_q1_backup_r.mean())
                ret_dict["debug/diff_r_q1_std"] = _tensor_to_scalar(diff_q1_backup_r.std())
                ret_dict["debug/diff_r_q2"] = _tensor_to_scalar(diff_q2_backup_r.mean())
                ret_dict["debug/diff_r_q2_std"] = _tensor_to_scalar(diff_q2_backup_r.std())
                ret_dict["debug/diff_q1pt_qpt"] = _tensor_to_scalar(diff_q1pt_qpt.mean())
                ret_dict["debug/diff_q2pt_qpt"] = _tensor_to_scalar(diff_q2pt_qpt.mean())
                ret_dict["debug/diff_q1_q1t_a2"] = _tensor_to_scalar(diff_q1_q1t_a2.mean())
                ret_dict["debug/diff_q2_q2t_a2"] = _tensor_to_scalar(diff_q2_q2t_a2.mean())
                ret_dict["debug/diff_q1_q1t_pi"] = _tensor_to_scalar(diff_q1_q1t_pi.mean())
                ret_dict["debug/diff_q2_q2t_pi"] = _tensor_to_scalar(diff_q2_q2t_pi.mean())
                ret_dict["debug/diff_q1_q1t_a"] = _tensor_to_scalar(diff_q1_q1t_a.mean())
                ret_dict["debug/diff_q2_q2t_a"] = _tensor_to_scalar(diff_q2_q2t_a.mean())
                ret_dict["debug/diff_q1pt_qpt_std"] = _tensor_to_scalar(diff_q1pt_qpt.std())
                ret_dict["debug/diff_q2pt_qpt_std"] = _tensor_to_scalar(diff_q2pt_qpt.std())
                ret_dict["debug/diff_q1_q1t_a2_std"] = _tensor_to_scalar(diff_q1_q1t_a2.std())
                ret_dict["debug/diff_q2_q2t_a2_std"] = _tensor_to_scalar(diff_q2_q2t_a2.std())
                ret_dict["debug/diff_q1_q1t_pi_std"] = _tensor_to_scalar(diff_q1_q1t_pi.std())
                ret_dict["debug/diff_q2_q2t_pi_std"] = _tensor_to_scalar(diff_q2_q2t_pi.std())
                ret_dict["debug/diff_q1_q1t_a_std"] = _tensor_to_scalar(diff_q1_q1t_a.std())
                ret_dict["debug/diff_q2_q2t_a_std"] = _tensor_to_scalar(diff_q2_q2t_a.std())
                ret_dict["debug/r"] = _tensor_to_scalar(r.detach().mean())
                ret_dict["debug/r_std"] = _tensor_to_scalar(r.detach().std())
                ret_dict["debug/d"] = _tensor_to_scalar(d.detach().mean())
                ret_dict["debug/d_std"] = _tensor_to_scalar(d.detach().std())
                ret_dict["debug/a_0"] = _tensor_to_scalar(a[:, 0].detach().mean())
                ret_dict["debug/a_0_std"] = _tensor_to_scalar(a[:, 0].detach().std())
                ret_dict["debug/a_1"] = _tensor_to_scalar(a[:, 1].detach().mean())
                ret_dict["debug/a_1_std"] = _tensor_to_scalar(a[:, 1].detach().std())
                ret_dict["debug/a_2"] = _tensor_to_scalar(a[:, 2].detach().mean())
                ret_dict["debug/a_2_std"] = _tensor_to_scalar(a[:, 2].detach().std())
                ret_dict["debug/a1_0"] = _tensor_to_scalar(pi[:, 0].detach().mean())
                ret_dict["debug/a1_0_std"] = _tensor_to_scalar(pi[:, 0].detach().std())
                ret_dict["debug/a1_1"] = _tensor_to_scalar(pi[:, 1].detach().mean())
                ret_dict["debug/a1_1_std"] = _tensor_to_scalar(pi[:, 1].detach().std())
                ret_dict["debug/a1_2"] = _tensor_to_scalar(pi[:, 2].detach().mean())
                ret_dict["debug/a1_2_std"] = _tensor_to_scalar(pi[:, 2].detach().std())
                ret_dict["debug/a2_0"] = _tensor_to_scalar(a2[:, 0].detach().mean())
                ret_dict["debug/a2_0_std"] = _tensor_to_scalar(a2[:, 0].detach().std())
                ret_dict["debug/a2_1"] = _tensor_to_scalar(a2[:, 1].detach().mean())
                ret_dict["debug/a2_1_std"] = _tensor_to_scalar(a2[:, 1].detach().std())
                ret_dict["debug/a2_2"] = _tensor_to_scalar(a2[:, 2].detach().mean())
                ret_dict["debug/a2_2_std"] = _tensor_to_scalar(a2[:, 2].detach().std())

        if self.learn_entropy_coef:
            ret_dict["loss_entropy_coef"] = loss_alpha.detach().item()
            ret_dict["entropy_coef"] = alpha_t.item()

        return ret_dict
