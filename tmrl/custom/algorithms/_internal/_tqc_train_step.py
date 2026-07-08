"""Extracted TQC training step — called from TQCAgent.train()."""

import itertools
import math

import numpy as np
import torch

from tmrl.custom.algorithms._common import (
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    autocast_context,
    clip_model_weights,
    polyak_update,
    project_simbav2_weights,
)


def _tqc_train_step(
    self,
    batch: tuple,
    epoch: int,
    batch_index: int,
    iters: int,
) -> dict:
    """Run one TQC training step on a sampled batch.

    Args:
        self: TQCAgent instance (passed as first arg to avoid class coupling).
        batch: Sampled experience tuple.
        epoch: Current epoch number.
        batch_index: Current batch index within the epoch.
        iters: Total number of batches per epoch.

    Returns:
        Dict of scalar metrics for logging.
    """
    self._training_step += 1

    if (
        self._training_step % self._nan_weight_check_interval == 0
        or self._consecutive_bad_steps >= 3
    ) and self._model_has_nan_weights():
        self._reinitialize_model()

    o = batch[0]
    batch_size = o[0].shape[0] if isinstance(o, (tuple, list)) else o.shape[0]

    if hasattr(self.model.actor, "reset_noise"):
        self.model.actor.reset_noise(batch_size)

    # Only terminated for bootstrap; truncated must bootstrap so value not underestimated.
    if len(batch) >= 7:
        o, a, r, o2, d, _trunc, info = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
            batch[6],
        )
    else:
        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        info = {}

    _tensors_to_check = [a, r, d]
    _obs_tensors = list(o) if isinstance(o, (tuple, list)) else [o]
    _obs2_tensors = list(o2) if isinstance(o2, (tuple, list)) else [o2]
    _has_bad = any(self._has_nan(t) for t in _tensors_to_check + _obs_tensors + _obs2_tensors)
    if _has_bad:
        if self._training_step <= 5 or self._training_step % 500 == 0:
            from loguru import logger

            logger.warning(
                " NaN detected in batch input data (step %d). Sanitizing obs/act/rew/done.",
                self._training_step,
            )
        o = self._sanitize_obs(o)
        o2 = self._sanitize_obs(o2)
        a = self._sanitize_tensor(a)
        r = self._sanitize_tensor(r)
        d = self._sanitize_tensor(d)

    # Scale rewards before Bellman backup: r = r * scale (<1 shrinks).
    if self.reward_normalize_scale != 1.0 and self.reward_normalize_scale > 0:
        r = r * self.reward_normalize_scale

    def autocast_ctx():
        return autocast_context(self.use_mixed_precision, self.amp_dtype)

    batch_size = r.shape[0]
    if self.n_steps > 1 and self.n_steps >= batch_size:
        raise ValueError(
            f"Invalid n-step config: n_steps ({self.n_steps}) must be smaller than "
            f"batch_size ({batch_size})."
        )
    burn_in_len = int(self.r2d2_burn_in)
    _seq_len_cfg = int(self.r2d2_sequence_length)
    if burn_in_len > 0 and _seq_len_cfg > 0 and burn_in_len >= _seq_len_cfg:
        raise ValueError(
            f"r2d2_burn_in ({burn_in_len}) >= r2d2_sequence_length ({_seq_len_cfg}). "
            "Burn-in must be < sequence length to leave active training steps."
        )

    if self.n_steps <= 1:
        truncated_batch_size = batch_size
        _seq_len_cfg = int(self.r2d2_sequence_length)
        if _seq_len_cfg > 0:
            seq_len = _seq_len_cfg
            step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
            valid_n_step = step_in_seq >= burn_in_len
        else:
            seq_len = 0
            valid_n_step = None
    else:
        truncated_batch_size = batch_size - self.n_steps
        seq_len = int(self.r2d2_sequence_length)
        if seq_len > 0:
            step_in_seq = torch.arange(truncated_batch_size, device=r.device) % seq_len
            valid_n_step = (step_in_seq + self.n_steps <= seq_len) & (step_in_seq >= burn_in_len)
        else:
            valid_n_step = None

    # ── 1. Actor forward ──
    with autocast_ctx():
        out = self.model.actor(o, return_pre_tanh_mean=True)
        pi, logp_pi, mu = out[0], out[1], out[2]
    logp_pi_safe = self._safe_logprob(logp_pi)

    # ── 2. Entropy coefficient ──
    alpha_loss = None
    if self.entropy_schedule == "cosine":
        raw_alpha = self._cosine_alpha(self._training_step)
        alpha_t = torch.tensor(max(raw_alpha, self.entropy_floor), device=pi.device)
    elif self.learn_entropy_coef:
        alpha_t = torch.exp(self.log_alpha.detach())
        target = self.target_entropy
        assert target is not None
        alpha_loss = -(self.log_alpha * (logp_pi_safe + target).detach()).mean()
    else:
        alpha_t = self.alpha_t  # type: ignore[assignment]

    if alpha_loss is not None:
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            _log_alpha_min = math.log(self.entropy_floor)
            self.log_alpha.clamp_(min=_log_alpha_min)

    # ── 3. Critic forward on BUFFER actions ──
    with autocast_ctx():
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

    # ── 4. N-step return computation ──
    n_step_return = None
    n_step_not_done = None
    if self.n_steps > 1:
        n_step_return, n_step_not_done = _compute_n_step_return_and_bootstrap_mask(
            r, d, self.gamma, self.n_steps
        )
        n_step_return = n_step_return[:truncated_batch_size]
        n_step_not_done = n_step_not_done[:truncated_batch_size]

    # ── 5. Target Q computation ──
    with torch.no_grad():
        with autocast_ctx():
            a2, logp_a2 = self.model.actor(o2)
        logp_a2 = self._safe_logprob(logp_a2)

        with autocast_ctx():
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)

        next_z = torch.stack((q1_pi_targ, q2_pi_targ), dim=1)
        sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
        effective_drop = self.total_quantiles_to_drop
        if self.dynamic_truncation_enabled:
            with torch.no_grad():
                var_current = float(sorted_z.var().item())
                self._trunc_var_history.append(var_current)
                if len(self._trunc_var_history) >= 20:
                    pct_val = float(
                        np.percentile(
                            self._trunc_var_history,
                            self.dynamic_truncation_variance_pct * 100,
                        )
                    )
                    if var_current > pct_val:
                        effective_drop = min(
                            self.quantiles_total - 2,
                            effective_drop + 2,
                        )
        sorted_z_part = sorted_z[:, : self.quantiles_total - effective_drop]
        at = alpha_t if alpha_t is not None else self.alpha_t
        q_pi_targ_full = sorted_z_part - at * logp_a2.reshape(-1, 1)

        if self.n_steps > 1:
            assert n_step_not_done is not None
            target_indices = torch.arange(truncated_batch_size, device=r.device) + self.n_steps - 1
            q_pi_targ = q_pi_targ_full[target_indices]
            not_done = n_step_not_done
        else:
            q_pi_targ = q_pi_targ_full[:truncated_batch_size]
            not_done = (1 - d).unsqueeze(-1)[:truncated_batch_size]

        tmp = q_pi_targ * not_done
        if self.n_steps > 1:
            assert n_step_return is not None
            k = q_pi_targ.shape[1]
            backup = n_step_return.expand(-1, k) + (self.gamma**self.n_steps) * tmp
            if seq_len > 0 and valid_n_step is not None:
                backup = backup * valid_n_step.float().unsqueeze(-1)
        else:
            backup = r[:truncated_batch_size].unsqueeze(-1) + self.gamma * tmp

    if self.per_td_enabled and isinstance(info, dict) and "is_weight" in info:
        is_weights = info["is_weight"].float().view(-1)
        is_weights = is_weights / (is_weights.max() + 1e-8)
    else:
        is_weights = torch.ones_like(r)

    # ── 6. Critic loss ──
    cur_z = torch.stack((q1, q2), dim=1)[:truncated_batch_size]
    if self.backup_clip_range > 0:
        backup = backup.clamp(-self.backup_clip_range, self.backup_clip_range)

    cur_z = cur_z.float()
    backup = backup.float()
    clamp_val = 1e4
    cur_z = torch.nan_to_num(cur_z, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(
        -clamp_val, clamp_val
    )
    backup = torch.nan_to_num(backup, nan=0.0, posinf=clamp_val, neginf=-clamp_val).clamp(
        -clamp_val, clamp_val
    )

    per_sample_loss = self._quantile_huber_per_sample(cur_z, backup)

    if self.n_steps > 1 and seq_len > 0 and valid_n_step is not None:
        per_sample_loss = per_sample_loss * is_weights[:truncated_batch_size]
        denom = valid_n_step.float().sum().clamp(min=1.0)
        critic_loss = (per_sample_loss * valid_n_step.float()).sum() / denom
    else:
        critic_loss = (per_sample_loss * is_weights[:truncated_batch_size]).mean()

    with torch.no_grad():
        q_cur = cur_z.float().mean(dim=(1, 2))
        q_targ = backup.float().mean(dim=1)
        td_errors_batch = (q_cur - q_targ).abs()
        if self.n_steps > 1 and seq_len > 0 and valid_n_step is not None:
            td_errors_batch = td_errors_batch * valid_n_step.float()

    def _is_bad_loss(loss_tensor):
        if loss_tensor is None:
            return False
        return bool(torch.isnan(loss_tensor).any().item() or torch.isinf(loss_tensor).any().item())

    # ── 7. Critic update ──
    critic_loss_bad = _is_bad_loss(critic_loss)
    self.critic_optimizer.zero_grad()
    if not critic_loss_bad:
        if self.use_mixed_precision:
            self.grad_scaler.scale(critic_loss).backward()
            self.grad_scaler.unscale_(self.critic_optimizer)
        else:
            critic_loss.backward()
        if self.grad_clip_critic > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
                self.grad_clip_critic,
            )
        else:
            critic_grad_norm = None
        if self.use_mixed_precision:
            self.grad_scaler.step(self.critic_optimizer)
        else:
            self.critic_optimizer.step()
    else:
        critic_grad_norm = None

    # ── 8. Actor loss ──
    self.model.q1.requires_grad_(False)
    self.model.q2.requires_grad_(False)

    with autocast_ctx():
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)

    q1_pi = q1_pi[:truncated_batch_size]
    q2_pi = q2_pi[:truncated_batch_size]
    logp_actor = logp_pi_safe[:truncated_batch_size]
    mu_actor = mu[:truncated_batch_size]
    pi_actor = pi[:truncated_batch_size]

    q_pi = torch.stack((q1_pi.float(), q2_pi.float()), dim=1).mean(2).mean(1)
    q_pi = torch.nan_to_num(q_pi, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

    # VCSE: scale alpha by critic std (more exploration in uncertain states)
    if self.vcse_enabled:
        q_stack = torch.stack((q1_pi.float().mean(1), q2_pi.float().mean(1)), dim=1)
        sigma_q = q_stack.std(dim=1)
        alpha_base = float(self.vcse_alpha_base)
        vcse_lam = float(self.vcse_lambda)
        alpha_per_sample = (
            alpha_t.float().squeeze() + vcse_lam * sigma_q
            if alpha_t.dim() > 0
            else alpha_t.float() + vcse_lam * sigma_q
        )
        alpha_per_sample = alpha_per_sample.clamp(min=alpha_base, max=2.0)
        actor_loss_unmasked = alpha_per_sample * logp_actor - q_pi
    else:
        actor_loss_unmasked = alpha_t.float() * logp_actor - q_pi
    _mean_penalty_coef = float(self.mean_penalty_coef)

    if seq_len > 0 and valid_n_step is not None:
        denom = valid_n_step.float().sum().clamp(min=1.0)
        actor_loss = (actor_loss_unmasked * valid_n_step.float()).sum() / denom
        mean_penalty = (
            _mean_penalty_coef * ((mu_actor**2).mean(-1) * valid_n_step.float()).sum() / denom
        )
    else:
        actor_loss = actor_loss_unmasked.mean()
        mean_penalty = _mean_penalty_coef * (mu_actor**2).mean()

    actor_loss = actor_loss + mean_penalty

    bc_lam = self._get_bc_lambda()
    if bc_lam > 0 and isinstance(info, dict) and "is_demo" in info:
        is_demo = info["is_demo"]
        if isinstance(is_demo, torch.Tensor) and is_demo.any():
            is_demo_t = is_demo[:truncated_batch_size]
            a_t = a[:truncated_batch_size]
            if is_demo_t.any():
                a_demo = a_t[is_demo_t]
                pi_demo = pi_actor[is_demo_t]
                bc_loss = ((pi_demo - a_demo) ** 2).mean()
                actor_loss = actor_loss + bc_lam * bc_loss

    # ── 9. Actor update ──
    actor_loss_bad = _is_bad_loss(actor_loss)
    self.actor_optimizer.zero_grad()
    if not actor_loss_bad:
        if self.use_mixed_precision:
            self.grad_scaler.scale(actor_loss).backward()
            self.grad_scaler.unscale_(self.actor_optimizer)
        else:
            actor_loss.backward()
        if self.grad_clip_actor > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(),
                self.grad_clip_actor,
            )
        else:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), float("inf")
            )
        if self.use_mixed_precision:
            self.grad_scaler.step(self.actor_optimizer)
            self.grad_scaler.update()
        else:
            self.actor_optimizer.step()
        if actor_grad_norm is not None and actor_grad_norm < 0.05:
            self._consecutive_low_grad_steps += 1
            if self._consecutive_low_grad_steps > 100 and self.learn_entropy_coef:
                with torch.no_grad():
                    self.log_alpha.fill_(math.log(0.01))
                self._consecutive_low_grad_steps = 0
                if self._training_step % 500 == 0 or self._training_step <= 10:
                    from loguru import logger

                    logger.warning(
                        " Emergency entropy reset: actor_grad_norm < 0.05 for >100 steps; "
                        "log_alpha set to log(0.01)."
                    )
        else:
            self._consecutive_low_grad_steps = 0
    else:
        actor_grad_norm = None

    project_simbav2_weights(self.model)

    # ── 10. Post-update: scheduler, clipping, target update ──
    if self.scheduler_name:
        self.actor_scheduler.step(epoch + batch_index / iters)
        self.critic_scheduler.step(epoch + batch_index / iters)
    if self.weight_clipping_enabled:
        clip_model_weights(self.model.actor, self.clip_weights_value)
        clip_model_weights(self.model.q1, self.clip_weights_value)
        clip_model_weights(self.model.q2, self.clip_weights_value)

    self.model.q1.requires_grad_(True)
    self.model.q2.requires_grad_(True)

    if critic_loss_bad or actor_loss_bad:
        self._consecutive_bad_steps += 1
    else:
        self._consecutive_bad_steps = 0

    if not critic_loss_bad and not actor_loss_bad:
        polyak_update(self.model.q1, self.model_target.q1, self.polyak)
        polyak_update(self.model.q2, self.model_target.q2, self.polyak)

    # ── PER: return TD errors and batch indices for priority update ──
    if self.per_td_enabled and isinstance(info, dict) and "batch_indices" in info:
        bi = info["batch_indices"]
        bi = bi.cpu().numpy() if isinstance(bi, torch.Tensor) else np.asarray(bi)
        td_np = td_errors_batch.cpu().numpy()
        if self.n_steps > 1 and len(bi) > truncated_batch_size:
            mean_td = float(np.mean(td_np))
            padding = np.full((len(bi) - truncated_batch_size,), mean_td, dtype=td_np.dtype)
            td_np = np.concatenate([td_np, padding])
        ret_dict_per = {"td_errors": td_np, "batch_indices": bi}
    else:
        ret_dict_per = {}

    # ── Logging ──
    with torch.no_grad():
        ret_dict = {}
        ret_dict["losses/actor"] = _tensor_to_scalar(actor_loss.detach())
        ret_dict["losses/critic"] = _tensor_to_scalar(critic_loss.detach())
        ret_dict["lrs/actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
        ret_dict["lrs/critic_lr"] = self.critic_optimizer.param_groups[0]["lr"]
        if self.bc_lambda > 0 or self.bc_anneal_steps_end > 0:
            ret_dict["bc/bc_lambda"] = self._get_bc_lambda()
        if critic_grad_norm is not None:
            ret_dict["debug/critic_grad_norm"] = float(critic_grad_norm)
        if actor_grad_norm is not None:
            ret_dict["debug/actor_grad_norm"] = float(actor_grad_norm)
        if self.wandb_debug:
            q1_targ_a = self.model_target.q1(o, a)[:truncated_batch_size]
            q2_targ_a = self.model_target.q2(o, a)[:truncated_batch_size]

            diff_q1_q1t_a = (q1[:truncated_batch_size] - q1_targ_a).detach()
            diff_q2_q2t_a = (q2[:truncated_batch_size] - q2_targ_a).detach()

            q1_t = q1[:truncated_batch_size]
            same_shape = q1_t.shape == backup.shape
            if same_shape:
                diff_q1_backup = (q1_t - backup).detach()
                diff_q2_backup = (q2[:truncated_batch_size] - backup).detach()
            else:
                diff_q1_backup = diff_q2_backup = None

            ret_dict["debug/log_pi"] = _tensor_to_scalar(logp_pi.detach().mean())
            ret_dict["debug/logp_a2"] = _tensor_to_scalar(logp_a2.detach().mean())
            ret_dict["debug/q_a1"] = _tensor_to_scalar(q_pi.detach().mean())
            ret_dict["debug/q_a1_targ"] = _tensor_to_scalar(q_pi_targ.detach().mean())
            ret_dict["debug/backup"] = _tensor_to_scalar(backup.detach().mean())
            ret_dict["debug/q1"] = _tensor_to_scalar(q1.detach().mean())
            ret_dict["debug/q2"] = _tensor_to_scalar(q2.detach().mean())
            ret_dict["debug/diff_q1"] = (
                _tensor_to_scalar(diff_q1_backup.mean()) if diff_q1_backup is not None else 0.0
            )
            ret_dict["debug/diff_q2"] = (
                _tensor_to_scalar(diff_q2_backup.mean()) if diff_q2_backup is not None else 0.0
            )
            ret_dict["debug/diff_q1_q1t_a"] = _tensor_to_scalar(diff_q1_q1t_a.mean())
            ret_dict["debug/diff_q2_q2t_a"] = _tensor_to_scalar(diff_q2_q2t_a.mean())

            ret_dict["debug/a_0"] = _tensor_to_scalar(a[:, 0].detach().mean())
            ret_dict["debug/a_1"] = _tensor_to_scalar(a[:, 1].detach().mean())
            ret_dict["debug/a_2"] = _tensor_to_scalar(a[:, 2].detach().mean())
            ret_dict["debug/a1_0"] = _tensor_to_scalar(pi[:, 0].detach().mean())
            ret_dict["debug/a1_1"] = _tensor_to_scalar(pi[:, 1].detach().mean())
            ret_dict["debug/a1_2"] = _tensor_to_scalar(pi[:, 2].detach().mean())

    if self.entropy_schedule == "cosine":
        ret_dict["entropy_coef"] = float(alpha_t)
    elif self.learn_entropy_coef and alpha_loss is not None:
        ret_dict["loss_entropy_coef"] = alpha_loss.detach().item()
        ret_dict["entropy_coef"] = torch.exp(self.log_alpha.detach()).item()
        ret_dict["lrs/entropy_lr"] = self.alpha_optimizer.param_groups[0]["lr"]
        if self.wandb_debug:
            ret_dict["debug/log_alpha"] = self.log_alpha.detach().item()

    ret_dict.update(ret_dict_per)  # type: ignore[arg-type]
    return ret_dict
