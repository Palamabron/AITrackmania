"""Extracted IQN training step — called from IQNAgent.train()."""

import torch
from einops import rearrange, repeat
from loguru import logger

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from tmrl.custom.algorithms._internal._common import (
    _tensor_to_scalar,
    autocast_context,
    project_simbav2_weights,
)
from tmrl.custom.algorithms._internal._iqn_schedules import (
    _munchausen_bonus_from_q,
    _quantile_huber_loss,
)
from tmrl.util import wandb_monotonic_step


def _iqn_train_step(self, batch: tuple) -> dict:
    """Run one IQN training step on a sampled batch.

    Args:
        self: IQNAgent instance (passed as first arg to avoid class coupling).
        batch: Tuple of ``(obs, action, reward, next_obs, done, ...)``.
               May include PER importance weights in batch[6]['is_weight'].

    Returns:
        Dict of scalar metrics for logging.
    """
    self._training_step += 1
    eps = self._update_epsilon()

    if self.noisy_linear:
        noise_scale = self._update_noise_scale()
        head = getattr(self.model, "head", None)
        if head is not None and hasattr(head, "reset_noise"):
            head.set_noise_scale(noise_scale)
            head.reset_noise()
        head_tgt = getattr(self.model_target, "head", None)
        if head_tgt is not None and hasattr(head_tgt, "reset_noise"):
            head_tgt.set_noise_scale(noise_scale)
            head_tgt.reset_noise()

    o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
    device = self.device or "cpu"
    batch_size = r.shape[0]

    o = self._sanitize_obs(o)
    o2 = self._sanitize_obs(o2)
    a = self._sanitize_tensor(a)
    r = self._sanitize_tensor(r)
    d = self._sanitize_tensor(d)

    info_dict: dict | None = batch[6] if len(batch) >= 7 and isinstance(batch[6], dict) else None

    if a.dim() >= 2 and a.shape[-1] == 3:
        from tmrl.custom.tm.utils.control.discrete import (
            build_brake_tap_action_table,
            continuous_control_to_discrete_indices_batch,
        )

        if self._legacy_action_table is None:
            _, self._legacy_action_table = build_brake_tap_action_table(
                n_steer=int(self.iqn_n_steer_bins)
            )
        idx = continuous_control_to_discrete_indices_batch(
            a.cpu().numpy(), self._legacy_action_table
        )
        a = torch.from_numpy(idx).to(device=a.device, dtype=torch.long)
    actions = a.long().squeeze(-1)

    # Munchausen bonus added in raw reward space, before any scale is applied,
    # so that bonus and environment reward are in the same units and both are
    # compressed together by reward_normalize_scale.  Applying the bonus after
    # scaling would leave it unscaled (up to 180x larger than the scaled reward
    # when reward_normalize_scale is small, e.g. 0.005).
    if self.munchausen_enabled:
        with torch.no_grad():
            q_curr = self.model.q_values(o, n_quantiles=self.n_quantiles_eval)
            munchausen_bonus = _munchausen_bonus_from_q(
                q_values=q_curr,
                actions=actions,
                tau=float(self.munchausen_tau),
                clip_min=float(self.munchausen_clip_min),
                clip_max=float(self.munchausen_clip_max),
            )
            # Reshape to r's exact shape: r can be (b,) or (b, 1) depending on
            # the memory; a blind unsqueeze would broadcast (b,) + (b, 1) to (b, b).
            # Match r's dtype too (q_curr may be bf16 under autocast).
            r = r + float(self.munchausen_alpha) * munchausen_bonus.to(dtype=r.dtype).reshape(
                r.shape
            )
    else:
        munchausen_bonus = torch.zeros(batch_size, device=device)

    # Scale rewards (and Munchausen bonus) before the Bellman backup.
    # Values < 1 shrink the combined signal (e.g. 0.005 → 0.5 % of original).
    if self.reward_normalize_scale != 1.0 and self.reward_normalize_scale > 0:
        r = r * self.reward_normalize_scale

    # Memory-side n-step: ``r`` is already the discounted n-step return,
    # ``o2`` the observation at the end of the window, and ``d`` the
    # terminated flag of the window's last step (episode boundaries are
    # handled by the memory, which never accumulates across them).
    # ``n_step_effective`` carries the per-sample window length (1 for
    # plain 1-step transitions). Bootstrap through truncation, never
    # through termination.
    n_eff = info_dict.get("n_step_effective", None) if info_dict is not None else None
    if n_eff is not None and self.n_steps > 1 and not self._warned_n_step_all_one:
        n_eff_t = torch.as_tensor(n_eff).float().reshape(-1).cpu()
        if n_eff_t.numel() > 0 and bool((n_eff_t <= 1.0).all()):
            logger.warning(
                "IQN n_steps={} but all sampled windows have n_step_effective=1 "
                "(frequent resets or very short episodes); using 1-step bootstrap "
                "for this batch. Will not repeat until a deeper regression occurs.",
                self.n_steps,
            )
            self._warned_n_step_all_one = True
    elif n_eff is None and self.n_steps > 1 and not self._warned_missing_n_step_metadata:
        logger.warning(
            "IQN n_steps={} but the replay batch carries no 'n_step_effective' metadata "
            "(this memory does not implement memory-side n-step returns); "
            "training with 1-step targets instead.",
            self.n_steps,
        )
        self._warned_missing_n_step_metadata = True

    n_step_return = r.squeeze(-1)
    bootstrap_mask = 1.0 - d.squeeze(-1)
    if n_eff is not None:
        n_eff_t = torch.as_tensor(n_eff, device=n_step_return.device).float().clamp(min=1.0)
        gamma_n = torch.pow(torch.tensor(float(self.gamma), device=n_step_return.device), n_eff_t)
    else:
        gamma_n = torch.full_like(bootstrap_mask, float(self.gamma) ** self.n_steps)

    def autocast_ctx():
        return autocast_context(self.use_mixed_precision, self.amp_dtype)

    with autocast_ctx():
        # sort_quantiles only sorts the sampled fractions BEFORE the forward
        # pass (stable diagnostics / monotonicity regularization). Network
        # outputs must NOT be re-sorted afterwards: each output quantile is
        # paired with its tau in the quantile-Huber loss, and sorting outputs
        # would assign gradients to the wrong fractions.
        tau = torch.rand(batch_size, self.n_quantiles_train, device=device)
        if self.sort_quantiles:
            tau, _ = torch.sort(tau, dim=1)
        current_quantiles, _, dueling_head_stats = self.model.forward_with_head_stats(o, tau=tau)
        action_idx = repeat(actions, "b -> b n 1", n=self.n_quantiles_train)
        current_q = current_quantiles.gather(2, action_idx).squeeze(2)

    with torch.no_grad():
        tau_prime = torch.rand(batch_size, self.n_quantiles_target, device=device)
        if self.sort_quantiles:
            tau_prime, _ = torch.sort(tau_prime, dim=1)

        with autocast_ctx():
            if self.double_dqn:
                online_q_next = self.model.q_values(o2, n_quantiles=self.n_quantiles_eval)
                next_actions = online_q_next.argmax(dim=-1)

            target_quantiles, _ = self.model_target(o2, tau=tau_prime)

        if not self.double_dqn:
            # Reuse the already-computed target quantiles for action selection:
            # mean over the tau_prime dimension → (B, N_actions) → argmax.
            # This avoids a redundant second forward pass through the target network.
            next_actions = target_quantiles.mean(dim=1).argmax(dim=-1)

        next_action_idx = repeat(next_actions, "b -> b n 1", n=self.n_quantiles_target)
        next_q = target_quantiles.gather(2, next_action_idx).squeeze(2)

        target = (
            rearrange(n_step_return, "b -> b 1")
            + rearrange(gamma_n * bootstrap_mask, "b -> b 1") * next_q
        )
        backup_clip = float(self.backup_clip_range)
        if backup_clip > 0.0:
            target = target.clamp(min=-backup_clip, max=backup_clip)

    # Compute the quantile-Huber loss in fp32 regardless of the autocast
    # dtype: the squared-delta term loses precision in bf16/fp16 and the
    # loss gradients are the training signal. (Matches the TQC critic.)
    current_for_loss = current_q.float()
    target_for_loss = target.float()

    is_weights = None
    if info_dict is not None:
        is_weights = info_dict.get("is_weight", None)
        if is_weights is not None:
            if not isinstance(is_weights, torch.Tensor):
                is_weights = torch.as_tensor(is_weights, device=device, dtype=torch.float32)
            else:
                is_weights = is_weights.to(device=device, dtype=torch.float32)

    loss_iqn = _quantile_huber_loss(
        current_for_loss, target_for_loss, tau, kappa=self.huber_kappa, is_weights=is_weights
    )

    if torch.isnan(loss_iqn).any() or torch.isinf(loss_iqn).any():
        logger.error(
            "NaN/Inf detected in IQN loss! current_q range=[{:.2f}, {:.2f}], "
            "target range=[{:.2f}, {:.2f}], skipping update",
            current_q.min().item(),
            current_q.max().item(),
            target.min().item(),
            target.max().item(),
        )
        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.update()
        return {
            "loss/iqn_loss": 0.0,
            "loss/total_loss": 0.0,
            "exploration/epsilon": self._epsilon,
            "debug/nan_detected": 1.0,
        }

    if self.n_quantiles_train > 1:
        dq = current_q[:, 1:] - current_q[:, :-1]
        if self.monotonicity_regularization:
            monotonic_penalty = torch.relu(-dq).mean()
            crossing_magnitude = monotonic_penalty.detach()
            crossing_rate = (dq.detach() < 0).float().mean()
        else:
            monotonic_penalty = torch.zeros((), device=current_q.device, dtype=current_q.dtype)
            with torch.no_grad():
                crossing_magnitude = torch.relu(-dq).mean()
                crossing_rate = (dq < 0).float().mean()
    else:
        monotonic_penalty = torch.zeros((), device=current_q.device, dtype=current_q.dtype)
        crossing_magnitude = torch.zeros((), device=current_q.device, dtype=current_q.dtype)
        crossing_rate = torch.zeros((), device=current_q.device, dtype=current_q.dtype)

    # DQfD large-margin loss on demo samples: push Q(s, a_demo) above every
    # other action by at least bc_margin. TD backups alone propagate demo
    # values too slowly to move the argmax across 78 actions.
    bc_lam = self._get_bc_lambda()
    bc_loss = torch.zeros((), device=current_q.device, dtype=torch.float32)
    demo_argmax_match = float("nan")
    demo_mask = None
    if info_dict is not None:
        raw_mask = info_dict.get("is_demo", None)
        if raw_mask is not None:
            demo_mask = torch.as_tensor(raw_mask, device=current_q.device).reshape(-1).bool()
    if bc_lam > 0.0 and demo_mask is not None and bool(demo_mask.any()):
        q_all = current_quantiles.float().mean(dim=1)  # (b, n_actions)
        a_col = actions.reshape(-1, 1)
        q_demo_action = q_all.gather(1, a_col).squeeze(1)
        # Margin added to every action except the demo action (where the
        # entry is replaced by Q(s,a_E) itself, so J_E >= 0 by construction).
        margined = (q_all + float(self.bc_margin)).scatter(1, a_col, q_demo_action.unsqueeze(1))
        per_sample_margin = margined.max(dim=1).values - q_demo_action
        bc_loss = per_sample_margin[demo_mask].mean()
        with torch.no_grad():
            demo_argmax_match = float(
                (q_all.argmax(dim=1) == actions.reshape(-1))[demo_mask].float().mean()
            )

    loss = loss_iqn + float(self.monotonicity_lambda) * monotonic_penalty + bc_lam * bc_loss

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        logger.error("NaN/Inf in total loss after monotonicity penalty, skipping update")
        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.update()
        return {
            "loss/iqn_loss": _tensor_to_scalar(loss_iqn),
            "loss/total_loss": 0.0,
            "loss/monotonicity_penalty": _tensor_to_scalar(monotonic_penalty),
            "exploration/epsilon": self._epsilon,
            "debug/nan_detected": 1.0,
        }

    # Single grad-norm pass: clip_grad_norm_ returns the total norm BEFORE
    # clipping and rescales in place when it exceeds the threshold, so the
    # post-clip norm is exactly min(pre, clip_val) — no extra passes needed.
    # clip_val=inf measures without scaling (grad_clip disabled).
    clip_val = float(self.grad_clip) if self.grad_clip > 0.0 else float("inf")
    self.optimizer.zero_grad()
    optimizer_stepped = False
    if self.use_mixed_precision:
        self.grad_scaler.scale(loss).backward()
        # Always unscale before grad norms / stabilizer / clip (even when grad_clip==0).
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm_pre_clip = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        )
        grad_norm_post_clip = min(grad_norm_pre_clip, clip_val)
        grad_norm = self._stabilize_grads(grad_norm_post_clip)
        scale_before = float(self.grad_scaler.get_scale())
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        optimizer_stepped = float(self.grad_scaler.get_scale()) >= scale_before
    else:
        loss.backward()
        grad_norm_pre_clip = float(
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_val)
        )
        grad_norm_post_clip = min(grad_norm_pre_clip, clip_val)
        grad_norm = self._stabilize_grads(grad_norm_post_clip)
        self.optimizer.step()
        optimizer_stepped = True

    if self._lr_scheduler is not None and optimizer_stepped:
        self._lr_scheduler.step()

    project_simbav2_weights(self.model)

    if self.soft_target_tau > 0.0:
        tau_polyak = float(self.soft_target_tau)
        with torch.no_grad():
            for p_tgt, p_src in zip(
                self.model_target.parameters(), self.model.parameters(), strict=False
            ):
                p_tgt.lerp_(p_src, tau_polyak)
    elif self._training_step % self.target_update_freq == 0:
        self.model_target.load_state_dict(self.model.state_dict())
        for p in self.model_target.parameters():
            p.requires_grad = False
        logger.debug(" Hard target update at step {}", self._training_step)

    iqn_loss_scalar = _tensor_to_scalar(loss_iqn)
    ret = {
        "loss/iqn_loss": iqn_loss_scalar,
        "loss/total_loss": _tensor_to_scalar(loss),
        "loss/monotonicity_penalty": _tensor_to_scalar(monotonic_penalty),
        "loss/bc_margin": _tensor_to_scalar(bc_loss),
        "bc/bc_lambda": bc_lam,
        "exploration/epsilon": eps,
        "q/mean_q": _tensor_to_scalar(current_q.mean()),
        "q/max_q": _tensor_to_scalar(current_q.max()),
        "q/min_q": _tensor_to_scalar(current_q.min()),
        "q/std_q": _tensor_to_scalar(current_q.std()),
        "debug/quantile_crossing_rate": _tensor_to_scalar(crossing_rate),
        "debug/quantile_crossing_magnitude": _tensor_to_scalar(crossing_magnitude),
        "debug/target_mean": _tensor_to_scalar(target.mean()),
        "debug/target_max": _tensor_to_scalar(target.max()),
        "debug/target_min": _tensor_to_scalar(target.min()),
        "debug/reward_mean": _tensor_to_scalar(r.mean()),
        "debug/reward_max": _tensor_to_scalar(r.max()),
        "debug/munchausen_bonus_mean": _tensor_to_scalar(munchausen_bonus.mean()),
        "debug/grad_norm_pre_clip": grad_norm_pre_clip,
        "debug/grad_norm_post_clip": grad_norm_post_clip,
        "debug/grad_norm": grad_norm,
        "debug/grad_ema_norm": self._grad_ema_norm(),
        "train/lr": float(self.optimizer.param_groups[0]["lr"]),
        "train/step": self._training_step,
    }
    if self.noisy_linear:
        ret["exploration/noise_scale"] = self._noise_scale
    if demo_argmax_match == demo_argmax_match:  # not NaN: demos present in batch
        ret["debug/demo_argmax_match"] = demo_argmax_match
    if dueling_head_stats is not None:
        value = dueling_head_stats["value"]
        advantage = dueling_head_stats["advantage"]
        centered_advantage = dueling_head_stats["centered_advantage"]
        adv_span = advantage.max(dim=-1).values - advantage.min(dim=-1).values
        ret.update(
            {
                "debug/dueling_value_mean": _tensor_to_scalar(value.mean()),
                "debug/dueling_value_std": _tensor_to_scalar(value.std(unbiased=False)),
                "debug/dueling_adv_mean": _tensor_to_scalar(advantage.mean()),
                "debug/dueling_adv_abs_mean": _tensor_to_scalar(advantage.abs().mean()),
                "debug/dueling_adv_std": _tensor_to_scalar(advantage.std(unbiased=False)),
                "debug/dueling_centered_adv_abs_mean": _tensor_to_scalar(
                    centered_advantage.abs().mean()
                ),
                "debug/dueling_adv_span_mean": _tensor_to_scalar(adv_span.mean()),
            }
        )
    if self.log_target_stats:
        with torch.no_grad():
            td_abs = (target.mean(dim=1) - current_q.mean(dim=1)).abs()
            td_p95 = torch.quantile(td_abs, 0.95) if td_abs.numel() > 1 else td_abs.max()
            ret["q/target_mean"] = _tensor_to_scalar(target.mean())
            ret["q/target_max"] = _tensor_to_scalar(target.max())
            ret["debug/td_abs_mean"] = _tensor_to_scalar(td_abs.mean())
            ret["debug/td_abs_p95"] = _tensor_to_scalar(td_p95)

    if wandb is not None and wandb.run is not None:
        wandb.log(ret, step=wandb_monotonic_step(self._training_step, wandb.run))

    return ret
