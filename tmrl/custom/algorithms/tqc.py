"""Truncated Quantile Critic (TQC) agent."""

import contextlib
import itertools
import math
from collections import deque
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

from tmrl.custom.algorithms._internal._common import (
    _amp_dtype,
    _amp_enabled,
    _make_optimizer,
    sanitize_obs,
    sanitize_tensor,
    set_seed,
)
from tmrl.custom.algorithms._internal._tqc_train_step import _tqc_train_step
from tmrl.custom.utils.nn_utils import copy_shared, no_grad
from tmrl.registry import ALGORITHMS
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


@ALGORITHMS.register("TQC")
@dataclass(eq=False)
class TQCAgent(TrainingAgent):
    """Truncated Quantile Critic (TQC) agent for continuous control.

    Implements TQC from "Controlling Overestimation Bias with Truncated Mixture of
    Continuous Distributional Quantile Critics" (Kuznetsov et al., 2020).

    All hyperparameters are required constructor arguments — values must be
    supplied explicitly by the config pipeline (no hidden numeric defaults).
    """

    # --- Required: core SAC/TQC hyperparameters ---
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
    top_quantiles_to_drop: int
    quantiles_number: int
    n_steps: int

    # --- Required: optimizer / regularisation ---
    actor_weight_decay: float
    critic_weight_decay: float
    adam_eps: float

    # --- Required: entropy schedule ---
    entropy_schedule: str
    entropy_floor: float
    entropy_cosine_t0: int
    entropy_cosine_tmult: float
    entropy_cosine_decay: float

    # --- Required: training stability ---
    reward_normalize_scale: float
    backup_clip_range: float
    grad_clip_actor: float
    grad_clip_critic: float
    weight_clipping_enabled: bool
    clip_weights_value: float
    mean_penalty_coef: float

    # --- Required: behavior cloning ---
    bc_lambda: float
    bc_lambda_start: float
    bc_lambda_end: float
    bc_anneal_steps_start: int
    bc_anneal_steps_end: int

    # --- Required: advanced TQC knobs ---
    dynamic_truncation_enabled: bool
    dynamic_truncation_variance_pct: float
    vcse_enabled: bool
    vcse_alpha_base: float
    vcse_lambda: float

    # --- Required: sequence replay (R2D2) ---
    r2d2_burn_in: int
    r2d2_sequence_length: int

    # --- Required: PER / debugging / scheduler ---
    per_td_enabled: bool
    wandb_debug: bool
    wandb_gradients: bool
    scheduler_name: str
    scheduler_t_0: int
    scheduler_t_mult: int
    scheduler_eta_min: float
    scheduler_last_epoch: int

    # --- Required: mixed precision ---
    mixed_precision: bool
    mixed_precision_dtype: str

    # --- Required: reproducibility ---
    seed: int

    # --- Structural defaults (None = auto-detect / optional) ---
    device: str | None = None
    target_entropy: float | None = None
    betas_actor: tuple[float, ...] | None = None
    betas_critic: tuple[float, ...] | None = None

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        """Build model, target, optimizers, and entropy coefficient (if learned)."""
        set_seed(self.seed)
        if self.n_steps == 1:
            logger.warning(
                "n_steps=1 is equivalent to n_steps=0 (standard 1-step TD); normalising to 0."
            )
            self.n_steps = 0
        action_space = self.action_space
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(" device TQC: {}", device)
        self.use_mixed_precision = _amp_enabled(device, self.mixed_precision)
        self.amp_dtype = _amp_dtype(self.mixed_precision_dtype)
        self._build_model_and_optimizers(device)

        if self.scheduler_name:
            self.actor_scheduler = CosineAnnealingWarmRestarts(
                self.actor_optimizer,
                self.scheduler_t_0,
                self.scheduler_t_mult,
                self.scheduler_eta_min,
                self.scheduler_last_epoch,
            )
            self.critic_scheduler = CosineAnnealingWarmRestarts(
                self.critic_optimizer,
                self.scheduler_t_0,
                self.scheduler_t_mult,
                self.scheduler_eta_min,
                self.scheduler_last_epoch,
            )

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape).astype(np.float32)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.entropy_schedule == "cosine":
            self.learn_entropy_coef = False
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)
            logger.info(
                " Entropy schedule: cosine (T0={}, Tmult={:.1f}, decay={:.2f}, floor={:.4f})",
                self.entropy_cosine_t0,
                self.entropy_cosine_tmult,
                self.entropy_cosine_decay,
                self.entropy_floor,
            )
        elif self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(
                True
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
            logger.info(" Entropy schedule: learnable (floor={:.4f})", self.entropy_floor)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(device)

        if self.wandb_gradients and wandb is not None:
            wandb.watch(self.model, log_freq=10)
        self._training_step = 0
        self._nan_weight_check_interval = 50
        self._consecutive_bad_steps = 0
        self._consecutive_low_grad_steps = 0
        self._trunc_var_history: deque[float] = deque(maxlen=500)

        if self.reward_normalize_scale != 1.0:
            logger.warning(
                "TQCAgent: reward_normalize_scale={:.4g} — rewards are MULTIPLIED by this "
                "factor. Previous versions divided; if you are loading an old config that "
                "used a large scale (e.g. 200) to shrink rewards, use the reciprocal "
                "(1/200 ≈ 0.005) to preserve the original effect.",
                self.reward_normalize_scale,
            )

    def _build_model_and_optimizers(self, device: str) -> None:
        """(Re)create model, target, optimizers, quantile counts, and grad scaler.

        Called from both ``__post_init__`` and ``_reinitialize_model``.
        ``self.use_mixed_precision`` and ``self.amp_dtype`` must be set before calling.
        """
        observation_space, action_space = self.observation_space, self.action_space
        model = self.model_cls(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))
        self.actor_optimizer = _make_optimizer(
            self.model.actor.parameters(),
            "adam",
            self.lr_actor,
            weight_decay=self.actor_weight_decay,
            eps=self.adam_eps,
            betas=self.betas_actor,
        )
        self.critic_optimizer = _make_optimizer(
            itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()),
            "adam",
            self.lr_critic,
            weight_decay=self.critic_weight_decay,
            eps=self.adam_eps,
            betas=self.betas_critic,
        )
        self.quantiles_total = self.model.q1.num_quantiles + self.model.q2.num_quantiles
        # Proper TQC truncation: drop d * N elements globally
        self.total_quantiles_to_drop = self.top_quantiles_to_drop * 2
        # GradScaler not recommended for bfloat16; use only for float16
        use_scaler = self.use_mixed_precision and (self.amp_dtype != torch.bfloat16)
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def _cosine_alpha(self, step: int) -> float:
        """Cosine annealing with warm restarts for entropy coefficient.

        Each cycle is T_mult longer than the previous one and the peak
        amplitude decays by ``decay`` per cycle, producing an envelope of
        decreasing exploration spikes.
        """
        t0 = max(1, self.entropy_cosine_t0)
        t_mult = self.entropy_cosine_tmult
        decay = self.entropy_cosine_decay
        alpha_max = float(self.alpha)
        alpha_min = self.entropy_floor
        t = step
        cycle = 0
        cycle_len = t0
        while t >= cycle_len:
            t -= cycle_len
            cycle += 1
            cycle_len = max(1, int(t0 * t_mult**cycle))
        amplitude = (alpha_max - alpha_min) * (decay**cycle)
        return alpha_min + 0.5 * amplitude * (1.0 + math.cos(math.pi * t / cycle_len))

    def _model_has_nan_weights(self) -> bool:
        """Check if any trainable parameter contains NaN."""
        return any(p.is_floating_point() and torch.isnan(p).any() for p in self.model.parameters())

    def _reinitialize_model(self) -> None:
        """Rebuild model, target, and optimizers from scratch when weights are corrupted."""
        logger.warning(" Model weights contain NaN — reinitializing model, target, and optimizers.")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._build_model_and_optimizers(device)
        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=device) * self.alpha).requires_grad_(
                True
            )
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        if hasattr(self, "model_nograd"):
            with contextlib.suppress(KeyError):
                del self.__dict__["model_nograd"]
        self._consecutive_bad_steps = 0
        self._consecutive_low_grad_steps = 0
        logger.info(" Model reinitialized successfully.")

    def get_actor(self) -> Any:
        """Return the current actor (policy) module for rollout workers."""
        return self.model_nograd.actor

    def _get_bc_lambda(self) -> float:
        """Current BC coefficient: constant or linear annealing from START to END over steps."""
        base = float(self.bc_lambda)
        step_end = int(self.bc_anneal_steps_end)
        if step_end <= 0:
            return base
        start_val = float(self.bc_lambda_start)
        end_val = float(self.bc_lambda_end)
        step_start = int(self.bc_anneal_steps_start)
        step = self._training_step
        if step <= step_start:
            return start_val
        if step >= step_end:
            return end_val
        frac = (step - step_start) / max(1, step_end - step_start)
        return start_val + (end_val - start_val) * frac

    @staticmethod
    def calculate_huber_loss(td_errors: torch.Tensor, k: float = 1.0) -> torch.Tensor:
        """Compute Huber loss element-wise."""
        loss = torch.where(
            td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k)
        )
        return loss

    def quantile_huber_loss_f(self, quantiles: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """Quantile Huber loss for TQC critic training. Uses FP32 for precision."""
        per_sample = self._quantile_huber_per_sample(quantiles, samples)
        return per_sample.mean()

    def _quantile_huber_per_sample(
        self, quantiles: torch.Tensor, samples: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample quantile Huber loss [batch]. Used for sequence-aware n-step masking."""
        quantiles = quantiles.float()
        samples = samples.float()
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]
        huber_loss = self.calculate_huber_loss(pairwise_delta)

        n_quantiles = quantiles.shape[2]
        tau = (
            torch.arange(n_quantiles, device=quantiles.device).float() / n_quantiles
            + 1 / 2 / n_quantiles
        )
        loss = torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss
        return loss.mean(dim=3).sum(dim=2).mean(dim=1)

    _sanitize_tensor = staticmethod(sanitize_tensor)
    _sanitize_obs = staticmethod(sanitize_obs)

    @staticmethod
    def _has_nan(t: torch.Tensor) -> bool:
        """Fast check: does the tensor contain any NaN?"""
        return bool(t.is_floating_point() and torch.isnan(t).any().item())

    @staticmethod
    def _safe_logprob(logp: torch.Tensor) -> torch.Tensor:
        """Make log-prob safe: nan_to_num first (clamp alone passes NaN through), then clamp."""
        return torch.nan_to_num(logp.float(), nan=-5.0, posinf=0.0, neginf=-50.0).clamp(-50.0, 0.0)

    def train(  # type: ignore[override]
        self, batch: tuple, epoch: int, batch_index: int, iters: int
    ) -> dict:
        return _tqc_train_step(self, batch, epoch, batch_index, iters)
