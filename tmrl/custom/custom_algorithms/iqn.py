"""IQN (Implicit Quantile Network) agent with Double DQN.

Implements DQN + Double DQN + IQN distributional RL + Dueling architecture
+ epsilon-greedy exploration + n-step returns.

References:
  - DQN: Mnih et al. 2015
  - Double DQN: van Hasselt et al. 2016
  - Dueling: Wang et al. 2016
  - IQN: Dabney et al. 2018
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import gymnasium
import torch
from einops import rearrange, repeat
from loguru import logger
from torch.optim import Adam

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from tmrl.custom.custom_algorithms._common import (
    _tensor_to_scalar,
    amp_setup,
    autocast_context,
    project_simbav2_weights,
    sanitize_obs,
    sanitize_tensor,
    set_seed,
)
from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import DQNActor, IQNQNetwork
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.custom.utils.optim import GradientStabilizer
from tmrl.registry import ALGORITHMS
from tmrl.training import TrainingAgent
from tmrl.util import cached_property, wandb_monotonic_step


def epsilon_cosine_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    t0: float = 50000.0,
    tmult: float = 1.5,
    decay: float = 0.8,
    initial_amplitude: float = 0.1,
    floor_frac: float = 0.0,
    floor_steps: int = 0,
    mode: str = "cosine",
) -> float:
    """Epsilon schedule for a given step (for plotting/debugging).

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Minimum epsilon floor value.
        t0: Initial cycle length.
        tmult: Cycle length multiplier (>1 for expanding cycles).
        decay: Amplitude decay factor per cycle.
        initial_amplitude: Initial oscillation amplitude.
        floor_frac: Fraction of cycle spent at floor (0-1).
        floor_steps: Explicit floor duration in steps (overrides floor_frac if >0).
        mode: Schedule mode - "cosine" or "ramp".

    Returns:
        Epsilon value for the given step.

    Note:
        Mode options:
        - "cosine": Damped sinusoid (full wave per cycle, peak->trough->peak).
        - "ramp": Half-cosine (peak->trough per cycle, then floor).
    """
    min_eps = epsilon_end
    floor_frac = max(0.0, min(1.0, floor_frac))
    floor_steps = max(0, floor_steps)

    if step <= 0.0:
        return epsilon_start

    if tmult <= 1.0:
        cycle_num = int(step // t0)
        step_in_cycle = step - cycle_num * t0
        cycle_length = t0
    else:
        ratio = 1.0 + step * (tmult - 1.0) / t0
        cycle_num = int(math.log(ratio) / math.log(tmult)) if ratio > 1.0 else 0
        cum_start = t0 * (tmult**cycle_num - 1.0) / (tmult - 1.0) if cycle_num > 0 else 0.0
        step_in_cycle = step - cum_start
        cycle_length = t0 * (tmult**cycle_num)

    if floor_steps > 0:
        floor_duration = min(floor_steps, cycle_length)
    else:
        floor_duration = floor_frac * cycle_length
    cosine_length = max(1e-9, cycle_length - floor_duration)

    if step_in_cycle >= cosine_length:
        return min_eps

    if mode == "ramp":
        current_amplitude = max(0.0, epsilon_start - min_eps) * (decay**cycle_num)
        angle = math.pi * (step_in_cycle / cosine_length)
    else:
        current_amplitude = max(0.0, initial_amplitude) * (decay**cycle_num)
        phase = step_in_cycle / cosine_length
        angle = 2.0 * math.pi * phase

    return min_eps + 0.5 * current_amplitude * (1.0 + math.cos(angle))


def epsilon_linear_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    decay_steps: float = 500000.0,
) -> float:
    """Linear decay from epsilon_start to epsilon_end (floor) over decay_steps.

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Final epsilon value (floor).
        decay_steps: Number of steps for full decay.

    Returns:
        Epsilon value for the given step.
    """
    if step <= 0.0:
        return epsilon_start
    if step >= decay_steps:
        return epsilon_end
    frac = step / decay_steps
    return epsilon_start + (epsilon_end - epsilon_start) * frac


def epsilon_cosine_anneal_schedule(
    step: float,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.005,
    decay_steps: float = 500000.0,
) -> float:
    """Cosine annealing (single period) from epsilon_start to epsilon_end over decay_steps.

    Args:
        step: Current training step.
        epsilon_start: Initial epsilon value.
        epsilon_end: Final epsilon value.
        decay_steps: Number of steps for full annealing.

    Returns:
        Epsilon value for the given step.
    """
    if step <= 0.0:
        return epsilon_start
    if step >= decay_steps:
        return epsilon_end
    frac = min(1.0, step / decay_steps)
    return epsilon_end + 0.5 * (epsilon_start - epsilon_end) * (1.0 + math.cos(math.pi * frac))


def _quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
    is_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantile Huber loss for IQN.

    Args:
        current_quantiles: (batch, N_tau, n_actions_selected=1) or (batch, N_tau).
        target_quantiles:  (batch, N_tau_prime).
        tau: (batch, N_tau) quantile fractions for current.
        kappa: Huber threshold.
        is_weights: (batch,) importance sampling weights for PER (optional).

    Returns:
        Scalar loss (sum over N_tau_prime, mean over N_tau, then mean over batch).
    """
    if current_quantiles.dim() == 2:
        current_quantiles = current_quantiles.unsqueeze(-1)
    if target_quantiles.dim() == 2:
        target_quantiles = target_quantiles.unsqueeze(1)

    delta = target_quantiles - current_quantiles
    abs_delta = delta.abs()

    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    tau_expanded = rearrange(tau, "b n -> b n 1")
    weight = torch.abs(tau_expanded - (delta.detach() < 0).float())
    # Sum over N_tau_prime (target), then mean over N_tau (current) — Dabney et al. 2018.
    per_sample_loss = (weight * huber).sum(dim=-1).mean(dim=-1)

    if is_weights is not None:
        # IS weights are already normalized in training_offline.py, so apply directly
        per_sample_loss = per_sample_loss * is_weights.squeeze()

    return per_sample_loss.mean()


def _munchausen_bonus_from_q(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    tau: float,
    clip_min: float,
    clip_max: float,
) -> torch.Tensor:
    """Compute clipped Munchausen log-policy bonus for selected actions.

    Args:
        q_values: Q-values for all actions (batch, n_actions).
        actions: Selected actions (batch,).
        tau: Temperature parameter for policy extraction.
        clip_min: Minimum clip value for log-policy.
        clip_max: Maximum clip value for log-policy.

    Returns:
        Clipped log-policy bonus for selected actions (batch,).
    """
    logits = q_values / tau
    log_policy = torch.log_softmax(logits, dim=-1)
    log_pi_a = log_policy.gather(1, actions.view(-1, 1)).squeeze(1)
    return log_pi_a.clamp(min=clip_min, max=clip_max)


@ALGORITHMS.register("IQN")
@dataclass(eq=False)
class IQNAgent(TrainingAgent):
    """IQN agent for discrete control with Double DQN and Dueling heads.

    Operates on a Discrete action space (single int index per step).
    The Q-network outputs quantile values for every action in one pass.

    All hyperparameters are required constructor arguments — values must be
    supplied explicitly by the config pipeline (no hidden numeric defaults).

    Critical Implementation Notes:
        - **N-step returns**: Computed memory-side (e.g. GenericTorchMemory with
          n_step_return > 1). The sampled reward is already the discounted n-step sum,
          next_obs is the window-end observation, and batch info carries
          ``n_step_effective`` so the bootstrap uses gamma**n_eff per sample.
        - **Value rescaling**: NEVER applied to quantiles during training (would distort
          distributional relationships). Quantile regression must occur in original value space.
        - **Munchausen bonus**: Added once to the (already aggregated) sampled reward,
          equivalent to adding it to the first reward of the window (gamma^0 = 1, per
          Vieillard et al. 2020). Uses online network for current policy.
        - **PER IS weights**: Expected to be pre-normalized by max(w) in the training loop
          before being passed to this agent's train() method.
        - **SimbaV2 projection**: Weights are projected BEFORE target update to ensure
          target network receives valid projected weights.
        - **NaN/Inf detection**: Loss is validated before backprop. If detected, update
          is skipped and logged to prevent gradient corruption.
    """

    observation_space: Any
    action_space: Any

    # Architecture
    hidden_dim: int
    num_blocks: int
    n_cos: int
    dueling: bool
    n_actions: int

    # IQN
    n_quantiles_train: int
    n_quantiles_target: int
    n_quantiles_eval: int

    # DQN
    gamma: float
    lr: float
    n_steps: int
    double_dqn: bool
    target_update_freq: int

    # Epsilon-greedy: cosine | ramp | cosine_anneal | linear
    epsilon_schedule_mode: str
    epsilon_start: float
    epsilon_end: float
    epsilon_decay_steps: float
    epsilon_cosine_t0: float
    epsilon_cosine_tmult: float
    epsilon_cosine_decay: float
    epsilon_cosine_initial_amplitude: float
    epsilon_cosine_floor_fraction: float
    epsilon_cosine_floor_steps: int

    # Smooth exploration: hold random action this many steps (DQNActor)
    explore_repeat_steps: int

    # Optimizer / regularisation
    weight_decay: float
    adam_eps: float
    grad_clip: float
    huber_kappa: float
    soft_target_tau: float
    log_target_stats: bool
    sort_quantiles: bool
    monotonicity_regularization: bool
    monotonicity_lambda: float
    munchausen_enabled: bool
    munchausen_alpha: float
    munchausen_tau: float
    munchausen_clip_min: float
    munchausen_clip_max: float

    # Previously hidden globals — now explicit
    iqn_n_steer_bins: int
    backup_clip_range: float
    reward_normalize_scale: float

    # Mixed precision
    mixed_precision: bool
    mixed_precision_dtype: str

    # --- Required: reproducibility ---
    seed: int

    # Structural defaults (None = auto-detect)
    device: str | None = None

    # DQfD large-margin classification loss on demo samples:
    #   J_E = max_a [Q(s,a) + bc_margin*1{a != a_E}] - Q(s,a_E)
    # Demos otherwise influence learning only through TD backups on their batch
    # share, which is far too slow to steer the argmax over 78 actions.
    # bc_lambda anneals from bc_lambda_start to bc_lambda_end over
    # [bc_anneal_steps_start, bc_anneal_steps_end] training steps;
    # bc_anneal_steps_end <= 0 uses the static bc_lambda (0.0 = off).
    bc_lambda: float = 0.0
    bc_lambda_start: float = 1.0
    bc_lambda_end: float = 0.01
    bc_anneal_steps_start: int = 0
    bc_anneal_steps_end: int = 0
    bc_margin: float = 0.5

    # Backbone architecture kwargs (must match worker — forwarded to IQNQNetwork)
    split_track_observation: bool = True
    track_encoder: str = "conv1d"
    use_rnn: bool = False
    rnn_hidden_size: int | None = None
    api_layernorm: bool = False
    use_simbav2: bool = False
    r2d2_sequence_length: int = 0
    r2d2_burn_in: int = 0
    gnn_hidden: int = 64
    gnn_layers: int = 3

    # NoisyNet: factorized Gaussian noise on DuelingHead output layers
    noisy_linear: bool = False
    noisy_std_init: float = 0.5
    noisy_eval_std: float = 0.01
    noisy_scale_start: float = 1.0
    noisy_scale_end: float = 0.05
    noisy_scale_decay_steps: int = 1_000_000

    # Gradient stabilizer (EMA rescale applied AFTER hard grad clip)
    grad_stabilizer_enabled: bool = True
    grad_stabilizer_ema_decay: float = 0.995

    # LR schedule: linear warmup then optional cosine decay (driven by training step count)
    lr_warmup_steps: int = 0
    lr_cosine_decay: bool = False
    lr_total_steps: int = 0
    lr_min: float = 1e-6

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def _iqn_network_kwargs(self) -> dict:
        """Common architecture kwargs shared between IQNQNetwork and DQNActor."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "n_cos": self.n_cos,
            "dueling": self.dueling,
            "n_actions": self.n_actions,
            "noisy": self.noisy_linear,
            "noisy_std_init": self.noisy_std_init,
            "split_track_observation": self.split_track_observation,
            "track_encoder": self.track_encoder,
            "use_rnn": self.use_rnn,
            "rnn_hidden_size": self.rnn_hidden_size,
            "api_layernorm": self.api_layernorm,
            "use_simbav2": self.use_simbav2,
            "r2d2_sequence_length": self.r2d2_sequence_length,
            "r2d2_burn_in": self.r2d2_burn_in,
            "gnn_hidden": self.gnn_hidden,
            "gnn_layers": self.gnn_layers,
        }

    def __post_init__(self) -> None:
        set_seed(self.seed)
        if int(self.n_steps) < 1:
            raise ValueError(f"IQN n_steps must be >= 1, got {self.n_steps}")
        if self.monotonicity_regularization and not self.sort_quantiles:
            raise ValueError("IQN monotonicity_regularization requires sort_quantiles=True.")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IQNQNetwork(self.observation_space, **self._iqn_network_kwargs()).to(device)

        self.model_target = no_grad(deepcopy(self.model))

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=float(self.adam_eps),
        )
        self.use_mixed_precision, self.amp_dtype, self.grad_scaler = amp_setup(
            device, self.mixed_precision, self.mixed_precision_dtype
        )

        self._training_step = 0
        self._warned_missing_n_step_metadata = False
        self._warned_n_step_all_one = False
        self._epsilon = self.epsilon_start
        self._noise_scale = float(self.noisy_scale_start)
        self._grad_stabilizer = (
            GradientStabilizer(ema_decay=float(self.grad_stabilizer_ema_decay))
            if self.grad_stabilizer_enabled
            else None
        )
        self._lr_scheduler = self._build_lr_scheduler()
        if self.munchausen_enabled and self.reward_normalize_scale != 1.0:
            logger.warning(
                "IQNAgent: munchausen_enabled=True with reward_normalize_scale={:.4g}. "
                "Q-values are trained in scaled-reward space; munchausen_tau={:.4g} should "
                "be calibrated relative to expected scaled Q-value magnitudes, not raw rewards.",
                self.reward_normalize_scale,
                self.munchausen_tau,
            )
        logger.info(
            "IQNAgent: n_actions={}, dueling={}, double={}, n_steps={}, gamma={:.3f}, "
            "eps mode={}, decay_steps={}, t0={}, tmult={}, decay={}, init_amp={}, "
            "floor_frac={}, floor_steps={}, huber_kappa={}, "
            "soft_target_tau={}, sort_quantiles={}, monotonicity={}, "
            "monotonicity_lambda={}, munchausen={}",
            self.n_actions,
            self.dueling,
            self.double_dqn,
            self.n_steps,
            self.gamma,
            self.epsilon_schedule_mode,
            self.epsilon_decay_steps,
            self.epsilon_cosine_t0,
            self.epsilon_cosine_tmult,
            self.epsilon_cosine_decay,
            self.epsilon_cosine_initial_amplitude,
            self.epsilon_cosine_floor_fraction,
            self.epsilon_cosine_floor_steps,
            self.huber_kappa,
            self.soft_target_tau,
            self.sort_quantiles,
            self.monotonicity_regularization,
            self.monotonicity_lambda,
            self.munchausen_enabled,
        )
        if self.bc_lambda > 0.0 and self.reward_normalize_scale != 1.0:
            expected_q_scale = self.reward_normalize_scale / max(1e-9, 1.0 - float(self.gamma))
            if self.bc_margin > expected_q_scale:
                logger.warning(
                    "IQNAgent: bc_margin={:.4g} may be too large relative to expected Q-value "
                    "scale (~{:.4g}) with reward_normalize_scale={:.4g}; "
                    "consider scaling bc_margin proportionally.",
                    self.bc_margin,
                    expected_q_scale,
                    self.reward_normalize_scale,
                )
        _obs = self.observation_space
        _obs_dim = (
            sum(math.prod(s.shape or ()) for s in _obs.spaces)
            if isinstance(_obs, gymnasium.spaces.Tuple)
            else math.prod(_obs.shape or ())
        )
        logger.info(
            "IQNAgent model fingerprint: observation_space total_dim={}",
            _obs_dim,
        )
        if self.noisy_linear:
            logger.info(
                "IQNAgent NoisyNet: std_init={}, scale_start={}, scale_end={}, "
                "scale_decay_steps={}",
                self.noisy_std_init,
                self.noisy_scale_start,
                self.noisy_scale_end,
                self.noisy_scale_decay_steps,
            )

    def _build_lr_scheduler(self):
        """Build a linear-warmup (+ optional cosine decay) LR scheduler, or None.

        The schedule is driven by the trainer's gradient-step counter (one ``.step()``
        per ``train()`` call). Returns None when neither warmup nor cosine decay is enabled,
        so the default behaviour (fixed ``iqn_lr``) is preserved.
        """
        if self.lr_warmup_steps <= 0 and not self.lr_cosine_decay:
            return None
        from torch.optim.lr_scheduler import LambdaLR

        return LambdaLR(self.optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step: int) -> float:
        """Multiplicative LR factor on the base ``iqn_lr`` for a given gradient step."""
        warmup = max(0, int(self.lr_warmup_steps))
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if self.lr_cosine_decay and self.lr_total_steps > warmup:
            progress = (step - warmup) / float(max(1, self.lr_total_steps - warmup))
            progress = min(1.0, max(0.0, progress))
            floor = (self.lr_min / self.lr) if self.lr > 0 else 0.0
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    def _stabilize_grads(self, fallback_norm: float) -> float:
        """Run the EMA gradient stabilizer if enabled, else return the post-clip norm."""
        if self._grad_stabilizer is None:
            return fallback_norm
        return self._grad_stabilizer.step(self.model.parameters())

    def _grad_ema_norm(self) -> float:
        return self._grad_stabilizer.ema_norm if self._grad_stabilizer is not None else 0.0

    def _update_noise_scale(self) -> float:
        """Linearly decay noise scale from start to end over decay steps."""
        if not self.noisy_linear or self.noisy_scale_decay_steps <= 0:
            return self._noise_scale
        t = float(self._training_step)
        frac = min(1.0, t / float(self.noisy_scale_decay_steps))
        self._noise_scale = self.noisy_scale_start + frac * (
            self.noisy_scale_end - self.noisy_scale_start
        )
        return self._noise_scale

    def _get_bc_lambda(self) -> float:
        """Current demo-margin coefficient: constant, or linear anneal START->END over steps."""
        base = float(self.bc_lambda)
        if base <= 0.0:
            return 0.0
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

    def _update_epsilon(self) -> float:
        """Update and return current epsilon value based on configured schedule.

        Returns:
            Current epsilon value for exploration.

        Note:
            Schedule modes:
            - cosine: Damped oscillation with expanding cycles
            - ramp: Half-cosine decay with expanding cycles
            - cosine_anneal: Single-period cosine decay to floor
            - linear: Linear decay to epsilon_end
        """
        min_eps = self.epsilon_end
        t = float(self._training_step)
        mode = (self.epsilon_schedule_mode or "cosine").strip().lower()

        if mode == "linear":
            self._epsilon = epsilon_linear_schedule(
                t,
                epsilon_start=self.epsilon_start,
                epsilon_end=min_eps,
                decay_steps=self.epsilon_decay_steps,
            )
            return self._epsilon
        if mode == "cosine_anneal":
            self._epsilon = epsilon_cosine_anneal_schedule(
                t,
                epsilon_start=self.epsilon_start,
                epsilon_end=min_eps,
                decay_steps=self.epsilon_decay_steps,
            )
            return self._epsilon

        # cosine | ramp
        if mode not in ("cosine", "ramp"):
            mode = "cosine"
        t0 = self.epsilon_cosine_t0
        tmult = self.epsilon_cosine_tmult
        decay = self.epsilon_cosine_decay
        initial_amplitude = max(0.0, self.epsilon_cosine_initial_amplitude)
        floor_frac = max(0.0, min(1.0, self.epsilon_cosine_floor_fraction))
        floor_steps = max(0, self.epsilon_cosine_floor_steps)
        self._epsilon = epsilon_cosine_schedule(
            t,
            epsilon_start=self.epsilon_start,
            epsilon_end=min_eps,
            t0=t0,
            tmult=tmult,
            decay=decay,
            initial_amplitude=initial_amplitude,
            floor_frac=floor_frac,
            floor_steps=floor_steps,
            mode=mode,
        )
        return self._epsilon

    def get_actor(self) -> DQNActor:
        """Return actor module with current Q-network weights + epsilon.

        Returns:
            DQNActor wrapper with synchronized weights and current epsilon.
        """
        wrapper = DQNActor(
            self.observation_space,
            self.action_space,
            **self._iqn_network_kwargs(),
            epsilon=self._epsilon,
            n_quantiles_eval=self.n_quantiles_eval,
            explore_repeat_steps=self.explore_repeat_steps,
            noisy_eval_std=self.noisy_eval_std,
        )
        wrapper.q_net.load_state_dict(self.model_nograd.state_dict())
        wrapper.set_epsilon(self._epsilon)
        if self.noisy_linear:
            wrapper.set_noise_scale(self._noise_scale)
        return wrapper

    _sanitize_tensor = staticmethod(sanitize_tensor)
    _sanitize_obs = staticmethod(sanitize_obs)

    def train(
        self,
        batch: tuple,
        epoch: int | None = None,
        batch_index: int | None = None,
        iters: int | None = None,
    ) -> dict:
        """Run one IQN training step on a sampled batch.

        Args:
            batch: Tuple of ``(obs, action, reward, next_obs, done, ...)``.
                   May include PER importance weights in batch[6]['is_weight'].
            epoch: Current epoch (unused, for API compat).
            batch_index: Current batch index (unused).
            iters: Total iterations (unused).

        Returns:
            Dict of scalar metrics for logging.

        Note:
            Critical implementation details:
            - Munchausen RL: Bonus is added to raw rewards BEFORE reward_normalize_scale
              is applied, so both are compressed together (Vieillard et al. 2020).
              Uses online network for current policy.
            - Value rescaling: NOT applied to quantiles during training to preserve
              distributional relationships. Quantile regression must occur in original
              value space. Rescaling would distort quantile relationships.
            - NaN/Inf detection: Validates loss before backprop to prevent gradient
              corruption. Skips update if NaN/Inf detected.
            - SimbaV2 projection: Weights are projected BEFORE target update to ensure
              target receives valid projected weights.
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

        info_dict: dict | None = (
            batch[6] if len(batch) >= 7 and isinstance(batch[6], dict) else None
        )

        if a.dim() >= 2 and a.shape[-1] == 3:
            from tmrl.custom.tm.utils.discrete_control import (
                build_brake_tap_action_table,
                continuous_control_to_discrete_indices_batch,
            )

            n_steer = int(self.iqn_n_steer_bins)
            _, table = build_brake_tap_action_table(n_steer=n_steer)
            idx = continuous_control_to_discrete_indices_batch(a.cpu().numpy(), table)
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
            gamma_n = torch.pow(
                torch.tensor(float(self.gamma), device=n_step_return.device), n_eff_t
            )
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
            current_quantiles, _, dueling_head_stats = self.model.forward_with_head_stats(
                o, tau=tau
            )
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
