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
from loguru import logger
from torch.optim import Adam

from tmrl.custom.algorithms._common import (
    amp_setup,
    sanitize_obs,
    sanitize_tensor,
    set_seed,
)
from tmrl.custom.algorithms._internal._iqn_schedules import (
    epsilon_cosine_anneal_schedule,
    epsilon_cosine_schedule,
    epsilon_linear_schedule,
)
from tmrl.custom.algorithms._internal._iqn_train_step import _iqn_train_step
from tmrl.custom.models.discrete_actions.iqn_discrete_q_network import DQNActor, IQNQNetwork
from tmrl.custom.utils.nn_utils import copy_shared, no_grad
from tmrl.custom.utils.optim import GradientStabilizer
from tmrl.registry import ALGORITHMS
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


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
        self._legacy_action_table: list | None = None
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
        if self.bc_lambda > 0.0 and int(self.bc_anneal_steps_end) > 0:
            logger.warning(
                "IQNAgent: bc_anneal_steps_end={} is set — during the anneal window bc_lambda "
                "acts only as an on/off gate (nonzero = enabled); the effective weight ranges "
                "from bc_lambda_start={:.4g} to bc_lambda_end={:.4g}. "
                "Set bc_anneal_steps_end=0 to use bc_lambda={:.4g} as a static weight.",
                self.bc_anneal_steps_end,
                self.bc_lambda_start,
                self.bc_lambda_end,
                self.bc_lambda,
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
        return _iqn_train_step(self, batch)
