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

from tmrl.custom.models.shared.neural_network_blocks import SimbaV2Backbone

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from tmrl.custom.custom_algorithms._common import (
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    amp_setup,
    autocast_context,
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

    mode:
      - "cosine": damped sinusoid (full wave per cycle, peak->trough->peak), no spikes.
      - "ramp": original half-cosine (odwrócone ReLU): peak->trough per cycle, then floor.
    """
    import math

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
        # Half-cosine (peak -> trough); amplitude scales as
        # (epsilon_start - epsilon_end) * decay**cycle
        current_amplitude = max(0.0, epsilon_start - min_eps) * (decay**cycle_num)
        angle = math.pi * (step_in_cycle / cosine_length)
    else:
        # cosine: pełna sinusoida (peak->trough->peak), init_amplitude * decay^cycle
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
    """Linear decay from epsilon_start to epsilon_end (floor) over decay_steps."""
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
    """Cosine annealing (single period) from epsilon_start to epsilon_end over decay_steps."""
    import math

    if step <= 0.0:
        return epsilon_start
    if step >= decay_steps:
        return epsilon_end
    frac = min(1.0, step / decay_steps)
    # 1 + cos(pi * frac): 2 -> 0 as frac 0 -> 1
    return epsilon_end + 0.5 * (epsilon_start - epsilon_end) * (1.0 + math.cos(math.pi * frac))


def _quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Quantile Huber loss for IQN.

    Args:
        current_quantiles: (batch, N_tau, n_actions_selected=1) or (batch, N_tau).
        target_quantiles:  (batch, N_tau_prime).
        tau: (batch, N_tau) quantile fractions for current.
        kappa: Huber threshold.

    Returns:
        Scalar loss.
    """
    if current_quantiles.dim() == 2:
        current_quantiles = current_quantiles.unsqueeze(-1)
    if target_quantiles.dim() == 2:
        target_quantiles = target_quantiles.unsqueeze(1)

    from einops import rearrange

    delta = target_quantiles - current_quantiles
    abs_delta = delta.abs()

    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )

    tau_expanded = rearrange(tau, "b n -> b n 1")
    weight = torch.abs(tau_expanded - (delta.detach() < 0).float())
    loss = (weight * huber).sum(dim=-1).mean(dim=-1)
    return loss.mean()


def _signed_value_rescale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Signed value transform used to tame large bootstrap targets."""
    abs_x = x.abs()
    return torch.sign(x) * (torch.sqrt(abs_x + 1.0) - 1.0) + eps * x


@ALGORITHMS.register("IQN")
@dataclass(eq=False)
class IQNAgent(TrainingAgent):
    """IQN agent for discrete control with Double DQN and Dueling heads.

    Operates on a Discrete action space (single int index per step).
    The Q-network outputs quantile values for every action in one pass.

    All hyperparameters are required constructor arguments — values must be
    supplied explicitly by the config pipeline (no hidden numeric defaults).
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
    use_value_rescaling: bool
    value_rescaling_eps: float
    soft_target_tau: float
    log_target_stats: bool

    # EDER diversity filtering (0 = disabled)
    eder_oversample_ratio: int

    # Previously hidden globals — now explicit
    iqn_n_steer_bins: int
    reward_normalize_scale: float
    backup_clip_range: float

    # Mixed precision
    mixed_precision: bool
    mixed_precision_dtype: str

    # Structural defaults (None = auto-detect)
    device: str | None = None

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

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self) -> None:
        set_seed()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IQNQNetwork(
            self.observation_space,
            n_actions=self.n_actions,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            n_cos=self.n_cos,
            dueling=self.dueling,
            split_track_observation=self.split_track_observation,
            track_encoder=self.track_encoder,
            use_rnn=self.use_rnn,
            rnn_hidden_size=self.rnn_hidden_size,
            api_layernorm=self.api_layernorm,
            use_simbav2=self.use_simbav2,
            r2d2_sequence_length=self.r2d2_sequence_length,
            r2d2_burn_in=self.r2d2_burn_in,
            gnn_hidden=self.gnn_hidden,
            gnn_layers=self.gnn_layers,
        ).to(device)

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
        self._epsilon = self.epsilon_start
        self._grad_stabilizer = GradientStabilizer(ema_decay=0.995)
        logger.info(
            "IQNAgent: n_actions={}, dueling={}, double={}, n_steps={}, gamma={:.3f}, "
            "eps mode={}, decay_steps={}, t0={}, tmult={}, decay={}, init_amp={}, "
            "floor_frac={}, floor_steps={}, huber_kappa={}, value_rescaling={}, "
            "rescale_eps={}, soft_target_tau={}",
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
            self.use_value_rescaling,
            self.value_rescaling_eps,
            self.soft_target_tau,
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

    def _update_epsilon(self) -> float:
        """Epsilon: cosine | ramp | cosine_anneal | linear.

        cosine_anneal: single period with decay to floor. linear uses epsilon_end as floor.
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
        """Return actor module with current Q-network weights + epsilon."""
        actor = self.model_nograd
        # Build a DQNActor wrapper for the worker
        wrapper = DQNActor(
            self.observation_space,
            self.action_space,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            n_cos=self.n_cos,
            dueling=self.dueling,
            n_actions=self.n_actions,
            epsilon=self._epsilon,
            n_quantiles_eval=self.n_quantiles_eval,
            explore_repeat_steps=self.explore_repeat_steps,
            split_track_observation=self.split_track_observation,
            track_encoder=self.track_encoder,
            use_rnn=self.use_rnn,
            rnn_hidden_size=self.rnn_hidden_size,
            api_layernorm=self.api_layernorm,
            use_simbav2=self.use_simbav2,
            r2d2_sequence_length=self.r2d2_sequence_length,
            r2d2_burn_in=self.r2d2_burn_in,
            gnn_hidden=self.gnn_hidden,
            gnn_layers=self.gnn_layers,
        )
        # Share weights: copy state dict from the no-grad model
        wrapper.q_net.load_state_dict(actor.state_dict())
        wrapper.set_epsilon(self._epsilon)
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
            epoch: Current epoch (unused, for API compat).
            batch_index: Current batch index (unused).
            iters: Total iterations (unused).

        Returns:
            Dict of scalar metrics for logging.
        """
        from einops import rearrange, repeat

        self._training_step += 1
        eps = self._update_epsilon()

        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        o = self._sanitize_obs(o)
        o2 = self._sanitize_obs(o2)
        a = self._sanitize_tensor(a)
        r = self._sanitize_tensor(r)
        d = self._sanitize_tensor(d)

        device = self.device or "cpu"
        batch_size = r.shape[0]
        # Replay may hold continuous [gas, brake, steer] (e.g. player runs); map to discrete.
        if a.dim() >= 2 and a.shape[-1] == 3:
            from tmrl.custom.tm.utils.discrete_control import (
                build_yosh_action_table,
                continuous_control_to_discrete_indices_batch,
            )

            n_steer = int(self.iqn_n_steer_bins)
            _, table = build_yosh_action_table(n_steer=n_steer)
            idx = continuous_control_to_discrete_indices_batch(a.cpu().numpy(), table)
            a = torch.from_numpy(idx).to(device=a.device, dtype=torch.long)
        actions = a.long().squeeze(-1)

        if self.eder_oversample_ratio >= 2:
            from tmrl.custom.utils.eder import greedy_kdpp_filter

            with torch.no_grad():
                tau_dummy = torch.full((batch_size, 1), 0.5, device=o[0].device)
                backbone = getattr(self.model, "backbone", self.model)
                feat = backbone(o, tau_dummy).squeeze(1)
            target_k = batch_size // self.eder_oversample_ratio
            keep = greedy_kdpp_filter(feat, target_k)
            o = tuple(t[keep] for t in o)
            o2 = tuple(t[keep] for t in o2)
            a = a[keep]
            r = r[keep]
            d = d[keep]
            actions = actions[keep]
            batch_size = target_k

        reward_scale = float(self.reward_normalize_scale)
        if reward_scale != 1.0 and reward_scale > 0:
            r = r / reward_scale

        if self.n_steps > 1:
            n_step_return, bootstrap_mask = _compute_n_step_return_and_bootstrap_mask(
                r, d, self.gamma, self.n_steps
            )
            n_step_return = n_step_return.squeeze(-1)
            bootstrap_mask = bootstrap_mask.squeeze(-1)
            gamma_n = self.gamma**self.n_steps
        else:
            n_step_return = r.squeeze(-1)
            bootstrap_mask = 1.0 - d.squeeze(-1)
            gamma_n = self.gamma

        def autocast_ctx():
            return autocast_context(self.use_mixed_precision, self.amp_dtype)

        with autocast_ctx():
            tau = torch.rand(batch_size, self.n_quantiles_train, device=device)
            current_quantiles, _ = self.model(o, tau=tau)
            action_idx = repeat(actions, "b -> b n 1", n=self.n_quantiles_train)
            current_q = current_quantiles.gather(2, action_idx).squeeze(2)

        with torch.no_grad():
            tau_prime = torch.rand(batch_size, self.n_quantiles_target, device=device)

            with autocast_ctx():
                if self.double_dqn:
                    online_q_next = self.model.q_values(o2, n_quantiles=self.n_quantiles_eval)
                    next_actions = online_q_next.argmax(dim=-1)
                else:
                    target_q_next = self.model_target.q_values(
                        o2, n_quantiles=self.n_quantiles_eval
                    )
                    next_actions = target_q_next.argmax(dim=-1)

                target_quantiles, _ = self.model_target(o2, tau=tau_prime)
            next_action_idx = repeat(next_actions, "b -> b n 1", n=self.n_quantiles_target)
            next_q = target_quantiles.gather(2, next_action_idx).squeeze(2)

            target = (
                rearrange(n_step_return, "b -> b 1")
                + gamma_n * rearrange(bootstrap_mask, "b -> b 1") * next_q
            )
            backup_clip = float(self.backup_clip_range)
            if backup_clip > 0.0:
                target = target.clamp(min=-backup_clip, max=backup_clip)

        current_for_loss = current_q
        target_for_loss = target
        if self.use_value_rescaling:
            current_for_loss = _signed_value_rescale(current_for_loss, self.value_rescaling_eps)
            target_for_loss = _signed_value_rescale(target_for_loss, self.value_rescaling_eps)

        loss = _quantile_huber_loss(
            current_for_loss, target_for_loss, tau, kappa=self.huber_kappa
        )

        self.optimizer.zero_grad()
        if self.use_mixed_precision:
            self.grad_scaler.scale(loss).backward()
            if self.grad_clip > 0.0:
                self.grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
            grad_norm = self._grad_stabilizer.step(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            if self.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))
            grad_norm = self._grad_stabilizer.step(self.model.parameters())
            self.optimizer.step()

        backbone = getattr(self.model, "backbone", None)
        if backbone is not None:
            inner = getattr(backbone, "backbone", backbone)
            if isinstance(inner, SimbaV2Backbone):
                inner.project_weights()

        if self.soft_target_tau > 0.0:
            tau_polyak = float(self.soft_target_tau)
            with torch.no_grad():
                for p_tgt, p_src in zip(
                    self.model_target.parameters(), self.model.parameters(), strict=False
                ):
                    p_tgt.data.lerp_(p_src.data, tau_polyak)
        elif self._training_step % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            for p in self.model_target.parameters():
                p.requires_grad = False

        iqn_loss_scalar = _tensor_to_scalar(loss)
        ret = {
            "loss/iqn_loss": iqn_loss_scalar,
            "exploration/epsilon": eps,
            "q/mean_q": _tensor_to_scalar(current_q.mean()),
            "q/max_q": _tensor_to_scalar(current_q.max()),
            "debug/grad_norm": grad_norm,
            "debug/grad_ema_norm": self._grad_stabilizer.ema_norm,
            "train/step": self._training_step,
        }
        if self.log_target_stats:
            with torch.no_grad():
                td_abs = (target.mean(dim=1) - current_q.mean(dim=1)).abs()
                td_p95 = (
                    torch.quantile(td_abs, 0.95) if td_abs.numel() > 1 else td_abs.max()
                )
                ret["q/target_mean"] = _tensor_to_scalar(target.mean())
                ret["q/target_max"] = _tensor_to_scalar(target.max())
                ret["debug/td_abs_mean"] = _tensor_to_scalar(td_abs.mean())
                ret["debug/td_abs_p95"] = _tensor_to_scalar(td_p95)

        if wandb is not None and wandb.run is not None:
            wandb.log(ret, step=wandb_monotonic_step(self._training_step, wandb.run))

        return ret
