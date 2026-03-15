"""IQN (Implicit Quantile Network) agent with Double DQN.

Implements DQN + Double DQN + IQN distributional RL + Dueling architecture
+ epsilon-greedy exploration + n-step returns.

References:
  - DQN: Mnih et al. 2015
  - Double DQN: van Hasselt et al. 2016
  - Dueling: Wang et al. 2016
  - IQN: Dabney et al. 2018
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger
from torch.optim import Adam

from tmrl.custom.models.model_blocks import SimbaV2Backbone

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

import tmrl.config.constants as cfg
from tmrl.custom.custom_algorithms._common import (
    _compute_n_step_return_and_bootstrap_mask,
    _tensor_to_scalar,
    set_seed,
)
from tmrl.custom.models.DQNNet import DQNActor, IQNQNetwork
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.custom.utils.optim import GradientStabilizer
from tmrl.training import TrainingAgent
from tmrl.util import cached_property


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


@dataclass(eq=False)
class IQNAgent(TrainingAgent):
    """IQN agent for discrete control with Double DQN and Dueling heads.

    Operates on a Discrete action space (single int index per step).
    The Q-network outputs quantile values for every action in one pass.
    """

    observation_space: Any = None
    action_space: Any = None
    device: str | None = None

    # Architecture
    hidden_dim: int = 256
    num_blocks: int = 3
    n_cos: int = 64
    dueling: bool = True
    n_actions: int = 78

    # IQN
    n_quantiles_train: int = 64
    n_quantiles_target: int = 64
    n_quantiles_eval: int = 32

    # DQN
    gamma: float = 0.99
    lr: float = 1e-4
    n_steps: int = 3
    double_dqn: bool = True
    target_update_freq: int = 1000

    # Epsilon-greedy (cosine annealing with amplitude decay)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.00005
    epsilon_decay_steps: int = 500000
    epsilon_cosine_cycle_runs: float = 200.0
    epsilon_cosine_decay: float = 0.9

    # Misc
    grad_clip: float = 10.0  # legacy; unused after GradientStabilizer integration
    weight_decay: float = 0.0

    # EDER diversity filtering (0 = disabled)
    eder_oversample_ratio: int = 0

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
        ).to(device)

        self.model_target = no_grad(deepcopy(self.model))

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        self._training_step = 0
        self._epsilon = self.epsilon_start
        self._grad_stabilizer = GradientStabilizer(ema_decay=0.995)
        logger.info(
            "IQNAgent: n_actions={}, dueling={}, double={}, n_steps={}, "
            "gamma={:.3f}, eps_start={:.2f}->{:.2f} over {} steps",
            self.n_actions,
            self.dueling,
            self.double_dqn,
            self.n_steps,
            self.gamma,
            self.epsilon_start,
            self.epsilon_end,
            self.epsilon_decay_steps,
        )

    def _update_epsilon(self, epoch_t: float | None = None) -> float:
        """Cosine annealing epsilon decay with amplitude decay.

        Lowest point: self.epsilon_end (IQN_EPSILON_END in config).
        Cycle length and per-cycle decay are set by epsilon_cosine_cycle_runs
        and epsilon_cosine_decay (IQN_EPSILON_COSINE_* in config).
        """
        import math
        min_eps = self.epsilon_end
        cycle_length = self.epsilon_cosine_cycle_runs
        decay = self.epsilon_cosine_decay
        
        if epoch_t is not None:
            t = epoch_t
        else:
            # Fallback assuming default 3000 steps per epoch
            t = self._training_step / 3000.0

        cycle_num = math.floor(t / cycle_length)
        step_in_cycle = t - (cycle_num * cycle_length)
        
        initial_amplitude = max(0.0, self.epsilon_start - min_eps)
        current_amplitude = initial_amplitude * (decay ** cycle_num)
        
        # Cosine annealing from max to min over the cycle
        angle = math.pi * (step_in_cycle / cycle_length)
        self._epsilon = min_eps + 0.5 * current_amplitude * (1.0 + math.cos(angle))
        
        return self._epsilon

    def get_actor(self):
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
        )
        # Share weights: copy state dict from the no-grad model
        wrapper.q_net.load_state_dict(actor.state_dict())
        wrapper.epsilon = self._epsilon
        return wrapper

    @staticmethod
    def _sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        return t

    @staticmethod
    def _sanitize_obs(obs):
        if isinstance(obs, torch.Tensor):
            return (
                torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                if obs.is_floating_point()
                else obs
            )
        return tuple(
            torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if isinstance(t, torch.Tensor) and t.is_floating_point()
            else t
            for t in obs
        )

    def train(self, batch, epoch=None, batch_index=None, iters=None):
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
        if epoch is not None and batch_index is not None and iters is not None:
            t = epoch + batch_index / iters
        elif epoch is not None:
            t = float(epoch)
        else:
            t = None
            
        eps = self._update_epsilon(t)

        o, a, r, o2, d = batch[0], batch[1], batch[2], batch[3], batch[4]
        o = self._sanitize_obs(o)
        o2 = self._sanitize_obs(o2)
        a = self._sanitize_tensor(a)
        r = self._sanitize_tensor(r)
        d = self._sanitize_tensor(d)

        device = self.device or "cpu"
        batch_size = r.shape[0]
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

        reward_scale = float(cfg.ALG_CONFIG.get("REWARD_NORMALIZE_SCALE", 1.0))
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

        tau = torch.rand(batch_size, self.n_quantiles_train, device=device)
        current_quantiles, _ = self.model(o, tau=tau)
        action_idx = repeat(actions, "b -> b n 1", n=self.n_quantiles_train)
        current_q = current_quantiles.gather(2, action_idx).squeeze(2)

        with torch.no_grad():
            tau_prime = torch.rand(batch_size, self.n_quantiles_target, device=device)

            if self.double_dqn:
                online_q_next = self.model.q_values(o2, n_quantiles=self.n_quantiles_eval)
                next_actions = online_q_next.argmax(dim=-1)
            else:
                target_q_next = self.model_target.q_values(o2, n_quantiles=self.n_quantiles_eval)
                next_actions = target_q_next.argmax(dim=-1)

            target_quantiles, _ = self.model_target(o2, tau=tau_prime)
            next_action_idx = repeat(next_actions, "b -> b n 1", n=self.n_quantiles_target)
            next_q = target_quantiles.gather(2, next_action_idx).squeeze(2)

            target = (
                rearrange(n_step_return, "b -> b 1")
                + gamma_n * rearrange(bootstrap_mask, "b -> b 1") * next_q
            )

        loss = _quantile_huber_loss(current_q, target, tau)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = self._grad_stabilizer.step(self.model.parameters())
        self.optimizer.step()

        backbone = getattr(self.model, "backbone", None)
        if backbone is not None:
            inner = getattr(backbone, "backbone", backbone)
            if isinstance(inner, SimbaV2Backbone):
                inner.project_weights()

        if self._training_step % self.target_update_freq == 0:
            self.model_target.load_state_dict(self.model.state_dict())
            for p in self.model_target.parameters():
                p.requires_grad = False

        ret = {
            "loss/iqn_loss": _tensor_to_scalar(loss),
            "exploration/epsilon": eps,
            "q/mean_q": _tensor_to_scalar(current_q.mean()),
            "q/max_q": _tensor_to_scalar(current_q.max()),
            "debug/grad_norm": grad_norm,
            "debug/grad_ema_norm": self._grad_stabilizer.ema_norm,
            "train/step": self._training_step,
        }

        if wandb is not None and wandb.run is not None:
            wandb.log(ret, step=self._training_step)

        return ret
