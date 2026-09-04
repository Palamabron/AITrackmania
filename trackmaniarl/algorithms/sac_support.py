from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.pytree import PyTree


@dataclass(frozen=True, slots=True)
class SACBatch:
    source: TrainingBatch
    observations: PyTree
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: PyTree
    discounts: torch.Tensor
    weights: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class EntropyConfig:
    initial_coefficient: float
    learning_rate: float
    mode: str


@dataclass(frozen=True, slots=True)
class EntropyRestoreTarget:
    log_alpha: torch.Tensor | None
    optimizer: torch.optim.Optimizer | None
    device: torch.device


def continuous_batch(batch: TrainingBatch) -> SACBatch:
    return SACBatch(
        batch,
        batch.observations,
        tensor(batch.actions, "actions").float(),
        tensor(batch.rewards, "rewards").float().reshape(-1),
        batch.next_observations,
        tensor(batch.bootstrap_discounts, "bootstrap_discounts").float().reshape(-1),
        batch.importance_weights if isinstance(batch.importance_weights, torch.Tensor) else None,
    )


def discrete_batch(batch: TrainingBatch) -> SACBatch:
    prepared = continuous_batch(batch)
    return SACBatch(
        prepared.source,
        prepared.observations,
        prepared.actions.long().reshape(-1),
        prepared.rewards,
        prepared.next_observations,
        prepared.discounts,
        prepared.weights,
    )


def tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor after feature collation")
    return value


def scalar_batch_output(value: Any, name: str, batch_size: int) -> torch.Tensor:
    output = tensor(value, name)
    if output.shape == (batch_size,):
        return output
    if output.shape == (batch_size, 1):
        return output.squeeze(-1)
    raise ValueError(
        f"{name} must have shape ({batch_size},) or ({batch_size}, 1), got {tuple(output.shape)}"
    )


def quantile_batch_output(value: Any, name: str, batch_size: int) -> torch.Tensor:
    output = tensor(value, name)
    if output.ndim == 2 and output.shape[0] == batch_size and output.shape[1] > 0:
        return output
    raise ValueError(f"{name} must have shape ({batch_size}, quantiles), got {tuple(output.shape)}")


def entropy_state(
    config: EntropyConfig, device: torch.device
) -> tuple[torch.Tensor | None, torch.optim.Optimizer | None]:
    if config.mode == "fixed":
        return None, None
    log_alpha = torch.tensor(config.initial_coefficient, device=device).log().requires_grad_()
    return log_alpha, torch.optim.Adam([log_alpha], lr=config.learning_rate)


def alpha_value(
    log_alpha: torch.Tensor | None, initial: float, device: torch.device
) -> torch.Tensor:
    if log_alpha is not None:
        return log_alpha.detach().exp()
    return torch.tensor(initial, device=device)


def restore_entropy_state(target: EntropyRestoreTarget, state: Mapping[str, Any]) -> None:
    saved_alpha = state["log_alpha"]
    saved_optimizer = state["alpha_optimizer"]
    if target.log_alpha is None:
        if saved_alpha is not None or saved_optimizer is not None:
            raise ValueError("checkpoint entropy mode does not match the learner")
        return
    if not isinstance(saved_alpha, torch.Tensor) or not isinstance(saved_optimizer, Mapping):
        raise ValueError("checkpoint is missing learned entropy state")
    if target.optimizer is None:
        raise RuntimeError("learned entropy requires an optimizer")
    target.log_alpha.data.copy_(saved_alpha.to(target.device))
    target.optimizer.load_state_dict(dict(saved_optimizer))


def freeze_modules(modules: Any) -> None:
    for module in modules:
        module.requires_grad_(False)


def unfreeze_modules(modules: Any) -> None:
    for module in modules:
        module.requires_grad_(True)
