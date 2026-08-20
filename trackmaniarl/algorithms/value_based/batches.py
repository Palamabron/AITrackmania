"""Canonical batch and sequence layout for discrete value learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trackmaniarl.core.data import TrainingBatch


@dataclass(frozen=True, slots=True)
class ValueBatchView:
    batch: TrainingBatch
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    masks: torch.Tensor
    sequence: bool
    batch_size: int
    time_steps: int
    n_step: int
    gamma: float

    @classmethod
    def from_batch(cls, batch: TrainingBatch) -> ValueBatchView:
        actions = _tensor(batch.actions, "actions").long()
        rewards = _tensor(batch.rewards, "rewards").float()
        discounts = _tensor(batch.bootstrap_discounts, "bootstrap_discounts").float()
        sequence = rewards.ndim == 2
        if not sequence:
            actions = actions.reshape(-1, 1)
            rewards = rewards.reshape(-1, 1)
            discounts = discounts.reshape(-1, 1)
        if actions.shape != rewards.shape or discounts.shape != rewards.shape:
            raise ValueError("actions, rewards and discounts must share (batch, time)")
        batch_size, time_steps = rewards.shape
        if sequence:
            if not isinstance(batch.masks, torch.Tensor) or batch.masks.shape != rewards.shape:
                raise ValueError("sequence batch requires boolean masks with shape (batch, time)")
            masks = batch.masks.bool()
            gamma = float(batch.metadata["gamma"])
            n_step = int(batch.metadata["n_step"])
        else:
            masks = torch.ones_like(rewards, dtype=torch.bool)
            gamma = 1.0
            n_step = 1
        if n_step < 1 or (n_step >= time_steps and sequence and time_steps > 1):
            raise ValueError("n_step must be positive and smaller than sequence length")
        return cls(
            batch,
            actions,
            rewards,
            discounts,
            masks,
            sequence,
            batch_size,
            time_steps,
            n_step,
            gamma,
        )

    def training_positions(self, burn_in: int) -> list[int]:
        if not self.sequence:
            if burn_in:
                raise ValueError("single-step batches require burn_in=0")
            return [0]
        if not 0 <= burn_in < self.time_steps:
            raise ValueError("burn_in must be in [0, sequence_length)")
        inner = list(range(burn_in, self.time_steps - self.n_step))
        return [*inner, self.time_steps - 1]

    def position_masks(self, positions: list[int]) -> torch.Tensor:
        return self.masks[:, positions]

    def position_actions(self, positions: list[int]) -> torch.Tensor:
        return self.actions[:, positions]

    def returns_and_discounts(self, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        returns: list[torch.Tensor] = []
        discounts: list[torch.Tensor] = []
        for position in positions:
            if position == self.time_steps - 1 or not self.sequence:
                returns.append(self.rewards[:, position])
                discounts.append(self.discounts[:, position])
                continue
            window = self.rewards[:, position : position + self.n_step]
            powers = self.gamma ** torch.arange(
                self.n_step, device=window.device, dtype=window.dtype
            )
            returns.append((window * powers).sum(dim=-1))
            discounts.append(torch.full_like(returns[-1], self.gamma**self.n_step))
        return torch.stack(returns, dim=1), torch.stack(discounts, dim=1)

    def priority_transition_ids(self) -> list[int]:
        configured = self.batch.metadata.get("priority_transition_ids")
        if configured is not None:
            return [int(value) for value in configured]
        return [int(value) for value in self.batch.transition_ids]


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    return value
