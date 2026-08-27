"""Canonical batch and sequence layout for discrete value learning."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trackmaniarl.core.data import TrainingBatch


@dataclass(frozen=True, slots=True)
class _BatchValues:
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    sequence: bool


@dataclass(frozen=True, slots=True)
class _BatchMetadata:
    masks: torch.Tensor
    gamma: float
    n_step: int


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
        values = _batch_values(batch)
        metadata = _batch_metadata(batch, values)
        _validate_layout(values, metadata)
        batch_size, time_steps = values.rewards.shape
        return cls(
            batch,
            values.actions,
            values.rewards,
            values.discounts,
            metadata.masks,
            values.sequence,
            batch_size,
            time_steps,
            metadata.n_step,
            metadata.gamma,
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


def _batch_values(batch: TrainingBatch) -> _BatchValues:
    actions = _tensor(batch.actions, "actions").long()
    rewards = _tensor(batch.rewards, "rewards").float()
    discounts = _tensor(batch.bootstrap_discounts, "bootstrap_discounts").float()
    sequence = rewards.ndim == 2
    if sequence:
        return _BatchValues(actions, rewards, discounts, sequence)
    return _BatchValues(
        actions.reshape(-1, 1), rewards.reshape(-1, 1), discounts.reshape(-1, 1), sequence
    )


def _batch_metadata(batch: TrainingBatch, values: _BatchValues) -> _BatchMetadata:
    if not values.sequence:
        return _BatchMetadata(torch.ones_like(values.rewards, dtype=torch.bool), 1.0, 1)
    if not isinstance(batch.masks, torch.Tensor) or batch.masks.shape != values.rewards.shape:
        raise ValueError("sequence batch requires boolean masks with shape (batch, time)")
    return _BatchMetadata(
        batch.masks.bool(), float(batch.metadata["gamma"]), int(batch.metadata["n_step"])
    )


def _validate_layout(values: _BatchValues, metadata: _BatchMetadata) -> None:
    if (
        values.actions.shape != values.rewards.shape
        or values.discounts.shape != values.rewards.shape
    ):
        raise ValueError("actions, rewards and discounts must share (batch, time)")
    time_steps = values.rewards.shape[1]
    if metadata.n_step < 1:
        raise ValueError("n_step must be positive and smaller than sequence length")
    if values.sequence and time_steps > 1 and metadata.n_step >= time_steps:
        raise ValueError("n_step must be positive and smaller than sequence length")
