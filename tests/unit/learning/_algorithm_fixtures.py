"""Shared models and batches for learner contract tests."""

from __future__ import annotations

from dataclasses import replace
from enum import StrEnum
from typing import Any

import torch
from torch import nn

from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models.actors import CategoricalActor, GaussianActor, GaussianActorConfig
from trackmaniarl.models.critics import (
    ContinuousQCritic,
    ContinuousValueCritic,
    QuantileCritic,
    QuantileCriticConfig,
)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 16), nn.SiLU())

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.layers(observation.float())


class StructuredEncoder(nn.Module):
    def forward(self, track: torch.Tensor, telemetry: torch.Tensor) -> torch.Tensor:
        return torch.cat((track, telemetry), dim=-1)


class ContinuousModel(nn.Module):
    def __init__(self, *, quantiles: int | None = None, critic_count: int = 2) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), GaussianActorConfig(16, 2))
        if quantiles is None:
            self.q1 = ContinuousQCritic(Encoder(), 16, 2)
            self.q2 = ContinuousQCritic(Encoder(), 16, 2)
        elif critic_count == 2:
            config = QuantileCriticConfig(16, 2, quantiles)
            self.q1 = QuantileCritic(Encoder(), config)
            self.q2 = QuantileCritic(Encoder(), config)
        else:
            self.critics = nn.ModuleList(
                [
                    QuantileCritic(Encoder(), QuantileCriticConfig(16, 2, quantiles))
                    for _ in range(critic_count)
                ]
            )


class ContinuousPpoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), GaussianActorConfig(16, 2))
        self.value = ContinuousValueCritic(Encoder(), 16)


class StructuredValue(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = StructuredEncoder()
        self.value = nn.Linear(4, 1)

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.value(self.encoder(**observation)).squeeze(-1)


class StructuredPpoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = GaussianActor(StructuredEncoder(), GaussianActorConfig(4, 2))
        self.value = StructuredValue()


class RedqModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), GaussianActorConfig(16, 2))
        self.critics = nn.ModuleList([ContinuousQCritic(Encoder(), 16, 2) for _ in range(3)])


class DiscreteValue(nn.Module):
    def __init__(self, action_count: int) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(16, action_count)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observation))


class DiscreteSacModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = CategoricalActor(Encoder(), 16, 3)
        self.q1 = DiscreteValue(3)
        self.q2 = DiscreteValue(3)


class ConstantActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action = nn.Parameter(torch.zeros(2))

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.action.expand(*observation.shape[:-1], 2)
        return actions, actions.sum(dim=-1) * 0.0


class ConstantQuantileCritic(nn.Module):
    def __init__(self, value: float, quantile_count: int = 5) -> None:
        super().__init__()
        self.quantiles = nn.Parameter(torch.full((quantile_count,), value))

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = self.quantiles.expand(*observation.shape[:-1], self.quantiles.shape[0])
        return values + action.sum(dim=-1, keepdim=True) * 0.0


class ConstantTqcModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = ConstantActor()
        self.critics = nn.ModuleList([ConstantQuantileCritic(1.0), ConstantQuantileCritic(3.0)])


class CheckpointScaler:
    def __init__(self, scale: float) -> None:
        self.current_scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.current_scale = float(state["current_scale"])


class BatchKind(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


def _batch(kind: BatchKind, batch_size: int = 8) -> TrainingBatch:
    observations = torch.randn(batch_size, 4)
    actions = (
        torch.randint(0, 3, (batch_size,))
        if kind is BatchKind.DISCRETE
        else torch.tanh(torch.randn(batch_size, 2))
    )
    return TrainingBatch(
        data=observations,
        observations=observations,
        actions=actions,
        rewards=torch.randn(batch_size),
        next_observations=torch.randn(batch_size, 4),
        terminated=torch.zeros(batch_size, dtype=torch.bool),
        truncated=torch.zeros(batch_size, dtype=torch.bool),
        bootstrap_discounts=torch.full((batch_size,), 0.99),
        transition_ids=list(range(batch_size)),
    )


def _sequence_batch(kind: BatchKind) -> TrainingBatch:
    batch = _batch(kind)
    observations = batch.observations.reshape(2, 4, 4)
    actions = batch.actions.reshape(2, 4, *batch.actions.shape[1:])
    return replace(
        batch,
        data=observations,
        observations=observations,
        actions=actions,
        rewards=batch.rewards.reshape(2, 4),
        next_observations=batch.next_observations.reshape(2, 4, 4),
        terminated=batch.terminated.reshape(2, 4),
        truncated=batch.truncated.reshape(2, 4),
        bootstrap_discounts=batch.bootstrap_discounts.reshape(2, 4),
        masks=torch.ones(2, 4, dtype=torch.bool),
        metadata={"sequence_length": 4},
    )


def _assert_update(learner: Any, batch: TrainingBatch) -> None:
    learner.execution = TorchExecutionConfig(device="cpu", precision="float32")
    learner.setup({"seed": 0})
    metrics, priorities = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert priorities.transition_ids == batch.transition_ids
    assert len(priorities.priorities) == len(batch.transition_ids)
    assert "scaler" in learner.state_dict()
