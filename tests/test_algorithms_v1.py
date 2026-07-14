"""CPU contract checks for the five first-class TMRL 1.0 learners."""

from __future__ import annotations

import torch
from tmrl.algorithms import (
    ImplicitQuantileQLearning,
    RandomizedEnsembleSAC,
    SoftActorCritic,
    StableDiscreteSoftActorCritic,
    TruncatedQuantileCritic,
)
from tmrl.algorithms._torch import polyak_update
from tmrl.core.data import TrainingBatch
from tmrl.models.actors import CategoricalActor, GaussianActor
from tmrl.models.critics import ContinuousQCritic, DiscreteQuantileNetwork, QuantileCritic
from torch import nn


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
    def __init__(self, *, quantiles: int | None = None) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), 16, 2)
        if quantiles is None:
            self.q1 = ContinuousQCritic(Encoder(), 16, 2)
            self.q2 = ContinuousQCritic(Encoder(), 16, 2)
        else:
            self.q1 = QuantileCritic(Encoder(), 16, 2, quantiles)
            self.q2 = QuantileCritic(Encoder(), 16, 2, quantiles)


class RedqModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), 16, 2)
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


def _batch(*, discrete: bool) -> TrainingBatch:
    observations = torch.randn(8, 4)
    actions = torch.randint(0, 3, (8,)) if discrete else torch.tanh(torch.randn(8, 2))
    return TrainingBatch(
        data=observations,
        observations=observations,
        actions=actions,
        rewards=torch.randn(8),
        next_observations=torch.randn(8, 4),
        terminated=torch.zeros(8, dtype=torch.bool),
        truncated=torch.zeros(8, dtype=torch.bool),
        bootstrap_discounts=torch.full((8,), 0.99),
        transition_ids=list(range(8)),
    )


def _assert_update(learner, batch: TrainingBatch) -> None:
    learner.setup({"seed": 0})
    metrics, priorities = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert priorities.transition_ids == batch.transition_ids
    assert len(priorities.priorities) == len(batch.transition_ids)


def test_continuous_learners_update_without_shape_or_target_errors() -> None:
    _assert_update(SoftActorCritic(ContinuousModel()), _batch(discrete=False))
    _assert_update(
        RandomizedEnsembleSAC(RedqModel(), policy_update_interval=1), _batch(discrete=False)
    )
    _assert_update(TruncatedQuantileCritic(ContinuousModel(quantiles=5)), _batch(discrete=False))


def test_discrete_learners_update_without_shape_or_target_errors() -> None:
    _assert_update(
        ImplicitQuantileQLearning(DiscreteQuantileNetwork(Encoder(), 16, 3, cosine_count=8)),
        _batch(discrete=True),
    )
    _assert_update(StableDiscreteSoftActorCritic(DiscreteSacModel()), _batch(discrete=True))


def test_iqn_network_is_batch_size_invariant() -> None:
    model = DiscreteQuantileNetwork(Encoder(), 16, 3, cosine_count=8)
    for batch_size in (1, 2, 256):
        output = model(torch.randn(batch_size, 4), torch.rand(batch_size, 7))
        assert output.shape == (batch_size, 7, 3)


def test_bundled_actors_support_tuple_and_mapping_encoder_inputs() -> None:
    track = torch.randn(3, 2)
    telemetry = torch.randn(3, 2)
    continuous = GaussianActor(StructuredEncoder(), 4, 2)
    discrete = CategoricalActor(StructuredEncoder(), 4, 3)
    assert continuous((track, telemetry))[0].shape == (3, 2)
    assert discrete({"track": track, "telemetry": telemetry})[0].shape == (3,)


def test_iqn_policy_accepts_an_unbatched_observation() -> None:
    learner = ImplicitQuantileQLearning(
        DiscreteQuantileNetwork(Encoder(), 16, 3, cosine_count=8),
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
    )
    learner.setup({"seed": 0})
    assert isinstance(learner.policy().act(torch.randn(4)), int)


class GreedyIQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros(()))

    def q_values(self, observation: torch.Tensor, quantile_count: int) -> torch.Tensor:
        del quantile_count
        return self.parameter.expand(observation.shape[0], 3)


def test_iqn_rollout_uses_epsilon_greedy_exploration_but_evaluation_is_greedy() -> None:
    learner = ImplicitQuantileQLearning(GreedyIQN(), exploration_epsilon=1.0)
    learner.setup({"seed": 0})
    policy = learner.policy()
    torch.manual_seed(0)
    explored_actions = {policy.act(torch.zeros(4)) for _ in range(32)}
    assert len(explored_actions) > 1
    assert policy.act(torch.zeros(4), deterministic=True) == 0


def test_polyak_update_copies_batch_norm_buffers() -> None:
    source = nn.BatchNorm1d(2)
    target = nn.BatchNorm1d(2)
    source.train()
    source(torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
    polyak_update(source, target, 0.5)
    assert torch.equal(target.running_mean, source.running_mean)
    assert torch.equal(target.running_var, source.running_var)
