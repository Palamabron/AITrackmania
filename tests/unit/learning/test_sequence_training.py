"""Contract tests for all-step recurrent sequence training and demo protection."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trackmaniarl.algorithms.implicit_quantile_q_learning import (
    ImplicitQuantileQLearning,
)
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler
from trackmaniarl.models.critics import DiscreteQuantileNetwork


def _transition(step: int, *, episode: str, terminal: bool, demo: bool = False) -> Transition:
    return Transition(
        observation=float(step),
        action=0.0,
        reward=1.0,
        next_observation=float(step + 1),
        terminated=terminal,
        truncated=False,
        episode_id=episode,
        step=step,
        info={"is_demo": demo},
    )


def _fill_episode(
    store: InMemoryReplayStore, episode: str, steps: int, *, demo: bool = False
) -> None:
    for step in range(steps):
        store.append(_transition(step, episode=episode, terminal=step == steps - 1, demo=demo))


def test_demo_transitions_survive_fifo_eviction() -> None:
    store = InMemoryReplayStore(capacity=16)
    _fill_episode(store, "demo-lap", 4, demo=True)
    for episode in range(10):
        _fill_episode(store, f"online-{episode}", 4)

    flags = store.demo_flags(store.available_ids())

    assert sum(flags) == 4
    demo_ids = [
        transition_id
        for transition_id, flag in zip(store.available_ids(), flags, strict=True)
        if flag
    ]
    resurrected = store.get(demo_ids)
    assert sorted(item.step for item in resurrected) == [0, 1, 2, 3]
    assert all(item.episode_id == "demo-lap" for item in resurrected)


def test_prioritized_sequence_masks_mark_left_padding() -> None:
    store = InMemoryReplayStore()
    _fill_episode(store, "episode-0", 3)
    sampler = PrioritizedSampler(IdentityFeaturePipeline(), seed=3)

    batch = sampler.sample(store, BatchRequest(batch_size=3, sequence_length=4, n_step=1))

    assert isinstance(batch.masks, torch.Tensor)
    assert batch.masks.shape == (3, 4)
    assert batch.metadata["gamma"] == pytest.approx(0.99)
    assert batch.metadata["n_step"] == 1
    assert len(batch.metadata["demo_flags"]) == 3
    for row in range(3):
        row_mask = batch.masks[row]
        assert bool(row_mask[-1])
        padding = int((~row_mask).sum())
        assert torch.equal(row_mask, torch.tensor([False] * padding + [True] * (4 - padding)))


class _SequenceEncoder(nn.Module):
    output_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, self.output_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encode_steps(observation)[:, -1]

    def encode_steps(self, observation: torch.Tensor) -> torch.Tensor:
        return self.projection(observation)


def _constant_model(constant: float) -> DiscreteQuantileNetwork:
    model = DiscreteQuantileNetwork(_SequenceEncoder(), 8, action_count=3, cosine_count=4)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(constant)
    return model


def _sequence_batch(
    *,
    batch_size: int = 2,
    sequence_length: int = 4,
    n_step: int = 1,
    gamma: float = 0.5,
    demo: bool = False,
    expert: bool | None = None,
) -> TrainingBatch:
    observations = torch.randn(batch_size, sequence_length, 2)
    next_observations = torch.randn(batch_size, sequence_length, 2)
    rewards = torch.arange(batch_size * sequence_length, dtype=torch.float32).reshape(
        batch_size, sequence_length
    )
    discounts = torch.full((batch_size, sequence_length), gamma)
    discounts[:, -1] = gamma**n_step
    return TrainingBatch(
        data={},
        observations=observations,
        actions=torch.zeros(batch_size, sequence_length, dtype=torch.int64),
        rewards=rewards,
        next_observations=next_observations,
        terminated=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        truncated=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        bootstrap_discounts=discounts,
        transition_ids=list(range(batch_size * sequence_length)),
        masks=torch.ones(batch_size, sequence_length, dtype=torch.bool),
        metadata={
            "gamma": gamma,
            "n_step": n_step,
            "priority_transition_ids": tuple(range(batch_size)),
            "demo_flags": tuple([demo] * batch_size),
            **({} if expert is None else {"expert_demo_flags": tuple([expert] * batch_size)}),
        },
    )


def _learner(model: DiscreteQuantileNetwork, **kwargs: object) -> ImplicitQuantileQLearning:
    learner = ImplicitQuantileQLearning(
        model,
        train_quantile_count=4,
        target_quantile_count=4,
        evaluation_quantile_count=4,
        execution={"device": "cpu"},
        **kwargs,  # type: ignore[arg-type]
    )
    learner.setup({})
    return learner


class _TailPreferenceModel(nn.Module):
    action_count = 2

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.quantiles: torch.Tensor | None = None

    def forward(self, observation: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
        del observation
        self.quantiles = quantiles.detach().cpu()
        values = torch.stack([1.0 - quantiles, quantiles], dim=-1)
        return values + self.bias

    def q_values(self, observation: torch.Tensor, quantile_count: int) -> torch.Tensor:
        del quantile_count
        return torch.tensor([[0.7, 0.6]], device=observation.device) + self.bias


def test_upper_cvar_changes_actor_and_evaluation_action_quantiles() -> None:
    learner = ImplicitQuantileQLearning(
        _TailPreferenceModel(),
        exploration_epsilon=0.0,
        online_quantile_distortion="upper_cvar",
        evaluation_quantile_distortion="upper_cvar",
        upper_cvar_alpha=0.25,
        evaluation_quantile_count=4,
        execution={"device": "cpu"},
    )
    learner.setup({})

    policy = learner.policy()

    assert policy.act(torch.zeros(1), deterministic=False) == 1
    assert policy.act(torch.zeros(1), deterministic=True) == 1
    assert isinstance(policy.model.quantiles, torch.Tensor)
    assert torch.allclose(
        policy.model.quantiles[0], torch.tensor([0.78125, 0.84375, 0.90625, 0.96875])
    )
    assert learner._masked_argmax(torch.tensor([[0.7, 0.6]])).item() == 0


def test_hard_target_syncs_only_at_the_configured_interval() -> None:
    learner = _learner(_constant_model(0.0), target_tau=0.0, target_update_interval=2)
    batch = _sequence_batch()

    first_metrics, _ = learner.update(batch)
    first_target = {
        name: value.clone() for name, value in learner.target_model.state_dict().items()
    }
    second_metrics, _ = learner.update(batch)

    assert first_metrics["debug/target_synced_fraction"] == 0.0
    assert first_metrics["debug/target_update_hard"] == 1.0
    assert first_metrics["debug/target_update_interval"] == 2.0
    assert any(
        not torch.equal(learner.model.state_dict()[name], value)
        for name, value in first_target.items()
    )
    assert second_metrics["debug/target_synced_fraction"] == 1.0
    for name, value in learner.model.state_dict().items():
        assert torch.equal(learner.target_model.state_dict()[name], value)


def test_sequence_update_trains_every_bootstrappable_position() -> None:
    learner = _learner(_constant_model(0.25))
    batch = _sequence_batch(gamma=0.5, n_step=1)

    metrics, priorities = learner.update(batch)

    assert metrics["debug/trained_positions"] == 4.0
    assert len(priorities.priorities) == 2
    assert list(priorities.transition_ids) == [0, 1]


def test_sequence_update_ignores_padded_positions() -> None:
    learner = _learner(_constant_model(0.25))
    batch = _sequence_batch(gamma=0.5, n_step=1)
    masked = TrainingBatch(
        data=batch.data,
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        next_observations=batch.next_observations,
        terminated=batch.terminated,
        truncated=batch.truncated,
        bootstrap_discounts=batch.bootstrap_discounts,
        transition_ids=batch.transition_ids,
        masks=torch.tensor([[False, False, True, True], [True, True, True, True]]),
        metadata=batch.metadata,
    )

    _, priorities = learner.update(masked)

    rewards = torch.as_tensor(batch.rewards)
    gamma, constant = 0.5, 0.25
    valid = [abs(float(rewards[0, 2]) + gamma * constant - constant)]
    final = abs(float(rewards[0, -1]) + gamma * constant - constant)
    td = [*valid, final]
    expected = 0.9 * max(td) + 0.1 * (sum(td) / len(td))
    assert priorities.priorities[0] == pytest.approx(expected, rel=1e-4)


def test_demonstration_margin_loss_penalizes_flat_q_values() -> None:
    margin = 0.8
    learner = _learner(
        _constant_model(0.25),
        demonstration_margin=margin,
        demonstration_margin_weight=1.0,
    )
    batch = _sequence_batch(demo=True)

    metrics, _ = learner.update(batch)

    assert metrics["loss/demonstration_margin"] == pytest.approx(margin, rel=1e-4)


def test_policy_action_mask_controls_greedy_exploration_and_bootstrap() -> None:
    model = _constant_model(0.0)
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([0.0, 1.0, 10.0]))
    learner = _learner(
        model,
        policy_action_ids=(0, 1),
        exploration_epsilon=1.0,
    )
    observation = torch.zeros(1, 2)
    deterministic = learner.policy().act(observation, deterministic=True)
    exploratory = {learner.policy().act(observation, deterministic=False) for _ in range(20)}

    assert deterministic == 1
    assert exploratory <= {0, 1}
    assert learner._masked_argmax(torch.tensor([[0.0, 1.0, 10.0]])).item() == 1
