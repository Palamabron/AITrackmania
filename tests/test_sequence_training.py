"""Contract tests for all-step recurrent sequence training and demo protection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from tmrl.algorithms.implicit_quantile_q_learning import (
    ImplicitQuantileQLearning,
    inverse_rescale_value,
    rescale_value,
)
from tmrl.core.builtins import IdentityFeaturePipeline
from tmrl.core.data import BatchRequest, TrainingBatch, Transition
from tmrl.core.replay import InMemoryReplayStore, PrioritizedSampler
from tmrl.models.critics import DiscreteQuantileNetwork
from tmrl.trackmania.reward import TrajectoryReward


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


def test_demo_protection_rejects_undersized_capacity() -> None:
    store = InMemoryReplayStore(capacity=4)
    _fill_episode(store, "demo-lap", 3, demo=True)

    def overflow() -> None:
        for episode in range(4):
            _fill_episode(store, f"online-{episode}", 4)

    with pytest.raises(RuntimeError, match="capacity is too small"):
        overflow()


def test_episode_index_is_pruned_after_full_eviction() -> None:
    store = InMemoryReplayStore(capacity=8)
    for episode in range(20):
        _fill_episode(store, f"episode-{episode}", 4)

    assert len(store._episode_names) <= 2
    assert len(store._episode_refcounts) <= 2


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


def test_sequence_update_trains_every_bootstrappable_position() -> None:
    learner = _learner(_constant_model(0.25))
    batch = _sequence_batch(gamma=0.5, n_step=1)

    metrics, priorities = learner.update(batch)

    assert metrics["debug/trained_positions"] == 4.0
    assert len(priorities.priorities) == 2
    assert list(priorities.transition_ids) == [0, 1]


def test_sequence_priorities_mix_max_and_mean_td_errors() -> None:
    constant = 0.25
    gamma = 0.5
    learner = _learner(_constant_model(constant))
    batch = _sequence_batch(gamma=gamma, n_step=1)

    _, priorities = learner.update(batch)

    rewards = torch.as_tensor(batch.rewards)
    for row in range(2):
        inner = [abs(float(rewards[row, i]) + gamma * constant - constant) for i in range(3)]
        final = abs(float(rewards[row, -1]) + gamma * constant - constant)
        td = [*inner, final]
        expected = 0.9 * max(td) + 0.1 * (sum(td) / len(td))
        assert priorities.priorities[row] == pytest.approx(expected, rel=1e-4)


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


def test_value_rescaling_is_invertible() -> None:
    values = torch.linspace(-50.0, 50.0, 101)
    assert torch.allclose(inverse_rescale_value(rescale_value(values)), values, atol=5e-3)


def test_rescaled_targets_stay_bounded_for_large_returns() -> None:
    learner = _learner(_constant_model(0.0), value_rescaling=True)
    batch = _sequence_batch(gamma=0.99, n_step=1)

    metrics, _ = learner.update(batch)

    assert metrics["debug/target_abs_max"] < 10.0


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


def test_margin_loss_is_absent_without_demo_samples() -> None:
    learner = _learner(
        _constant_model(0.25),
        demonstration_margin_weight=1.0,
    )
    batch = _sequence_batch(demo=False)

    metrics, _ = learner.update(batch)

    assert metrics["loss/demonstration_margin"] == 0.0


def test_reward_progress_index_cannot_jump_across_folded_track() -> None:
    points = np.asarray([[float(x), 0.0, 0.0] for x in range(300)], dtype=np.float32)
    reward = TrajectoryReward(
        points,
        nearest_forward_points=500,
        max_projected_speed_mps=50.0,
        max_time_delta_s=1.0,
    )
    reward.reset(points[0])

    result = reward.step(points[250], finish_ui_active=False)

    assert not result.terminated
    assert reward.progress_m <= 50.0
