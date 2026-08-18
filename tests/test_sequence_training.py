"""Contract tests for all-step recurrent sequence training and demo protection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from trackmaniarl.algorithms.implicit_quantile_q_learning import (
    ImplicitQuantileQLearning,
    inverse_rescale_value,
    rescale_value,
)
from trackmaniarl.core.builtins import IdentityFeaturePipeline
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition
from trackmaniarl.core.replay import InMemoryReplayStore, PrioritizedSampler
from trackmaniarl.models.critics import DiscreteQuantileNetwork
from trackmaniarl.trackmania.reward import TrajectoryReward


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


def test_margin_loss_ignores_non_expert_recovery_demonstrations() -> None:
    learner = _learner(
        _constant_model(0.25),
        demonstration_margin_weight=1.0,
    )
    batch = _sequence_batch(demo=True, expert=False)

    metrics, _ = learner.update(batch)

    assert metrics["loss/demonstration_margin"] == 0.0


def test_demonstration_td_weight_excludes_demo_td_loss_and_priority() -> None:
    learner = _learner(
        _constant_model(0.25),
        demonstration_margin_weight=1.0,
        demonstration_td_weight=0.0,
    )
    batch = _sequence_batch(demo=False)
    batch.metadata["demo_flags"] = (True, False)
    batch.metadata["expert_demo_flags"] = (True, False)

    metrics, priorities = learner.update(batch)

    assert metrics["debug/demonstration_td_weight"] == 0.0
    assert priorities.priorities[0] == 0.0
    assert priorities.priorities[1] > 0.0
    assert metrics["loss/demonstration_margin"] > 0.0


def test_demonstration_td_weight_scales_an_all_demo_batch() -> None:
    full = _learner(_constant_model(0.25), demonstration_td_weight=1.0)
    quarter = _learner(_constant_model(0.25), demonstration_td_weight=0.25)
    batch = _sequence_batch(demo=True)

    torch.manual_seed(17)
    full_metrics, full_priorities = full.update(batch)
    torch.manual_seed(17)
    quarter_metrics, quarter_priorities = quarter.update(batch)

    assert quarter_metrics["loss/iqn"] == pytest.approx(0.25 * full_metrics["loss/iqn"], rel=1e-5)
    assert quarter_priorities.priorities == pytest.approx(
        [0.25 * priority for priority in full_priorities.priorities], rel=1e-5
    )


def test_demonstration_cross_entropy_supervises_expert_actions_without_td_loss() -> None:
    learner = _learner(
        _constant_model(0.0),
        demonstration_cross_entropy_weight=1.0,
        demonstration_td_weight=0.0,
    )
    batch = _sequence_batch(demo=True, expert=True)

    metrics, priorities = learner.update(batch)

    assert metrics["loss/iqn"] == 0.0
    assert metrics["loss/demonstration_cross_entropy"] == pytest.approx(
        torch.log(torch.tensor(3.0)).item(), rel=1e-4
    )
    assert 0.0 <= metrics["debug/demonstration_action_accuracy"] <= 1.0
    assert metrics["loss/total"] == pytest.approx(
        metrics["loss/demonstration_cross_entropy"], rel=1e-4
    )
    assert all(priority == 0.0 for priority in priorities.priorities)


def test_demonstration_cross_entropy_uses_all_completed_demonstrations() -> None:
    learner = _learner(
        _constant_model(0.0),
        demonstration_cross_entropy_weight=1.0,
    )

    metrics, _ = learner.update(_sequence_batch(demo=True, expert=False))

    assert metrics["loss/demonstration_cross_entropy"] > 0.0


def test_demonstration_td_weight_must_be_a_fraction() -> None:
    with pytest.raises(ValueError, match="auxiliary loss parameters"):
        _learner(_constant_model(0.25), demonstration_td_weight=1.1)


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


def test_policy_action_mask_preserves_full_head_checkpoint_compatibility() -> None:
    original = _learner(_constant_model(0.0))
    state = original.state_dict()
    masked = _learner(_constant_model(0.0), policy_action_ids=(0, 1))

    masked.load_state_dict(state)

    assert masked.model is not None
    assert masked.model.action_count == 3


def test_policy_action_mask_logs_excluded_q_diagnostics() -> None:
    learner = _learner(_constant_model(0.25), policy_action_ids=(0, 1))

    metrics, _ = learner.update(_sequence_batch())

    assert metrics["debug/q_allowed_max_mean"] == pytest.approx(0.25)
    assert metrics["debug/q_excluded_max_mean"] == pytest.approx(0.25)
    assert metrics["debug/q_excluded_advantage_mean"] == pytest.approx(0.0)
    assert metrics["debug/greedy_masked_out_fraction"] == pytest.approx(0.0)


def test_policy_anchor_uses_and_persists_the_loaded_checkpoint() -> None:
    source = _learner(_constant_model(0.0))
    with torch.no_grad():
        source.model.head.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))
        source.target_model.load_state_dict(source.model.state_dict())
    anchored = _learner(_constant_model(0.0), policy_anchor_weight=1.0)

    anchored.load_state_dict(source.state_dict())
    assert anchored.policy_anchor_model is not None
    original_anchor = {
        name: value.clone() for name, value in anchored.policy_anchor_model.state_dict().items()
    }
    with torch.no_grad():
        anchored.model.head.bias.add_(torch.tensor([0.5, 0.0, -0.5]))

    metrics, _ = anchored.update(_sequence_batch())
    resumed = _learner(_constant_model(0.0), policy_anchor_weight=1.0)
    resumed.load_state_dict(anchored.state_dict())

    assert metrics["loss/policy_anchor"] > 0.0
    assert metrics["loss/total"] > metrics["loss/iqn"]
    assert resumed.policy_anchor_model is not None
    for name, value in original_anchor.items():
        assert torch.equal(resumed.policy_anchor_model.state_dict()[name], value)


def test_policy_anchor_is_disabled_during_offline_pretraining() -> None:
    learner = _learner(_constant_model(0.0), policy_anchor_weight=1.0)
    with torch.no_grad():
        learner.model.head.bias.add_(torch.tensor([0.5, 0.0, -0.5]))

    learner.begin_offline_pretraining()
    metrics, _ = learner.update(_sequence_batch())
    learner.end_offline_pretraining()

    assert metrics["loss/policy_anchor"] > 0.0
    assert metrics["loss/policy_anchor_weighted"] == 0.0


def test_policy_anchor_can_remain_active_during_offline_pretraining() -> None:
    learner = _learner(
        _constant_model(0.0),
        policy_anchor_weight=1.0,
        policy_anchor_during_offline_pretraining=True,
    )
    with torch.no_grad():
        learner.model.head.bias.add_(torch.tensor([0.5, 0.0, -0.5]))

    learner.begin_offline_pretraining()
    metrics, _ = learner.update(_sequence_batch())
    learner.end_offline_pretraining()

    assert metrics["loss/policy_anchor"] > 0.0
    assert metrics["loss/policy_anchor_weighted"] > 0.0


def test_external_policy_anchor_overrides_resumed_anchor(tmp_path: Path) -> None:
    source = _learner(_constant_model(0.0))
    with torch.no_grad():
        source.model.head.bias.copy_(torch.tensor([2.0, 1.0, 0.0]))
        source.target_model.load_state_dict(source.model.state_dict())
    checkpoint = tmp_path / "anchor.pt"
    torch.save({"learner": source.state_dict()}, checkpoint)
    anchored = _learner(
        _constant_model(0.0),
        policy_anchor_weight=1.0,
        policy_anchor_checkpoint="anchor.pt",
        base_dir=tmp_path,
    )
    resumed = source.state_dict()
    resumed["policy_anchor_model"] = _constant_model(-3.0).state_dict()

    anchored.load_state_dict(resumed)

    assert anchored.policy_anchor_model is not None
    for name, value in source.model.state_dict().items():
        assert torch.equal(anchored.policy_anchor_model.state_dict()[name], value)


def test_external_policy_anchor_requires_a_nonzero_weight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires policy_anchor_weight"):
        ImplicitQuantileQLearning(
            _constant_model(0.0),
            policy_anchor_checkpoint=tmp_path / "anchor.pt",
            execution={"device": "cpu"},
        )


def test_external_policy_anchor_rejects_an_incompatible_action_contract(tmp_path: Path) -> None:
    source = _learner(_constant_model(0.0), policy_action_ids=(0, 1))
    checkpoint = tmp_path / "anchor.pt"
    torch.save({"learner": source.state_dict()}, checkpoint)

    with pytest.raises(ValueError, match="action contract"):
        _learner(
            _constant_model(0.0),
            policy_action_ids=(1, 2),
            policy_anchor_weight=1.0,
            policy_anchor_checkpoint="anchor.pt",
            base_dir=tmp_path,
        )


def test_external_policy_anchor_rejects_an_incompatible_model(tmp_path: Path) -> None:
    unmasked_checkpoint = tmp_path / "unmasked-anchor.pt"
    unmasked_source = _learner(_constant_model(0.0))
    torch.save({"learner": unmasked_source.state_dict()}, unmasked_checkpoint)
    incompatible = DiscreteQuantileNetwork(_SequenceEncoder(), 8, action_count=4, cosine_count=4)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _learner(
            incompatible,
            policy_anchor_weight=1.0,
            policy_anchor_checkpoint="unmasked-anchor.pt",
            base_dir=tmp_path,
        )


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
