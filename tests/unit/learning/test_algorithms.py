"""CPU contract checks for the five first-class TrackmaniaRL 1.0 learners."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from trackmaniarl.algorithms import (
    AdaptiveGradientClipper,
    ImplicitQuantileQLearning,
    ProximalPolicyOptimization,
    RandomizedEnsembleSAC,
    SoftActorCritic,
    StableDiscreteSoftActorCritic,
    TruncatedQuantileCritic,
)
from trackmaniarl.algorithms._torch import polyak_update, weighted_mean
from trackmaniarl.algorithms.execution import TorchExecutionConfig
from trackmaniarl.algorithms.implicit_quantile_q_learning import implicit_quantile_huber_loss
from trackmaniarl.algorithms.proximal_policy_optimization import generalized_advantage_estimate
from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.models import (
    HypersphericalLinear,
    SimbaV2Backbone,
    project_hyperspherical_weights,
)
from trackmaniarl.models.actors import CategoricalActor, GaussianActor
from trackmaniarl.models.critics import (
    ContinuousQCritic,
    ContinuousValueCritic,
    DiscreteQuantileNetwork,
    QuantileCritic,
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
    def __init__(self, *, quantiles: int | None = None) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), 16, 2)
        if quantiles is None:
            self.q1 = ContinuousQCritic(Encoder(), 16, 2)
            self.q2 = ContinuousQCritic(Encoder(), 16, 2)
        else:
            self.q1 = QuantileCritic(Encoder(), 16, 2, quantiles)
            self.q2 = QuantileCritic(Encoder(), 16, 2, quantiles)


class ContinuousPpoModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = GaussianActor(Encoder(), 16, 2)
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
        self.actor = GaussianActor(StructuredEncoder(), 4, 2)
        self.value = StructuredValue()


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
    learner.execution = TorchExecutionConfig(device="cpu", precision="float32")
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


def test_ppo_updates_from_behavior_policy_sequences() -> None:
    learner = ProximalPolicyOptimization(
        ContinuousPpoModel(),
        update_epochs=2,
        minibatch_size=8,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0, "total_transitions": 32})
    observations = torch.randn(4, 4, 4)
    next_observations = torch.randn(4, 4, 4)
    with torch.no_grad():
        actions, log_probabilities, latent_actions = learner.model.actor.sample_with_latent(
            observations
        )
        values = learner.model.value(observations)
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=actions,
        rewards=torch.randn(4, 4),
        next_observations=next_observations,
        terminated=torch.zeros(4, 4, dtype=torch.bool),
        truncated=torch.zeros(4, 4, dtype=torch.bool),
        bootstrap_discounts=torch.full((4, 4), 0.99),
        transition_ids=list(range(16)),
        metadata={
            "behavior_log_probabilities": log_probabilities,
            "behavior_values": values,
            "behavior_latent_actions": latent_actions,
        },
    )

    metrics = learner.update(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert 0.0 <= metrics["state/clip_fraction"] <= 1.0
    assert metrics["state/learning_rate"] == pytest.approx(3e-4)

    annealed = learner.update(batch)

    assert annealed["state/learning_rate"] == pytest.approx(1.5e-4)
    assert learner.state_dict()["observation_normalizer"]["moments"]


def test_ppo_gae_stops_recursion_at_episode_end_but_bootstraps_truncation() -> None:
    advantages, returns = generalized_advantage_estimate(
        rewards=torch.tensor([[1.0, 2.0]]),
        values=torch.zeros(1, 2),
        next_values=torch.tensor([[0.0, 5.0]]),
        bootstrap_discounts=torch.tensor([[0.9, 0.9]]),
        episode_ends=torch.tensor([[False, True]]),
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, torch.tensor([[6.85, 6.5]]))
    assert torch.equal(returns, advantages)


def test_ppo_policy_records_behavior_statistics() -> None:
    learner = ProximalPolicyOptimization(
        ContinuousPpoModel(), execution={"device": "cpu", "precision": "float32"}
    )
    learner.setup({"seed": 0})

    action, info = learner.policy().act_with_info(torch.randn(4))

    assert action.shape == (2,)
    assert set(info) == {
        "_trackmaniarl_behavior_log_probability",
        "_trackmaniarl_behavior_value",
        "_trackmaniarl_behavior_latent_action",
    }


def test_ppo_updates_from_structured_observation_sequences() -> None:
    learner = ProximalPolicyOptimization(
        StructuredPpoModel(),
        update_epochs=1,
        minibatch_size=4,
        execution={"device": "cpu", "precision": "float32"},
    )
    learner.setup({"seed": 0})
    observations = {
        "track": torch.randn(2, 3, 2),
        "telemetry": torch.randn(2, 3, 2),
    }
    next_observations = {key: torch.randn_like(value) for key, value in observations.items()}
    with torch.no_grad():
        actions, log_probabilities, latent_actions = learner.model.actor.sample_with_latent(
            observations
        )
        values = learner.model.value(observations)
    batch = TrainingBatch(
        data=observations,
        observations=observations,
        actions=actions,
        rewards=torch.randn(2, 3),
        next_observations=next_observations,
        terminated=torch.zeros(2, 3, dtype=torch.bool),
        truncated=torch.zeros(2, 3, dtype=torch.bool),
        bootstrap_discounts=torch.full((2, 3), 0.99),
        transition_ids=list(range(6)),
        metadata={
            "behavior_log_probabilities": log_probabilities,
            "behavior_values": values,
            "behavior_latent_actions": latent_actions,
        },
    )

    metrics = learner.update(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


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


def test_iqn_loss_matches_paper_quantile_reduction() -> None:
    predictions = torch.zeros(1, 2)
    targets = torch.full((1, 3), 2.0)
    quantiles = torch.full((1, 2), 0.5)

    loss = implicit_quantile_huber_loss(predictions, targets, quantiles)

    assert torch.equal(loss, torch.tensor([0.75]))


def test_importance_weights_are_normalized_after_per_sample_iqn_reduction() -> None:
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 0.5])

    loss = weighted_mean(losses, weights)

    assert loss == torch.tensor(5.0 / 3.0)


def test_bundled_actors_support_tuple_and_mapping_encoder_inputs() -> None:
    track = torch.randn(3, 2)
    telemetry = torch.randn(3, 2)
    continuous = GaussianActor(StructuredEncoder(), 4, 2)
    discrete = CategoricalActor(StructuredEncoder(), 4, 3)
    assert continuous((track, telemetry))[0].shape == (3, 2)
    assert discrete({"track": track, "telemetry": telemetry})[0].shape == (3,)


def test_gaussian_actor_scores_actions_in_asymmetric_environment_bounds() -> None:
    actor = GaussianActor(
        Encoder(),
        16,
        3,
        action_low=(0.0, 0.0, -1.0),
        action_high=(1.0, 1.0, 1.0),
    )
    observations = torch.randn(8, 4)

    actions, sampled_log_probabilities, latent_actions = actor.sample_with_latent(observations)
    evaluated_log_probabilities, _ = actor.evaluate_latent_actions(observations, latent_actions)

    assert torch.all((actions[:, :2] >= 0.0) & (actions[:, :2] <= 1.0))
    assert torch.all((actions[:, 2] >= -1.0) & (actions[:, 2] <= 1.0))
    assert torch.allclose(evaluated_log_probabilities, sampled_log_probabilities, atol=1e-5)


def test_gaussian_actor_rescores_exact_latent_when_action_saturates() -> None:
    actor = GaussianActor(nn.Identity(), 1, 1)
    actor.mean.weight.data.zero_()
    actor.mean.bias.data.fill_(12.0)
    actor.log_std.weight.data.zero_()
    actor.log_std.bias.data.fill_(-5.0)

    action, sampled_log_probability, latent_action = actor.sample_with_latent(torch.zeros(1, 1))
    evaluated_log_probability, _ = actor.evaluate_latent_actions(torch.zeros(1, 1), latent_action)

    assert action.item() == 1.0
    assert torch.allclose(evaluated_log_probability, sampled_log_probability)


def test_gaussian_actor_entropy_accounts_for_squashing() -> None:
    actor = GaussianActor(nn.Identity(), 1, 1)
    actor.mean.weight.data.zero_()
    actor.log_std.weight.data.zero_()
    actor.log_std.bias.data.zero_()
    observation = torch.zeros(128, 1)
    latent_actions = torch.zeros_like(observation)

    torch.manual_seed(3)
    actor.mean.bias.data.zero_()
    _, centered_entropy = actor.evaluate_latent_actions(observation, latent_actions)
    torch.manual_seed(3)
    actor.mean.bias.data.fill_(8.0)
    _, saturated_entropy = actor.evaluate_latent_actions(observation, latent_actions)

    assert saturated_entropy.mean() < centered_entropy.mean() - 5.0


def test_iqn_policy_accepts_an_unbatched_observation() -> None:
    learner = ImplicitQuantileQLearning(
        DiscreteQuantileNetwork(Encoder(), 16, 3, cosine_count=8),
        train_quantile_count=8,
        target_quantile_count=8,
        evaluation_quantile_count=8,
        execution={"device": "cpu", "precision": "float32"},
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
    learner = ImplicitQuantileQLearning(
        GreedyIQN(),
        exploration_epsilon=1.0,
        execution={"device": "cpu", "precision": "float32"},
    )
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


def test_simbav2_backbone_preserves_shape_and_unit_feature_norm() -> None:
    backbone = SimbaV2Backbone(input_dim=6, hidden_dim=16, block_count=2)

    output = backbone(torch.randn(8, 6))

    assert output.shape == (8, 16)
    assert torch.allclose(output.norm(dim=-1), torch.ones(8), atol=1e-5)


def test_hyperspherical_weights_can_be_projected_after_an_optimizer_step() -> None:
    backbone = SimbaV2Backbone(input_dim=4, hidden_dim=8, block_count=1)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=0.1)
    backbone(torch.randn(4, 4)).square().sum().backward()
    optimizer.step()

    project_hyperspherical_weights(backbone)

    layers = [module for module in backbone.modules() if isinstance(module, HypersphericalLinear)]
    assert layers
    for layer in layers:
        assert torch.allclose(
            layer.weight.norm(dim=1),
            torch.ones(layer.weight.shape[0]),
            atol=1e-5,
        )


def test_adaptive_gradient_clipper_limits_spikes_and_restores_state() -> None:
    parameter = nn.Parameter(torch.ones(2))
    clipper = AdaptiveGradientClipper(decay=0.5, warmup_steps=0, clip_factor=1.0)
    parameter.grad = torch.ones_like(parameter)
    baseline = clipper([parameter])
    parameter.grad = torch.full_like(parameter, 100.0)

    spike = clipper([parameter])

    assert baseline.coefficient == 1.0
    assert spike.clipped
    assert parameter.grad.norm() == torch.tensor(spike.ema_norm)
    restored = AdaptiveGradientClipper()
    restored.load_state_dict(clipper.state_dict())
    assert restored.step_count == clipper.step_count
    assert restored.ema_norm == clipper.ema_norm
