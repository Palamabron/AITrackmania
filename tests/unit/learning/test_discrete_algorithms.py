"""Discrete learner, model, and optimizer contract tests."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.unit.learning._algorithm_fixtures import (
    BatchKind,
    DiscreteSacModel,
    Encoder,
    StructuredEncoder,
    _assert_update,
    _batch,
)
from trackmaniarl.algorithms import (
    AdaptiveGradientClipper,
    StableDiscreteSoftActorCritic,
)
from trackmaniarl.algorithms._torch import polyak_update, weighted_mean
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models import (
    HypersphericalLinear,
    SimbaV2Backbone,
    project_hyperspherical_weights,
)
from trackmaniarl.models.actors import CategoricalActor, GaussianActor, GaussianActorConfig
from trackmaniarl.models.encoders import ConvolutionalSensorEncoder


def test_discrete_sac_updates_without_shape_or_target_errors() -> None:
    _assert_update(StableDiscreteSoftActorCritic(DiscreteSacModel()), _batch(BatchKind.DISCRETE))


def test_convolutional_encoder_rejects_a_zero_width_first_convolution() -> None:
    with pytest.raises(ValueError, match="hidden_dim at least two"):
        ConvolutionalSensorEncoder(channels=3, output_dim=8, hidden_dim=1)


def test_importance_weights_are_normalized_after_per_sample_reduction() -> None:
    losses = torch.tensor([1.0, 3.0])
    weights = torch.tensor([1.0, 0.5])

    loss = weighted_mean(losses, weights)

    assert loss == torch.tensor(5.0 / 3.0)


def _assert_bundled_actors_support_tuple_and_mapping_encoder_inputs() -> None:
    track = torch.randn(3, 2)
    telemetry = torch.randn(3, 2)
    continuous = GaussianActor(StructuredEncoder(), GaussianActorConfig(4, 2))
    discrete = CategoricalActor(StructuredEncoder(), 4, 3)
    assert continuous((track, telemetry))[0].shape == (3, 2)
    assert discrete({"track": track, "telemetry": telemetry})[0].shape == (3,)


def _assert_categorical_actor_evaluation_selects_argmax() -> None:
    actor = CategoricalActor(nn.Identity(), 2, 3)
    with torch.no_grad():
        actor.logits.weight.zero_()
        actor.logits.bias.copy_(torch.tensor([1.0, 3.0, 2.0]))

    actions, _ = actor(torch.zeros(4, 2), mode=PolicyMode.EVALUATION)

    assert torch.equal(actions, torch.ones(4, dtype=torch.long))


def test_bundled_actor_inputs_and_categorical_evaluation() -> None:
    _assert_bundled_actors_support_tuple_and_mapping_encoder_inputs()
    _assert_categorical_actor_evaluation_selects_argmax()


def _assert_gaussian_actor_evaluation_uses_distribution_mean() -> None:
    actor = GaussianActor(nn.Identity(), GaussianActorConfig(2, 1))
    with torch.no_grad():
        actor.mean.weight.zero_()
        actor.mean.bias.fill_(0.5)

    _, _, latent = actor.sample_with_latent(torch.zeros(4, 2), mode=PolicyMode.EVALUATION)

    assert torch.equal(latent, torch.full((4, 1), 0.5))


def _assert_gaussian_actor_scores_actions_in_asymmetric_environment_bounds() -> None:
    actor = GaussianActor(
        Encoder(),
        GaussianActorConfig(
            16,
            3,
            action_low=(0.0, 0.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        ),
    )
    observations = torch.randn(8, 4)

    actions, sampled_log_probabilities, latent_actions = actor.sample_with_latent(observations)
    evaluated_log_probabilities, _ = actor.evaluate_latent_actions(observations, latent_actions)

    assert torch.all((actions[:, :2] >= 0.0) & (actions[:, :2] <= 1.0))
    assert torch.all((actions[:, 2] >= -1.0) & (actions[:, 2] <= 1.0))
    assert torch.allclose(evaluated_log_probabilities, sampled_log_probabilities, atol=1e-5)


def test_gaussian_actor_evaluation_and_asymmetric_bounds() -> None:
    _assert_gaussian_actor_evaluation_uses_distribution_mean()
    _assert_gaussian_actor_scores_actions_in_asymmetric_environment_bounds()


def _assert_gaussian_actor_rescores_exact_latent_when_action_saturates() -> None:
    actor = GaussianActor(nn.Identity(), GaussianActorConfig(1, 1))
    actor.mean.weight.data.zero_()
    actor.mean.bias.data.fill_(12.0)
    actor.log_std.weight.data.zero_()
    actor.log_std.bias.data.fill_(-5.0)

    action, sampled_log_probability, latent_action = actor.sample_with_latent(torch.zeros(1, 1))
    evaluated_log_probability, _ = actor.evaluate_latent_actions(torch.zeros(1, 1), latent_action)

    assert action.item() == 1.0
    assert torch.allclose(evaluated_log_probability, sampled_log_probability)


def _assert_gaussian_actor_entropy_accounts_for_squashing() -> None:
    actor = GaussianActor(nn.Identity(), GaussianActorConfig(1, 1))
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


def test_gaussian_actor_saturation_corrections() -> None:
    _assert_gaussian_actor_rescores_exact_latent_when_action_saturates()
    _assert_gaussian_actor_entropy_accounts_for_squashing()


def test_polyak_update_copies_batch_norm_buffers() -> None:
    source = nn.BatchNorm1d(2)
    target = nn.BatchNorm1d(2)
    source.train()
    source(torch.tensor([[2.0, 4.0], [6.0, 8.0]]))
    polyak_update(source, target, 0.5)
    assert torch.equal(target.running_mean, source.running_mean)
    assert torch.equal(target.running_var, source.running_var)


def _assert_simbav2_backbone_preserves_shape_and_unit_feature_norm() -> None:
    backbone = SimbaV2Backbone(input_dim=6, hidden_dim=16, block_count=2)

    output = backbone(torch.randn(8, 6))

    assert output.shape == (8, 16)
    assert torch.allclose(output.norm(dim=-1), torch.ones(8), atol=1e-5)


def _assert_hyperspherical_weights_project_after_optimizer_step() -> None:
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


def test_simbav2_shape_and_hyperspherical_projection() -> None:
    _assert_simbav2_backbone_preserves_shape_and_unit_feature_norm()
    _assert_hyperspherical_weights_project_after_optimizer_step()


def test_adaptive_gradient_clipper_limits_spikes_and_restores_state() -> None:
    parameter = nn.Parameter(torch.ones(2))
    clipper = AdaptiveGradientClipper(decay=0.5, warmup_steps=0, clip_factor=1.0)
    parameter.grad = torch.ones_like(parameter)
    baseline = clipper([parameter])
    prior_ema = baseline.ema_norm
    parameter.grad = torch.full_like(parameter, 100.0)

    spike = clipper([parameter])

    assert baseline.coefficient == 1.0
    assert spike.clipped
    assert float(parameter.grad.norm()) == pytest.approx(prior_ema)
    assert spike.ema_norm > prior_ema
    restored = AdaptiveGradientClipper()
    restored.load_state_dict(clipper.state_dict())
    assert restored.step_count == clipper.step_count
    assert restored.ema_norm == clipper.ema_norm
