"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.trackmania.imitation_learning import (
    SAMPLE_WEIGHT_KEY,
    BehaviorCloningLap,
    BehaviorCloningLearner,
    BehaviorCloningPolicy,
    LidarBehaviorCloningModel,
    augment_behavior_cloning_laps,
    horizontal_flip_observation,
)
from trackmaniarl.trackmania.imitation_learning._learner_metrics import ClassificationBatch


def _classification_learner() -> BehaviorCloningLearner:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4),
        label_smoothing=0.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    return learner


@dataclass(frozen=True)
class _LossInputs:
    logits: torch.Tensor
    targets: torch.Tensor
    class_weights: torch.Tensor
    observations: dict[str, torch.Tensor]


def _partition_inputs() -> _LossInputs:
    return _LossInputs(
        torch.tensor([[3.0, -1.0], [-2.0, 2.0], [0.5, -0.5], [-1.0, 1.0]]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0.5, 2.0]),
        {
            "expert_previous_action": torch.tensor([2, 0, 0, 1]),
            SAMPLE_WEIGHT_KEY: torch.tensor([1.0, 4.0, 0.5, 3.0]),
        },
    )


def _loss_terms(
    learner: BehaviorCloningLearner, inputs: _LossInputs
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = ClassificationBatch(
        inputs.logits, inputs.targets, inputs.class_weights, inputs.observations
    )
    return learner._classification_loss_terms(batch)


def _partitioned_loss(learner: BehaviorCloningLearner, inputs: _LossInputs) -> torch.Tensor:
    terms = []
    for indices in (slice(0, 1), slice(1, 4)):
        observations = {key: value[indices] for key, value in inputs.observations.items()}
        batch = ClassificationBatch(
            inputs.logits[indices], inputs.targets[indices], inputs.class_weights, observations
        )
        terms.append(learner._classification_loss_terms(batch))
    numerator = terms[0][0] + terms[1][0]
    denominator = terms[0][1] + terms[1][1]
    return numerator / denominator


def _history_observation(
    batch_size: int, history_length: int, telemetry_dim: int
) -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.zeros((batch_size, history_length, 4, 8)),
        "lidar_mask": torch.ones((batch_size, history_length, 8), dtype=torch.bool),
        "telemetry": torch.zeros((batch_size, history_length, telemetry_dim)),
    }


def _clock_model() -> LidarBehaviorCloningModel:
    return LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=6,
        spatial_bins=4,
        telemetry_group_dims=(6,),
        masked_telemetry_indices=(3, 5),
    ).eval()


def _clock_observations() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    baseline = {
        "lidar": torch.zeros((2, 4, 8)),
        "lidar_mask": torch.ones((2, 8), dtype=torch.bool),
        "telemetry": torch.randn((2, 6)),
    }
    shifted = {key: value.clone() for key, value in baseline.items()}
    shifted["telemetry"][..., 3] = 0.75
    shifted["telemetry"][..., 5] = -0.9
    return baseline, shifted


def _flicker_model() -> LidarBehaviorCloningModel:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        spatial_bins=4,
        previous_action_conditioning=True,
        minimum_action_hold_steps=2,
        switch_logit_margin=0.25,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[0] = 0.1
    return model


def _policy_observation() -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.zeros((4, 8)),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.zeros(26),
    }


def _assert_loss_prioritizes_weighted_recovery_states() -> None:
    learner = _classification_learner()
    logits = torch.tensor([[-3.0, 3.0], [3.0, -3.0]])
    targets = torch.tensor([0, 0])
    previous_actions = {"expert_previous_action": torch.tensor([2, 2])}
    weighted_observations = {**previous_actions, SAMPLE_WEIGHT_KEY: torch.tensor([8.0, 1.0])}
    weighted = learner._classification_loss(
        ClassificationBatch(logits, targets, torch.ones(2), weighted_observations)
    )
    unweighted = learner._classification_loss(
        ClassificationBatch(logits, targets, torch.ones(2), previous_actions)
    )

    assert weighted > unweighted


def _assert_weighted_loss_is_batch_partition_invariant() -> None:
    learner = _classification_learner()
    inputs = _partition_inputs()
    full = _loss_terms(learner, inputs)
    full_loss = full[0] / full[1]
    assert torch.allclose(full_loss, _partitioned_loss(learner, inputs))


def test_behavior_cloning_weighted_loss_contracts() -> None:
    _assert_loss_prioritizes_weighted_recovery_states()
    _assert_weighted_loss_is_batch_partition_invariant()


def _assert_model_encodes_a_temporal_history() -> None:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=23,
        history_length=8,
        spatial_bins=4,
    )
    observation = {
        "lidar": torch.zeros((2, 8, 4, 8)),
        "lidar_mask": torch.ones((2, 8, 8), dtype=torch.bool),
        "telemetry": torch.zeros((2, 8, 23)),
    }

    logits = model(observation)

    assert logits.shape == (2, 7)


def _assert_recurrent_policy_acts_from_a_feature_history() -> None:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=23,
        history_length=8,
        spatial_bins=4,
    )
    policy = BehaviorCloningPolicy(model, torch.device("cpu"))
    observation = {
        "lidar": torch.zeros((8, 4, 8)),
        "lidar_mask": torch.ones((8, 8), dtype=torch.bool),
        "telemetry": torch.zeros((8, 23)),
    }

    action = policy.act(observation, mode=PolicyMode.EVALUATION)

    assert 0 <= action < model.action_count


def test_behavior_cloning_temporal_model_and_policy() -> None:
    _assert_model_encodes_a_temporal_history()
    _assert_recurrent_policy_acts_from_a_feature_history()


def _assert_model_can_share_a_feedforward_simba_history_encoder() -> None:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=23,
        history_length=3,
        spatial_bins=4,
        encoder_hidden_dim=32,
        encoder_output_dim=32,
        simba_backbone={"hidden_dim": 32, "block_count": 1, "expansion": 2},
    )
    observation = _history_observation(2, 3, 23)

    logits = model(observation)

    assert logits.shape == (2, 7)
    assert isinstance(model.temporal, IdentityTemporalCore)


def _assert_model_masks_race_clock_features() -> None:
    torch.manual_seed(7)
    model = _clock_model()
    baseline, shifted = _clock_observations()

    with torch.inference_mode():
        baseline_logits = model(baseline)
        shifted_logits = model(shifted)

    assert torch.equal(baseline_logits, shifted_logits)


def test_behavior_cloning_encoder_and_feature_masking() -> None:
    _assert_model_can_share_a_feedforward_simba_history_encoder()
    _assert_model_masks_race_clock_features()


def test_behavior_cloning_policy_rejects_low_margin_action_flicker() -> None:
    model = _flicker_model()
    policy = BehaviorCloningPolicy(model, torch.device("cpu"))
    observation = _policy_observation()

    assert policy.act(observation) == 0
    with torch.no_grad():
        model.head.bias[1] = 0.2
    assert policy.act(observation) == 0
    with torch.no_grad():
        model.head.bias[1] = 1.0
    assert policy.act(observation) == 1


def _assert_horizontal_flip_reflects_actions_and_local_features() -> None:
    observation = {
        "lidar": torch.arange(128, dtype=torch.float32).reshape(2, 8, 8),
        "lidar_mask": torch.ones((2, 8), dtype=torch.bool),
        "telemetry": torch.arange(92, dtype=torch.float32).reshape(2, 46),
    }
    lap = BehaviorCloningLap((observation,) * 7, torch.arange(7, dtype=torch.long))

    augmented = augment_behavior_cloning_laps([lap], (0, 1, 3, 39, 72, 73, 75))
    reflected = augmented[1].observations[0]

    assert len(augmented) == 2
    assert augmented[1].labels.tolist() == [4, 5, 6, 3, 0, 1, 2]
    assert torch.equal(reflected["lidar"][..., 0, :], -observation["lidar"][..., 2, :])
    assert torch.equal(reflected["lidar"][..., 1, :], observation["lidar"][..., 3, :])
    assert torch.equal(reflected["telemetry"][..., 6], -observation["telemetry"][..., 6])
    assert torch.equal(reflected["telemetry"][..., 10], observation["telemetry"][..., 11])
    assert torch.equal(reflected["telemetry"][..., 34], -observation["telemetry"][..., 36])


def _assert_horizontal_flip_preserves_unknown_tensor_features() -> None:
    observation = {
        "lidar": torch.zeros((8, 8)),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.zeros(46),
        "future_feature": torch.tensor([1.0, 2.0]),
    }
    lap = BehaviorCloningLap((observation,), torch.tensor([0]))

    reflected = augment_behavior_cloning_laps([lap], (0, 1, 3, 39, 72, 73, 75))[1]

    assert torch.equal(reflected.observations[0]["future_feature"], observation["future_feature"])
    assert reflected.observations[0]["future_feature"] is not observation["future_feature"]


def _assert_horizontal_flip_supports_masked_control_history() -> None:
    observation = {
        "lidar": torch.zeros((3, 8, 8)),
        "lidar_mask": torch.ones((3, 8), dtype=torch.bool),
        "telemetry": torch.arange(147, dtype=torch.float32).reshape(3, 49),
    }

    reflected = horizontal_flip_observation(observation)

    assert torch.equal(reflected["telemetry"][..., 17], -observation["telemetry"][..., 17])
    assert torch.equal(reflected["telemetry"][..., 21], -observation["telemetry"][..., 21])
    assert torch.equal(reflected["telemetry"][..., 37], -observation["telemetry"][..., 39])


def test_behavior_cloning_horizontal_flip_contracts() -> None:
    _assert_horizontal_flip_reflects_actions_and_local_features()
    _assert_horizontal_flip_preserves_unknown_tensor_features()
    _assert_horizontal_flip_supports_masked_control_history()
