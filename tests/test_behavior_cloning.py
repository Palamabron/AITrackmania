"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

import torch

from tmrl.trackmania.behavior_cloning import (
    BehaviorCloningLap,
    BehaviorCloningLearner,
    LidarBehaviorCloningModel,
    augment_behavior_cloning_laps,
    class_weights,
    flatten_behavior_cloning_laps,
    split_behavior_cloning_laps,
)


def _observation(value: float) -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.full((4, 8), value),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.full((26,), value),
    }


def _lap(label: int) -> BehaviorCloningLap:
    return BehaviorCloningLap((_observation(float(label)),), torch.tensor([label]))


def test_behavior_cloning_model_emits_one_logit_per_compact_action() -> None:
    model = LidarBehaviorCloningModel(action_ids=(0, 1, 3, 39, 72, 73, 75), spatial_bins=4)
    observation = {
        "lidar": torch.zeros((2, 4, 8)),
        "lidar_mask": torch.ones((2, 8), dtype=torch.bool),
        "telemetry": torch.zeros((2, 26)),
    }

    logits = model(observation)

    assert logits.shape == (2, 7)
    assert model.action_count == 7


def test_behavior_cloning_model_encodes_a_temporal_history() -> None:
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


def test_lap_split_is_disjoint_and_retains_entire_laps() -> None:
    laps = [_lap(index % 2) for index in range(10)]

    training, validation = split_behavior_cloning_laps(laps, seed=7)
    observations, labels = flatten_behavior_cloning_laps(training)

    assert len(training) == 8
    assert len(validation) == 2
    assert {id(lap) for lap in training}.isdisjoint({id(lap) for lap in validation})
    assert len(observations) == len(labels) == 8


def test_class_weights_are_bounded_inverse_square_root_frequencies() -> None:
    weights = class_weights(torch.tensor([0, 0, 0, 0, 1]), action_count=2)

    assert torch.all(weights >= 0.5)
    assert torch.all(weights <= 3.0)
    assert weights[1] > weights[0]


def test_behavior_cloning_reduces_learning_rate_on_validation_plateau() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1, 3, 39, 72, 73, 75)),
        learning_rate=1e-3,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=1,
        execution={"device": "cpu"},
    )
    learner.setup({})

    learner.step_scheduler(1.0)
    learner.step_scheduler(1.0)
    reduced = learner.step_scheduler(1.0)
    state = learner.state_dict()
    restored = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1, 3, 39, 72, 73, 75)),
        execution={"device": "cpu"},
    )
    restored.setup({})
    restored.load_state_dict(state)

    assert reduced == 5e-4
    assert restored.current_learning_rate() == reduced


def test_behavior_cloning_clips_gradients_and_reports_the_unclipped_norm() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(
            action_ids=(0, 1, 3, 39, 72, 73, 75),
            spatial_bins=4,
        ),
        gradient_clip_norm=0.05,
        execution={"device": "cpu"},
    )
    learner.setup({})
    observations = {
        "lidar": torch.randn((8, 4, 8)),
        "lidar_mask": torch.ones((8, 8), dtype=torch.bool),
        "telemetry": torch.randn((8, 26)),
    }

    metrics = learner.train_batch(
        observations,
        torch.arange(8) % 7,
        torch.ones(7),
    )
    gradient_norm = torch.linalg.vector_norm(
        torch.stack(
            [
                parameter.grad.detach().norm()
                for parameter in learner.model.parameters()
                if parameter.grad is not None
            ]
        )
    )

    assert metrics["gradient_norm"] > 0.0
    assert gradient_norm <= 0.05001


def test_behavior_cloning_horizontal_flip_reflects_actions_and_local_features() -> None:
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
