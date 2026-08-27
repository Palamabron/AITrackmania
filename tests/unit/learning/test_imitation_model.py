"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
import torch

from tests.unit.learning._imitation_fixtures import (
    _CheckpointScaler,
    _lap,
    _observation,
)
from trackmaniarl.models.backbones import HypersphericalLinear
from trackmaniarl.trackmania.imitation_learning import (
    BehaviorCloningLap,
    BehaviorCloningLearner,
    LidarBehaviorCloningModel,
    class_weights,
    split_behavior_cloning_laps,
)


class _SeededModelFactory:
    def build(self) -> LidarBehaviorCloningModel:
        return LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4)


def _initialized_state() -> Mapping[str, torch.Tensor]:
    learner = BehaviorCloningLearner(
        model_factory=_SeededModelFactory(), execution={"device": "cpu"}
    )
    learner.setup({"seed": 31})
    assert learner.model is not None
    return {key: value.clone() for key, value in learner.model.state_dict().items()}


def _checkpoint_case() -> tuple[BehaviorCloningLearner, Mapping[str, Any]]:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4),
        execution={"device": "cpu"},
    )
    learner.setup({"seed": 11})
    learner.bind_dataset("dataset-a")
    learner.scaler = _CheckpointScaler(17.0)
    return learner, learner.state_dict()


def _simba_model() -> LidarBehaviorCloningModel:
    return LidarBehaviorCloningModel(
        action_ids=(0, 3),
        telemetry_dim=23,
        history_length=3,
        spatial_bins=4,
        encoder_hidden_dim=32,
        encoder_output_dim=32,
        simba_backbone={"hidden_dim": 32, "block_count": 1, "expansion": 2},
    )


def _simba_observations() -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.zeros((2, 3, 4, 8)),
        "lidar_mask": torch.ones((2, 3, 8), dtype=torch.bool),
        "telemetry": torch.zeros((2, 3, 23)),
    }


def _assert_split_keeps_the_elite_lap_in_training() -> None:
    laps = [
        BehaviorCloningLap(
            (_observation(float(index)),),
            torch.tensor([index]),
            quality_weight=1.0 if index == 0 else 0.2,
        )
        for index in range(13)
    ]

    training, validation = split_behavior_cloning_laps(laps, seed=17)

    assert any(lap is laps[0] for lap in training)
    assert not any(lap is laps[0] for lap in validation)


def _assert_split_is_disjoint_and_deterministic(lap_count: int) -> None:
    laps = [_lap(index) for index in range(lap_count)]

    first_training, first_validation = split_behavior_cloning_laps(laps, seed=19)
    second_training, second_validation = split_behavior_cloning_laps(laps, seed=19)

    assert first_training == second_training
    assert first_validation == second_validation
    assert first_training
    assert first_validation
    assert not {id(lap) for lap in first_training} & {id(lap) for lap in first_validation}
    assert len(first_training) + len(first_validation) == lap_count


def test_behavior_cloning_split_contracts() -> None:
    _assert_split_keeps_the_elite_lap_in_training()
    for lap_count in (3, 4, 11):
        _assert_split_is_disjoint_and_deterministic(lap_count)


def test_behavior_cloning_split_rejects_one_recovery_episode() -> None:
    with pytest.raises(ValueError, match="at least two complete episodes"):
        split_behavior_cloning_laps([_lap(0)], seed=1)


def test_behavior_cloning_class_weights_allow_a_larger_safe_action_contract() -> None:
    weights = class_weights(torch.tensor([3, 3, 7]), 10, power=0.5)

    assert weights.shape == (10,)
    assert torch.isfinite(weights).all()
    assert weights[7] > weights[3]


def _assert_setup_seeds_model_initialization() -> None:
    first = _initialized_state()
    torch.manual_seed(999)
    second = _initialized_state()

    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def _assert_checkpoint_restores_rng_and_scaler() -> None:
    learner, state = _checkpoint_case()
    assert "scaler" in state
    expected = torch.rand(4)
    torch.manual_seed(999)
    learner.scaler = _CheckpointScaler(99.0)

    learner.load_state_dict(state)

    assert learner.scaler.current_scale == 17.0
    assert torch.equal(torch.rand(4), expected)


def test_behavior_cloning_deterministic_setup_and_checkpoint_restore() -> None:
    _assert_setup_seeds_model_initialization()
    _assert_checkpoint_restores_rng_and_scaler()


def _assert_checkpoint_rejects_other_dataset() -> None:
    learner, state = _checkpoint_case()

    learner.bind_dataset("dataset-b")

    with pytest.raises(ValueError, match="dataset fingerprint"):
        learner.load_state_dict(state)


def _assert_checkpoint_requires_field(field: str) -> None:
    learner, state = _checkpoint_case()
    incomplete = dict(state)
    incomplete.pop(field)

    with pytest.raises((KeyError, ValueError), match=field):
        learner.load_state_dict(incomplete)


def test_behavior_cloning_checkpoint_requires_current_fields() -> None:
    fields = (
        "schema_version",
        "scheduler",
        "scaler",
        "rng",
        "policy_action_ids",
        "dataset_fingerprint",
    )
    for field in fields:
        _assert_checkpoint_requires_field(field)


def _assert_checkpoint_rejects_other_schema() -> None:
    learner, state = _checkpoint_case()
    incompatible = {**state, "schema_version": "unsupported"}

    with pytest.raises(ValueError, match="unsupported behavior-cloning checkpoint schema"):
        learner.load_state_dict(incompatible)


def test_behavior_cloning_checkpoint_rejects_incompatible_identity() -> None:
    _assert_checkpoint_rejects_other_dataset()
    _assert_checkpoint_rejects_other_schema()


def _assert_steering_loss_penalizes_wrong_direction() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(3, 39, 75), spatial_bins=4),
        steering_auxiliary_loss_weight=1.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    correct = learner._steering_loss(torch.tensor([[3.0, 0.0, -3.0]]), torch.tensor([0]))
    wrong = learner._steering_loss(torch.tensor([[-3.0, 0.0, 3.0]]), torch.tensor([0]))

    assert correct < wrong


def _assert_steering_loss_penalizes_wrong_analog_magnitude() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(3, 9, 39, 75), spatial_bins=4),
        steering_auxiliary_loss_weight=1.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    correct = learner._steering_loss(torch.tensor([[0.0, 3.0, 0.0, -3.0]]), torch.tensor([1]))
    full_left = learner._steering_loss(torch.tensor([[3.0, 0.0, 0.0, -3.0]]), torch.tensor([1]))

    assert correct < full_left


def test_steering_auxiliary_loss_penalizes_incorrect_controls() -> None:
    _assert_steering_loss_penalizes_wrong_direction()
    _assert_steering_loss_penalizes_wrong_analog_magnitude()


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


def test_factorized_behavior_cloning_head_shares_steering_and_drive_evidence() -> None:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 3, 39, 75),
        spatial_bins=4,
        factorized_action_head=True,
    )
    head = cast(Any, model.head)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.steering.bias[0] = 2.0
        head.drive_mode.bias[3] = 1.0

    logits = head(torch.zeros((1, model.encoder.output_dim)))

    assert torch.equal(logits, torch.tensor([[2.0, 3.0, 1.0, 1.0]]))


def test_behavior_cloning_reprojects_simba_weights_after_optimizer_step() -> None:
    model = _simba_model()
    learner = BehaviorCloningLearner(model, execution={"device": "cpu"})
    learner.setup({})
    layers = [layer for layer in model.modules() if isinstance(layer, HypersphericalLinear)]
    with torch.no_grad():
        for layer in layers:
            layer.weight.mul_(2.0)
    learner.train_batch(_simba_observations(), torch.tensor([0, 1]), torch.ones(2))

    assert layers
    for layer in layers:
        assert torch.allclose(layer.weight.norm(dim=1), torch.ones(layer.weight.shape[0]))
