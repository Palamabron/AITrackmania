"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.imitation_learning import (
    INTERVENTION_KEY,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    BehaviorCloningLap,
    BehaviorCloningLearner,
    BehaviorCloningPolicy,
    LidarBehaviorCloningModel,
    augment_behavior_cloning_laps,
    load_behavior_cloning_laps,
    load_behavior_cloning_recovery,
    save_behavior_cloning_recovery,
    split_behavior_cloning_laps,
)


class _RecoveryPipeline:
    def reset_episode(self) -> None:
        return None

    def transform_observation(self, observation: object) -> dict[str, torch.Tensor]:
        values = torch.as_tensor(observation, dtype=torch.float32)
        return {
            "lidar": torch.zeros((4, 8)),
            "lidar_mask": torch.ones(8, dtype=torch.bool),
            "telemetry": values[:26],
        }

    def collate(self, transitions: list[object]) -> list[object]:
        return transitions


def test_behavior_cloning_rejects_explicit_decision_interval_mismatch(
    tmp_path: Path,
) -> None:
    frames = np.zeros((2, 33), dtype=np.float32)
    frames[:, 3] = [20.0, 40.0]
    frames[-1, 2] = 1.0
    demonstration = Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=20.0,
        frames=frames,
        actions=np.asarray([39], dtype=np.int64),
        controls=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        finish_time_s=0.04,
    )
    paths = [save_demonstration(tmp_path / f"demo-{index}", demonstration) for index in range(3)]

    with pytest.raises(ValueError, match="decision interval 20ms"):
        load_behavior_cloning_laps(
            paths,
            _RecoveryPipeline(),
            (0, 1, 3, 39, 72, 73, 75),
            expected_action_repeat_frames=1,
            expected_decision_interval_ms=10.0,
        )


def test_behavior_cloning_rejects_sparse_recording_during_ingestion(tmp_path: Path) -> None:
    frames = np.zeros((3, 33), dtype=np.float32)
    frames[:, 3] = [0.0, 10.0, 70.0]
    frames[-1, 2] = 1.0
    demonstration = Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=10.0,
        frames=frames,
        actions=np.asarray([39, 39], dtype=np.int64),
        controls=np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        finish_time_s=0.07,
    )
    paths = [save_demonstration(tmp_path / f"sparse-{index}", demonstration) for index in range(3)]

    with pytest.raises(ValueError, match="telemetry cadence is too sparse"):
        load_behavior_cloning_laps(
            paths,
            _RecoveryPipeline(),
            (0, 1, 3, 39, 72, 73, 75),
            expected_decision_interval_ms=10.0,
        )


def _observation(value: float) -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.full((4, 8), value),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.full((26,), value),
    }


def _lap(label: int) -> BehaviorCloningLap:
    return BehaviorCloningLap((_observation(float(label)),), torch.tensor([label]))


def test_behavior_cloning_split_keeps_the_elite_lap_in_training() -> None:
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


@pytest.mark.parametrize("lap_count", [3, 4, 11])
def test_behavior_cloning_split_is_disjoint_and_deterministic(lap_count: int) -> None:
    laps = [_lap(index) for index in range(lap_count)]

    first_training, first_validation = split_behavior_cloning_laps(laps, seed=19)
    second_training, second_validation = split_behavior_cloning_laps(laps, seed=19)

    assert first_training == second_training
    assert first_validation == second_validation
    assert first_training
    assert first_validation
    assert not {id(lap) for lap in first_training} & {id(lap) for lap in first_validation}
    assert len(first_training) + len(first_validation) == lap_count


def test_behavior_cloning_split_rejects_one_recovery_episode() -> None:
    with pytest.raises(ValueError, match="at least two complete episodes"):
        split_behavior_cloning_laps([_lap(0)], seed=1)


def test_behavior_cloning_setup_seeds_model_initialization() -> None:
    def initialize() -> dict[str, torch.Tensor]:
        learner = BehaviorCloningLearner(
            model_factory=type(
                "Factory",
                (),
                {
                    "build": lambda self: LidarBehaviorCloningModel(
                        action_ids=(0, 1), spatial_bins=4
                    )
                },
            )(),
            execution={"device": "cpu"},
        )
        learner.setup({"seed": 31})
        assert learner.model is not None
        return {key: value.clone() for key, value in learner.model.state_dict().items()}

    first = initialize()
    torch.manual_seed(999)
    second = initialize()

    assert first.keys() == second.keys()
    assert all(torch.equal(first[key], second[key]) for key in first)


def test_behavior_cloning_checkpoint_restores_rng_and_rejects_other_dataset() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4),
        execution={"device": "cpu"},
    )
    learner.setup({"seed": 11})
    learner.bind_dataset("dataset-a")
    state = learner.state_dict()
    expected = torch.rand(4)
    torch.manual_seed(999)

    learner.load_state_dict(state)

    assert torch.equal(torch.rand(4), expected)
    learner.bind_dataset("dataset-b")
    with pytest.raises(ValueError, match="dataset fingerprint"):
        learner.load_state_dict(state)


def test_steering_auxiliary_loss_penalizes_wrong_direction() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(3, 39, 75), spatial_bins=4),
        steering_auxiliary_loss_weight=1.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    correct = learner._steering_loss(torch.tensor([[3.0, 0.0, -3.0]]), torch.tensor([0]))
    wrong = learner._steering_loss(torch.tensor([[-3.0, 0.0, 3.0]]), torch.tensor([0]))

    assert correct < wrong


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


def test_weighted_recovery_round_trip_preserves_dagger_metadata(tmp_path: Path) -> None:
    frames = np.arange(3 * 33, dtype=np.float32).reshape(3, 33)
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "weighted-recovery",
        frames,
        np.asarray([0, 3, 6], dtype=np.int64),
        np.asarray([True, False, False]),
        action_ids,
        sample_weights=np.asarray([0.25, 3.0, 6.0], dtype=np.float32),
        student_actions=np.asarray([0, 2, 4], dtype=np.int64),
        interventions=np.asarray([False, True, True]),
        state_errors=np.asarray([0.1, 0.8, 1.4], dtype=np.float32),
    )

    observations = load_behavior_cloning_recovery([path], _RecoveryPipeline(), action_ids)[
        0
    ].observations

    assert [float(item[SAMPLE_WEIGHT_KEY]) for item in observations] == [0.25, 3.0, 6.0]
    assert [int(item[STUDENT_ACTION_KEY]) for item in observations] == [0, 2, 4]
    assert [bool(item[INTERVENTION_KEY]) for item in observations] == [False, True, True]
    assert [float(item[STATE_ERROR_KEY]) for item in observations] == pytest.approx([0.1, 0.8, 1.4])


def test_recovery_populates_previous_action_for_conditioned_model(tmp_path: Path) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "conditioned-recovery",
        np.zeros((3, 33), dtype=np.float32),
        np.asarray([2, 4, 1], dtype=np.int64),
        np.asarray([True, False, False]),
        action_ids,
    )

    lap = load_behavior_cloning_recovery(
        [path],
        _RecoveryPipeline(),
        action_ids,
        previous_action_conditioning=True,
    )[0]

    assert [int(item["previous_action"]) for item in lap.observations] == [7, 2, 4]


def test_behavior_cloning_loss_prioritizes_weighted_recovery_states() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4),
        label_smoothing=0.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    logits = torch.tensor([[-3.0, 3.0], [3.0, -3.0]])
    targets = torch.tensor([0, 0])
    observations = {
        "expert_previous_action": torch.tensor([2, 2]),
        SAMPLE_WEIGHT_KEY: torch.tensor([8.0, 1.0]),
    }

    weighted = learner._classification_loss(logits, targets, torch.ones(2), observations)
    unweighted = learner._classification_loss(
        logits,
        targets,
        torch.ones(2),
        {"expert_previous_action": torch.tensor([2, 2])},
    )

    assert weighted > unweighted


def test_behavior_cloning_weighted_loss_is_batch_partition_invariant() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(0, 1), spatial_bins=4),
        label_smoothing=0.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    logits = torch.tensor([[3.0, -1.0], [-2.0, 2.0], [0.5, -0.5], [-1.0, 1.0]])
    targets = torch.tensor([0, 0, 1, 1])
    class_weights = torch.tensor([0.5, 2.0])
    observations = {
        "expert_previous_action": torch.tensor([2, 0, 0, 1]),
        SAMPLE_WEIGHT_KEY: torch.tensor([1.0, 4.0, 0.5, 3.0]),
    }

    full = learner._classification_loss_terms(logits, targets, class_weights, observations)
    partitions = [
        learner._classification_loss_terms(
            logits[indices],
            targets[indices],
            class_weights,
            {key: value[indices] for key, value in observations.items()},
        )
        for indices in (slice(0, 1), slice(1, 4))
    ]

    full_loss = full[0] / full[1]
    partitioned_numerator = partitions[0][0] + partitions[1][0]
    partitioned_denominator = partitions[0][1] + partitions[1][1]
    partitioned_loss = partitioned_numerator / partitioned_denominator
    assert torch.allclose(full_loss, partitioned_loss)


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


def test_behavior_cloning_model_masks_race_clock_features() -> None:
    torch.manual_seed(7)
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=6,
        spatial_bins=4,
        telemetry_group_dims=(6,),
        masked_telemetry_indices=(3, 5),
    ).eval()
    baseline = {
        "lidar": torch.zeros((2, 4, 8)),
        "lidar_mask": torch.ones((2, 8), dtype=torch.bool),
        "telemetry": torch.randn((2, 6)),
    }
    shifted = {key: value.clone() for key, value in baseline.items()}
    shifted["telemetry"][..., 3] = 0.75
    shifted["telemetry"][..., 5] = -0.9

    with torch.inference_mode():
        baseline_logits = model(baseline)
        shifted_logits = model(shifted)

    assert torch.equal(baseline_logits, shifted_logits)


def test_behavior_cloning_policy_rejects_low_margin_action_flicker() -> None:
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
    policy = BehaviorCloningPolicy(model, torch.device("cpu"))
    observation = {
        "lidar": torch.zeros((4, 8)),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.zeros(26),
    }

    assert policy.act(observation) == 0
    with torch.no_grad():
        model.head.bias[1] = 0.2
    assert policy.act(observation) == 0
    with torch.no_grad():
        model.head.bias[1] = 1.0
    assert policy.act(observation) == 1


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


def test_behavior_cloning_horizontal_flip_preserves_unknown_tensor_features() -> None:
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
