"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from trackmaniarl.models.backbones import HypersphericalLinear
from trackmaniarl.models.temporal import IdentityTemporalCore
from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
)
from trackmaniarl.trackmania.demonstrations import Demonstration, save_demonstration
from trackmaniarl.trackmania.imitation_learning import (
    INTERVENTION_KEY,
    RECOVERY_DATASET_FORMAT_V1,
    RECOVERY_DATASET_FORMAT_V2,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    BehaviorCloningLap,
    BehaviorCloningLearner,
    BehaviorCloningPolicy,
    LidarBehaviorCloningModel,
    RecoveryContract,
    RecoveryProvenance,
    augment_behavior_cloning_laps,
    class_weights,
    horizontal_flip_observation,
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


def _recovery_contract(
    *,
    map_uid: str = "test-map",
    geometry_sha256: str = "a" * 64,
    action_repeat_frames: int = 1,
    decision_interval_ms: float | None = 10.0,
) -> RecoveryContract:
    return RecoveryContract(
        map_uid=map_uid,
        geometry_sha256=geometry_sha256,
        action_repeat_frames=action_repeat_frames,
        decision_interval_ms=decision_interval_ms,
        control_alignment="frame_start",
    )


def _recovery_provenance() -> RecoveryProvenance:
    return RecoveryProvenance(
        _recovery_contract(),
        source_demonstration_sha256="b" * 64,
        source_checkpoint_sha256="c" * 64,
    )


def _rewrite_recovery_archive(
    path: Path,
    updates: dict[str, np.ndarray | None],
) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    for key, value in updates.items():
        if value is None:
            payload.pop(key)
        else:
            payload[key] = value
    np.savez_compressed(path, **payload)


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


def test_behavior_cloning_can_lead_labels_for_delayed_observations(tmp_path: Path) -> None:
    frames = np.zeros((6, 33), dtype=np.float32)
    frames[:, 3] = np.arange(6, dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    actions = np.asarray([0, 3, 39, 72, 75], dtype=np.int64)
    _, table = build_brake_tap_action_table()
    demonstration = Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        decision_interval_ms=10.0,
        frames=frames,
        actions=actions,
        controls=np.asarray([table[action] for action in actions], dtype=np.float32),
        finish_time_s=0.05,
    )
    paths = [save_demonstration(tmp_path / f"lead-{index}", demonstration) for index in range(3)]

    laps = load_behavior_cloning_laps(
        paths,
        _RecoveryPipeline(),
        (0, 1, 3, 39, 72, 73, 75),
        expected_decision_interval_ms=10.0,
        action_lead_ms=20.0,
    )

    assert laps[0].labels.tolist() == [3, 4, 6, 6, 6]


def test_behavior_cloning_can_ingest_aggregated_control_windows(tmp_path: Path) -> None:
    frames = np.zeros((6, 33), dtype=np.float32)
    frames[:, 3] = np.arange(6, dtype=np.float32) * 10.0
    frames[-1, 2] = 1.0
    controls = np.asarray(
        [[1.0, 0.0, -1.0]] * 2 + [[1.0, 0.0, 1.0]] * 3,
        dtype=np.float32,
    )
    _, table = build_brake_tap_action_table()
    demonstration = Demonstration(
        map_uid="test-map",
        geometry_sha256="a" * 64,
        action_repeat_frames=1,
        frames=frames,
        actions=continuous_control_to_discrete_indices_batch(controls, table),
        controls=controls,
        finish_time_s=0.05,
    )
    paths = [
        save_demonstration(tmp_path / f"aggregate-{index}", demonstration) for index in range(3)
    ]

    laps = load_behavior_cloning_laps(
        paths,
        _RecoveryPipeline(),
        tuple(range(78)),
        expected_decision_interval_ms=50.0,
        aggregate_controls=True,
    )

    assert laps[0].labels.tolist() == [45]


def _observation(value: float) -> dict[str, torch.Tensor]:
    return {
        "lidar": torch.full((4, 8), value),
        "lidar_mask": torch.ones(8, dtype=torch.bool),
        "telemetry": torch.full((26,), value),
    }


class _CheckpointScaler:
    def __init__(self, scale: float) -> None:
        self.current_scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"current_scale": self.current_scale}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.current_scale = float(state["current_scale"])


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


def test_behavior_cloning_class_weights_allow_a_larger_safe_action_contract() -> None:
    weights = class_weights(torch.tensor([3, 3, 7]), 10, power=0.5)

    assert weights.shape == (10,)
    assert torch.isfinite(weights).all()
    assert weights[7] > weights[3]


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
    learner.scaler = _CheckpointScaler(17.0)
    state = learner.state_dict()
    assert "scaler" in state
    expected = torch.rand(4)
    torch.manual_seed(999)
    learner.scaler = _CheckpointScaler(99.0)

    learner.load_state_dict(state)

    assert learner.scaler.current_scale == 17.0
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


def test_steering_auxiliary_loss_penalizes_wrong_analog_magnitude() -> None:
    learner = BehaviorCloningLearner(
        LidarBehaviorCloningModel(action_ids=(3, 9, 39, 75), spatial_bins=4),
        steering_auxiliary_loss_weight=1.0,
        execution={"device": "cpu"},
    )
    learner.setup({})
    correct = learner._steering_loss(torch.tensor([[0.0, 3.0, 0.0, -3.0]]), torch.tensor([1]))
    full_left = learner._steering_loss(torch.tensor([[3.0, 0.0, 0.0, -3.0]]), torch.tensor([1]))

    assert correct < full_left


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
    model = LidarBehaviorCloningModel(
        action_ids=(0, 3),
        telemetry_dim=23,
        history_length=3,
        spatial_bins=4,
        encoder_hidden_dim=32,
        encoder_output_dim=32,
        simba_backbone={"hidden_dim": 32, "block_count": 1, "expansion": 2},
    )
    learner = BehaviorCloningLearner(model, execution={"device": "cpu"})
    learner.setup({})
    layers = [layer for layer in model.modules() if isinstance(layer, HypersphericalLinear)]
    with torch.no_grad():
        for layer in layers:
            layer.weight.mul_(2.0)
    observations = {
        "lidar": torch.zeros((2, 3, 4, 8)),
        "lidar_mask": torch.ones((2, 3, 8), dtype=torch.bool),
        "telemetry": torch.zeros((2, 3, 23)),
    }

    learner.train_batch(observations, torch.tensor([0, 1]), torch.ones(2))

    assert layers
    for layer in layers:
        assert torch.allclose(layer.weight.norm(dim=1), torch.ones(layer.weight.shape[0]))


def test_weighted_recovery_round_trip_preserves_dagger_metadata(tmp_path: Path) -> None:
    frames = np.arange(3 * 33, dtype=np.float32).reshape(3, 33)
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "weighted-recovery",
        frames,
        np.asarray([0, 3, 6], dtype=np.int64),
        np.asarray([True, False, False]),
        action_ids,
        provenance=_recovery_provenance(),
        sample_weights=np.asarray([0.25, 3.0, 6.0], dtype=np.float32),
        student_actions=np.asarray([0, 2, 4], dtype=np.int64),
        interventions=np.asarray([False, True, True]),
        state_errors=np.asarray([0.1, 0.8, 1.4], dtype=np.float32),
    )

    observations = load_behavior_cloning_recovery(
        [path],
        _RecoveryPipeline(),
        action_ids,
        expected_contract=_recovery_contract(),
        expected_source_demonstration_sha256=frozenset({"b" * 64}),
    )[0].observations

    assert [float(item[SAMPLE_WEIGHT_KEY]) for item in observations] == [0.25, 3.0, 6.0]
    assert [int(item[STUDENT_ACTION_KEY]) for item in observations] == [0, 2, 4]
    assert [bool(item[INTERVENTION_KEY]) for item in observations] == [False, True, True]
    assert [float(item[STATE_ERROR_KEY]) for item in observations] == pytest.approx([0.1, 0.8, 1.4])
    with np.load(path, allow_pickle=False) as data:
        assert str(data["source_demonstration_sha256"].item()) == "b" * 64
        assert str(data["source_checkpoint_sha256"].item()) == "c" * 64


def test_recovery_populates_previous_action_for_conditioned_model(tmp_path: Path) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "conditioned-recovery",
        np.zeros((3, 33), dtype=np.float32),
        np.asarray([2, 4, 1], dtype=np.int64),
        np.asarray([True, False, False]),
        action_ids,
        provenance=_recovery_provenance(),
    )

    lap = load_behavior_cloning_recovery(
        [path],
        _RecoveryPipeline(),
        action_ids,
        expected_contract=_recovery_contract(),
        expected_source_demonstration_sha256=frozenset({"b" * 64}),
        previous_action_conditioning=True,
    )[0]

    assert [int(item["previous_action"]) for item in lap.observations] == [7, 2, 4]


def test_recovery_rejects_the_same_source_twice(tmp_path: Path) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "duplicate-recovery",
        np.zeros((1, 33), dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        action_ids,
        provenance=_recovery_provenance(),
    )

    with pytest.raises(ValueError, match="paths must be unique"):
        load_behavior_cloning_recovery(
            [path, path],
            _RecoveryPipeline(),
            action_ids,
            expected_contract=_recovery_contract(),
            expected_source_demonstration_sha256=frozenset({"b" * 64}),
        )


@pytest.mark.parametrize(
    ("expected_contract", "message"),
    [
        (_recovery_contract(map_uid="another-map"), "map UID"),
        (_recovery_contract(geometry_sha256="d" * 64), "geometry"),
        (_recovery_contract(decision_interval_ms=20.0), "decision interval"),
        (
            _recovery_contract(action_repeat_frames=2, decision_interval_ms=None),
            "action repeat",
        ),
        (_recovery_contract(decision_interval_ms=None), "decision interval"),
    ],
)
def test_recovery_rejects_incompatible_provenance(
    tmp_path: Path,
    expected_contract: RecoveryContract,
    message: str,
) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "incompatible-recovery",
        np.zeros((1, 33), dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        action_ids,
        provenance=_recovery_provenance(),
    )

    with pytest.raises(ValueError, match=message):
        load_behavior_cloning_recovery(
            [path],
            _RecoveryPipeline(),
            action_ids,
            expected_contract=expected_contract,
            expected_source_demonstration_sha256=frozenset({"b" * 64}),
        )


@pytest.mark.parametrize(
    "format_name",
    [RECOVERY_DATASET_FORMAT_V1, RECOVERY_DATASET_FORMAT_V2],
)
def test_recovery_rejects_legacy_format_without_provenance(
    tmp_path: Path,
    format_name: str,
) -> None:
    path = tmp_path / "legacy-recovery.npz"
    np.savez_compressed(path, format=np.asarray(format_name))

    with pytest.raises(ValueError, match="predates map and timing provenance; regenerate"):
        load_behavior_cloning_recovery(
            [path],
            _RecoveryPipeline(),
            (0, 1, 3, 39, 72, 73, 75),
            expected_contract=_recovery_contract(),
            expected_source_demonstration_sha256=frozenset({"b" * 64}),
        )


def test_recovery_rejects_unexpected_source_demonstration(tmp_path: Path) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "unexpected-source",
        np.zeros((1, 33), dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        action_ids,
        provenance=_recovery_provenance(),
    )

    with pytest.raises(ValueError, match="not present in --demo inputs"):
        load_behavior_cloning_recovery(
            [path],
            _RecoveryPipeline(),
            action_ids,
            expected_contract=_recovery_contract(),
            expected_source_demonstration_sha256=frozenset({"d" * 64}),
        )


def test_recovery_rejects_non_finite_frames_before_save(tmp_path: Path) -> None:
    frames = np.zeros((1, 33), dtype=np.float32)
    frames[0, 0] = np.nan

    with pytest.raises(ValueError, match="frames must be finite"):
        save_behavior_cloning_recovery(
            tmp_path / "non-finite",
            frames,
            np.asarray([0], dtype=np.int64),
            np.asarray([True]),
            (0, 1, 3, 39, 72, 73, 75),
            provenance=_recovery_provenance(),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sample_weight": None}, "missing metadata"),
        (
            {"action_repeat_frames": np.asarray(1.5, dtype=np.float64)},
            "action repeat must be an integer",
        ),
        (
            {"frames": np.full((1, 33), np.inf, dtype=np.float32)},
            "non-finite values",
        ),
    ],
)
def test_recovery_rejects_corrupted_v3_archive(
    tmp_path: Path,
    updates: dict[str, np.ndarray | None],
    message: str,
) -> None:
    action_ids = (0, 1, 3, 39, 72, 73, 75)
    path = save_behavior_cloning_recovery(
        tmp_path / "corrupted-v3",
        np.zeros((1, 33), dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        action_ids,
        provenance=_recovery_provenance(),
    )
    _rewrite_recovery_archive(path, updates)

    with pytest.raises(ValueError, match=message):
        load_behavior_cloning_recovery(
            [path],
            _RecoveryPipeline(),
            action_ids,
            expected_contract=_recovery_contract(),
            expected_source_demonstration_sha256=frozenset({"b" * 64}),
        )


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


def test_recurrent_behavior_cloning_policy_acts_from_a_feature_history() -> None:
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

    action = policy.act(observation, deterministic=True)

    assert 0 <= action < model.action_count


def test_behavior_cloning_model_can_share_a_feedforward_simba_history_encoder() -> None:
    model = LidarBehaviorCloningModel(
        action_ids=(0, 1, 3, 39, 72, 73, 75),
        telemetry_dim=23,
        history_length=3,
        spatial_bins=4,
        encoder_hidden_dim=32,
        encoder_output_dim=32,
        simba_backbone={"hidden_dim": 32, "block_count": 1, "expansion": 2},
    )
    observation = {
        "lidar": torch.zeros((2, 3, 4, 8)),
        "lidar_mask": torch.ones((2, 3, 8), dtype=torch.bool),
        "telemetry": torch.zeros((2, 3, 23)),
    }

    logits = model(observation)

    assert logits.shape == (2, 7)
    assert isinstance(model.temporal, IdentityTemporalCore)


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


def test_behavior_cloning_horizontal_flip_supports_masked_control_history() -> None:
    observation = {
        "lidar": torch.zeros((3, 8, 8)),
        "lidar_mask": torch.ones((3, 8), dtype=torch.bool),
        "telemetry": torch.arange(147, dtype=torch.float32).reshape(3, 49),
    }

    reflected = horizontal_flip_observation(observation)

    assert torch.equal(reflected["telemetry"][..., 17], -observation["telemetry"][..., 17])
    assert torch.equal(reflected["telemetry"][..., 21], -observation["telemetry"][..., 21])
    assert torch.equal(reflected["telemetry"][..., 37], -observation["telemetry"][..., 39])
