"""Compact behavior-cloning components preserve the TrackMania action contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.unit.learning._imitation_fixtures import (
    _recovery_contract,
    _recovery_save_request,
    _RecoveryContractOverrides,
    _RecoveryPipeline,
    _rewrite_recovery_archive,
)
from trackmaniarl.trackmania.imitation_learning import (
    INTERVENTION_KEY,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    RecoveryArrays,
    RecoveryContract,
    RecoveryLoadRequest,
    RecoveryMetadata,
    load_behavior_cloning_recovery,
    save_behavior_cloning_recovery,
)

_ACTION_IDS = (0, 1, 3, 39, 72, 73, 75)


def _standard_arrays() -> RecoveryArrays:
    return RecoveryArrays(
        np.zeros((1, 33), dtype=np.float32),
        np.asarray([0], dtype=np.int64),
        np.asarray([True]),
        _ACTION_IDS,
    )


def _save_standard(path: Path) -> Path:
    return save_behavior_cloning_recovery(_recovery_save_request(path, _standard_arrays()))


def _load_request(
    paths: Path | list[Path], contract: RecoveryContract | None = None, source: str = "b" * 64
) -> RecoveryLoadRequest:
    path_list = paths if isinstance(paths, list) else [paths]
    return RecoveryLoadRequest(
        path_list,
        _RecoveryPipeline(),
        _ACTION_IDS,
        contract or _recovery_contract(),
        frozenset({source}),
    )


def _weighted_recovery(path: Path) -> Path:
    arrays = RecoveryArrays(
        np.arange(3 * 33, dtype=np.float32).reshape(3, 33),
        np.asarray([0, 3, 6], dtype=np.int64),
        np.asarray([True, False, False]),
        _ACTION_IDS,
    )
    metadata = RecoveryMetadata(
        np.asarray([0.25, 3.0, 6.0], dtype=np.float32),
        np.asarray([0, 2, 4], dtype=np.int64),
        np.asarray([False, True, True]),
        np.asarray([0.1, 0.8, 1.4], dtype=np.float32),
    )
    return save_behavior_cloning_recovery(_recovery_save_request(path, arrays, metadata))


def _assert_archive_provenance(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        assert str(data["source_demonstration_sha256"].item()) == "b" * 64
        assert str(data["source_checkpoint_sha256"].item()) == "c" * 64


def _conditioned_recovery(path: Path) -> Path:
    arrays = RecoveryArrays(
        np.zeros((3, 33), dtype=np.float32),
        np.asarray([2, 4, 1], dtype=np.int64),
        np.asarray([True, False, False]),
        _ACTION_IDS,
    )
    return save_behavior_cloning_recovery(_recovery_save_request(path, arrays))


def _conditioned_load_request(path: Path) -> RecoveryLoadRequest:
    return RecoveryLoadRequest(
        [path],
        _RecoveryPipeline(),
        _ACTION_IDS,
        _recovery_contract(),
        frozenset({"b" * 64}),
        previous_action_conditioning=True,
    )


def _assert_weighted_round_trip_preserves_dagger_metadata(tmp_path: Path) -> None:
    path = _weighted_recovery(tmp_path / "weighted-recovery")

    observations = load_behavior_cloning_recovery(_load_request(path))[0].observations

    assert [float(item[SAMPLE_WEIGHT_KEY]) for item in observations] == [0.25, 3.0, 6.0]
    assert [int(item[STUDENT_ACTION_KEY]) for item in observations] == [0, 2, 4]
    assert [bool(item[INTERVENTION_KEY]) for item in observations] == [False, True, True]
    assert [float(item[STATE_ERROR_KEY]) for item in observations] == pytest.approx([0.1, 0.8, 1.4])
    _assert_archive_provenance(path)


def _assert_populates_previous_action_for_conditioned_model(tmp_path: Path) -> None:
    path = _conditioned_recovery(tmp_path / "conditioned-recovery")

    lap = load_behavior_cloning_recovery(_conditioned_load_request(path))[0]

    assert [int(item["previous_action"]) for item in lap.observations] == [7, 2, 4]


def test_recovery_round_trip_preserves_training_metadata(tmp_path: Path) -> None:
    _assert_weighted_round_trip_preserves_dagger_metadata(tmp_path)
    _assert_populates_previous_action_for_conditioned_model(tmp_path)


def test_recovery_rejects_the_same_source_twice(tmp_path: Path) -> None:
    path = _save_standard(tmp_path / "duplicate-recovery")

    with pytest.raises(ValueError, match="paths must be unique"):
        load_behavior_cloning_recovery(_load_request([path, path]))


_PROVENANCE_CASES = (
    (_recovery_contract(_RecoveryContractOverrides(map_uid="another-map")), "map UID"),
    (
        _recovery_contract(_RecoveryContractOverrides(geometry_sha256="d" * 64)),
        "geometry",
    ),
    (
        _recovery_contract(_RecoveryContractOverrides(decision_interval_ms=20.0)),
        "decision interval",
    ),
    (
        _recovery_contract(
            _RecoveryContractOverrides(action_repeat_frames=2, decision_interval_ms=None)
        ),
        "action repeat",
    ),
    (
        _recovery_contract(_RecoveryContractOverrides(decision_interval_ms=None)),
        "decision interval",
    ),
)


def _assert_rejects_incompatible_provenance(
    tmp_path: Path,
    expected_contract: RecoveryContract,
    message: str,
) -> None:
    path = _save_standard(tmp_path / "incompatible-recovery")

    with pytest.raises(ValueError, match=message):
        load_behavior_cloning_recovery(_load_request(path, expected_contract))


def test_recovery_rejects_incompatible_provenance(tmp_path: Path) -> None:
    for expected_contract, message in _PROVENANCE_CASES:
        _assert_rejects_incompatible_provenance(tmp_path, expected_contract, message)


def test_recovery_rejects_unexpected_source_demonstration(tmp_path: Path) -> None:
    path = _save_standard(tmp_path / "unexpected-source")

    with pytest.raises(ValueError, match="not present in --demo inputs"):
        load_behavior_cloning_recovery(_load_request(path, source="d" * 64))


def test_recovery_rejects_non_finite_frames_before_save(tmp_path: Path) -> None:
    frames = np.zeros((1, 33), dtype=np.float32)
    frames[0, 0] = np.nan

    with pytest.raises(ValueError, match="frames must be finite"):
        save_behavior_cloning_recovery(
            _recovery_save_request(
                tmp_path / "non-finite",
                RecoveryArrays(
                    frames,
                    np.asarray([0], dtype=np.int64),
                    np.asarray([True]),
                    (0, 1, 3, 39, 72, 73, 75),
                ),
            )
        )


def _assert_rejects_corrupted_archive(
    tmp_path: Path,
    updates: dict[str, np.ndarray | None],
    message: str,
) -> None:
    path = _save_standard(tmp_path / "corrupted-v3")
    _rewrite_recovery_archive(path, updates)

    with pytest.raises(ValueError, match=message):
        load_behavior_cloning_recovery(_load_request(path))


def test_recovery_rejects_corrupted_v3_archive(tmp_path: Path) -> None:
    cases = (
        ({"sample_weight": None}, "missing metadata"),
        (
            {"action_repeat_frames": np.asarray(1.5, dtype=np.float64)},
            "action repeat must be an integer",
        ),
        (
            {"frames": np.full((1, 33), np.inf, dtype=np.float32)},
            "non-finite values",
        ),
    )
    for updates, message in cases:
        _assert_rejects_corrupted_archive(tmp_path, updates, message)
