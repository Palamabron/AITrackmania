from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from trackmaniarl.trackmania.imitation_learning._data_types import (
    INTERVENTION_KEY,
    RECOVERY_DATASET_FORMAT,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    RecoveryContract,
    RecoveryProvenance,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import RecoveryArchive
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoveryArrays as _RecoveryArrays,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoveryMetadata as _RecoveryMetadata,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoveryReadRequest as _RecoveryReadRequest,
)
from trackmaniarl.trackmania.imitation_learning._recovery_types import (
    RecoverySaveRequest as _RecoverySaveRequest,
)


def save_behavior_cloning_recovery(request: _RecoverySaveRequest) -> Path:
    """Persist DAgger states with compact expert labels."""

    _validate_recovery_arrays(request.arrays)
    _validate_recovery_metadata(request.metadata, request.arrays)
    target = _recovery_path(request.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **_recovery_archive_values(request))
    return target


def _recovery_path(path: str | Path) -> Path:
    target = Path(path)
    return target if target.suffix.lower() == ".npz" else target.with_suffix(".npz")


def _recovery_archive_values(request: _RecoverySaveRequest) -> dict[str, Any]:
    contract = request.provenance.contract
    arrays = request.arrays
    values = _recovery_metadata_values(request.metadata, arrays)
    return {
        "format": np.asarray(RECOVERY_DATASET_FORMAT),
        "map_uid": np.asarray(contract.map_uid),
        "geometry_sha256": np.asarray(contract.geometry_sha256),
        "action_repeat_frames": np.asarray(contract.action_repeat_frames, dtype=np.int32),
        "decision_interval_ms": np.asarray(contract.decision_interval_ms or 0.0),
        "control_alignment": np.asarray(contract.control_alignment),
        "source_demonstration_sha256": np.asarray(request.provenance.source_demonstration_sha256),
        "source_checkpoint_sha256": np.asarray(request.provenance.source_checkpoint_sha256 or ""),
        "frames": arrays.frames.astype(np.float32, copy=False),
        "labels": arrays.labels.astype(np.int64, copy=False),
        "episode_starts": arrays.episode_starts.astype(np.bool_, copy=False),
        "action_ids": np.asarray(arrays.action_ids, dtype=np.int64),
        **values,
    }


def _recovery_metadata_values(
    metadata: _RecoveryMetadata, arrays: _RecoveryArrays
) -> dict[str, np.ndarray]:
    count = len(arrays.frames)
    return {
        "sample_weight": _metadata_or_default(
            metadata.sample_weights, np.ones(count, dtype=np.float32)
        ),
        "student_action": _metadata_or_default(
            metadata.student_actions, np.full(count, len(arrays.action_ids), dtype=np.int64)
        ),
        "intervention": _metadata_or_default(
            metadata.interventions, np.zeros(count, dtype=np.bool_)
        ),
        "state_error": _metadata_or_default(
            metadata.state_errors, np.zeros(count, dtype=np.float32)
        ),
    }


def read_recovery_archive(request: _RecoveryReadRequest) -> RecoveryArchive:
    with np.load(request.path, allow_pickle=False) as data:
        _validate_archive_identity(data, request.path)
        provenance = _load_recovery_provenance(data, request.path)
        _validate_recovery_source(request, provenance)
        arrays = _load_recovery_arrays(data, request)
        metadata = _load_recovery_metadata(data, arrays)
    _validate_loaded_arrays(arrays, request.path)
    return RecoveryArchive(
        request.path,
        arrays.frames,
        arrays.labels,
        arrays.episode_starts,
        metadata,
    )


def _validate_recovery_source(
    request: _RecoveryReadRequest, provenance: RecoveryProvenance
) -> None:
    _validate_recovery_contract(provenance.contract, request.expected_contract, request.path)
    if provenance.source_demonstration_sha256 not in request.expected_source_hashes:
        raise ValueError(
            f"recovery source demonstration is not present in --demo inputs: {request.path}"
        )


def _load_recovery_arrays(data: Any, request: _RecoveryReadRequest) -> _RecoveryArrays:
    required = {"action_ids", "frames", "labels", "episode_starts"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"recovery archive is missing data {sorted(missing)}: {request.path}")
    stored_ids = tuple(int(value) for value in data["action_ids"].tolist())
    if stored_ids != request.action_ids:
        raise ValueError(f"recovery action IDs do not match the model: {request.path}")
    return _RecoveryArrays(
        np.asarray(data["frames"], dtype=np.float32),
        np.asarray(data["labels"], dtype=np.int64),
        np.asarray(data["episode_starts"], dtype=np.bool_),
        request.action_ids,
    )


def _metadata_or_default(values: np.ndarray | None, default: np.ndarray) -> np.ndarray:
    return default if values is None else values.astype(default.dtype, copy=False)


def _validate_recovery_arrays(arrays: _RecoveryArrays) -> None:
    if arrays.frames.ndim != 2 or arrays.frames.shape[1] != 33:
        raise ValueError("recovery frames must have shape (steps, 33)")
    if not np.isfinite(arrays.frames).all():
        raise ValueError("recovery frames must be finite")
    if arrays.labels.shape != (len(arrays.frames),):
        raise ValueError("recovery labels and episode starts must match frames")
    if arrays.episode_starts.shape != (len(arrays.frames),):
        raise ValueError("recovery labels and episode starts must match frames")
    if len(arrays.frames) < 1 or not bool(arrays.episode_starts[0]):
        raise ValueError("recovery data must begin with an episode start")
    if np.any(arrays.labels < 0) or np.any(arrays.labels >= len(arrays.action_ids)):
        raise ValueError("recovery data contains an invalid compact action")


def _validate_recovery_metadata(metadata: _RecoveryMetadata, arrays: _RecoveryArrays) -> None:
    sample_count = len(arrays.frames)
    _validate_sample_weights(metadata.sample_weights, sample_count)
    _validate_student_actions(metadata.student_actions, sample_count, len(arrays.action_ids))
    if metadata.interventions is not None and metadata.interventions.shape != (sample_count,):
        raise ValueError("recovery interventions must match frames")
    _validate_state_errors(metadata.state_errors, sample_count)


def _validate_sample_weights(values: np.ndarray | None, sample_count: int) -> None:
    if values is not None and (
        values.shape != (sample_count,) or not np.isfinite(values).all() or np.any(values <= 0.0)
    ):
        raise ValueError("recovery sample weights must be finite, positive, and match frames")


def _validate_student_actions(
    values: np.ndarray | None, sample_count: int, action_count: int
) -> None:
    if values is not None and (
        values.shape != (sample_count,) or np.any(values < 0) or np.any(values >= action_count)
    ):
        raise ValueError("recovery student actions must be compact actions matching frames")


def _validate_state_errors(values: np.ndarray | None, sample_count: int) -> None:
    if values is not None and (
        values.shape != (sample_count,) or not np.isfinite(values).all() or np.any(values < 0.0)
    ):
        raise ValueError("recovery state errors must be finite, non-negative, and match frames")


def _load_recovery_metadata(data: Any, arrays: _RecoveryArrays) -> dict[str, np.ndarray]:
    required = {"sample_weight", "student_action", "intervention", "state_error"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"recovery archive is missing metadata {sorted(missing)}")
    metadata = _loaded_metadata(data, len(arrays.action_ids))
    _validate_recovery_metadata(metadata, arrays)
    return {
        SAMPLE_WEIGHT_KEY: cast(np.ndarray, metadata.sample_weights),
        STUDENT_ACTION_KEY: np.asarray(data["student_action"], dtype=np.int64),
        INTERVENTION_KEY: cast(np.ndarray, metadata.interventions),
        STATE_ERROR_KEY: cast(np.ndarray, metadata.state_errors),
    }


def _loaded_metadata(data: Any, action_count: int) -> _RecoveryMetadata:
    students = np.asarray(data["student_action"], dtype=np.int64)
    validated_students = None if bool(np.all(students == action_count)) else students
    return _RecoveryMetadata(
        np.asarray(data["sample_weight"], dtype=np.float32),
        validated_students,
        np.asarray(data["intervention"], dtype=np.bool_),
        np.asarray(data["state_error"], dtype=np.float32),
    )


def _load_recovery_provenance(data: Any, path: Path) -> RecoveryProvenance:
    _validate_provenance_keys(data, path)
    raw_action_repeat = data["action_repeat_frames"].item()
    action_repeat = _recovery_action_repeat(raw_action_repeat, path)
    try:
        contract = _loaded_contract(data, action_repeat)
        return RecoveryProvenance(
            contract,
            str(data["source_demonstration_sha256"].item()),
            str(data["source_checkpoint_sha256"].item()) or None,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"recovery archive has invalid provenance: {path}") from exc


def _validate_provenance_keys(data: Any, path: Path) -> None:
    required = {
        "map_uid",
        "geometry_sha256",
        "action_repeat_frames",
        "decision_interval_ms",
        "control_alignment",
        "source_demonstration_sha256",
        "source_checkpoint_sha256",
    }
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"recovery archive is missing provenance {sorted(missing)}: {path}")


def _recovery_action_repeat(raw_action_repeat: Any, path: Path) -> int:
    if isinstance(raw_action_repeat, (bool, np.bool_)) or not isinstance(
        raw_action_repeat, (int, np.integer)
    ):
        raise ValueError(f"recovery action repeat must be an integer: {path}")
    return int(raw_action_repeat)


def _loaded_contract(data: Any, action_repeat: int) -> RecoveryContract:
    interval_ms = float(data["decision_interval_ms"].item())
    return RecoveryContract(
        map_uid=str(data["map_uid"].item()),
        geometry_sha256=str(data["geometry_sha256"].item()),
        action_repeat_frames=action_repeat,
        decision_interval_ms=interval_ms or None,
        control_alignment=str(data["control_alignment"].item()),
    )


def _validate_recovery_contract(
    actual: RecoveryContract,
    expected: RecoveryContract,
    path: Path,
) -> None:
    _validate_contract_identity(actual, expected, path)
    _validate_contract_interval(actual, expected, path)


def _validate_contract_identity(
    actual: RecoveryContract, expected: RecoveryContract, path: Path
) -> None:
    comparisons = (
        (actual.map_uid, expected.map_uid, "map UID", "training map"),
        (actual.geometry_sha256, expected.geometry_sha256, "geometry", "feature geometry"),
        (
            actual.action_repeat_frames,
            expected.action_repeat_frames,
            "action repeat",
            "environment",
        ),
        (actual.control_alignment, expected.control_alignment, "control alignment", "dataset"),
    )
    for actual_value, expected_value, field, owner in comparisons:
        if actual_value != expected_value:
            raise ValueError(f"recovery {field} does not match the {owner}: {path}")


def _validate_contract_interval(
    actual: RecoveryContract, expected: RecoveryContract, path: Path
) -> None:
    actual_interval, expected_interval = actual.decision_interval_ms, expected.decision_interval_ms
    if actual_interval is None and expected_interval is None:
        return
    if (
        actual_interval is None
        or expected_interval is None
        or not np.isclose(actual_interval, expected_interval, rtol=0.0, atol=0.05)
    ):
        raise ValueError(f"recovery decision interval does not match the environment: {path}")


def _validate_archive_identity(data: Any, path: Path) -> None:
    if "format" not in data.files:
        raise ValueError(f"recovery archive is missing its format marker: {path}")
    format_name = str(data["format"].item())
    if format_name != RECOVERY_DATASET_FORMAT:
        raise ValueError(f"unsupported behavior-cloning recovery format: {path}")


def _validate_loaded_arrays(arrays: _RecoveryArrays, path: Path) -> None:
    frames, labels, starts = arrays.frames, arrays.labels, arrays.episode_starts
    if frames.ndim != 2 or frames.shape[1] != 33:
        raise ValueError(f"recovery frames have an invalid shape: {path}")
    if not np.isfinite(frames).all():
        raise ValueError(f"recovery frames contain non-finite values: {path}")
    if labels.shape != (len(frames),) or starts.shape != (len(frames),):
        raise ValueError(f"recovery labels and episode starts do not match frames: {path}")
    if np.any(labels < 0) or np.any(labels >= len(arrays.action_ids)):
        raise ValueError(f"recovery data contains an invalid compact action: {path}")
    boundaries = np.flatnonzero(starts)
    if not len(boundaries) or boundaries[0] != 0:
        raise ValueError(f"recovery data does not begin with an episode: {path}")
