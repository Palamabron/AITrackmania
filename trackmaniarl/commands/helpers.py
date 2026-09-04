from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.environment import TrackmaniaEnvironmentConfig
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.imitation_learning import RecoveryContract


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _recovery_contract(
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> RecoveryContract:
    return RecoveryContract(
        map_uid=geometry.map_uid,
        geometry_sha256=geometry.sha256,
        action_repeat_frames=config.action_repeat_frames,
        decision_interval_ms=config.decision_interval_ms,
        control_alignment="frame_start",
    )


def _compact_action_ids(spec: RunSpec) -> tuple[int, ...]:
    environment = spec.components.environment
    if environment is None:
        raise ValueError("behavior cloning requires components.environment")
    config = environment.kwargs["config"]
    if not isinstance(config, dict):
        raise TypeError("environment.config must be a mapping")
    raw_ids = config["compact_action_ids"]
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("behavior cloning requires environment.config.compact_action_ids")
    return tuple(int(action) for action in raw_ids)


def _learner_context(run: Any) -> dict[str, Any]:
    return {"seed": run.spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}


def _training_learner_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "learner"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"training checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["schema_version"] not in {"1.0", "2.0"}:
        raise ValueError("unsupported training checkpoint schema")
    learner = checkpoint["learner"]
    if not isinstance(learner, Mapping):
        raise TypeError("training checkpoint learner state must be a mapping")
    return learner


def _behavior_policy_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    required = {"schema_version", "learner"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"behavior-cloning checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["schema_version"] != "trackmaniarl-bc-policy-v2":
        raise ValueError("unsupported behavior-cloning policy checkpoint schema")
    learner = checkpoint["learner"]
    if not isinstance(learner, Mapping):
        raise TypeError("behavior-cloning policy state must be a mapping")
    return learner
