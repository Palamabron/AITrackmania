from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch

from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.protocol import transition_from_wire


def _validate_submit_payload(value: Mapping[str, Any], codec: WireCodec) -> None:
    _validate_fields(value, _SUBMIT_FIELDS, "rollout")
    _validate_submit_identity(value)
    transitions, episodes, evaluations = _submit_collections(value)
    _validate_submit_collections(transitions, episodes, evaluations)
    _validate_evaluation_snapshot(value, evaluations, codec)


def _validate_submit_identity(value: Mapping[str, Any]) -> None:
    _required_nonempty_string(value, "actor_id")
    _required_nonempty_string(value, "session_id")
    _required_integer(value, "sequence", minimum=0)
    _required_integer(value, "policy_version", minimum=-1)


def _submit_collections(
    value: Mapping[str, Any],
) -> tuple[list[Any], list[Any], list[Any]]:
    transitions = _required_list(value, "transitions")
    episodes = _required_list(value, "episodes")
    evaluations = _required_list(value, "evaluations")
    return transitions, episodes, evaluations


def _validate_submit_collections(
    transitions: list[Any], episodes: list[Any], evaluations: list[Any]
) -> None:
    for item in transitions:
        _validate_wire_transition(item)
    for summary in episodes:
        _validate_episode_summary(summary)
    for summary in evaluations:
        _validate_evaluation_summary(summary)


def _validate_evaluation_snapshot(
    value: Mapping[str, Any], evaluations: list[Any], codec: WireCodec
) -> None:
    snapshot = value["evaluation_snapshot"]
    if not isinstance(snapshot, bytes):
        raise TypeError("evaluation_snapshot must be bytes")
    if not evaluations:
        if snapshot:
            raise ValueError("evaluation_snapshot requires evaluations")
        return
    if not snapshot:
        raise ValueError("evaluations require an evaluation_snapshot")
    policy_state = codec.decode(snapshot)
    if not isinstance(policy_state, Mapping):
        raise TypeError("evaluation_snapshot must decode to a mapping")
    versions = {_required_integer(item, "policy_version", minimum=0) for item in evaluations}
    if len(versions) != 1:
        raise ValueError("evaluation_snapshot cannot cover mixed policy versions")
    _validate_finite_tree(policy_state, "evaluation_snapshot")


def _validate_wire_transition(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("transitions must contain mappings")
    _validate_transition_fields(value)
    _validate_transition_identity(value)
    _validate_transition_values(value)
    transition_from_wire(value)


def _validate_transition_fields(value: Mapping[str, Any]) -> None:
    _validate_fields(value, _TRANSITION_FIELDS, "transition")
    if not isinstance(value["terminated"], bool) or not isinstance(value["truncated"], bool):
        raise TypeError("transition terminal flags must be booleans")
    if not isinstance(value["info"], Mapping):
        raise TypeError("transition info must be a mapping")


def _validate_transition_identity(value: Mapping[str, Any]) -> None:
    episode_id = value["episode_id"]
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise TypeError("transition episode_id must be a non-empty string or null")
    step = value["step"]
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise TypeError("transition step must be a non-negative integer or null")


def _validate_transition_values(value: Mapping[str, Any]) -> None:
    _validate_numeric_tree(value["observation"], "transition observation")
    _validate_numeric_tree(value["action"], "transition action")
    _validate_numeric_tree(value["next_observation"], "transition next_observation")
    _validate_finite_number(value["reward"], "transition reward")
    projected = value["info"].get("sampling/projected_lap_time_s")
    if projected is not None:
        _validate_finite_number(projected, "projected lap time")


def _validate_episode_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("episodes must contain mappings")
    missing = ({"finished", "termination"} | _EPISODE_NUMERIC_FIELDS) - value.keys()
    if missing:
        raise ValueError(f"episode summary is missing {sorted(missing)}")
    _validate_episode_fields(value)
    steps = _required_integer(value, "steps", minimum=1)
    _validate_observability_summary(value, "episode", steps)
    _validate_finite_tree(value, "episode summary")


def _validate_episode_fields(value: Mapping[str, Any]) -> None:
    _validate_binary_flag(value["finished"], "episode finished")
    if not isinstance(value["termination"], str):
        raise TypeError("episode termination must be a string")
    for key in _EPISODE_NUMERIC_FIELDS:
        _validate_finite_number(value[key], f"episode {key}")


def _validate_evaluation_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("evaluations must contain mappings")
    _validate_binary_flag(value["finished"], "evaluation finished")
    _validate_finite_number(value["finish_time_s"], "evaluation finish_time_s")
    steps = _required_integer(value, "steps", minimum=0)
    _required_integer(value, "policy_version", minimum=0)
    _validate_observability_summary(value, "evaluation", steps)
    _validate_finite_tree(value, "evaluation summary")


def _validate_observability_summary(value: Mapping[str, Any], name: str, steps: int) -> None:
    observed = {
        key: _validate_nonnegative_number(value[key], f"{name} {key}")
        for key in _OBSERVABILITY_FIELDS
        if key in value
    }
    fraction = _skipped_frame_fraction(value, name)
    if steps == 0 and (any(observed.values()) or fraction > 0.0):
        raise ValueError(f"{name} timing and frame metrics require at least one step")
    _validate_skipped_frame_counts(observed, name)


def _skipped_frame_fraction(value: Mapping[str, Any], name: str) -> float:
    key = "telemetry_steps_with_skipped_frames_fraction"
    if key not in value:
        return 0.0
    fraction = _validate_nonnegative_number(value[key], f"{name} {key}")
    if fraction > 1.0:
        raise ValueError(f"{name} {key} must be at most one")
    return fraction


def _validate_skipped_frame_counts(observed: Mapping[str, float], name: str) -> None:
    total = observed.get("telemetry_skipped_frames_total")
    maximum = observed.get("telemetry_skipped_frames_max")
    if total is not None and not total.is_integer():
        raise ValueError(f"{name} skipped frame total must be an integer")
    if maximum is not None and not maximum.is_integer():
        raise ValueError(f"{name} skipped frame maximum must be an integer")
    if total is not None and maximum is not None and maximum > total:
        raise ValueError(f"{name} skipped frame maximum cannot exceed its total")


def _validate_nonnegative_number(value: Any, name: str) -> float:
    _validate_finite_number(value, name)
    scalar = float(value)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return scalar


def _required_list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value[key]
    if not isinstance(result, list):
        raise TypeError(f"{key} must be a list")
    return result


def _required_nonempty_string(value: Mapping[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _required_integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise TypeError(f"{key} must be an integer >= {minimum}")
    return cast(int, result)


def _validate_fields(value: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"{name} fields differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


def _validate_numeric_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        _validate_numeric_children(value.values(), name)
        return
    if isinstance(value, (tuple, list)):
        _validate_numeric_children(value, name)
        return
    if isinstance(value, (torch.Tensor, np.ndarray)):
        _validate_numeric_array(value, name)
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, float, np.number)):
        _validate_finite_number(value, name)
        return
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _validate_numeric_children(values: Any, name: str) -> None:
    for item in values:
        _validate_numeric_tree(item, name)


def _validate_numeric_array(value: torch.Tensor | np.ndarray[Any, Any], name: str) -> None:
    if value.dtype in {torch.bool, np.bool_}:
        return
    finite = (
        torch.isfinite(value).all() if isinstance(value, torch.Tensor) else np.isfinite(value).all()
    )
    if not bool(finite):
        raise ValueError(f"{name} contains non-finite values")


def _validate_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (torch.Tensor, np.ndarray, bool, int, float, np.number)):
        _validate_numeric_tree(value, name)
        return
    if value is not None and not isinstance(value, (str, bytes)):
        raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _validate_finite_number(value: Any, name: str) -> None:
    numeric = isinstance(value, (int, float, np.number, torch.Tensor))
    if isinstance(value, bool) or not numeric:
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, torch.Tensor) and value.numel() != 1:
        raise TypeError(f"{name} must be scalar")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")


def _validate_binary_flag(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        return
    _validate_finite_number(value, name)
    if float(value) not in {0.0, 1.0}:
        raise ValueError(f"{name} must be boolean or numeric zero/one")


_SUBMIT_FIELDS = frozenset(
    {
        "protocol_version",
        "fingerprint",
        "actor_id",
        "session_id",
        "sequence",
        "policy_version",
        "transitions",
        "episodes",
        "evaluations",
        "evaluation_snapshot",
    }
)

_TRANSITION_FIELDS = frozenset(
    {
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "info",
        "episode_id",
        "step",
    }
)

_EPISODE_NUMERIC_FIELDS = {
    "finish_time_s",
    "progress_pct",
    "return",
    "reward/time",
    "reward/pace",
    "reward/pbrs",
    "reward/progress",
    "reward/projected_velocity",
    "reward/projected_speed",
    "reward/steering_delta",
    "reward/collision",
    "collision/count",
    "collision/detected_count",
    "reward/terminal",
    "reward/time_attack_terminal",
    "velocity/ratio_mean",
    "velocity/ratio_max",
    "race_time_s",
    "exploration_epsilon",
}

_OBSERVABILITY_FIELDS = {
    "timing/policy_inference_ms_mean",
    "timing/policy_inference_ms_max",
    "controller_apply_ms_mean",
    "controller_apply_ms_max",
    "telemetry_wait_ms_mean",
    "telemetry_wait_ms_max",
    "telemetry_skipped_frames_total",
    "telemetry_skipped_frames_mean",
    "telemetry_skipped_frames_max",
}
