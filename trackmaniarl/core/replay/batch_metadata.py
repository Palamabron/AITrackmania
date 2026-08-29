"""Behavior-policy metadata extracted from replay transitions."""

from __future__ import annotations

from typing import Any

import torch

from trackmaniarl.core.data import Transition

_BEHAVIOR_KEYS = {
    "behavior_log_probabilities": "_trackmaniarl_behavior_log_probability",
    "behavior_values": "_trackmaniarl_behavior_value",
    "behavior_latent_actions": "_trackmaniarl_behavior_latent_action",
}


def _behavior_metadata(transitions: list[Transition]) -> dict[str, Any]:
    result = _demonstration_metadata(transitions)
    for output_key, info_key in _BEHAVIOR_KEYS.items():
        values = [transition.info.get(info_key) for transition in transitions]
        if all(value is not None for value in values):
            result[output_key] = torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in values]
            )
    return result


def _demonstration_metadata(transitions: list[Transition]) -> dict[str, Any]:
    return {
        "demo_flags": tuple(
            bool(transition.info.get("is_demo", False)) for transition in transitions
        ),
        "demonstration_steering_switches": tuple(
            bool(transition.info.get("demonstration_steering_switch", False))
            for transition in transitions
        ),
        "demonstration_steering_switch_distances": tuple(
            int(transition.info.get("demonstration_steering_switch_distance", 1_000_000))
            for transition in transitions
        ),
    }
