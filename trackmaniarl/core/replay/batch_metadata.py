"""Behavior-policy metadata extracted from replay transitions."""

from __future__ import annotations

import torch

from trackmaniarl.core.data import Transition

_BEHAVIOR_KEYS = {
    "behavior_log_probabilities": "_trackmaniarl_behavior_log_probability",
    "behavior_values": "_trackmaniarl_behavior_value",
    "behavior_latent_actions": "_trackmaniarl_behavior_latent_action",
}


def _behavior_metadata(transitions: list[Transition]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for output_key, info_key in _BEHAVIOR_KEYS.items():
        values = [transition.info.get(info_key) for transition in transitions]
        if all(value is not None for value in values):
            result[output_key] = torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in values]
            )
    return result
