from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from trackmaniarl.trackmania.imitation_learning.model import LidarBehaviorCloningModel


@dataclass(frozen=True, slots=True)
class CheckpointComponents:
    model: LidarBehaviorCloningModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau
    scaler: Any
    dataset_fingerprint: str | None


def capture_checkpoint(components: CheckpointComponents) -> Mapping[str, Any]:
    return {
        "schema_version": "trackmaniarl-bc-checkpoint-v2",
        "model": components.model.state_dict(),
        "optimizer": components.optimizer.state_dict(),
        "scheduler": components.scheduler.state_dict(),
        "scaler": components.scaler.state_dict(),
        "policy_action_ids": components.model.action_ids,
        "dataset_fingerprint": components.dataset_fingerprint,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "accelerator": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def restore_checkpoint(components: CheckpointComponents, state: Mapping[str, Any]) -> None:
    _validate_checkpoint_contract(components, state)
    components.model.load_state_dict(state["model"])
    components.optimizer.load_state_dict(state["optimizer"])
    components.scheduler.load_state_dict(state["scheduler"])
    _restore_scaler(components, state)
    rng = state["rng"]
    if not isinstance(rng, Mapping):
        raise ValueError("checkpoint is missing RNG state")
    _restore_rng_state(rng)


def _validate_checkpoint_contract(
    components: CheckpointComponents, state: Mapping[str, Any]
) -> None:
    if state["schema_version"] != "trackmaniarl-bc-checkpoint-v2":
        raise ValueError("unsupported behavior-cloning checkpoint schema")
    saved_action_ids = state["policy_action_ids"]
    if tuple(saved_action_ids) != components.model.action_ids:
        raise ValueError("behavior-cloning checkpoint action contract does not match")
    saved_dataset = state["dataset_fingerprint"]
    if saved_dataset is not None and not isinstance(saved_dataset, str):
        raise ValueError("behavior-cloning checkpoint has an invalid dataset fingerprint")
    if (
        components.dataset_fingerprint is not None
        and saved_dataset != components.dataset_fingerprint
    ):
        raise ValueError("behavior-cloning checkpoint dataset fingerprint does not match")


def _restore_scaler(components: CheckpointComponents, state: Mapping[str, Any]) -> None:
    scaler = state["scaler"]
    if not isinstance(scaler, Mapping):
        raise ValueError("checkpoint is missing gradient scaler state")
    components.scaler.load_state_dict(dict(scaler))


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch", "accelerator"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"checkpoint RNG state is missing: {', '.join(sorted(missing))}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    accelerator = state["accelerator"]
    if torch.cuda.is_available() and accelerator:
        torch.cuda.set_rng_state_all(accelerator)
