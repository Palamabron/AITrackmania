"""Checkpoint 2.0 schema validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CHECKPOINT_SCHEMA_VERSION = "2.0"
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "architecture_fingerprint",
        "online",
        "target",
        "optimizers",
        "objectives",
        "training",
        "runtime",
    }
)
_MODULE_KEYS = frozenset({"encoder", "temporal", "head", "strategy"})


def validate_checkpoint_v2(state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("resume requires checkpoint schema 2.0")
    missing = _REQUIRED_KEYS - set(state)
    unexpected = set(state) - _REQUIRED_KEYS
    if missing or unexpected:
        raise ValueError(
            f"checkpoint 2.0 keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for model_name in ("online", "target"):
        modules = state[model_name]
        if not isinstance(modules, Mapping) or set(modules) != _MODULE_KEYS:
            raise ValueError(f"checkpoint {model_name} must contain all composite modules")
    training = state["training"]
    if not isinstance(training, Mapping) or not {"update_count", "rng", "schedules"} <= set(
        training
    ):
        raise ValueError("checkpoint training state is incomplete")
