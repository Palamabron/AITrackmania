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


def validate_policy_checkpoint_v2(state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("policy evaluation requires checkpoint schema 2.0")
    if not isinstance(state.get("architecture_fingerprint"), str):
        raise ValueError("checkpoint is missing the architecture fingerprint")
    modules = state.get("online")
    if not isinstance(modules, Mapping) or set(modules) != _MODULE_KEYS:
        raise ValueError("checkpoint online must contain all composite modules")


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
    validate_policy_checkpoint_v2(state)
    for model_name in ("online", "target"):
        modules = state[model_name]
        if not isinstance(modules, Mapping) or set(modules) != _MODULE_KEYS:
            raise ValueError(f"checkpoint {model_name} must contain all composite modules")
    training = state["training"]
    if not isinstance(training, Mapping) or not {
        "update_count",
        "scaler",
        "rng",
        "schedules",
    } <= set(training):
        raise ValueError("checkpoint training state is incomplete")
