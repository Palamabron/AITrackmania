"""Checkpoint 2.0 schema validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
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
_CHECKPOINT_FAMILIES = (
    (
        "fastest-eval",
        re.compile(r"fastest-eval-policy-(?P<policy>\d{8,})-at-update-(?P<update>\d{8,})\.pt"),
    ),
    (
        "best-eval",
        re.compile(r"best-eval-policy-(?P<policy>\d{8,})-at-update-(?P<update>\d{8,})\.pt"),
    ),
    ("distributed", re.compile(r"distributed-update-(?P<update>\d{8,})\.pt")),
    ("local", re.compile(r"update-(?P<update>\d{8,})\.pt")),
)


@dataclass(frozen=True, slots=True)
class CheckpointRetentionResult:
    family: str
    removed: tuple[Path, ...]


def prune_checkpoint_family(
    saved_path: Path, checkpoint_dir: Path, keep: int
) -> CheckpointRetentionResult:
    if isinstance(keep, bool) or keep < 1:
        raise ValueError("checkpoint retention count must be a positive integer")
    directory = _validated_checkpoint_directory(checkpoint_dir)
    saved = saved_path.resolve(strict=True)
    if saved.parent != directory:
        raise ValueError("saved checkpoint is outside the configured checkpoint directory")
    family, pattern = _checkpoint_family(saved.name)
    candidates = _family_candidates(directory, pattern)
    removed = tuple(path for _, path in candidates[:-keep])
    for path in removed:
        path.unlink()
    return CheckpointRetentionResult(family, removed)


def _validated_checkpoint_directory(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.is_symlink():
        raise ValueError("checkpoint directory must not be a symbolic link")
    directory = checkpoint_dir.resolve(strict=True)
    if not directory.is_dir() or directory.name != "checkpoints":
        raise ValueError("checkpoint retention requires an exact checkpoints directory")
    return directory


def _checkpoint_family(name: str) -> tuple[str, re.Pattern[str]]:
    for family, pattern in _CHECKPOINT_FAMILIES:
        if pattern.fullmatch(name) is not None:
            return family, pattern
    raise ValueError(f"unsupported checkpoint filename for retention: {name}")


def _family_candidates(
    directory: Path, pattern: re.Pattern[str]
) -> list[tuple[tuple[int, int], Path]]:
    candidates: list[tuple[tuple[int, int], Path]] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        _validate_checkpoint_file(path, directory)
        policy = int(match.groupdict().get("policy", 0))
        candidates.append(((int(match.group("update")), policy), path))
    return sorted(candidates)


def _validate_checkpoint_file(path: Path, directory: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint retention found an unsafe path: {path}")
    if path.resolve(strict=True).parent != directory:
        raise ValueError(f"checkpoint retention path escaped its directory: {path}")


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
    _validate_checkpoint_keys(state)
    validate_policy_checkpoint_v2(state)
    _validate_model_states(state)
    _validate_training_state(state)


def _validate_checkpoint_keys(state: Mapping[str, Any]) -> None:
    missing = _REQUIRED_KEYS - set(state)
    unexpected = set(state) - _REQUIRED_KEYS
    if missing or unexpected:
        raise ValueError(
            f"checkpoint 2.0 keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def _validate_model_states(state: Mapping[str, Any]) -> None:
    for model_name in ("online", "target"):
        modules = state[model_name]
        if not isinstance(modules, Mapping) or set(modules) != _MODULE_KEYS:
            raise ValueError(f"checkpoint {model_name} must contain all composite modules")


def _validate_training_state(state: Mapping[str, Any]) -> None:
    training = state["training"]
    if not isinstance(training, Mapping) or not {
        "update_count",
        "scaler",
        "rng",
        "schedules",
    } <= set(training):
        raise ValueError("checkpoint training state is incomplete")
