from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, TypedDict

from trackmaniarl.core.data import TrainingBatch


class BatchCore(TypedDict):
    data: Any
    observations: Any
    actions: Any
    rewards: Any
    next_observations: Any
    terminated: Any
    truncated: Any
    bootstrap_discounts: Any


def transform_batch(batch: TrainingBatch, transform: Callable[[Any], Any]) -> TrainingBatch:
    core = transform_batch_core(batch, transform)
    importance = transform_optional(batch.importance_weights, transform)
    masks = transform_optional(batch.masks, transform)
    return replace(batch, **core, importance_weights=importance, masks=masks)


def transform_batch_core(batch: TrainingBatch, transform: Callable[[Any], Any]) -> BatchCore:
    return {
        "data": transform(batch.data),
        "observations": transform(batch.observations),
        "actions": transform(batch.actions),
        "rewards": transform(batch.rewards),
        "next_observations": transform(batch.next_observations),
        "terminated": transform(batch.terminated),
        "truncated": transform(batch.truncated),
        "bootstrap_discounts": transform(batch.bootstrap_discounts),
    }


def transform_optional(value: Any | None, transform: Callable[[Any], Any]) -> Any | None:
    return transform(value) if value is not None else None
