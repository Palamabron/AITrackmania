"""Shape recurrent replay batches for sequence learners."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from trackmaniarl.core.data import TrainingBatch
from trackmaniarl.core.replay.batches import _reshape_sequence_batch


@dataclass(frozen=True, slots=True)
class _SequenceBatchShape:
    batch: TrainingBatch
    batch_size: int
    length: int
    next_observations: Any


def _reshape_training_batch(shape: _SequenceBatchShape) -> TrainingBatch:
    batch = shape.batch
    return replace(
        batch,
        data=_reshape_value(shape, batch.data),
        observations=_reshape_value(shape, batch.observations),
        actions=_reshape_value(shape, batch.actions),
        rewards=_reshape_value(shape, batch.rewards),
        next_observations=_reshape_value(shape, shape.next_observations),
        terminated=_reshape_value(shape, batch.terminated),
        truncated=_reshape_value(shape, batch.truncated),
        bootstrap_discounts=_reshape_value(shape, batch.bootstrap_discounts),
        masks=torch.ones((shape.batch_size, shape.length), dtype=torch.bool),
        metadata=_reshaped_metadata(shape),
    )


def _reshape_value(shape: _SequenceBatchShape, value: Any) -> Any:
    return _reshape_sequence_batch(value, shape.batch_size, shape.length)


def _reshaped_metadata(shape: _SequenceBatchShape) -> dict[str, Any]:
    recurrent = {
        key: value.reshape(shape.batch_size, shape.length, *value.shape[1:])
        for key, value in shape.batch.metadata.items()
        if key in _RECURRENT_METADATA and isinstance(value, torch.Tensor)
    }
    return {**shape.batch.metadata, **recurrent}


_RECURRENT_METADATA = {
    "behavior_log_probabilities",
    "behavior_values",
    "behavior_latent_actions",
}
