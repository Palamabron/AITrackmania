"""Batch concatenation utilities for training_offline."""

from typing import Any

import torch


def _concat_batches(batches: list[Any]) -> Any:
    """Concatenate multiple training batches along the batch dimension (dim 0).

    Each batch has the same structure as from memory.sample(): (obs, actions, rewards,
    next_obs, dones, ...) where obs/next_obs may be tuples of tensors. Used when
    BATCHES_PER_STEP > 1 to run multiple R2D2 batches through the model in one step.

    Examples of structure mismatch that trigger errors:
        - Worker A has USE_IMAGES=True (obs tuple length 3), Worker B has False (length 2)
        - Corrupted network packet caused truncated observation tuple
        - Mixed worker configurations with different TRACK_CURVATURE_OBS settings

    Args:
        batches: List of batch tuples from memory.sample(), all must have identical structure

    Returns:
        Single batch with all samples concatenated along dim 0

    Raises:
        ValueError: When batch structures don't match across samples (different top-level length)
        RuntimeError: When tuple lengths differ (avoids silent data corruption). This indicates
            incompatible worker configurations or corrupted data in the replay buffer.
    """
    if len(batches) == 1:
        return batches[0]
    n_top = len(batches[0])
    for bi, b in enumerate(batches):
        if len(b) != n_top:
            raise ValueError(
                f"_concat_batches: batch structure mismatch: batch 0 has {n_top} "
                f"elements, batch {bi} has {len(b)}. Ensure all replay samples have "
                "the same format (e.g. same obs tuple length, no mixed worker configs)."
            )
    out: list[Any] = []
    for i in range(n_top):
        elem = batches[0][i]
        if isinstance(elem, (list, tuple)):
            n_inner = min(len(b[i]) for b in batches)
            n_inner_max = max(len(b[i]) for b in batches)
            if n_inner != len(elem) or n_inner_max != len(elem):
                raise RuntimeError(
                    f"_concat_batches: tuple length mismatch at index {i}: batch 0 has "
                    f"{len(elem)} elements, min across batches is {n_inner}. Refusing to "
                    "truncate (would silently corrupt training). Ensure all workers use "
                    "the same observation format (e.g. USE_IMAGES) and no corrupted packets. "
                    "Timeouts and validation in retrieve_data() plus interface handling of "
                    "telemetry_invalid/position_patched are the first line of defense against "
                    "corrupted samples entering the replay buffer."
                )
            out.append(
                type(elem)(torch.cat([b[i][j] for b in batches], dim=0) for j in range(n_inner))
            )
        elif isinstance(elem, torch.Tensor):
            out.append(torch.cat([b[i] for b in batches], dim=0))
        elif isinstance(elem, dict):
            merged: dict[str, Any] = {}
            for key in elem:
                vals = [b[i][key] for b in batches]
                if isinstance(vals[0], torch.Tensor):
                    merged[key] = torch.cat(vals, dim=0)
                elif isinstance(vals[0], (bool, int, float)):
                    merged[key] = vals[0]
                else:
                    merged[key] = vals[0]
            out.append(merged)
        else:
            out.append(torch.cat([torch.as_tensor(b[i]) for b in batches], dim=0))
    return type(batches[0])(out)
