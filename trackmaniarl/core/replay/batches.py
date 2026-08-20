"""Replay batch collation, sequence validation, and n-step return construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, TrainingBatch, Transition, TransitionId
from trackmaniarl.core.pytree import tree_collate, tree_map


def _is_contiguous_episode(indices: list[TransitionId], transitions: list[Transition]) -> bool:
    episode_id = transitions[0].episode_id
    if episode_id is None:
        return False
    for previous_index, current_index, previous, current in zip(
        indices[:-1], indices[1:], transitions[:-1], transitions[1:], strict=True
    ):
        if current.episode_id != episode_id or current_index != previous_index + 1:
            return False
        if previous.step is not None and current.step != previous.step + 1:
            return False
        if previous.terminated or previous.truncated:
            return False
    return True


def _is_contiguous_rollout(indices: list[TransitionId], transitions: list[Transition]) -> bool:
    for previous_index, current_index, previous, current in zip(
        indices[:-1], indices[1:], transitions[:-1], transitions[1:], strict=True
    ):
        if current_index != previous_index + 1:
            return False
        same_episode = current.episode_id == previous.episode_id
        if same_episode and previous.step is not None and current.step != previous.step + 1:
            return False
        if same_episode and (previous.terminated or previous.truncated):
            return False
        if not same_episode and not (previous.terminated or previous.truncated):
            return False
        if not same_episode and current.step not in {0, None}:
            return False
    return True


def _eligible_n_step_ids(store: ReplayStore, request: BatchRequest) -> list[TransitionId]:
    """Return starts whose target is complete or ends at a real episode boundary."""

    incremental = getattr(store, "eligible_transition_ids", None)
    if callable(incremental):
        return cast(list[TransitionId], incremental(request.n_step))
    transition_ids = store.available_ids()
    available = dict(zip(transition_ids, store.get(transition_ids), strict=True))
    return [
        transition_id
        for transition_id in transition_ids
        if _has_complete_n_step(transition_id, available, request)
    ]


def _has_complete_n_step(
    transition_id: TransitionId,
    available: Mapping[TransitionId, Transition],
    request: BatchRequest,
) -> bool:
    """Reject live replay tails, whose future rewards have not arrived yet."""

    first = available[transition_id]
    for offset in range(request.n_step):
        candidate = available.get(transition_id + offset)
        if candidate is None or candidate.episode_id != first.episode_id:
            return False
        if (
            candidate.step is not None
            and first.step is not None
            and candidate.step != first.step + offset
        ):
            return False
        if candidate.terminated or candidate.truncated:
            return True
    return True


def _reshape_sequence_batch(value: Any, batch_size: int, sequence_length: int) -> Any:
    """Restore ``(batch, time, ...)`` layout after replay gathers contiguous IDs."""

    flattened_size = batch_size * sequence_length

    def reshape(leaf: Any) -> Any:
        if (
            hasattr(leaf, "shape")
            and hasattr(leaf, "reshape")
            and leaf.shape[:1] == (flattened_size,)
        ):
            return leaf.reshape(batch_size, sequence_length, *leaf.shape[1:])
        return leaf

    return tree_map(reshape, value)


def _history_padding_masks(histories: list[list[TransitionId]]) -> torch.Tensor:
    """Mark left-padded history positions, which repeat the first real transition."""

    sequence_length = len(histories[0])
    masks = torch.ones((len(histories), sequence_length), dtype=torch.bool)
    for row, history in enumerate(histories):
        padding = sequence_length - len(set(history))
        if padding:
            masks[row, :padding] = False
    return masks


def _reshape_batch_sequences(
    batch: TrainingBatch,
    batch_size: int,
    sequence_length: int,
    *,
    masks: torch.Tensor | None = None,
) -> TrainingBatch:
    return replace(
        batch,
        data=_reshape_sequence_batch(batch.data, batch_size, sequence_length),
        observations=_reshape_sequence_batch(batch.observations, batch_size, sequence_length),
        actions=_reshape_sequence_batch(batch.actions, batch_size, sequence_length),
        rewards=_reshape_sequence_batch(batch.rewards, batch_size, sequence_length),
        next_observations=_reshape_sequence_batch(
            batch.next_observations, batch_size, sequence_length
        ),
        terminated=_reshape_sequence_batch(batch.terminated, batch_size, sequence_length),
        truncated=_reshape_sequence_batch(batch.truncated, batch_size, sequence_length),
        bootstrap_discounts=_reshape_sequence_batch(
            batch.bootstrap_discounts, batch_size, sequence_length
        ),
        masks=masks
        if masks is not None
        else torch.ones((batch_size, sequence_length), dtype=torch.bool),
    )


def _make_batch(
    store: ReplayStore,
    pipeline: Any,
    transition_ids: list[TransitionId],
    request: BatchRequest,
    *,
    importance_weights: tuple[float, ...] | None = None,
    masks: Any = None,
    metadata: Mapping[str, Any] | None = None,
    bootstrap_stride: int = 1,
) -> TrainingBatch:
    """Build a batch whose n-step returns are derived from replay order, not batch order.

    With ``bootstrap_stride > 1`` the ids form contiguous groups of that length and
    only the last id of each group receives a full n-step return; the earlier ids
    are recurrent context whose reward fields the learner never consumes.
    """

    resolver = getattr(store, "n_step_ids", None)
    requested_ids: list[TransitionId] = []
    horizons: list[list[TransitionId]] = []
    seen: set[TransitionId] = set()
    for index, transition_id in enumerate(transition_ids):
        needs_return = bootstrap_stride == 1 or index % bootstrap_stride == bootstrap_stride - 1
        if not needs_return:
            horizon = [transition_id]
        elif callable(resolver):
            horizon = cast(list[TransitionId], resolver(transition_id, request.n_step))
        else:
            horizon = [transition_id + offset for offset in range(request.n_step)]
        horizons.append(horizon)
        for candidate in horizon:
            if candidate not in seen and store.contains(candidate):
                seen.add(candidate)
                requested_ids.append(candidate)
    available = dict(zip(requested_ids, store.get(requested_ids), strict=True))
    n_step = [
        _n_step_transition(transition_id, available, request, horizon=horizon)
        for transition_id, horizon in zip(transition_ids, horizons, strict=True)
    ]
    transitions = [item[0] for item in n_step]
    discounts = [item[1] for item in n_step]
    behavior = _behavior_metadata(transitions)
    data = pipeline.collate(transitions)
    standard = (
        data
        if isinstance(data, Mapping)
        and data.get("_trackmaniarl_batch_collated") is True
        and {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminated",
            "truncated",
        }.issubset(data)
        else None
    )
    batch_data = (
        {
            key: value
            for key, value in standard.items()
            if key
            not in {
                "_trackmaniarl_batch_collated",
                "observations",
                "actions",
                "rewards",
                "next_observations",
                "terminated",
                "truncated",
            }
        }
        if standard is not None
        else data
    )
    return TrainingBatch(
        data=batch_data,
        observations=standard["observations"]
        if standard is not None
        else tree_collate([item.observation for item in transitions]),
        actions=standard["actions"]
        if standard is not None
        else tree_collate([item.action for item in transitions]),
        rewards=standard["rewards"]
        if standard is not None
        else tree_collate([item.reward for item in transitions]),
        next_observations=standard["next_observations"]
        if standard is not None
        else tree_collate([item.next_observation for item in transitions]),
        terminated=standard["terminated"]
        if standard is not None
        else tree_collate([item.terminated for item in transitions]),
        truncated=standard["truncated"]
        if standard is not None
        else tree_collate([item.truncated for item in transitions]),
        bootstrap_discounts=tree_collate(discounts),
        transition_ids=transition_ids,
        importance_weights=tree_collate(importance_weights)
        if importance_weights is not None
        else None,
        masks=masks,
        metadata={**dict(metadata or {}), **behavior},
    )


def _behavior_metadata(transitions: list[Transition]) -> dict[str, torch.Tensor]:
    keys = {
        "behavior_log_probabilities": "_trackmaniarl_behavior_log_probability",
        "behavior_values": "_trackmaniarl_behavior_value",
        "behavior_latent_actions": "_trackmaniarl_behavior_latent_action",
    }
    result: dict[str, torch.Tensor] = {}
    for output_key, info_key in keys.items():
        values = [transition.info.get(info_key) for transition in transitions]
        if all(value is not None for value in values):
            result[output_key] = torch.stack(
                [torch.as_tensor(value, dtype=torch.float32) for value in values]
            )
    return result


def _n_step_transition(
    transition_id: TransitionId,
    available: Mapping[TransitionId, Transition],
    request: BatchRequest,
    *,
    horizon: list[TransitionId] | None = None,
) -> tuple[Transition, float]:
    """Return the episode-safe discounted n-step transition beginning at ``transition_id``."""

    first = available[transition_id]
    current = first
    reward = 0.0
    discount = 1.0
    effective_steps = 0
    terminated = False
    truncated = False
    ordered_ids = horizon or [transition_id + offset for offset in range(request.n_step)]
    for offset, current_id in enumerate(ordered_ids):
        candidate = available.get(current_id)
        if candidate is None or candidate.episode_id != first.episode_id:
            break
        if (
            candidate.step is not None
            and first.step is not None
            and candidate.step != first.step + offset
        ):
            break
        current = candidate
        reward += discount * candidate.reward
        effective_steps += 1
        terminated = candidate.terminated
        truncated = candidate.truncated
        if terminated or truncated:
            break
        discount *= request.gamma
    if effective_steps == 0:
        raise RuntimeError(f"Transition {transition_id} is no longer available")
    bootstrap_discount = 0.0 if terminated else request.gamma**effective_steps
    return (
        Transition(
            observation=first.observation,
            action=first.action,
            reward=reward,
            next_observation=current.next_observation,
            terminated=terminated,
            truncated=truncated,
            info=first.info,
            episode_id=first.episode_id,
            step=first.step,
        ),
        bootstrap_discount,
    )
