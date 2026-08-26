"""Uniform, sequential, on-policy, and demonstration replay sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, floor
from typing import Any

import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch, Transition
from trackmaniarl.core.pytree import tree_collate
from trackmaniarl.core.replay.batches import (
    _eligible_n_step_ids,
    _has_complete_n_step,
    _is_contiguous_rollout,
    _make_batch,
    _n_step_horizon,
    _reshape_sequence_batch,
    _sequence_target_observation_histories,
)
from trackmaniarl.core.replay.store import _is_demo


class UniformSampler:
    """Reference sampler suitable for custom project templates and smoke tests."""

    supports_sequence_sampling = False

    def __init__(self, pipeline: Any, seed: int = 0) -> None:
        self.pipeline = pipeline
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("UniformSampler supports sequence_length=1; use a sequence sampler")
        fast_sample = getattr(store, "sample_eligible_ids", None)
        if callable(fast_sample):
            transition_ids = fast_sample(request.n_step, request.batch_size, self._rng)
        else:
            transition_ids = _eligible_n_step_ids(store, request)
            if len(transition_ids) < request.batch_size:
                raise RuntimeError(
                    f"Need {request.batch_size} complete n-step transitions, replay has "
                    f"{len(transition_ids)}"
                )
            transition_ids = self._rng.sample(transition_ids, request.batch_size)
        return _make_batch(store, self.pipeline, transition_ids, request)

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update  # Uniform sampling intentionally ignores priority feedback.

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class SequenceSampler:
    """Samples only contiguous transitions from one identified episode."""

    supports_sequence_sampling = True

    def __init__(self, pipeline: Any, sequence_length: int, seed: int = 0) -> None:
        if sequence_length < 2:
            raise ValueError("SequenceSampler requires sequence_length >= 2")
        self.pipeline = pipeline
        self.sequence_length = sequence_length
        self._rng = random.Random(seed)
        self._cached_store: ReplayStore | None = None
        self._cached_request: tuple[int, int] | None = None
        self._cached_revision: int | None = None
        self._cached_ids: tuple[int, ...] | None = None
        self._cached_starts: list[int] = []

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        length = request.sequence_length if request.sequence_length > 1 else self.sequence_length
        starts = self._window_starts(store, request, length)
        if len(starts) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} valid sequences, replay has {len(starts)}"
            )
        selected = [
            list(range(start, start + length))
            for start in self._rng.sample(starts, request.batch_size)
        ]
        flattened = [transition_id for window in selected for transition_id in window]
        final_ids = [window[-1] for window in selected]
        histories, horizons = _selected_sequence_transitions(store, selected, request)
        next_histories = _sequence_target_observation_histories(histories, horizons, length)
        batch = _make_batch(
            store,
            self.pipeline,
            flattened,
            request,
            metadata={
                "sampling": "sequence",
                "sequence_length": length,
                "n_step": request.n_step,
                "gamma": request.gamma,
                "priority_transition_ids": tuple(final_ids),
            },
            bootstrap_stride=length,
        )
        return replace(
            batch,
            data=_reshape_sequence_batch(batch.data, request.batch_size, length),
            observations=_reshape_sequence_batch(batch.observations, request.batch_size, length),
            actions=_reshape_sequence_batch(batch.actions, request.batch_size, length),
            rewards=_reshape_sequence_batch(batch.rewards, request.batch_size, length),
            next_observations=_reshape_sequence_batch(
                tree_collate(
                    [observation for history in next_histories for observation in history]
                ),
                request.batch_size,
                length,
            ),
            terminated=_reshape_sequence_batch(batch.terminated, request.batch_size, length),
            truncated=_reshape_sequence_batch(batch.truncated, request.batch_size, length),
            bootstrap_discounts=_reshape_sequence_batch(
                batch.bootstrap_discounts, request.batch_size, length
            ),
            masks=torch.ones((request.batch_size, length), dtype=torch.bool),
            metadata={
                **batch.metadata,
                **{
                    key: value.reshape(request.batch_size, length, *value.shape[1:])
                    for key, value in batch.metadata.items()
                    if key
                    in {
                        "behavior_log_probabilities",
                        "behavior_values",
                        "behavior_latent_actions",
                    }
                    and isinstance(value, torch.Tensor)
                },
            },
        )

    def _window_starts(
        self,
        store: ReplayStore,
        request: BatchRequest,
        length: int,
    ) -> list[int]:
        request_key = (length, request.n_step)
        ordered: list[int] | None = None
        revision: int | None = None
        identifier: tuple[int, ...] | None = None
        changes_since = getattr(store, "changes_since", None)
        same_index = self._cached_store is store and self._cached_request == request_key
        if callable(changes_since):
            revision, changes = changes_since(self._cached_revision if same_index else None)
            if same_index and changes == []:
                return self._cached_starts
        else:
            ordered = store.available_ids()
            identifier = tuple(ordered)
            if same_index and identifier == self._cached_ids:
                return self._cached_starts
        ordered = store.available_ids() if ordered is None else ordered
        if len(ordered) < length:
            raise RuntimeError(f"Need at least {length} transitions for sequence sampling")
        transitions = store.get(ordered)
        available = dict(zip(ordered, transitions, strict=True))
        starts = _sequence_window_starts(ordered, transitions, available, request, length)
        self._cached_store = store
        self._cached_request = request_key
        self._cached_revision = revision
        self._cached_ids = identifier
        self._cached_starts = starts
        return starts

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])
        self._cached_store = None
        self._cached_request = None
        self._cached_revision = None
        self._cached_ids = None
        self._cached_starts = []


def _sequence_window_starts(
    ordered: list[int],
    transitions: list[Transition],
    available: Mapping[int, Transition],
    request: BatchRequest,
    length: int,
) -> list[int]:
    eligible = [
        _has_complete_n_step(transition_id, available, request) for transition_id in ordered
    ]
    starts: list[int] = []
    contiguous = 1
    ineligible = 0
    for index, transition in enumerate(transitions):
        if index:
            contiguous = (
                contiguous + 1
                if _extends_episode(
                    ordered[index - 1],
                    ordered[index],
                    transitions[index - 1],
                    transition,
                )
                else 1
            )
        ineligible += int(not eligible[index])
        if index >= length:
            ineligible -= int(not eligible[index - length])
        if index >= length - 1 and contiguous >= length and not ineligible:
            starts.append(ordered[index - length + 1])
    return starts


def _extends_episode(
    previous_id: int,
    transition_id: int,
    previous: Transition,
    transition: Transition,
) -> bool:
    return bool(
        transition.episode_id is not None
        and transition.episode_id == previous.episode_id
        and transition_id == previous_id + 1
        and (previous.step is None or transition.step == previous.step + 1)
        and not previous.terminated
        and not previous.truncated
    )


def _selected_sequence_transitions(
    store: ReplayStore,
    selected: list[list[int]],
    request: BatchRequest,
) -> tuple[list[list[Transition]], list[list[Transition]]]:
    final_ids = [window[-1] for window in selected]
    resolver = getattr(store, "n_step_ids", None)
    horizon_ids = (
        [resolver(transition_id, request.n_step) for transition_id in final_ids]
        if callable(resolver)
        else [
            [
                candidate
                for candidate in range(transition_id, transition_id + request.n_step)
                if store.contains(candidate)
            ]
            for transition_id in final_ids
        ]
    )
    requested = list(
        dict.fromkeys(
            transition_id for group in [*selected, *horizon_ids] for transition_id in group
        )
    )
    available = dict(zip(requested, store.get(requested), strict=True))
    histories = [[available[transition_id] for transition_id in window] for window in selected]
    horizons = (
        [[available[transition_id] for transition_id in horizon] for horizon in horizon_ids]
        if callable(resolver)
        else [_n_step_horizon(transition_id, available, request) for transition_id in final_ids]
    )
    return histories, horizons


class OnPolicySequenceSampler:
    """Collate the latest fixed-length on-policy rollout."""

    on_policy_rollouts = True
    supports_sequence_sampling = True

    def __init__(self, pipeline: Any, seed: int = 0) -> None:
        self.pipeline = pipeline
        self.seed = seed

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.batch_size != 1 or request.n_step != 1:
            raise ValueError("OnPolicySequenceSampler requires batch_size=1 and n_step=1")
        ordered = store.available_ids()
        if not ordered:
            raise RuntimeError("On-policy replay is empty")
        if len(ordered) < request.sequence_length:
            raise RuntimeError(
                f"On-policy replay has {len(ordered)} transitions, need {request.sequence_length}"
            )
        transition_ids = ordered[-request.sequence_length :]
        values = store.get(transition_ids)
        if not _is_contiguous_rollout(transition_ids, values):
            raise RuntimeError("Latest on-policy rollout is not contiguous")
        batch = _make_batch(
            store,
            self.pipeline,
            transition_ids,
            request,
            metadata={"sampling": "on_policy", "sequence_length": request.sequence_length},
        )
        length = request.sequence_length
        return replace(
            batch,
            data=_reshape_sequence_batch(batch.data, 1, length),
            observations=_reshape_sequence_batch(batch.observations, 1, length),
            actions=_reshape_sequence_batch(batch.actions, 1, length),
            rewards=_reshape_sequence_batch(batch.rewards, 1, length),
            next_observations=_reshape_sequence_batch(batch.next_observations, 1, length),
            terminated=_reshape_sequence_batch(batch.terminated, 1, length),
            truncated=_reshape_sequence_batch(batch.truncated, 1, length),
            bootstrap_discounts=_reshape_sequence_batch(batch.bootstrap_discounts, 1, length),
            masks=torch.ones((1, length), dtype=torch.bool),
            metadata={
                **batch.metadata,
                **{
                    key: value.reshape(1, length, *value.shape[1:])
                    for key, value in batch.metadata.items()
                    if key
                    in {
                        "behavior_log_probabilities",
                        "behavior_values",
                        "behavior_latent_actions",
                    }
                    and isinstance(value, torch.Tensor)
                },
            },
        )

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.seed = int(state["seed"])


class DemoMixSampler:
    """Uniform sampler with explicit, bounded demonstration mixing."""

    supports_sequence_sampling = False

    def __init__(
        self,
        pipeline: Any,
        *,
        min_demo_fraction: float = 0.0,
        max_demo_fraction: float = 1.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= min_demo_fraction <= max_demo_fraction <= 1.0:
            raise ValueError("demo fractions must satisfy 0 <= min <= max <= 1")
        self.pipeline = pipeline
        self.min_demo_fraction = min_demo_fraction
        self.max_demo_fraction = max_demo_fraction
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("DemoMixSampler supports sequence_length=1")
        transition_ids = _eligible_n_step_ids(store, request)
        if len(transition_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
            )
        flags = getattr(store, "demo_flags", None)
        if callable(flags):
            demos = [
                transition_id
                for transition_id, is_demo in zip(
                    transition_ids, flags(transition_ids), strict=True
                )
                if is_demo
            ]
        else:
            items = store.get(transition_ids)
            demos = [
                transition_id
                for transition_id, value in zip(transition_ids, items, strict=True)
                if _is_demo(value.info)
            ]
        demo_indices = set(demos)
        online = [
            transition_id for transition_id in transition_ids if transition_id not in demo_indices
        ]
        minimum = ceil(self.min_demo_fraction * request.batch_size)
        maximum = floor(self.max_demo_fraction * request.batch_size)
        demo_count = min(maximum, len(demos))
        if demo_count < minimum:
            raise RuntimeError(
                f"Need {minimum} demo transitions for this batch, replay has {len(demos)}"
            )
        online_count = request.batch_size - demo_count
        if len(online) < online_count:
            raise RuntimeError(f"Need {online_count} online transitions, replay has {len(online)}")
        chosen = self._rng.sample(demos, demo_count) + self._rng.sample(online, online_count)
        self._rng.shuffle(chosen)
        return _make_batch(
            store,
            self.pipeline,
            chosen,
            request,
            metadata={"sampling": "demo_mix", "demo_fraction": demo_count / request.batch_size},
        )

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])
