"""Uniform, sequential, on-policy, and demonstration replay sampling."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace
from math import ceil, floor
from typing import Any

import torch

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch
from trackmaniarl.core.replay.batches import (
    _eligible_n_step_ids,
    _has_complete_n_step,
    _is_contiguous_episode,
    _is_contiguous_rollout,
    _make_batch,
    _reshape_sequence_batch,
)
from trackmaniarl.core.replay.store import _is_demo


class UniformSampler:
    """Reference sampler suitable for custom project templates and smoke tests."""

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

    def __init__(self, pipeline: Any, sequence_length: int, seed: int = 0) -> None:
        if sequence_length < 2:
            raise ValueError("SequenceSampler requires sequence_length >= 2")
        self.pipeline = pipeline
        self.sequence_length = sequence_length
        self._rng = random.Random(seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        length = request.sequence_length if request.sequence_length > 1 else self.sequence_length
        ordered = store.available_ids()
        if len(ordered) < length:
            raise RuntimeError(f"Need at least {length} transitions for sequence sampling")
        transitions = store.get(ordered)
        available = dict(zip(ordered, transitions, strict=True))
        windows: list[list[int]] = []
        for offset in range(len(ordered) - length + 1):
            indices = ordered[offset : offset + length]
            values = transitions[offset : offset + length]
            if _is_contiguous_episode(indices, values) and all(
                _has_complete_n_step(transition_id, available, request) for transition_id in indices
            ):
                windows.append(indices)
        if len(windows) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} valid sequences, replay has {len(windows)}"
            )
        selected = self._rng.sample(windows, request.batch_size)
        flattened = [transition_id for window in selected for transition_id in window]
        batch = _make_batch(
            store,
            self.pipeline,
            flattened,
            request,
            metadata={"sampling": "sequence", "sequence_length": length},
        )
        return replace(
            batch,
            data=_reshape_sequence_batch(batch.data, request.batch_size, length),
            observations=_reshape_sequence_batch(batch.observations, request.batch_size, length),
            actions=_reshape_sequence_batch(batch.actions, request.batch_size, length),
            rewards=_reshape_sequence_batch(batch.rewards, request.batch_size, length),
            next_observations=_reshape_sequence_batch(
                batch.next_observations, request.batch_size, length
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

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class OnPolicySequenceSampler:
    """Collate the latest fixed-length on-policy rollout."""

    on_policy_rollouts = True

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
