"""Replay sampling with an explicit demonstration ratio."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, NotRequired, TypedDict, Unpack

from trackmaniarl.core.contracts import ReplayStore
from trackmaniarl.core.data import BatchRequest, PriorityUpdate, TrainingBatch
from trackmaniarl.core.replay.batches import _BatchBuild, _eligible_n_step_ids, _make_batch
from trackmaniarl.core.replay.store import _is_demo


class DemoMixSampler:
    """Uniform sampler with explicit, bounded demonstration mixing."""

    supports_sequence_sampling = False

    def __init__(self, pipeline: Any, **options: Unpack[_DemoMixOptions]) -> None:
        config = _DemoMixConfig.from_options(options)
        if not 0.0 <= config.min_demo_fraction <= config.max_demo_fraction <= 1.0:
            raise ValueError("demo fractions must satisfy 0 <= min <= max <= 1")
        self.pipeline = pipeline
        self.min_demo_fraction = config.min_demo_fraction
        self.max_demo_fraction = config.max_demo_fraction
        self._rng = random.Random(config.seed)

    def sample(self, store: ReplayStore, request: BatchRequest) -> TrainingBatch:
        if request.sequence_length != 1:
            raise ValueError("DemoMixSampler supports sequence_length=1")
        transition_ids = _eligible_n_step_ids(store, request)
        if len(transition_ids) < request.batch_size:
            raise RuntimeError(
                f"Need {request.batch_size} transitions, replay has {len(transition_ids)}"
            )
        partition = self._partition(store, transition_ids)
        chosen, demo_count = self._choose(partition, request.batch_size)
        self._rng.shuffle(chosen)
        metadata = {"sampling": "demo_mix", "demo_fraction": demo_count / request.batch_size}
        build = _BatchBuild(store, self.pipeline, chosen, request, metadata=metadata)
        return _make_batch(build)

    def _partition(self, store: ReplayStore, transition_ids: list[int]) -> _DemoPartition:
        demos = self._demo_ids(store, transition_ids)
        demo_indices = set(demos)
        online = [item for item in transition_ids if item not in demo_indices]
        return _DemoPartition(demos, online)

    def _choose(self, partition: _DemoPartition, batch_size: int) -> tuple[list[int], int]:
        demo_count = self._demo_count(len(partition.demos), batch_size)
        online_count = batch_size - demo_count
        if len(partition.online) < online_count:
            raise RuntimeError(
                f"Need {online_count} online transitions, replay has {len(partition.online)}"
            )
        chosen = self._rng.sample(partition.demos, demo_count)
        chosen.extend(self._rng.sample(partition.online, online_count))
        return chosen, demo_count

    @staticmethod
    def _demo_ids(store: ReplayStore, transition_ids: list[int]) -> list[int]:
        flags = getattr(store, "demo_flags", None)
        if callable(flags):
            return [
                transition_id
                for transition_id, is_demo in zip(
                    transition_ids, flags(transition_ids), strict=True
                )
                if is_demo
            ]
        items = store.get(transition_ids)
        return [
            transition_id
            for transition_id, value in zip(transition_ids, items, strict=True)
            if _is_demo(value.info)
        ]

    def _demo_count(self, available: int, batch_size: int) -> int:
        minimum = ceil(self.min_demo_fraction * batch_size)
        maximum = floor(self.max_demo_fraction * batch_size)
        count = min(maximum, available)
        if count < minimum:
            raise RuntimeError(
                f"Need {minimum} demo transitions for this batch, replay has {available}"
            )
        return count

    def update_priorities(self, update: PriorityUpdate) -> None:
        del update

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self._rng.getstate()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._rng.setstate(state["rng"])


class _DemoMixOptions(TypedDict):
    min_demo_fraction: NotRequired[float]
    max_demo_fraction: NotRequired[float]
    seed: NotRequired[int]


@dataclass(frozen=True, slots=True)
class _DemoMixConfig:
    min_demo_fraction: float
    max_demo_fraction: float
    seed: int

    @classmethod
    def from_options(cls, options: _DemoMixOptions) -> _DemoMixConfig:
        return cls(
            min_demo_fraction=options.get("min_demo_fraction", 0.0),
            max_demo_fraction=options.get("max_demo_fraction", 1.0),
            seed=options.get("seed", 0),
        )


@dataclass(frozen=True, slots=True)
class _DemoPartition:
    demos: list[int]
    online: list[int]
