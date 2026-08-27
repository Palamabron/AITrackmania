from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.data import BatchRequest, TrainingBatch
from trackmaniarl.core.runtime import ResolvedRun

ROLLOUT_QUEUE_MAXSIZE = 64


@dataclass(slots=True)
class _Counters:
    transitions: int = 0
    episodes: int = 0
    finishes: int = 0
    best_finish_time_s: float = 0.0
    evaluations: int = 0
    evaluation_finishes: int = 0
    evaluation_bucket_finishes: dict[str, int] = field(default_factory=dict)
    updates: int = 0
    update_credit: float = 0.0
    journal_applied_frontier: int = 0
    policy_version: int = 0
    actor_sequences: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedBatch:
    batch: TrainingBatch
    preparation_s: float


class _BatchPrefetcher:
    """Checkpoint-safe batch source without speculative sampler or transfer state."""

    def __init__(self, run: ResolvedRun) -> None:
        self.run = run

    def next(self, request: BatchRequest) -> tuple[TrainingBatch, float, float]:
        prepared = self._prepare(request)
        return prepared.batch, prepared.preparation_s, 0.0

    def _prepare(self, request: BatchRequest) -> _PreparedBatch:
        started = perf_counter()
        batch = self.run.sampler.sample(self.run.replay_store, request)
        return _PreparedBatch(batch, perf_counter() - started)

    def close(self) -> None:
        return


class _MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.maximums: dict[str, float] = {}

    def add(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            numeric = float(value)
            if key.endswith("_max"):
                self.maximums[key] = max(self.maximums.get(key, numeric), numeric)
            else:
                self.values[key] = self.values.get(key, 0.0) + numeric
                self.counts[key] = self.counts.get(key, 0) + 1

    def flush(self) -> dict[str, float]:
        if not self.values and not self.maximums:
            return {}
        output = {key: value / self.counts[key] for key, value in self.values.items()}
        output.update(self.maximums)
        self.values.clear()
        self.counts.clear()
        self.maximums.clear()
        return output


@dataclass(frozen=True, slots=True)
class _CheckpointWrite:
    state: Mapping[str, Any]
    path: Path
    on_saved: Callable[[], None] | None
    on_failed: Callable[[BaseException], None] | None


@dataclass(frozen=True, slots=True)
class _RolloutRejection:
    reason: str
    details: Mapping[str, object] = field(default_factory=dict)


class _AsyncCheckpointWriter:
    def __init__(self, codec: Any) -> None:
        self.codec = codec
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trackmaniarl-checkpoint"
        )
        self.pending: Any = None

    def submit(self, request: _CheckpointWrite) -> None:
        self.wait()
        self.pending = self.executor.submit(self._save, request)

    def _save(self, request: _CheckpointWrite) -> None:
        try:
            self.codec.save(request.state, request.path)
            sync_checkpoint_path(request.path)
            if request.on_saved is not None:
                request.on_saved()
        except BaseException as exc:
            if request.on_failed is not None:
                request.on_failed(exc)
            raise

    def wait(self) -> None:
        pending = self.pending
        self.pending = None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        self.wait()
        self.executor.shutdown(wait=True, cancel_futures=False)


def state_dict(component: object) -> Mapping[str, object]:
    method = getattr(component, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{type(component).__name__}.state_dict() must return a mapping")
    return cast(Mapping[str, object], state)


def load_state_dict(component: object, state: object) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint component state must be a mapping")
    method = getattr(component, "load_state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no load_state_dict()")
    method(state)


def snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_value(item) for item in value)
    return deepcopy(value)
