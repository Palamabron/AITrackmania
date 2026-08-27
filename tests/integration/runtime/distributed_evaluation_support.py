from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Logger,
    _Pipeline,
    _SlowLearner,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_support import _Counters
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig


class _OrderedLearner(_SlowLearner):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def update(self, batch: Any) -> Mapping[str, float]:
        self.events.append("update")
        return super().update(batch)


def _distributed_components() -> dict[str, dict[str, str]]:
    return {
        "learner": {"class_path": "tests.fake:SlowLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
    }


def _drain_training() -> dict[str, float | int]:
    return {
        "total_transitions": 6,
        "batch_size": 1,
        "warmup_transitions": 1,
        "updates_per_transition": 1.0,
        "checkpoint_interval_updates": 1000,
    }


def _drain_spec(tmp_path: Path) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "drain-first",
            "artifacts_dir": str(tmp_path),
            "components": _distributed_components(),
            "training": _drain_training(),
        }
    )


def _stopped_coordinator(tmp_path: Path, events: list[str]) -> Coordinator:
    spec = _drain_spec(tmp_path)
    run = _drain_run(spec, events, tmp_path)
    stop = threading.Event()
    stop.set()
    config = CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint", external_stop=stop)
    return Coordinator(run, config)


def _drain_run(spec: RunSpec, events: list[str], tmp_path: Path) -> ResolvedRun:
    pipeline = _Pipeline()
    return ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "drain-first",
        learner=_OrderedLearner(events),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )


class _RecordingLogger:
    def __init__(self, events: list[tuple[str, dict[str, Any]]]) -> None:
        self.events = events

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del step
        self.events.append((event, dict(payload)))

    def close(self) -> None:
        return


@dataclass(slots=True)
class _CheckpointProbe:
    directory: Path
    checkpoints: list[int]

    def save(
        self,
        *,
        policy_state: Mapping[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> Path:
        del policy_state, policy_version
        self.checkpoints.append(0)
        return self.directory / "best.pt"


def _evaluation_coordinator(
    tmp_path: Path, events: list[tuple[str, dict[str, Any]]], checkpoints: list[int]
) -> Coordinator:
    coordinator = object.__new__(Coordinator)
    coordinator.run = _evaluation_run(events)
    coordinator.counters = _Counters()
    coordinator._last_ingest_at = time.monotonic()
    coordinator._rollouts = Queue()
    coordinator._lock = threading.RLock()
    coordinator._recovering = False
    coordinator._time_buckets = (40.0, 38.0, 36.0)
    coordinator._best_evaluation = None
    coordinator._evaluation_policy_states = {}
    coordinator._checkpoints = []
    coordinator._checkpoint = _CheckpointProbe(tmp_path, checkpoints).save
    return coordinator


def _evaluation_run(events: list[tuple[str, dict[str, Any]]]) -> SimpleNamespace:
    training = SimpleNamespace(
        warmup_transitions=1,
        updates_per_transition=1.0,
        evaluate_every_episodes=None,
    )
    spec = SimpleNamespace(
        distributed=SimpleNamespace(max_update_credit=512),
        evaluation=SimpleNamespace(min_finish_rate=1.0),
        training=training,
    )
    return SimpleNamespace(
        replay_store=InMemoryReplayStore(), spec=spec, logger=_RecordingLogger(events)
    )


class _EvaluationIngestor:
    def __init__(self, coordinator: Coordinator) -> None:
        self.coordinator = coordinator
        self.journal_row = 0

    def ingest(self, evaluations: list[dict[str, Any]]) -> None:
        self.journal_row += 1
        payload = {
            "actor_id": "actor",
            "session_id": "session",
            "sequence": 0,
            "policy_version": 0,
            "transitions": [],
            "episodes": [],
            "evaluations": evaluations,
            "evaluation_snapshot": b"",
        }
        self.coordinator._ingest(payload, self.journal_row)


class _EvaluationResult(StrEnum):
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class _EvaluationCase:
    finish_time_s: float
    result: _EvaluationResult
    steps: int
    policy_latency: float


def _finished_evaluation(finish_time_s: float) -> dict[str, Any]:
    case = _EvaluationCase(finish_time_s, _EvaluationResult.FINISHED, 10, 1.0)
    return _evaluation_summary(case)


def _failed_evaluation() -> dict[str, Any]:
    return _evaluation_summary(_EvaluationCase(0.0, _EvaluationResult.FAILED, 30, 3.0))


def _evaluation_summary(case: _EvaluationCase) -> dict[str, Any]:
    finished = float(case.result is _EvaluationResult.FINISHED)
    summary = {
        "finished": finished,
        "finish_time_s": case.finish_time_s,
        "policy_version": 41,
        "q_margin/start_mean": 0.5,
        "control/gas_fraction": 0.75,
        "control/brake_fraction": 0.2,
        "control/brake_tap_fraction": 0.1,
        "control/steer_abs_mean": 0.6,
        "steps": case.steps,
    }
    summary.update(_evaluation_timing(case))
    summary.update(_progress_metrics())
    return summary


def _evaluation_timing(case: _EvaluationCase) -> dict[str, float]:
    finished = case.result is _EvaluationResult.FINISHED
    return {
        "timing/policy_inference_ms_mean": case.policy_latency,
        "controller_apply_ms_mean": 2.0 if finished else 4.0,
        "telemetry_wait_ms_mean": 5.0 if finished else 7.0,
        "telemetry_skipped_frames_total": 2.0 if finished else 3.0,
        "telemetry_skipped_frames_max": 2.0 if finished else 3.0,
        "telemetry_steps_with_skipped_frames_fraction": 0.2 if finished else 0.1,
    }


def _progress_metrics() -> dict[str, float]:
    return {
        "progress_bin/90_100/action_count": 2.0,
        "progress_bin/90_100/action_entropy": 0.5,
        "progress_bin/90_100/action_coverage": 0.25,
        "progress_bin/90_100/q_margin_mean": 1.5,
        "progress_bin/90_100/q_margin_min": 0.75,
        "progress_bin/90_100/q_max_mean": 3.0,
    }


def _queue_backlog(coordinator: Coordinator) -> None:
    for chunk in range(3):
        coordinator._rollouts.put((chunk + 1, time.monotonic()))
