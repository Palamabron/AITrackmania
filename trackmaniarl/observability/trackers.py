"""Optional remote tracker adapter; local JSONL remains the source of truth."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from queue import Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Any, TypedDict, Unpack
from uuid import uuid4

from trackmaniarl.observability.wandb_metrics import (
    _IGNORED_REMOTE_EVENTS,
    _event_metrics,
    _flat_name,
    _load_wandb_key_from_dotenv,
)

_METRIC_AXES = (
    "trainer/update",
    "env/transitions",
    "env/episode",
    "eval/batch",
    "expert/transitions",
    "runtime/elapsed_s",
)
_METRIC_PATTERNS = {
    "trainer/update": (
        "checkpoint_best/*",
        "checkpoint_fastest/*",
        "imitation_train/*",
        "imitation_validation/*",
        "learner/*",
        "outcome/*",
        "performance/*",
        "replay/*",
    ),
    "env/transitions": ("pipeline/*",),
    "env/episode": ("episode/*",),
    "eval/batch": ("evaluation/*",),
    "expert/transitions": ("expert/*",),
    "runtime/elapsed_s": ("health/*", "system/*"),
}
_HEALTH_COUNTERS = {
    "actor/timeout": "actor_timeouts",
    "distributed/rollout_rejected": "rollouts_rejected",
    "distributed/wal_error": "wal_errors",
    "distributed/wal_recovery": "wal_recoveries",
    "run/failure": "run_failures",
    "train/checkpoint": "checkpoints_queued",
    "train/checkpoint_completed": "checkpoints_completed",
    "train/checkpoint_failed": "checkpoint_failures",
}


class _WandbKwargs(TypedDict, total=False):
    entity: str | None
    run_dir: str | None
    run_id: str | None
    config: Mapping[str, Any] | None
    queue_size: int
    attempt_id: str | None
    resumed_from: str | None


@dataclass(frozen=True, slots=True)
class _WandbOptions:
    project: str
    entity: str | None = None
    run_dir: str | None = None
    run_id: str | None = None
    config: Mapping[str, Any] | None = None
    queue_size: int = 10_000
    attempt_id: str | None = None
    resumed_from: str | None = None


@dataclass(frozen=True, slots=True)
class _EventContext:
    name: str
    payload: Mapping[str, Any]
    step: int | None


class WandbTracker:
    """Bounded asynchronous W&B projection of the neutral event stream."""

    def __init__(self, project: str, **kwargs: Unpack[_WandbKwargs]) -> None:
        options = _WandbOptions(project, **kwargs)
        if options.queue_size < 1:
            raise ValueError("W&B queue_size must be positive")
        _load_wandb_key_from_dotenv(options.run_dir)
        self._load_client()
        self._start_run(options)
        self._initialize_state(options.queue_size)

    def _load_client(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[wandb] to configure WandbTracker") from exc
        self._wandb: Any = wandb

    def _start_run(self, options: _WandbOptions) -> None:
        self._started_at = monotonic()
        self._attempt_id = options.attempt_id or uuid4().hex
        self._run = self._init_run(options)
        self._define_metrics()

    def _initialize_state(self, queue_size: int) -> None:
        self._events: Queue[dict[str, Any] | None] = Queue(maxsize=queue_size)
        self._lock = Lock()
        self._enabled = True
        self._failed = False
        self._dropped_events = 0
        self._worker_errors = 0
        self._evaluation_batches = 0
        self._episode_count = 0
        self._latest_ingest: dict[str, Any] = {}
        self._heartbeats: dict[str, tuple[float, int]] = {}
        self._event_counts: dict[str, int] = {}
        self._worker = Thread(target=self._send_events, name="trackmaniarl-wandb", daemon=True)
        self._worker.start()

    @property
    def dropped_events(self) -> int:
        with self._lock:
            return self._dropped_events

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        with self._lock:
            values = self._project(event, payload, step)
            enabled = self._enabled
        if not enabled or not values:
            return
        try:
            self._events.put_nowait(values)
        except Full:
            self._record_drop()

    def close(self) -> None:
        try:
            self._events.put(None, timeout=5.0)
        except Full:
            self._record_drop()
        else:
            self._worker.join(timeout=10.0)
        if self._worker.is_alive():
            self._failed = True
            print("W&B worker did not stop within 10 seconds.", flush=True)
        self._run.finish(exit_code=int(self._failed))

    def _run_config(self, options: _WandbOptions) -> dict[str, Any]:
        run_config = dict(options.config) if options.config is not None else {}
        run_config["observability/attempt_id"] = self._attempt_id
        if options.resumed_from is not None:
            run_config["observability/resumed_from"] = options.resumed_from
        return run_config

    def _init_run(self, options: _WandbOptions) -> Any:
        run = self._wandb.init(
            project=options.project,
            entity=options.entity,
            dir=options.run_dir,
            name=options.run_id,
            group=options.run_id,
            job_type="training",
            config=self._run_config(options),
            reinit="finish_previous",
            settings=self._wandb.Settings(
                console="wrap", x_graphql_timeout_seconds=10, x_service_wait=5
            ),
        )
        return self._validate_run(run)

    @staticmethod
    def _validate_run(run: Any) -> Any:
        if run is None:
            raise RuntimeError("W&B did not create a run")
        if getattr(run, "url", None):
            print(f"Weights & Biases run: {run.url}", flush=True)
        return run

    def _define_metrics(self) -> None:
        for axis in _METRIC_AXES:
            self._run.define_metric(axis)
        for axis, names in _METRIC_PATTERNS.items():
            for name in names:
                self._run.define_metric(name, step_metric=axis)

    def _project(self, event: str, payload: Mapping[str, Any], step: int | None) -> dict[str, Any]:
        if event == "distributed/ingest":
            self._remember_ingest(payload)
            return {}
        if event == "actor/heartbeat":
            self._remember_heartbeat(payload)
            return {}
        if event in _IGNORED_REMOTE_EVENTS:
            return {}
        if event == "run/failure":
            self._failed = True
        values = _event_metrics(event, payload)
        values.update(self._health_metrics(event, payload))
        self._add_axes(values, _EventContext(event, payload, step))
        return values

    def _remember_ingest(self, payload: Mapping[str, Any]) -> None:
        for key in ("transitions", "utd"):
            if key in payload:
                self._latest_ingest[key] = payload[key]
        for key in ("policy_lag_updates", "queue_delay_s", "rollout_queue_depth"):
            if key in payload:
                self._latest_ingest[key] = max(
                    float(payload[key]), float(self._latest_ingest.get(key, 0.0))
                )

    def _remember_heartbeat(self, payload: Mapping[str, Any]) -> None:
        actor_id = str(payload.get("actor_id", ""))
        if actor_id:
            self._heartbeats[actor_id] = (monotonic(), int(payload.get("spool_bytes", 0)))

    def _health_metrics(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        if event == "train/update":
            values.update(self._pipeline_snapshot())
            values["health/tracker_dropped_events"] = self._dropped_events
            values["health/tracker_worker_errors"] = self._worker_errors
            values.update(
                {key: value for key, value in payload.items() if key.startswith("health/wal_")}
            )
        values.update(self._counter_metrics(event, payload))
        return values

    def _counter_metrics(self, event: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        counter_name = _HEALTH_COUNTERS.get(event)
        if counter_name is not None:
            self._event_counts[event] = self._event_counts.get(event, 0) + 1
            values[f"health/{counter_name}"] = self._event_counts[event]
        if event == "actor/timeout" and "silence_s" in payload:
            values["health/max_heartbeat_age_s"] = payload["silence_s"]
        return values

    def _pipeline_snapshot(self) -> dict[str, Any]:
        values = {_flat_name("pipeline", key): value for key, value in self._latest_ingest.items()}
        cutoff = monotonic() - 30.0
        self._heartbeats = {
            actor: item for actor, item in self._heartbeats.items() if item[0] >= cutoff
        }
        values["health/active_actors"] = len(self._heartbeats)
        values["health/spool_bytes"] = sum(item[1] for item in self._heartbeats.values())
        for key in ("policy_lag_updates", "queue_delay_s", "rollout_queue_depth"):
            self._latest_ingest.pop(key, None)
        return values

    def _add_axes(self, values: dict[str, Any], context: _EventContext) -> None:
        if not values:
            return
        values["runtime/elapsed_s"] = monotonic() - self._started_at
        values.update(self._event_axis(context))
        transitions = context.payload.get("transitions", self._latest_ingest.get("transitions"))
        if transitions is not None:
            values["env/transitions"] = int(transitions)

    def _event_axis(self, context: _EventContext) -> dict[str, int]:
        if context.name == "train/episode":
            index = int(context.payload.get("index", 0))
            self._episode_count = max(self._episode_count + 1, index)
            return {"env/episode": self._episode_count}
        if context.name in {"eval/summary", "eval/suite"}:
            self._evaluation_batches += 1
            return {"eval/batch": self._evaluation_batches}
        if context.name in {"diagnose/expert", "diagnose/expert_progress"}:
            return {"expert/transitions": int(context.payload.get("count", 0))}
        return {"trainer/update": int(context.step or 0)}

    def _record_drop(self) -> None:
        with self._lock:
            self._dropped_events += 1
            first = self._dropped_events == 1
        if first:
            print("W&B event queue is full; remote metrics are being dropped.", flush=True)

    def _send_events(self) -> None:
        while True:
            values = self._events.get()
            try:
                if values is None:
                    return
                with self._lock:
                    enabled = self._enabled
                if not enabled:
                    continue
                self._run.log(values)
            except Exception as exc:
                with self._lock:
                    self._enabled = False
                    self._failed = True
                    self._worker_errors += 1
                print(f"W&B logging disabled after {type(exc).__name__}: {exc}", flush=True)
            finally:
                self._events.task_done()
