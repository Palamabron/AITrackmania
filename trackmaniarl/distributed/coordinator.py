"""Central asynchronous rollout coordinator and learner loop."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Full, Queue
from statistics import fmean, median
from threading import RLock
from time import monotonic, perf_counter, sleep
from typing import Any, cast

import grpc
import numpy as np
import torch
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.core.data import BatchRequest, TrainingBatch
from trackmaniarl.core.runtime import ResolvedRun, prepare_run, resolve_run
from trackmaniarl.core.spec import DEFAULT_EVALUATION_TIME_BUCKETS_S, RunSpec
from trackmaniarl.core.training import TrainingResult
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.journal import JournalPayloadConflictError, RolloutJournal
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    SERVICE,
    authenticate,
    deserialize_message,
    require_distributed_token,
    require_loopback_bind,
    run_fingerprint,
    serialize_message,
    transition_from_wire,
)
from trackmaniarl.trackmania.diagnostics import aggregate_progress_bins

logger = logging.getLogger(__name__)

_ROLLOUT_QUEUE_MAXSIZE = 64


def _bucket_key(bucket: float) -> str:
    return f"sub_{bucket:g}"


def _evaluation_rank(
    finish_rate: float, median_time_s: float, required_finish_rate: float
) -> tuple[float, float, float]:
    """Rank release-qualified policies by time, otherwise by reliability."""

    if finish_rate >= required_finish_rate:
        return 1.0, -median_time_s, finish_rate
    return 0.0, finish_rate, -median_time_s


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
class _PendingRollout:
    value: Mapping[str, Any]
    row_id: int
    enqueued_at: float


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


class _AsyncCheckpointWriter:
    def __init__(self, codec: Any) -> None:
        self.codec = codec
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trackmaniarl-checkpoint"
        )
        self.pending: Any = None

    def submit(
        self,
        state: Mapping[str, Any],
        path: Path,
        on_saved: Callable[[], None] | None = None,
        on_failed: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.wait()
        self.pending = self.executor.submit(self._save, state, path, on_saved, on_failed)

    def _save(
        self,
        state: Mapping[str, Any],
        path: Path,
        on_saved: Callable[[], None] | None,
        on_failed: Callable[[BaseException], None] | None,
    ) -> None:
        try:
            self.codec.save(state, path)
            sync_checkpoint_path(path)
            if on_saved is not None:
                on_saved()
        except BaseException as exc:
            if on_failed is not None:
                on_failed(exc)
            raise

    def wait(self) -> None:
        pending = self.pending
        self.pending = None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        self.wait()
        self.executor.shutdown(wait=True, cancel_futures=False)


class Coordinator:
    """Own the replay, learner and network-facing rollout journal."""

    def __init__(
        self,
        run: ResolvedRun,
        *,
        bind: str,
        token: str,
        fingerprint: str,
        resume_checkpoint: Path | None = None,
        reset_replay: bool = False,
        external_stop: Any | None = None,
        demo_paths: tuple[Path, ...] = (),
    ) -> None:
        self.token = require_distributed_token(token)
        if getattr(run.learner, "on_policy", False):
            raise ValueError(
                "Distributed training does not support on-policy learners; "
                "use trackmaniarl train for PPO"
            )
        self.run = run
        self.bind = require_loopback_bind(bind)
        self.fingerprint = fingerprint
        self.resume_checkpoint = resume_checkpoint
        self.reset_replay = reset_replay
        self.external_stop = external_stop
        self.demo_paths = demo_paths
        self.codec = WireCodec(run.spec.distributed.max_message_bytes)
        self.journal = RolloutJournal(run.run_dir / "distributed" / "rollouts.sqlite3")
        self.counters = _Counters()
        self._lock = RLock()
        self._policy_payload = b""
        self._last_policy_publish = 0.0
        self._last_policy_update = -1
        self._server: grpc.Server | None = None
        self._rpc_executor: ThreadPoolExecutor | None = None
        self._checkpoints: list[Path] = []
        self._last_progress_print = 0
        self._last_heartbeats: dict[str, float] = {}
        self._timed_out_actors: set[str] = set()
        self._evaluation_due: set[str] = set()
        self._last_ingest_at = monotonic()
        self._started_at = monotonic()
        self._rollouts: Queue[object] = Queue(maxsize=_ROLLOUT_QUEUE_MAXSIZE)
        self._journal_enqueued_at: dict[int, float] = {}
        evaluation = getattr(run.spec, "evaluation", None)
        self._time_buckets = (
            evaluation.time_buckets_s
            if evaluation is not None
            else DEFAULT_EVALUATION_TIME_BUCKETS_S
        )
        self._metrics = _MetricAccumulator()
        self._metric_window_started = monotonic()
        self._last_metric_credit = 0.0
        self._growing_credit_windows = 0
        self._last_logging_s = 0.0
        self._last_metric_transitions = 0
        self._best_evaluation: tuple[float, float, float] | None = None
        self._evaluation_policy_states: dict[int, Mapping[str, Any]] = {}
        self._consecutive_evaluation_passes = 0
        self._evaluation_stop_reason: str | None = None
        self._recovering = False
        self._checkpoint_writer = _AsyncCheckpointWriter(run.checkpoint_codec)

    def run_forever(self) -> TrainingResult:
        try:
            return self._run_forever()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            self._log_run_failure("distributed_training", exc)
            raise
        finally:
            self._close_runtime()

    def _run_forever(self) -> TrainingResult:
        self._prepare_training()
        if self.resume_checkpoint is not None:
            logger.info("Restoring checkpoint: %s", self.resume_checkpoint)
            self.restore_checkpoint(self.resume_checkpoint, reset_replay=self.reset_replay)
            restored = (
                "learner state only; replay and runtime counters reset"
                if self.reset_replay
                else "full state"
            )
            logger.info(
                "Checkpoint restored (%s): transitions=%d, updates=%d",
                restored,
                self.counters.transitions,
                self.counters.updates,
            )
        elif self.journal.has_history():
            raise RuntimeError(
                f"run_id {self.run.spec.run_id!r} has prior rollout data in "
                f"{self.journal.path}; resume with --checkpoint or choose a new run_id"
            )
        self._import_demonstrations()
        if self.resume_checkpoint is None or self.demo_paths:
            self._offline_pretrain()
        self._publish_policy(force=True)
        self._start_server()
        logger.info(
            "Async learner ready (pid=%d): run_id=%s, gRPC bind=%s, target_transitions=%d",
            os.getpid(),
            self.run.spec.run_id,
            self.bind,
            self.run.spec.training.total_transitions,
        )
        try:
            self._learn()
            if self.run.spec.training.save_final_checkpoint:
                self._checkpoints.append(self._checkpoint())
            self._checkpoint_writer.wait()
            return TrainingResult(
                self.counters.episodes,
                self.counters.transitions,
                self.counters.updates,
                tuple(self._checkpoints),
                None,
            )
        except KeyboardInterrupt:
            if self.run.spec.training.save_final_checkpoint:
                self._checkpoints.append(self._checkpoint())
            self._checkpoint_writer.wait()
            raise

    def run_offline_pretraining(self) -> TrainingResult:
        """Train only from configured demonstrations without opening the actor server."""

        try:
            return self._run_offline_pretraining()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            self._log_run_failure("offline_pretraining", exc)
            raise
        finally:
            self._close_runtime()

    def _run_offline_pretraining(self) -> TrainingResult:

        if self.run.spec.training.offline_pretrain_updates == 0:
            raise ValueError("offline pretraining requires offline_pretrain_updates > 0")
        if not self.demo_paths:
            raise ValueError("offline pretraining requires at least one demonstration")
        checkpoints: list[Path] = []
        self._prepare_training()
        if self.journal.has_history():
            raise RuntimeError(
                f"run_id {self.run.spec.run_id!r} has prior rollout data in "
                f"{self.journal.path}; choose a new run_id"
            )
        self._import_demonstrations()
        self._offline_pretrain()
        if self.run.spec.training.save_final_checkpoint:
            checkpoints.append(self._checkpoint())
        self._checkpoint_writer.wait()
        return TrainingResult(
            self.counters.episodes,
            self.counters.transitions,
            self.counters.updates,
            tuple(checkpoints),
            None,
        )

    def _log_run_failure(self, phase: str, exc: BaseException) -> None:
        self.run.logger.log(
            "run/failure",
            {
                "phase": phase,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            step=self.counters.updates,
        )

    def _close_runtime(self) -> None:
        if self._server is not None:
            self._server.stop(grace=2).wait(timeout=5)
        if self._rpc_executor is not None:
            self._rpc_executor.shutdown(wait=True, cancel_futures=True)
        self._checkpoint_writer.close()
        self.journal.close()

    def _prepare_training(self) -> None:
        self.run.learner.setup(
            {
                "seed": self.run.spec.seed,
                "run_dir": self.run.run_dir,
                "model_factory": self.run.model_factory,
            }
        )
        prepare_run(self.run)
        self._log_execution()

    def _start_server(self) -> None:
        options = (
            ("grpc.max_receive_message_length", self.run.spec.distributed.max_message_bytes),
            ("grpc.max_send_message_length", self.run.spec.distributed.max_message_bytes),
        )
        executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="trackmaniarl-grpc")
        server = grpc.server(executor, options=options)
        handlers = {
            "Register": grpc.unary_unary_rpc_method_handler(
                self._register,
                request_deserializer=deserialize_message,
                response_serializer=serialize_message,
            ),
            "Submit": grpc.unary_unary_rpc_method_handler(
                self._submit,
                request_deserializer=deserialize_message,
                response_serializer=serialize_message,
            ),
            "Policy": grpc.unary_unary_rpc_method_handler(
                self._policy,
                request_deserializer=deserialize_message,
                response_serializer=serialize_message,
            ),
            "Heartbeat": grpc.unary_unary_rpc_method_handler(
                self._heartbeat,
                request_deserializer=deserialize_message,
                response_serializer=serialize_message,
            ),
        }
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, handlers),))
        if server.add_insecure_port(self.bind) == 0:
            executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"could not bind distributed learner to {self.bind}")
        server.start()
        self._server = server
        self._rpc_executor = executor

    def _import_demonstrations(self) -> None:
        if not self.demo_paths:
            return
        factory = self.run.environment_factory
        loader = getattr(factory, "load_demonstration", None)
        if not callable(loader):
            raise ValueError("configured environment does not support replay demonstrations")
        logger.info("Importing %d demonstration file(s) into replay...", len(self.demo_paths))
        imported = 0
        finish_times: list[float] = []
        for path in self.demo_paths:
            transitions = loader(path, self.run.feature_pipeline)
            for transition in transitions:
                self.run.replay_store.append(transition)
            imported += len(transitions)
            finish_times.append(float(transitions[0].info["sampling/projected_lap_time_s"]))
            logger.info("Imported demonstration %s: %d transitions", path, len(transitions))
        logger.info(
            "Demonstration import complete: %d transitions from %d file(s)",
            imported,
            len(self.demo_paths),
        )
        self.run.logger.log(
            "train/demonstrations",
            {
                "files": len(self.demo_paths),
                "transitions": imported,
                "best_finish_time_s": min(finish_times),
                "replay_size": len(self.run.replay_store),
            },
            step=self.counters.updates,
        )

    def _offline_pretrain(self) -> None:
        updates = self.run.spec.training.offline_pretrain_updates
        if updates == 0:
            return
        if not self.demo_paths:
            raise ValueError("offline_pretrain_updates requires at least one demonstration")
        spec = self.run.spec.training
        footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
        if len(self.run.replay_store) < footprint:
            raise RuntimeError(
                "offline demonstration replay is too small for the configured batch footprint"
            )
        started = perf_counter()
        metrics: list[Mapping[str, float]] = []
        progress_interval = min(25, updates)
        begin = getattr(self.run.learner, "begin_offline_pretraining", None)
        end = getattr(self.run.learner, "end_offline_pretraining", None)
        if callable(begin):
            begin()
        try:
            for index in range(1, updates + 1):
                batch = self.run.sampler.sample(
                    self.run.replay_store, spec.batch_request(beta=spec.replay_beta(0))
                )
                result = self.run.learner.update(batch)
                values, priorities = result if isinstance(result, tuple) else (result, None)
                if priorities is not None:
                    self.run.sampler.update_priorities(priorities)
                self.counters.updates += 1
                metrics.append(values)
                if index % progress_interval == 0 or index == updates:
                    logger.info("Offline pretraining progress: updates=%d/%d", index, updates)
        finally:
            if callable(end):
                end()
        summary = {
            key: fmean(float(values[key]) for values in metrics if key in values)
            for key in {key for values in metrics for key in values}
        }
        self.run.logger.log(
            "train/offline_pretrain",
            {
                **summary,
                "updates": updates,
                "replay_size": len(self.run.replay_store),
                "duration_s": perf_counter() - started,
            },
            step=self.counters.updates,
        )
        logger.info(
            "Offline pretraining complete: updates=%d, replay=%d, duration=%.1fs",
            updates,
            len(self.run.replay_store),
            perf_counter() - started,
        )

    def _request(
        self, request: BytesValue, context: grpc.ServicerContext[Any, Any]
    ) -> Mapping[str, Any]:
        authenticate(context, self.token)
        value = self.codec.decode(request.value)
        if not isinstance(value, Mapping):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "request must be a mapping")
        if value.get("protocol_version") != PROTOCOL_VERSION:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "protocol version mismatch")
        if value.get("fingerprint") != self.fingerprint:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "run fingerprint mismatch")
        return cast(Mapping[str, Any], value)

    def _response(self, value: Mapping[str, Any]) -> BytesValue:
        return BytesValue(value=self.codec.encode(value))

    def _log_rollout_rejected(
        self,
        value: Mapping[str, Any],
        reason: str,
        **details: object,
    ) -> None:
        self.run.logger.log(
            "distributed/rollout_rejected",
            {
                "actor_id": str(value["actor_id"]),
                "session_id": str(value["session_id"]),
                "sequence": int(value["sequence"]),
                "reason": reason,
                **details,
            },
            step=self.counters.updates,
        )

    def _log_wal_error(self, operation: str, exc: BaseException) -> None:
        self.run.logger.log(
            "distributed/wal_error",
            {
                "operation": operation,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "journal_path": str(self.journal.path),
                "journal_applied_frontier": self.counters.journal_applied_frontier,
            },
            step=self.counters.updates,
        )

    def _journal_rows(self, watermark: int, operation: str) -> Iterator[tuple[int, bytes]]:
        try:
            yield from self.journal.rows_after(watermark)
        except Exception as exc:
            self._log_wal_error(operation, exc)
            raise

    def _decode_journal_payload(self, payload: bytes, operation: str) -> Mapping[str, Any]:
        try:
            value = self.codec.decode(payload)
            if not isinstance(value, Mapping):
                raise ValueError("journal chunk must decode to a mapping")
        except Exception as exc:
            self._log_wal_error(operation, exc)
            raise
        return value

    def _register(self, request: BytesValue, context: grpc.ServicerContext[Any, Any]) -> BytesValue:
        value = self._request(request, context)
        actor_id = str(value["actor_id"])
        with self._lock:
            self._last_heartbeats[actor_id] = monotonic()
            self._timed_out_actors.discard(actor_id)
        profile = self.journal.actor_profile(
            actor_id, len(self.run.spec.distributed.epsilon_profiles)
        )
        self.run.logger.log(
            "actor/registered",
            {
                "actor_id": actor_id,
                "session_id": value["session_id"],
                "epsilon_profile": profile,
            },
            step=self.counters.updates,
        )
        with self._lock:
            return self._response(
                {
                    "accepted": True,
                    "policy_version": self.counters.policy_version,
                    "epsilon": self._epsilon(profile),
                    "stop": self._should_stop(),
                }
            )

    def _submit(self, request: BytesValue, context: grpc.ServicerContext[Any, Any]) -> BytesValue:
        value = self._request(request, context)
        try:
            _validate_submit_payload(value, self.codec)
        except (KeyError, TypeError, ValueError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"invalid rollout payload: {exc}")
            raise AssertionError("gRPC abort returned") from exc
        policy_version = int(value["policy_version"])
        with self._lock:
            lag = max(0, self.counters.updates - policy_version)
            stop = self._should_stop()
        if value["transitions"] and lag > self.run.spec.distributed.hard_policy_lag_updates:
            self._log_rollout_rejected(
                value,
                "hard_policy_lag",
                policy_lag_updates=lag,
                hard_policy_lag_updates=self.run.spec.distributed.hard_policy_lag_updates,
            )
            return self._response(
                {
                    "accepted": False,
                    "reason": "hard_policy_lag",
                    "force_refresh": True,
                    "stop": stop,
                }
            )
        session_id = str(value["session_id"])
        sequence = int(value["sequence"])
        try:
            row_id, inserted = self.journal.append(session_id, sequence, request.value)
        except JournalPayloadConflictError as exc:
            self._log_rollout_rejected(value, "payload_conflict")
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            raise AssertionError("gRPC abort returned") from exc
        except Exception as exc:
            self._log_wal_error("append", exc)
            raise
        if inserted:
            # SQLite is the durable queue. A missed wake-up is harmless because
            # the learner polls the ordered journal on every loop iteration.
            with suppress(Full):
                self._rollouts.put_nowait((row_id, monotonic()))
        with self._lock:
            actor_id = str(value["actor_id"])
            evaluate = actor_id in self._evaluation_due and self.counters.policy_version > 0
            if evaluate:
                self._evaluation_due.discard(actor_id)
            evaluation_version = self.counters.policy_version
            evaluation_snapshot = self._policy_payload if evaluate else b""
            if evaluate:
                policy_state = self.codec.decode(evaluation_snapshot)
                if not isinstance(policy_state, Mapping):
                    raise ValueError("published policy snapshot must decode to a mapping")
                self._evaluation_policy_states[evaluation_version] = _snapshot_value(policy_state)
                while len(self._evaluation_policy_states) > 16:
                    self._evaluation_policy_states.pop(next(iter(self._evaluation_policy_states)))
        return self._response(
            {
                "accepted": True,
                "duplicate": not inserted,
                "force_refresh": lag > self.run.spec.distributed.soft_policy_lag_updates,
                "stop": stop,
                "policy_lag_updates": lag,
                "evaluate": evaluate,
                "evaluation_policy_version": evaluation_version,
                "evaluation_snapshot": evaluation_snapshot,
            }
        )

    def _policy(self, request: BytesValue, context: grpc.ServicerContext[Any, Any]) -> BytesValue:
        value = self._request(request, context)
        profile = self.journal.actor_profile(
            str(value["actor_id"]), len(self.run.spec.distributed.epsilon_profiles)
        )
        with self._lock:
            current = int(value.get("current_version", -1))
            return self._response(
                {
                    "policy_version": self.counters.policy_version,
                    "snapshot": self._policy_payload
                    if current != self.counters.policy_version
                    else b"",
                    "epsilon": self._epsilon(profile),
                    "stop": self._should_stop(),
                }
            )

    def _heartbeat(
        self, request: BytesValue, context: grpc.ServicerContext[Any, Any]
    ) -> BytesValue:
        value = self._request(request, context)
        actor_id = str(value["actor_id"])
        with self._lock:
            self._last_heartbeats[actor_id] = monotonic()
            self._timed_out_actors.discard(actor_id)
        self.run.logger.log(
            "actor/heartbeat",
            {
                "actor_id": actor_id,
                "policy_version": value["policy_version"],
                "spool_bytes": value["spool_bytes"],
            },
            step=self.counters.updates,
        )
        return self._response({"stop": self._should_stop()})

    def _epsilon(self, profile: int) -> float:
        spec = self.run.spec.distributed
        schedule_progress = (
            self.counters.transitions
            if spec.epsilon_decay_updates is None
            else self.counters.updates
        )
        schedule_length = spec.epsilon_decay_updates or spec.epsilon_decay_transitions
        fraction = min(1.0, schedule_progress / schedule_length)
        scheduled = spec.epsilon_start + fraction * (spec.epsilon_final - spec.epsilon_start)
        return scheduled * spec.epsilon_profiles[profile]

    def _ingest(self, value: Mapping[str, Any], row_id: int) -> None:
        if row_id <= self.counters.journal_applied_frontier:
            raise ValueError("journal rows must be ingested in strictly increasing order")
        transitions = [transition_from_wire(item) for item in value["transitions"]]
        now = monotonic()
        elapsed = max(now - self._last_ingest_at, 1e-6)
        ingest_fps = len(transitions) / elapsed
        self._last_ingest_at = now
        before = self.counters.transitions
        for transition in transitions:
            replay_info = self._replay_info(transition.info)
            self.run.replay_store.append(replace(transition, info=replay_info))
        self.counters.transitions += len(transitions)
        ready = self.run.spec.training.warmup_transitions
        newly_trainable = max(0, self.counters.transitions - ready) - max(0, before - ready)
        self.counters.update_credit = min(
            self.run.spec.distributed.max_update_credit,
            self.counters.update_credit
            + newly_trainable * self.run.spec.training.updates_per_transition,
        )
        session_id = str(value["session_id"])
        self.counters.actor_sequences[session_id] = max(
            self.counters.actor_sequences.get(session_id, -1),
            int(value["sequence"]),
        )
        for summary in value.get("episodes", []):
            self.counters.episodes += 1
            finished = bool(summary["finished"])
            finish_time_s = float(summary["finish_time_s"])
            if finished:
                self.counters.finishes += 1
                if (
                    self.counters.best_finish_time_s == 0.0
                    or finish_time_s < self.counters.best_finish_time_s
                ):
                    self.counters.best_finish_time_s = finish_time_s
            if not self._recovering:
                self._log_episode(value, summary)
                interval = self.run.spec.training.evaluate_every_episodes
                if interval is not None and self.counters.episodes % interval == 0:
                    with self._lock:
                        self._evaluation_due.add(str(value["actor_id"]))
        evaluations = [dict(summary) for summary in value.get("evaluations", [])]
        evaluation_snapshot = value.get("evaluation_snapshot", b"")
        if evaluations and evaluation_snapshot:
            if not isinstance(evaluation_snapshot, bytes):
                raise ValueError("evaluation snapshot must be bytes")
            policy_state = self.codec.decode(evaluation_snapshot)
            if not isinstance(policy_state, Mapping):
                raise ValueError("evaluation snapshot must decode to a mapping")
            versions = {int(summary.get("policy_version", 0)) for summary in evaluations}
            if len(versions) != 1:
                raise ValueError("evaluation snapshot cannot cover mixed policy versions")
            with self._lock:
                self._evaluation_policy_states[versions.pop()] = _snapshot_value(policy_state)
        for summary in evaluations:
            self.counters.evaluations += 1
            finished = bool(summary["finished"])
            finish_time_s = float(summary["finish_time_s"])
            self.counters.evaluation_finishes += int(finished)
            bucket_metrics: dict[str, float] = {}
            for bucket in self._time_buckets:
                key = _bucket_key(bucket)
                hit = finished and finish_time_s < bucket
                count = self.counters.evaluation_bucket_finishes.get(key, 0) + int(hit)
                self.counters.evaluation_bucket_finishes[key] = count
                bucket_metrics[key] = float(hit)
                bucket_metrics[f"{key}_rate"] = count / self.counters.evaluations
            if not self._recovering:
                self.run.logger.log(
                    "eval/episode",
                    {
                        **summary,
                        "index": self.counters.evaluations,
                        "finish_rate": self.counters.evaluation_finishes
                        / self.counters.evaluations,
                        **bucket_metrics,
                        "actor_id": value["actor_id"],
                    },
                    step=self.counters.updates,
                )
        self.counters.journal_applied_frontier = row_id
        if evaluations and not self._recovering:
            self._finish_evaluation_batch(evaluations)
        if not self._recovering:
            self.run.logger.log(
                "distributed/ingest",
                {
                    "actor_id": value["actor_id"],
                    "chunk_transitions": len(transitions),
                    "transitions": self.counters.transitions,
                    "replay_size": len(self.run.replay_store),
                    "ingest_fps": ingest_fps,
                    "policy_lag_updates": max(
                        0, self.counters.updates - int(value["policy_version"])
                    ),
                    "utd": self.counters.updates
                    / max(
                        1,
                        self.counters.transitions - self.run.spec.training.warmup_transitions,
                    ),
                    "queue_delay_s": max(0.0, now - float(value.get("_enqueued_at", now))),
                    "rollout_queue_depth": self._rollouts.qsize(),
                },
                step=self.counters.updates,
            )

    @staticmethod
    def _replay_info(info: Mapping[str, Any]) -> dict[str, Any]:
        replay_info: dict[str, Any] = {}
        if "is_demo" in info or info.get("source") == "demo":
            replay_info["is_demo"] = bool(
                info.get("is_demo", False) or info.get("source") == "demo"
            )
        if "sampling/projected_lap_time_s" in info:
            replay_info["sampling/projected_lap_time_s"] = float(
                info["sampling/projected_lap_time_s"]
            )
        return replay_info

    def _log_episode(self, value: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        self.run.logger.log(
            "train/episode",
            {
                **summary,
                "index": self.counters.episodes,
                "finish_count": self.counters.finishes,
                "finish_rate": self.counters.finishes / self.counters.episodes,
                "best_finish_time_s": self.counters.best_finish_time_s,
                "actor_id": value["actor_id"],
                "replay_size": len(self.run.replay_store),
            },
            step=self.counters.updates,
        )
        progress_bins = _progress_bin_metrics(summary)
        if progress_bins:
            self.run.logger.log(
                "train/progress_bin",
                progress_bins,
                step=self.counters.updates,
            )
        logger.info(
            "Actor %s episode %d: progress=%.1f%%, return=%.3f, "
            "reward(time=%.3f, pace=%.3f, pbrs=%.3f, progress=%.3f, projected_velocity=%.3f, "
            "projected_speed=%.3f, steering_delta=%.3f, collision=%.3f (%d/%d), "
            "terminal=%.3f), velocity_ratio(mean=%.3f, max=%.3f), steps=%d, "
            "race=%.2fs, epsilon=%.3f, policy=%d, q_margin(start=%.2f, min=%.2f), "
            "termination=%s",
            value["actor_id"],
            self.counters.episodes,
            float(summary["progress_pct"]),
            float(summary["return"]),
            float(summary["reward/time"]),
            float(summary["reward/pace"]),
            float(summary["reward/pbrs"]),
            float(summary["reward/progress"]),
            float(summary["reward/projected_velocity"]),
            float(summary["reward/projected_speed"]),
            float(summary["reward/steering_delta"]),
            float(summary["reward/collision"]),
            int(summary["collision/count"]),
            int(summary["collision/detected_count"]),
            float(summary["reward/terminal"]),
            float(summary["velocity/ratio_mean"]),
            float(summary["velocity/ratio_max"]),
            int(summary["steps"]),
            float(summary["race_time_s"]),
            float(summary["exploration_epsilon"]),
            int(summary.get("policy_version", 0)),
            float(summary.get("q_margin/start_mean", 0.0)),
            float(summary.get("q_margin/min", 0.0)),
            summary["termination"],
        )

    def _learn(self) -> None:
        spec = self.run.spec.training
        footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
        ready = max(spec.warmup_transitions, footprint)
        prefetcher = _BatchPrefetcher(self.run)
        try:
            while not self._external_stop_requested() and (
                not self._should_stop()
                or (
                    self._evaluation_stop_reason is None
                    and len(self.run.replay_store) >= ready
                    and self.counters.update_credit >= 1.0
                )
                or self.journal.has_rows_after(self.counters.journal_applied_frontier)
            ):
                did_update = False
                self._check_actor_timeouts()
                # Ingest the whole backlog every iteration: a standing queue
                # would otherwise train the learner on minutes-old transitions
                # and inflate the measured actor policy lag by the queue delay.
                self._drain_rollouts(_ROLLOUT_QUEUE_MAXSIZE)
                if self._evaluation_stop_reason is not None:
                    break
                if (
                    self._can_update()
                    and len(self.run.replay_store) >= ready
                    and self.counters.update_credit >= 1.0
                ):
                    request = spec.batch_request(beta=spec.replay_beta(self.counters.transitions))
                    batch, preparation_s, wait_s = prefetcher.next(request)
                    update_started = perf_counter()
                    result = self.run.learner.update(batch)
                    update_finished = perf_counter()
                    metrics, priorities = result if isinstance(result, tuple) else (result, None)
                    if priorities is not None:
                        self.run.sampler.update_priorities(priorities)
                    self.counters.updates += 1
                    self.counters.update_credit -= 1.0
                    did_update = True
                    self._metrics.add(
                        {
                            **metrics,
                            "timing/replay_sample_s": preparation_s,
                            "timing/replay_wait_s": wait_s,
                            "timing/learner_update_s": update_finished - update_started,
                        }
                    )
                    self._emit_metrics_if_ready()
                    if (
                        self.counters.updates == 1
                        or self.counters.updates - self._last_progress_print >= 100
                    ):
                        self._last_progress_print = self.counters.updates
                        logger.info(
                            "Async training progress: transitions=%d/%d, updates=%d, "
                            "replay=%d, credit=%.1f",
                            self.counters.transitions,
                            spec.total_transitions,
                            self.counters.updates,
                            len(self.run.replay_store),
                            self.counters.update_credit,
                        )
                    if (
                        spec.checkpoint_interval_updates is not None
                        and self.counters.updates % spec.checkpoint_interval_updates == 0
                    ):
                        self._checkpoints.append(self._checkpoint())
                    self._publish_policy()
                if not did_update:
                    sleep(0.005)
        finally:
            prefetcher.close()

    def _drain_rollouts(self, limit: int) -> None:
        for _ in range(limit):
            try:
                wake = self._rollouts.get_nowait()
            except Empty:
                break
            if (
                isinstance(wake, tuple)
                and len(wake) == 2
                and isinstance(wake[0], int)
                and wake[0] > self.counters.journal_applied_frontier
            ):
                self._journal_enqueued_at[wake[0]] = float(wake[1])
            self._rollouts.task_done()
        frontier = self.counters.journal_applied_frontier
        rows = self._journal_rows(frontier, "drain")
        for applied, (row_id, payload) in enumerate(rows, start=1):
            value = self._decode_journal_payload(payload, "drain_decode")
            queued_at = self._journal_enqueued_at.pop(row_id, None)
            materialized = dict(value)
            if queued_at is not None:
                materialized["_enqueued_at"] = queued_at
            self._ingest(materialized, row_id)
            if applied >= limit:
                return

    def _finish_evaluation_batch(self, summaries: list[dict[str, Any]]) -> None:
        stats = _evaluation_batch_stats(summaries, self._time_buckets)
        self.run.logger.log("eval/summary", stats, step=self.counters.updates)
        progress_bins = _progress_bin_metrics(stats)
        if progress_bins:
            self.run.logger.log("eval/progress_bin", progress_bins, step=self.counters.updates)
        logger.info(
            "Deterministic evaluation @update %d: %d/%d finished, mean=%.2fs, "
            "best=%.2fs, policy_version=%d",
            self.counters.updates,
            int(stats["finished_trials"]),
            int(stats["trials"]),
            stats["finish_time_mean_s"],
            stats["finish_time_best_s"],
            int(stats["policy_version"]),
        )
        self._record_best_evaluation(
            float(stats["finish_rate"]),
            float(stats["finish_time_median_s"]),
            float(stats["finish_time_mean_s"]),
            int(stats["policy_version"]),
        )
        self._record_evaluation_stop(stats)

    def _record_evaluation_stop(self, stats: Mapping[str, float]) -> None:
        training = self.run.spec.training
        required_finish_rate = getattr(training, "evaluation_stop_min_finish_rate", None)
        maximum_median_s = getattr(training, "evaluation_stop_median_s", None)
        required_batches = getattr(training, "evaluation_stop_consecutive_batches", None)
        if required_finish_rate is None or maximum_median_s is None or required_batches is None:
            return
        passed = (
            stats["finish_rate"] >= required_finish_rate
            and stats["finish_time_median_s"] <= maximum_median_s
        )
        self._consecutive_evaluation_passes = (
            self._consecutive_evaluation_passes + 1 if passed else 0
        )
        if self._consecutive_evaluation_passes < required_batches:
            return
        self._evaluation_stop_reason = (
            "evaluation target passed "
            f"{self._consecutive_evaluation_passes} consecutive times: "
            f"finish_rate={stats['finish_rate']:.3f}, "
            f"median_finish_time_s={stats['finish_time_median_s']:.3f}"
        )
        self.run.logger.log(
            "train/early_stop",
            {
                "reason": self._evaluation_stop_reason,
                "consecutive_passes": self._consecutive_evaluation_passes,
                "finish_rate": stats["finish_rate"],
                "median_finish_time_s": stats["finish_time_median_s"],
            },
            step=self.counters.updates,
        )
        logger.info("Stopping training: %s", self._evaluation_stop_reason)

    def _record_best_evaluation(
        self,
        finish_rate: float,
        median_time_s: float,
        mean_time_s: float,
        policy_version: int,
    ) -> None:
        if self._recovering:
            return
        suite = getattr(self.run.spec, "evaluation", None)
        required_finish_rate = 1.0 if suite is None else suite.min_finish_rate
        if finish_rate < required_finish_rate:
            return
        candidate = _evaluation_rank(finish_rate, median_time_s, required_finish_rate)
        if self._best_evaluation is not None and candidate <= self._best_evaluation:
            return
        self._best_evaluation = candidate
        lock = getattr(self, "_lock", None)
        if lock is None:
            policy_state = getattr(self, "_evaluation_policy_states", {}).get(policy_version)
        else:
            with lock:
                policy_state = getattr(self, "_evaluation_policy_states", {}).get(policy_version)
        path = (
            self._checkpoint(policy_state=policy_state, policy_version=policy_version)
            if policy_state is not None
            else self._checkpoint()
        )
        self._checkpoints.append(path)
        self.run.logger.log(
            "eval/best_checkpoint",
            {
                "finish_rate": finish_rate,
                "finish_time_median_s": median_time_s,
                "finish_time_mean_s": mean_time_s,
                "release_qualified": 1.0,
                "policy_version": policy_version,
                "exact_policy": float(policy_state is not None),
                "path": str(path),
            },
            step=self.counters.updates,
        )

    def _emit_metrics_if_ready(self) -> None:
        interval = self.run.spec.training.metrics_interval_updates
        if self.counters.updates % interval != 0:
            return
        now = monotonic()
        elapsed = max(now - self._metric_window_started, 1e-6)
        window_transitions = self.counters.transitions - self._last_metric_transitions
        transitions_per_s = window_transitions / elapsed
        updates_per_s = interval / elapsed
        target_updates_per_s = transitions_per_s * self.run.spec.training.updates_per_transition
        replay_capacity = int(getattr(self.run.replay_store, "capacity", 0))
        payload: dict[str, object] = {
            **self._metrics.flush(),
            "replay_size": len(self.run.replay_store),
            "replay_fill_fraction": (
                len(self.run.replay_store) / replay_capacity if replay_capacity else 0.0
            ),
            "update_credit": self.counters.update_credit,
            "rollout_queue_depth": self._rollouts.qsize(),
            "updates_per_s": updates_per_s,
            "transitions_per_s": transitions_per_s,
            "cumulative_transitions_per_s": self.counters.transitions
            / max(now - self._started_at, 1e-6),
            "target_updates_per_s": target_updates_per_s,
            "update_backlog_s": self.counters.update_credit / max(updates_per_s, 1e-6),
            "episodes": self.counters.episodes,
            "finish_rate": self.counters.finishes / max(1, self.counters.episodes),
            "per_beta": self.run.spec.training.replay_beta(self.counters.transitions),
            "timing/logging_s": self._last_logging_s,
        }
        if target_updates_per_s > 0.0:
            payload["update_throughput_ratio"] = updates_per_s / target_updates_per_s
        execution = getattr(self.run.learner, "execution_manifest", None)
        if callable(execution):
            payload["execution"] = dict(execution())
        try:
            import torch

            if torch.cuda.is_available():
                payload["accelerator_memory_bytes"] = torch.cuda.memory_allocated()
        except ImportError:
            pass
        if self.counters.update_credit > self._last_metric_credit:
            self._growing_credit_windows += 1
        else:
            self._growing_credit_windows = 0
        if self._growing_credit_windows >= 5:
            payload["warning"] = "update credit has grown for five metric windows"
        self._last_metric_credit = self.counters.update_credit
        self._metric_window_started = now
        self._last_metric_transitions = self.counters.transitions
        logging_started = perf_counter()
        self.run.logger.log("train/update", payload, step=self.counters.updates)
        self._last_logging_s = perf_counter() - logging_started

    def _check_actor_timeouts(self) -> None:
        now = monotonic()
        timeout = self.run.spec.distributed.actor_timeout_s
        with self._lock:
            heartbeats = tuple(self._last_heartbeats.items())
        for actor_id, heartbeat in heartbeats:
            if now - heartbeat <= timeout:
                continue
            with self._lock:
                self._last_heartbeats.pop(actor_id, None)
                self._timed_out_actors.discard(actor_id)
            self.run.logger.log(
                "actor/timeout",
                {"actor_id": actor_id, "silence_s": now - heartbeat},
                step=self.counters.updates,
            )

    def _has_active_actor(self) -> bool:
        with self._lock:
            return bool(set(self._last_heartbeats) - self._timed_out_actors)

    def _can_update(self) -> bool:
        return not self._external_stop_requested() and (
            self._has_active_actor()
            or self.counters.transitions >= self.run.spec.training.total_transitions
        )

    def _publish_policy(self, *, force: bool = False) -> None:
        now = monotonic()
        if not force and (
            self.counters.updates == self._last_policy_update
            or now - self._last_policy_publish < self.run.spec.distributed.policy_refresh_s
        ):
            return
        publish_started = perf_counter()
        policy = self.run.learner.policy()
        if not isinstance(policy, ReplicablePolicy):
            raise TypeError("distributed training requires learner.policy() to be ReplicablePolicy")
        payload = self.codec.encode(dict(policy.export_state()))
        with self._lock:
            self._policy_payload = payload
            self.counters.policy_version = self.counters.updates
        self._last_policy_update = self.counters.updates
        self._last_policy_publish = now
        self.run.logger.log(
            "distributed/policy_published",
            {
                "policy_version": self.counters.policy_version,
                "timing/policy_publish_s": perf_counter() - publish_started,
            },
            step=self.counters.updates,
        )

    def _should_stop(self) -> bool:
        return (
            self.counters.transitions >= self.run.spec.training.total_transitions
            or getattr(self, "_evaluation_stop_reason", None) is not None
            or self._external_stop_requested()
        )

    def _external_stop_requested(self) -> bool:
        return bool(self.external_stop is not None and self.external_stop.is_set())

    def _checkpoint(
        self,
        *,
        policy_state: Mapping[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> Path:
        checkpoint_started = perf_counter()
        name = (
            f"best-eval-policy-{policy_version:08d}-at-update-{self.counters.updates:08d}.pt"
            if policy_state is not None and policy_version is not None
            else f"distributed-update-{self.counters.updates:08d}.pt"
        )
        path = self.run.run_dir / "checkpoints" / name
        learner_state = self.run.learner.state_dict()
        if policy_state is not None:
            exact_state = getattr(self.run.learner, "state_dict_for_policy", None)
            if not callable(exact_state):
                raise TypeError("learner cannot build an exact evaluated-policy checkpoint")
            learner_state = exact_state(policy_state)
        state = {
            "schema_version": "2.0",
            "journal_contract_version": 2,
            "journal_id": self.journal.identity,
            "run_fingerprint": self.fingerprint,
            "learner": _snapshot_value(learner_state),
            "replay_store": _snapshot_value(_state_dict(self.run.replay_store)),
            "sampler": _snapshot_value(_state_dict(self.run.sampler)),
            "distributed": {
                "transitions": self.counters.transitions,
                "episodes": self.counters.episodes,
                "finishes": self.counters.finishes,
                "best_finish_time_s": self.counters.best_finish_time_s,
                "evaluations": self.counters.evaluations,
                "evaluation_finishes": self.counters.evaluation_finishes,
                "evaluation_bucket_finishes": dict(self.counters.evaluation_bucket_finishes),
                "updates": self.counters.updates,
                "update_credit": self.counters.update_credit,
                "journal_applied_frontier": self.counters.journal_applied_frontier,
                "policy_version": self.counters.policy_version,
                "actor_sequences": dict(self.counters.actor_sequences),
            },
            "evaluated_policy_version": policy_version,
        }
        applied_frontier = self.counters.journal_applied_frontier
        checkpoint_update = self.counters.updates

        def saved() -> None:
            try:
                self.journal.prune(applied_frontier)
            except Exception as exc:
                self._log_wal_error("prune", exc)
                raise
            self.run.logger.log(
                "train/checkpoint_completed",
                {
                    "path": str(path),
                    "journal_applied_frontier": applied_frontier,
                    "duration_s": perf_counter() - checkpoint_started,
                },
                step=checkpoint_update,
            )

        def failed(exc: BaseException) -> None:
            self.run.logger.log(
                "train/checkpoint_failed",
                {
                    "path": str(path),
                    "journal_applied_frontier": applied_frontier,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                step=checkpoint_update,
            )

        self._checkpoint_writer.submit(
            state,
            path,
            saved,
            failed,
        )
        self.run.logger.log(
            "train/checkpoint",
            {
                "path": str(path),
                "timing/checkpoint_snapshot_s": perf_counter() - checkpoint_started,
            },
            step=self.counters.updates,
        )
        logger.info("Checkpoint queued: %s", path)
        return path

    def restore_checkpoint(self, path: Path, *, reset_replay: bool = False) -> None:
        """Restore a checkpoint, optionally retaining only the learner state."""

        self._restore(path, reset_replay=reset_replay)

    def _restore(self, path: Path, *, reset_replay: bool) -> None:
        state = self.run.checkpoint_codec.load(path)
        if state.get("schema_version") != "2.0":
            raise ValueError("async runtime only resumes distributed checkpoint schema 2.0")
        if reset_replay and self.journal.has_history():
            raise RuntimeError(
                f"cannot reset replay while {self.journal.path} contains rollout data; "
                "choose a new run_id so stale journal rows cannot enter a later resume"
            )
        if reset_replay:
            self.run.learner.load_state_dict(state["learner"])
            self.counters = _Counters()
            return
        if state.get("journal_contract_version") != 2:
            raise ValueError(
                "distributed checkpoint predates the contiguous WAL frontier contract; "
                "resume with --reset-replay or use a new run"
            )
        if state.get("run_fingerprint") != self.fingerprint:
            raise ValueError("distributed checkpoint run fingerprint mismatch")
        distributed = dict(state["distributed"])
        if "journal_applied_frontier" not in distributed:
            raise ValueError("distributed checkpoint has no contiguous journal applied frontier")
        frontier = int(distributed["journal_applied_frontier"])
        try:
            self.journal.validate_checkpoint(state.get("journal_id"), frontier)
        except Exception as exc:
            self._log_wal_error("checkpoint_validation", exc)
            raise
        self.run.learner.load_state_dict(state["learner"])
        self.counters = _Counters(**distributed)
        self.counters.update_credit = min(
            self.counters.update_credit,
            float(self.run.spec.distributed.max_update_credit),
        )
        _load_state_dict(self.run.replay_store, state["replay_store"])
        _load_state_dict(self.run.sampler, state["sampler"])
        self._recover_journal(self.counters.journal_applied_frontier)

    def _recover_journal(self, watermark: int) -> None:
        self._recovering = True
        recovered_rows = 0
        recovered_transitions = 0
        try:
            for row_id, payload in self._journal_rows(watermark, "recovery"):
                value = self._decode_journal_payload(payload, "recovery_decode")
                self._ingest(value, row_id)
                recovered_rows += 1
                recovered_transitions += len(value["transitions"])
        finally:
            self._recovering = False
        if recovered_rows:
            self.run.logger.log(
                "distributed/wal_recovery",
                {
                    "rows": recovered_rows,
                    "transitions": recovered_transitions,
                    "from_frontier": watermark,
                    "to_frontier": self.counters.journal_applied_frontier,
                },
                step=self.counters.updates,
            )

    def _log_execution(self) -> None:
        execution = getattr(self.run.learner, "execution_manifest", None)
        if callable(execution):
            self.run.logger.log(
                "train/execution",
                dict(execution()),
                step=self.counters.updates,
            )


def learner_process_entry(
    config_path: str,
    bind: str,
    token: str,
    resume_checkpoint: str | None = None,
    reset_replay: bool = False,
    external_stop: Any | None = None,
    demo_paths: tuple[str, ...] = (),
) -> None:
    """Spawn-safe learner entrypoint used by both local and remote launchers."""

    path = Path(config_path).resolve()
    spec = RunSpec.from_yaml(path)
    run = resolve_run(spec, base_dir=path.parent)
    try:
        Coordinator(
            run,
            bind=bind,
            token=token,
            fingerprint=run_fingerprint(spec, path.parent),
            resume_checkpoint=Path(resume_checkpoint) if resume_checkpoint else None,
            reset_replay=reset_replay,
            external_stop=external_stop,
            demo_paths=tuple(Path(item) for item in demo_paths),
        ).run_forever()
    finally:
        run.logger.close()


def _evaluation_batch_stats(
    summaries: list[dict[str, Any]], time_buckets_s: tuple[float, ...]
) -> dict[str, float]:
    if not summaries:
        raise ValueError("deterministic evaluation batch must not be empty")
    policy_versions = {int(item.get("policy_version", 0)) for item in summaries}
    if len(policy_versions) != 1:
        raise ValueError("deterministic evaluation batch mixed policy versions")
    finished_times = sorted(
        float(item["finish_time_s"]) for item in summaries if bool(item["finished"])
    )
    failure_progress = [
        float(item.get("progress_pct", 0.0)) for item in summaries if not bool(item["finished"])
    ]
    trials = len(summaries)
    stats = {
        "trials": float(trials),
        "finished_trials": float(len(finished_times)),
        "finish_rate": len(finished_times) / trials,
        "finish_time_mean_s": fmean(finished_times) if finished_times else 0.0,
        "finish_time_median_s": median(finished_times) if finished_times else 0.0,
        "finish_time_best_s": finished_times[0] if finished_times else 0.0,
        **{
            f"{_bucket_key(bucket)}_rate": sum(1 for time_s in finished_times if time_s < bucket)
            / trials
            for bucket in time_buckets_s
        },
        "failure_progress_mean_pct": fmean(failure_progress) if failure_progress else 100.0,
        "failure_progress_median_pct": median(failure_progress) if failure_progress else 100.0,
        "failure_progress_best_pct": max(failure_progress) if failure_progress else 100.0,
        "collision_rate": sum(
            int(float(item.get("collision/count", 0.0)) > 0.0) for item in summaries
        )
        / trials,
        "control_brake_fraction_mean": fmean(
            float(item.get("control/brake_fraction", 0.0)) for item in summaries
        ),
        "control_brake_tap_fraction_mean": fmean(
            float(item.get("control/brake_tap_fraction", 0.0)) for item in summaries
        ),
        "control_gas_fraction_mean": fmean(
            float(item.get("control/gas_fraction", 0.0)) for item in summaries
        ),
        "control_steer_abs_mean": fmean(
            float(item.get("control/steer_abs_mean", 0.0)) for item in summaries
        ),
        "off_track_rate": sum(
            int(str(item.get("termination", "")) == "off_track") for item in summaries
        )
        / trials,
        "telemetry_error_rate": sum(
            int(str(item.get("termination", "")) == "telemetry_error") for item in summaries
        )
        / trials,
        "projected_velocity_ratio_mean": fmean(
            float(item.get("velocity/ratio_mean", 0.0)) for item in summaries
        ),
        "policy_version": float(policy_versions.pop()),
        "q_margin_start_mean": fmean(
            float(item.get("q_margin/start_mean", 0.0)) for item in summaries
        ),
    }
    stats.update(aggregate_progress_bins(_progress_bin_summary(item) for item in summaries))
    return stats


def _progress_bin_summary(summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    bins: dict[str, dict[str, float]] = {}
    for key, value in summary.items():
        prefix, separator, suffix = key.partition("progress_bin/")
        if prefix or not separator:
            continue
        name, metric_separator, metric = suffix.partition("/")
        if not metric_separator:
            continue
        bins.setdefault(name, {})[metric] = float(value)
    return bins


def _progress_bin_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        key.removeprefix("progress_bin/"): float(value)
        for key, value in summary.items()
        if key.startswith("progress_bin/")
    }


def _state_dict(component: object) -> Mapping[str, object] | None:
    method = getattr(component, "state_dict", None)
    return cast(Mapping[str, object], method()) if callable(method) else None


def _validate_submit_payload(value: Mapping[str, Any], codec: WireCodec) -> None:
    actor_id = _required_nonempty_string(value, "actor_id")
    session_id = _required_nonempty_string(value, "session_id")
    del actor_id, session_id
    _required_integer(value, "sequence", minimum=0)
    _required_integer(value, "policy_version", minimum=-1)
    transitions = _required_list(value, "transitions")
    episodes = _required_list(value, "episodes")
    evaluations = _optional_list(value, "evaluations")
    for item in transitions:
        _validate_wire_transition(item)
    for summary in episodes:
        _validate_episode_summary(summary)
    for summary in evaluations:
        _validate_evaluation_summary(summary)
    snapshot = value.get("evaluation_snapshot", b"")
    if not isinstance(snapshot, bytes):
        raise TypeError("evaluation_snapshot must be bytes")
    if snapshot:
        if not evaluations:
            raise ValueError("evaluation_snapshot requires evaluations")
        policy_state = codec.decode(snapshot)
        if not isinstance(policy_state, Mapping):
            raise TypeError("evaluation_snapshot must decode to a mapping")
        versions = {
            _required_integer(summary, "policy_version", minimum=0) for summary in evaluations
        }
        if len(versions) != 1:
            raise ValueError("evaluation_snapshot cannot cover mixed policy versions")
        _validate_finite_tree(policy_state, "evaluation_snapshot")


def _validate_wire_transition(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("transitions must contain mappings")
    required = {
        "observation",
        "action",
        "reward",
        "next_observation",
        "terminated",
        "truncated",
        "info",
        "episode_id",
        "step",
    }
    missing = required - value.keys()
    if missing:
        raise ValueError(f"transition is missing {sorted(missing)}")
    if not isinstance(value["terminated"], bool) or not isinstance(value["truncated"], bool):
        raise TypeError("transition terminal flags must be booleans")
    if not isinstance(value["info"], Mapping):
        raise TypeError("transition info must be a mapping")
    episode_id = value["episode_id"]
    if episode_id is not None and (not isinstance(episode_id, str) or not episode_id):
        raise TypeError("transition episode_id must be a non-empty string or null")
    step = value["step"]
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise TypeError("transition step must be a non-negative integer or null")
    _validate_numeric_tree(value["observation"], "transition observation")
    _validate_numeric_tree(value["action"], "transition action")
    _validate_numeric_tree(value["next_observation"], "transition next_observation")
    _validate_finite_number(value["reward"], "transition reward")
    projected = value["info"].get("sampling/projected_lap_time_s")
    if projected is not None:
        _validate_finite_number(projected, "projected lap time")
    transition_from_wire(value)


def _validate_episode_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("episodes must contain mappings")
    numeric = {
        "finish_time_s",
        "progress_pct",
        "return",
        "reward/time",
        "reward/pace",
        "reward/pbrs",
        "reward/progress",
        "reward/projected_velocity",
        "reward/projected_speed",
        "reward/steering_delta",
        "reward/collision",
        "collision/count",
        "collision/detected_count",
        "reward/terminal",
        "velocity/ratio_mean",
        "velocity/ratio_max",
        "steps",
        "race_time_s",
        "exploration_epsilon",
    }
    missing = ({"finished", "termination"} | numeric) - value.keys()
    if missing:
        raise ValueError(f"episode summary is missing {sorted(missing)}")
    _validate_binary_flag(value["finished"], "episode finished")
    if not isinstance(value["termination"], str):
        raise TypeError("episode termination must be a string")
    for key in numeric:
        _validate_finite_number(value[key], f"episode {key}")
    _validate_finite_tree(value, "episode summary")


def _validate_evaluation_summary(value: object) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("evaluations must contain mappings")
    _validate_binary_flag(value.get("finished"), "evaluation finished")
    _validate_finite_number(value.get("finish_time_s"), "evaluation finish_time_s")
    if "policy_version" in value:
        _required_integer(value, "policy_version", minimum=0)
    _validate_finite_tree(value, "evaluation summary")


def _required_list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value[key]
    if not isinstance(result, list):
        raise TypeError(f"{key} must be a list")
    return result


def _optional_list(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key, [])
    if not isinstance(result, list):
        raise TypeError(f"{key} must be a list")
    return result


def _required_nonempty_string(value: Mapping[str, Any], key: str) -> str:
    result = value[key]
    if not isinstance(result, str) or not result:
        raise TypeError(f"{key} must be a non-empty string")
    return result


def _required_integer(value: Mapping[str, Any], key: str, *, minimum: int) -> int:
    result = value[key]
    if isinstance(result, bool) or not isinstance(result, int) or result < minimum:
        raise TypeError(f"{key} must be an integer >= {minimum}")
    return int(result)


def _validate_numeric_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_numeric_tree(item, name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_numeric_tree(item, name)
        return
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
        return
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return
        if not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} contains non-finite values")
        return
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, (int, float, np.number)):
        _validate_finite_number(value, name)
        return
    raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _validate_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_finite_tree(item, name)
        return
    if isinstance(value, (torch.Tensor, np.ndarray, bool, int, float, np.number)):
        _validate_numeric_tree(value, name)
        return
    if value is not None and not isinstance(value, (str, bytes)):
        raise TypeError(f"{name} contains unsupported {type(value).__name__}")


def _validate_finite_number(value: Any, name: str) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")


def _validate_binary_flag(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        return
    _validate_finite_number(value, name)
    if float(value) not in {0.0, 1.0}:
        raise ValueError(f"{name} must be boolean or numeric zero/one")


def _load_state_dict(component: object, state: object) -> None:
    if state is None:
        return
    method = getattr(component, "load_state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no load_state_dict()")
    method(state)


def _snapshot_value(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    return deepcopy(value)
