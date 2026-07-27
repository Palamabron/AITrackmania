"""Central asynchronous rollout coordinator and learner loop."""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Queue
from statistics import fmean, median
from threading import RLock
from time import monotonic, perf_counter, sleep
from typing import Any, cast

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from tmrl.core.contracts import ReplicablePolicy
from tmrl.core.data import BatchRequest, TrainingBatch
from tmrl.core.runtime import ResolvedRun, prepare_run, resolve_run
from tmrl.core.spec import RunSpec
from tmrl.core.training import TrainingResult
from tmrl.distributed.codec import WireCodec
from tmrl.distributed.journal import RolloutJournal
from tmrl.distributed.protocol import (
    PROTOCOL_VERSION,
    SERVICE,
    authenticate,
    deserialize_message,
    require_loopback_bind,
    run_fingerprint,
    serialize_message,
    transition_from_wire,
)


@dataclass(slots=True)
class _Counters:
    transitions: int = 0
    episodes: int = 0
    finishes: int = 0
    best_finish_time_s: float = 0.0
    evaluations: int = 0
    evaluation_finishes: int = 0
    evaluation_sub40: int = 0
    evaluation_sub38: int = 0
    evaluation_sub36: int = 0
    updates: int = 0
    update_credit: float = 0.0
    journal_watermark: int = 0
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
    def __init__(self, run: ResolvedRun) -> None:
        self.run = run
        self.prepare = getattr(run.learner, "prepare_batch", None)
        self.enabled = bool(getattr(run.sampler, "thread_safe_prefetch", False))
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmrl-batch-prefetch")
            if self.enabled
            else None
        )
        self.pending: Future[_PreparedBatch] | None = None

    def next(self, request: BatchRequest) -> tuple[TrainingBatch, float, float]:
        waited_at = perf_counter()
        prepared = self.pending.result() if self.pending is not None else self._prepare(request)
        wait_s = perf_counter() - waited_at
        if self.executor is not None:
            self.pending = self.executor.submit(self._prepare, request)
        return prepared.batch, prepared.preparation_s, wait_s

    def _prepare(self, request: BatchRequest) -> _PreparedBatch:
        started = perf_counter()
        batch = self.run.sampler.sample(self.run.replay_store, request)
        if callable(self.prepare):
            batch = self.prepare(batch)
        return _PreparedBatch(batch, perf_counter() - started)

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


class _MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.maximums: dict[str, float] = {}
        self.count = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            numeric = float(value)
            if key.endswith("_max"):
                self.maximums[key] = max(self.maximums.get(key, numeric), numeric)
            else:
                self.values[key] = self.values.get(key, 0.0) + numeric
        self.count += 1

    def flush(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        output = {key: value / self.count for key, value in self.values.items()}
        output.update(self.maximums)
        self.values.clear()
        self.maximums.clear()
        self.count = 0
        return output


class _AsyncCheckpointWriter:
    def __init__(self, codec: Any) -> None:
        self.codec = codec
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tmrl-checkpoint")
        self.pending: Any = None

    def submit(self, state: Mapping[str, Any], path: Path) -> None:
        self.wait()
        self.pending = self.executor.submit(self.codec.save, state, path)

    def wait(self) -> None:
        if self.pending is not None:
            self.pending.result()
            self.pending = None

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
    ) -> None:
        self.run = run
        self.bind = require_loopback_bind(bind)
        self.token = token
        self.fingerprint = fingerprint
        self.resume_checkpoint = resume_checkpoint
        self.reset_replay = reset_replay
        self.external_stop = external_stop
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
        self._rollouts: Queue[_PendingRollout] = Queue()
        self._metrics = _MetricAccumulator()
        self._metric_window_started = monotonic()
        self._last_metric_credit = 0.0
        self._growing_credit_windows = 0
        self._last_logging_s = 0.0
        self._last_metric_transitions = 0
        self._best_evaluation: tuple[float, float] | None = None
        self._evaluation_policy_states: dict[int, Mapping[str, Any]] = {}
        self._recovering = False
        self._checkpoint_writer = _AsyncCheckpointWriter(run.checkpoint_codec)

    def run_forever(self) -> TrainingResult:
        self.run.learner.setup(
            {
                "seed": self.run.spec.seed,
                "run_dir": self.run.run_dir,
                "model_factory": self.run.model_factory,
            }
        )
        prepare_run(self.run)
        self._log_execution()
        if self.resume_checkpoint is not None:
            print(f"Restoring checkpoint: {self.resume_checkpoint}", flush=True)
            self.restore_checkpoint(self.resume_checkpoint, reset_replay=self.reset_replay)
            restored = (
                "learner state only; replay and runtime counters reset"
                if self.reset_replay
                else "full state"
            )
            print(
                f"Checkpoint restored ({restored}): "
                f"transitions={self.counters.transitions}, updates={self.counters.updates}",
                flush=True,
            )
        else:
            self._recover_journal(0)
        self._publish_policy(force=True)
        self._start_server()
        print(
            f"Async learner ready (pid={os.getpid()}): run_id={self.run.spec.run_id}, "
            f"gRPC bind={self.bind}, target_transitions="
            f"{self.run.spec.training.total_transitions}",
            flush=True,
        )
        try:
            self._learn()
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
            self._checkpoints.append(self._checkpoint())
            self._checkpoint_writer.wait()
            raise
        finally:
            if self._server is not None:
                self._server.stop(grace=2).wait(timeout=5)
            if self._rpc_executor is not None:
                self._rpc_executor.shutdown(wait=True, cancel_futures=True)
            self._checkpoint_writer.close()
            self.journal.close()

    def _start_server(self) -> None:
        options = (
            ("grpc.max_receive_message_length", self.run.spec.distributed.max_message_bytes),
            ("grpc.max_send_message_length", self.run.spec.distributed.max_message_bytes),
        )
        executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tmrl-grpc")
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
        policy_version = int(value["policy_version"])
        with self._lock:
            lag = max(0, self.counters.updates - policy_version)
            stop = self._should_stop()
        if lag > self.run.spec.distributed.hard_policy_lag_updates:
            return self._response(
                {
                    "accepted": False,
                    "reason": "hard_policy_lag",
                    "force_refresh": True,
                    "stop": stop,
                }
            )
        row_id, inserted = self.journal.append(
            str(value["session_id"]), int(value["sequence"]), request.value
        )
        if inserted:
            self._rollouts.put(_PendingRollout(value, row_id, monotonic()))
        with self._lock:
            actor_id = str(value["actor_id"])
            evaluate = actor_id in self._evaluation_due
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
        if self.counters.transitions < self.run.spec.training.warmup_transitions:
            return 1.0
        spec = self.run.spec.distributed
        elapsed = self.counters.transitions - self.run.spec.training.warmup_transitions
        fraction = min(1.0, elapsed / spec.epsilon_decay_transitions)
        scheduled = spec.epsilon_start + fraction * (spec.epsilon_final - spec.epsilon_start)
        return scheduled * spec.epsilon_profiles[profile]

    def _ingest(self, value: Mapping[str, Any], row_id: int) -> None:
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
        self.counters.journal_watermark = max(self.counters.journal_watermark, row_id)
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
            sub40 = finished and finish_time_s < 40.0
            sub38 = finished and finish_time_s < 38.0
            sub36 = finished and finish_time_s < 36.0
            self.counters.evaluation_finishes += int(finished)
            self.counters.evaluation_sub40 += int(sub40)
            self.counters.evaluation_sub38 += int(sub38)
            self.counters.evaluation_sub36 += int(sub36)
            if not self._recovering:
                self.run.logger.log(
                    "eval/episode",
                    {
                        **summary,
                        "index": self.counters.evaluations,
                        "finish_rate": self.counters.evaluation_finishes
                        / self.counters.evaluations,
                        "sub40": float(sub40),
                        "sub40_rate": self.counters.evaluation_sub40 / self.counters.evaluations,
                        "sub38": float(sub38),
                        "sub38_rate": self.counters.evaluation_sub38 / self.counters.evaluations,
                        "sub36": float(sub36),
                        "sub36_rate": self.counters.evaluation_sub36 / self.counters.evaluations,
                        "actor_id": value["actor_id"],
                    },
                    step=self.counters.updates,
                )
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
        progress = float(info.get("progress_pct", 0.0))
        race_time_s = float(info.get("race_time_ms", 0.0)) / 1_000.0
        if progress >= 10.0 and race_time_s > 0.0:
            replay_info["sampling/projected_lap_time_s"] = race_time_s * 100.0 / progress
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
        print(
            f"Actor {value['actor_id']} episode {self.counters.episodes}: "
            f"progress={float(summary['progress_pct']):.1f}%, "
            f"return={float(summary['return']):.3f}, "
            f"reward(time={float(summary['reward/time']):.3f}, "
            f"pbrs={float(summary['reward/pbrs']):.3f}, "
            f"progress={float(summary['reward/progress']):.3f}, "
            f"projected_velocity={float(summary['reward/projected_velocity']):.3f}, "
            f"projected_speed={float(summary['reward/projected_speed']):.3f}, "
            f"steering_delta={float(summary['reward/steering_delta']):.3f}, "
            f"collision={float(summary['reward/collision']):.3f} "
            f"({int(summary['collision/count'])}/"
            f"{int(summary['collision/detected_count'])}), "
            f"terminal={float(summary['reward/terminal']):.3f}), "
            f"velocity_ratio(mean={float(summary['velocity/ratio_mean']):.3f}, "
            f"max={float(summary['velocity/ratio_max']):.3f}), "
            f"steps={int(summary['steps'])}, "
            f"race={float(summary['race_time_s']):.2f}s, "
            f"epsilon={float(summary['exploration_epsilon']):.3f}, "
            f"policy={int(summary.get('policy_version', 0))}, "
            f"q_margin(start={float(summary.get('q_margin/start_mean', 0.0)):.2f}, "
            f"min={float(summary.get('q_margin/min', 0.0)):.2f}), "
            f"termination={summary['termination']}",
            flush=True,
        )

    def _learn(self) -> None:
        spec = self.run.spec.training
        footprint = spec.batch_size * spec.sequence_length + spec.n_step - 1
        ready = max(spec.warmup_transitions, footprint)
        prefetcher = _BatchPrefetcher(self.run)
        try:
            while not self._external_stop_requested() and (
                not self._should_stop()
                or (len(self.run.replay_store) >= ready and self.counters.update_credit >= 1.0)
                or not self._rollouts.empty()
            ):
                did_update = False
                self._check_actor_timeouts()
                # Ingest the whole backlog every iteration: a standing queue
                # would otherwise train the learner on minutes-old transitions
                # and inflate the measured actor policy lag by the queue delay.
                self._drain_rollouts(max(1, self._rollouts.qsize()))
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
                        print(
                            "Async training progress: "
                            f"transitions={self.counters.transitions}/"
                            f"{spec.total_transitions}, updates={self.counters.updates}, "
                            f"replay={len(self.run.replay_store)}, "
                            f"credit={self.counters.update_credit:.1f}",
                            flush=True,
                        )
                    if self.counters.updates % spec.checkpoint_interval_updates == 0:
                        self._checkpoints.append(self._checkpoint())
                    self._publish_policy()
                if not did_update:
                    sleep(0.005)
        finally:
            prefetcher.close()

    def _drain_rollouts(self, limit: int) -> None:
        for _ in range(limit):
            try:
                pending = self._rollouts.get_nowait()
            except Empty:
                return
            value = dict(pending.value)
            value["_enqueued_at"] = pending.enqueued_at
            self._ingest(value, pending.row_id)
            self._rollouts.task_done()

    def _finish_evaluation_batch(self, summaries: list[dict[str, Any]]) -> None:
        stats = _evaluation_batch_stats(summaries)
        self.run.logger.log("eval/summary", stats, step=self.counters.updates)
        print(
            f"Deterministic evaluation @update {self.counters.updates}: "
            f"{int(stats['finished_trials'])}/{int(stats['trials'])} finished, "
            f"mean={stats['finish_time_mean_s']:.2f}s, "
            f"best={stats['finish_time_best_s']:.2f}s, "
            f"policy_version={int(stats['policy_version'])}",
            flush=True,
        )
        self._record_best_evaluation(
            float(stats["finish_rate"]),
            float(stats["finish_time_mean_s"]),
            int(stats["policy_version"]),
        )

    def _record_best_evaluation(
        self, finish_rate: float, mean_time_s: float, policy_version: int
    ) -> None:
        if self._recovering:
            return
        candidate = (finish_rate, -mean_time_s)
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
                "finish_time_mean_s": mean_time_s,
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
            "update_throughput_ratio": updates_per_s / max(target_updates_per_s, 1e-6),
            "update_backlog_s": self.counters.update_credit / max(updates_per_s, 1e-6),
            "episodes": self.counters.episodes,
            "finish_rate": self.counters.finishes / max(1, self.counters.episodes),
            "per_beta": self.run.spec.training.replay_beta(self.counters.transitions),
            "timing/logging_s": self._last_logging_s,
        }
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
            if now - heartbeat <= timeout or actor_id in self._timed_out_actors:
                continue
            self._timed_out_actors.add(actor_id)
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
            "learner": _snapshot_value(learner_state),
            "replay_store": _state_dict(self.run.replay_store),
            "sampler": _state_dict(self.run.sampler),
            "distributed": {
                "transitions": self.counters.transitions,
                "episodes": self.counters.episodes,
                "finishes": self.counters.finishes,
                "best_finish_time_s": self.counters.best_finish_time_s,
                "evaluations": self.counters.evaluations,
                "evaluation_finishes": self.counters.evaluation_finishes,
                "evaluation_sub40": self.counters.evaluation_sub40,
                "evaluation_sub38": self.counters.evaluation_sub38,
                "evaluation_sub36": self.counters.evaluation_sub36,
                "updates": self.counters.updates,
                "update_credit": self.counters.update_credit,
                "journal_watermark": self.counters.journal_watermark,
                "policy_version": self.counters.policy_version,
                "actor_sequences": dict(self.counters.actor_sequences),
            },
            "evaluated_policy_version": policy_version,
        }
        self._checkpoint_writer.submit(state, path)
        self.run.logger.log(
            "train/checkpoint",
            {
                "path": str(path),
                "timing/checkpoint_snapshot_s": perf_counter() - checkpoint_started,
            },
            step=self.counters.updates,
        )
        print(f"Checkpoint queued: {path}", flush=True)
        return path

    def restore_checkpoint(self, path: Path, *, reset_replay: bool = False) -> None:
        """Restore a checkpoint, optionally retaining only the learner state."""

        self._restore(path, reset_replay=reset_replay)

    def _restore(self, path: Path, *, reset_replay: bool) -> None:
        state = self.run.checkpoint_codec.load(path)
        if state.get("schema_version") != "2.0":
            raise ValueError("async runtime only resumes distributed checkpoint schema 2.0")
        self.run.learner.load_state_dict(state["learner"])
        if reset_replay:
            self.counters = _Counters()
            return
        distributed = state["distributed"]
        self.counters = _Counters(**distributed)
        self.counters.update_credit = min(
            self.counters.update_credit,
            float(self.run.spec.distributed.max_update_credit),
        )
        _load_state_dict(self.run.replay_store, state["replay_store"])
        _load_state_dict(self.run.sampler, state["sampler"])
        self._recover_journal(self.counters.journal_watermark)

    def _recover_journal(self, watermark: int) -> None:
        self._recovering = True
        try:
            for row_id, payload in self.journal.rows_after(watermark):
                value = self.codec.decode(payload)
                if not isinstance(value, Mapping):
                    raise ValueError("journal chunk must decode to a mapping")
                self._ingest(value, row_id)
        finally:
            self._recovering = False

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
        ).run_forever()
    finally:
        run.logger.close()


def _evaluation_batch_stats(summaries: list[dict[str, Any]]) -> dict[str, float]:
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
    return {
        "trials": float(trials),
        "finished_trials": float(len(finished_times)),
        "finish_rate": len(finished_times) / trials,
        "finish_time_mean_s": fmean(finished_times) if finished_times else 0.0,
        "finish_time_median_s": median(finished_times) if finished_times else 0.0,
        "finish_time_best_s": finished_times[0] if finished_times else 0.0,
        "sub40_rate": sum(1 for time_s in finished_times if time_s < 40.0) / trials,
        "sub38_rate": sum(1 for time_s in finished_times if time_s < 38.0) / trials,
        "sub36_rate": sum(1 for time_s in finished_times if time_s < 36.0) / trials,
        "failure_progress_mean_pct": fmean(failure_progress) if failure_progress else 100.0,
        "failure_progress_median_pct": median(failure_progress) if failure_progress else 100.0,
        "failure_progress_best_pct": max(failure_progress) if failure_progress else 100.0,
        "collision_rate": sum(
            int(float(item.get("collision/count", 0.0)) > 0.0) for item in summaries
        )
        / trials,
        "off_track_rate": sum(
            int(str(item.get("termination", "")) == "off_track") for item in summaries
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


def _state_dict(component: object) -> Mapping[str, object] | None:
    method = getattr(component, "state_dict", None)
    return cast(Mapping[str, object], method()) if callable(method) else None


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
