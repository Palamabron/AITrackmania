"""Central asynchronous rollout coordinator and learner loop."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import RLock
from time import monotonic
from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import DEFAULT_EVALUATION_TIME_BUCKETS_S
from trackmaniarl.core.training import TrainingResult
from trackmaniarl.distributed import (
    coordinator_checkpoint,
    coordinator_evaluation,
    coordinator_ingest,
    coordinator_learning,
    coordinator_policy,
    coordinator_rpc,
    coordinator_runtime,
    coordinator_support,
)
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig, ReplayRestoreMode
from trackmaniarl.distributed.journal import RolloutJournal
from trackmaniarl.distributed.protocol import (
    require_distributed_token,
    require_loopback_bind,
)


class Coordinator:
    """Own the replay, learner and network-facing rollout journal."""

    def __init__(self, run: ResolvedRun, config: CoordinatorConfig) -> None:
        require_distributed_token(config.token)
        if getattr(run.learner, "on_policy", False):
            raise ValueError(
                "Distributed training does not support on-policy learners; "
                "use trackmaniarl train for PPO"
            )
        self._configure(run, config)
        self._initialize_transport()
        self._initialize_metrics()
        self._initialize_evaluation()

    def _configure(self, run: ResolvedRun, config: CoordinatorConfig) -> None:
        self.token = config.token
        self.run = run
        self.bind = require_loopback_bind(config.bind)
        self.fingerprint = config.fingerprint
        self.resume_checkpoint = config.resume_checkpoint
        self.restore_mode = config.restore_mode
        self.external_stop = config.external_stop
        self.demo_paths = config.demo_paths
        self.codec = WireCodec(run.spec.distributed.max_message_bytes)
        self.journal = RolloutJournal(run.run_dir / "distributed" / "rollouts.sqlite3")
        self.counters = coordinator_support._Counters()

    def _initialize_transport(self) -> None:
        self._lock = RLock()
        self._policy_payload = b""
        self._last_policy_publish = 0.0
        self._last_policy_update = -1
        self._server: grpc.Server | None = None
        self._bound_port: int | None = None
        self._rpc_executor: ThreadPoolExecutor | None = None
        self._checkpoints: list[Path] = []
        self._last_progress_print = 0
        self._last_heartbeats: dict[str, float] = {}
        self._timed_out_actors: set[str] = set()
        self._evaluation_due: set[str] = set()
        self._last_ingest_at = monotonic()
        self._started_at = monotonic()
        self._rollouts: Queue[tuple[int, float]] = Queue(
            maxsize=coordinator_support.ROLLOUT_QUEUE_MAXSIZE
        )
        self._journal_enqueued_at: dict[int, float] = {}

    def _initialize_metrics(self) -> None:
        self._metrics = coordinator_support._MetricAccumulator()
        self._metric_window_started = monotonic()
        self._last_metric_credit = 0.0
        self._growing_credit_windows = 0
        self._last_logging_s = 0.0
        self._last_metric_transitions = 0

    def _initialize_evaluation(self) -> None:
        evaluation = self.run.spec.evaluation
        self._time_buckets = (
            evaluation.time_buckets_s
            if evaluation is not None
            else DEFAULT_EVALUATION_TIME_BUCKETS_S
        )
        self._best_evaluation: tuple[float, float, float] | None = None
        self._evaluation_policy_states: dict[int, Mapping[str, Any]] = {}
        self._consecutive_evaluation_passes = 0
        self._evaluation_stop_reason: str | None = None
        self._recovering = False
        self._checkpoint_writer = coordinator_support._AsyncCheckpointWriter(
            self.run.checkpoint_codec
        )

    @property
    def bound_port(self) -> int:
        if self._bound_port is None:
            raise RuntimeError("distributed learner has not bound its gRPC port")
        return self._bound_port

    def run_forever(self) -> TrainingResult:
        return coordinator_runtime.run_forever(self)

    def run_offline_pretraining(self) -> TrainingResult:
        """Train only from configured demonstrations without opening the actor server."""

        return coordinator_runtime.run_offline_pretraining(self)

    def _log_run_failure(self, phase: str, exc: BaseException) -> None:
        coordinator_runtime.log_run_failure(self, phase, exc)

    def _close_runtime(self) -> None:
        coordinator_runtime.close_runtime(self)

    def _prepare_training(self) -> None:
        coordinator_runtime.prepare_training(self)

    def _start_server(self) -> None:
        coordinator_rpc.start_server(self)

    def _import_demonstrations(self) -> None:
        coordinator_learning.import_demonstrations(self)

    def _offline_pretrain(self) -> None:
        coordinator_learning.offline_pretrain(self)

    def _request(
        self,
        request: BytesValue,
        context: grpc.ServicerContext[Any, Any],
    ) -> Mapping[str, Any]:
        return coordinator_rpc.request(self, request, context)

    def _response(
        self,
        value: Mapping[str, Any],
        context: grpc.ServicerContext[Any, Any],
    ) -> BytesValue:
        return coordinator_rpc.response(self, value, context)

    def _log_rollout_rejected(
        self,
        value: Mapping[str, Any],
        rejection: coordinator_support._RolloutRejection,
    ) -> None:
        coordinator_rpc.log_rollout_rejected(self, value, rejection)

    def _log_wal_error(self, operation: str, exc: BaseException) -> None:
        coordinator_ingest.log_wal_error(self, operation, exc)

    def _journal_rows(self, watermark: int, operation: str) -> Iterator[tuple[int, bytes]]:
        return coordinator_ingest.journal_rows(self, watermark, operation)

    def _decode_journal_payload(self, payload: bytes, operation: str) -> Mapping[str, Any]:
        return coordinator_ingest.decode_journal_payload(self, payload, operation)

    def _register(
        self,
        request: BytesValue,
        context: grpc.ServicerContext[Any, Any],
    ) -> BytesValue:
        return coordinator_rpc.register(self, request, context)

    def _submit(
        self,
        request: BytesValue,
        context: grpc.ServicerContext[Any, Any],
    ) -> BytesValue:
        return coordinator_rpc.submit(self, request, context)

    def _policy(
        self,
        request: BytesValue,
        context: grpc.ServicerContext[Any, Any],
    ) -> BytesValue:
        return coordinator_rpc.policy(self, request, context)

    def _heartbeat(
        self,
        request: BytesValue,
        context: grpc.ServicerContext[Any, Any],
    ) -> BytesValue:
        return coordinator_rpc.heartbeat(self, request, context)

    def _epsilon(self, profile: int) -> float:
        return coordinator_rpc.epsilon(self, profile)

    def _ingest(self, value: Mapping[str, Any], row_id: int) -> None:
        coordinator_ingest.ingest(self, value, row_id)

    def _log_episode(self, value: Mapping[str, Any], summary: Mapping[str, Any]) -> None:
        coordinator_ingest.log_episode(self, value, summary)

    def _learn(self) -> None:
        coordinator_learning.learn(self)

    def _drain_rollouts(self, limit: int) -> None:
        coordinator_ingest.drain_rollouts(self, limit)

    def _finish_evaluation_batch(self, summaries: list[dict[str, Any]]) -> None:
        coordinator_evaluation.finish_evaluation_batch(self, summaries)

    def _record_evaluation_stop(self, stats: Mapping[str, float]) -> None:
        coordinator_evaluation.record_evaluation_stop(self, stats)

    def _record_best_evaluation(
        self, candidate: coordinator_evaluation._EvaluationCandidate
    ) -> None:
        coordinator_evaluation.record_best_evaluation(self, candidate)

    def _emit_metrics_if_ready(self) -> None:
        coordinator_learning.emit_metrics_if_ready(self)

    def _check_actor_timeouts(self) -> None:
        coordinator_learning.check_actor_timeouts(self)

    def _has_active_actor(self) -> bool:
        return coordinator_learning.has_active_actor(self)

    def _can_update(self) -> bool:
        return coordinator_learning.can_update(self)

    def _publish_policy(
        self,
        mode: coordinator_policy.PolicyPublicationMode = (
            coordinator_policy.PolicyPublicationMode.SCHEDULED
        ),
    ) -> None:
        coordinator_learning.publish_policy(self, mode)

    def _should_stop(self) -> bool:
        return coordinator_learning.should_stop(self)

    def _external_stop_requested(self) -> bool:
        return coordinator_learning.external_stop_requested(self)

    def _checkpoint(
        self,
        *,
        policy_state: Mapping[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> Path:
        return coordinator_checkpoint.checkpoint(
            self, policy_state=policy_state, policy_version=policy_version
        )

    def restore_checkpoint(
        self, path: Path, mode: ReplayRestoreMode = ReplayRestoreMode.FULL
    ) -> None:
        coordinator_checkpoint.restore_checkpoint(self, path, mode)

    def _recover_journal(self, watermark: int) -> None:
        coordinator_ingest.recover_journal(self, watermark)

    def _log_execution(self) -> None:
        coordinator_learning.log_execution(self)
