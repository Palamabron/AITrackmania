"""Continuous rollout actor with disk spooling and atomic policy refresh."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import socket
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from queue import Queue
from time import monotonic
from typing import Any

import torch

import trackmaniarl.distributed.actor_background as actor_background
import trackmaniarl.distributed.actor_collection as actor_collection
import trackmaniarl.distributed.actor_evaluation as actor_evaluation
import trackmaniarl.distributed.actor_spool as actor_spool
import trackmaniarl.distributed.actor_transport as actor_transport
from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.contracts import ReplicablePolicy
from trackmaniarl.core.runtime import _instantiate
from trackmaniarl.core.spec import ActorExecutionSpec, ComponentSpec, RunSpec
from trackmaniarl.distributed.actor_errors import (
    ActorBackgroundError as ActorBackgroundError,
)
from trackmaniarl.distributed.actor_errors import (
    ActorEnvironmentError as ActorEnvironmentError,
)
from trackmaniarl.distributed.actor_errors import (
    ActorRuntimeError as ActorRuntimeError,
)
from trackmaniarl.distributed.actor_requests import ActorProcessRequest
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    require_distributed_token,
    run_fingerprint,
)

logger = logging.getLogger(__name__)


def _actor_learner_spec(
    learner: ComponentSpec, override: ActorExecutionSpec | None
) -> ComponentSpec:
    if override is None:
        return learner
    kwargs = dict(learner.kwargs)
    configured = kwargs.get("execution") or {}
    if not isinstance(configured, Mapping):
        raise TypeError("components.learner.kwargs.execution must be a mapping")
    execution = dict(configured)
    execution.update(
        device=override.device,
        precision=override.precision,
    )
    kwargs["execution"] = execution
    return learner.model_copy(update={"kwargs": kwargs})


def _configure_actor_threads(override: ActorExecutionSpec | None) -> None:
    if override is not None and override.torch_threads is not None:
        torch.set_num_threads(override.torch_threads)


class ActorRuntime:
    """Drive one environment while networking and learner updates remain asynchronous."""

    def __init__(self, config: ActorProcessRequest) -> None:
        self.token = require_distributed_token(config.token)
        self.config_path = Path(config.config_path).resolve()
        self.base_dir = self.config_path.parent
        self.spec = RunSpec.from_yaml(self.config_path)
        self.target = config.target
        self.actor_id = config.actor_id or f"{socket.gethostname()}-{os.getpid()}"
        self.external_stop = config.external_stop
        self._initialize_runtime_state()
        self._initialize_spool()
        self._initialize_transport()

    def _initialize_runtime_state(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.codec = WireCodec(self.spec.distributed.max_message_bytes)
        self.fingerprint = run_fingerprint(self.spec, self.base_dir)
        self.stop = threading.Event()
        self.stop_reason = "running"
        self.force_refresh = threading.Event()
        self.evaluate = threading.Event()
        self._evaluation_request_lock = threading.Lock()
        self._evaluation_request: tuple[bytes, int] | None = None
        self._evaluation_index = 0
        self.queue: Queue[Path] = Queue()

    def _initialize_spool(self) -> None:
        actor_directory = re.sub(r"[^A-Za-z0-9_.-]", "_", self.actor_id)
        self.spool_dir = self._spool_directory(actor_directory)
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self._recover_spool_temporaries()
        self._spool_lock = threading.Lock()
        self._spool_bytes_total = self._scan_spool_bytes()
        self.sequence = self._next_sequence()

    def _initialize_transport(self) -> None:
        self.client = actor_transport.Client(self.target, self.token, self.codec)
        self._replica_learner: Any = None
        self._policy_ref: actor_transport.PolicyReference | None = None
        self._background_failure_lock = threading.Lock()
        self._background_failure: ActorBackgroundError | None = None

    def _spool_directory(self, actor_directory: str) -> Path:
        return Path(
            self.base_dir
            / self.spec.artifacts_dir
            / self.spec.run_id
            / "actors"
            / actor_directory
            / "spool"
        )

    def run_forever(self) -> None:
        pipeline, environment_factory = self._components()
        initial = self._register()
        self._initialize_policy(initial)
        senders = actor_background.start_background_workers(self)
        environment = environment_factory.create(seed=self._actor_seed())
        self._collect_environment(environment, pipeline, senders)
        self._raise_background_failure()

    def _initialize_policy(self, initial: Mapping[str, Any]) -> None:
        logger.info(
            "Actor %s (pid=%d) registered with learner at %s; collecting rollouts (epsilon=%.4f)",
            self.actor_id,
            os.getpid(),
            self.target,
            float(initial["epsilon"]),
        )
        self._policy_ref = actor_transport.PolicyReference(
            self._new_policy(), float(initial["epsilon"]), -1
        )
        self._refresh_policy()
        for path in sorted(self.spool_dir.glob("*.rollout")):
            self.queue.put(path)

    def _collect_environment(
        self, environment: Any, pipeline: Any, senders: list[threading.Thread]
    ) -> None:
        try:
            self._collect(environment, pipeline)
        finally:
            self._close_environment(environment)
            self.stop.set()
            self._join_senders(senders)
            self.client.close()
            logger.info("Actor %s stopped: %s", self.actor_id, self.stop_reason)

    @staticmethod
    def _close_environment(environment: Any) -> None:
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _join_senders(senders: list[threading.Thread]) -> None:
        deadline = monotonic() + 10.0
        for sender in senders:
            sender.join(timeout=max(0.0, deadline - monotonic()))

    def _components(self) -> tuple[Any, Any]:
        _configure_actor_threads(self.spec.distributed.actor_execution)
        pipeline = _instantiate(self.spec.components.feature_pipeline, base_dir=self.base_dir)
        environment = self._environment_factory()
        model_factory = self._model_factory()
        self._replica_learner = self._create_replica_learner(model_factory)
        self._replica_learner.setup(self._replica_context(model_factory))
        self._log_replica_execution()
        return pipeline, environment

    def _environment_factory(self) -> Any:
        environment = self.spec.components.environment
        if environment is None:
            raise ValueError("distributed actor requires components.environment")
        return _instantiate(environment, base_dir=self.base_dir)

    def _model_factory(self) -> Any:
        model_factory = self.spec.components.model_factory
        return _instantiate(model_factory) if model_factory is not None else None

    def _create_replica_learner(self, model_factory: Any) -> Any:
        learner_spec = _actor_learner_spec(
            self.spec.components.learner,
            self.spec.distributed.actor_execution,
        )
        return _instantiate(
            learner_spec,
            seed=self._actor_seed(),
            model_factory=model_factory,
            base_dir=self.base_dir,
        )

    def _replica_context(self, model_factory: Any) -> dict[str, Any]:
        return {
            "seed": self._actor_seed(),
            "model_factory": model_factory,
            "restoring_checkpoint": True,
        }

    def _log_replica_execution(self) -> None:
        manifest = getattr(self._replica_learner, "execution_manifest", None)
        if not callable(manifest):
            return
        execution = manifest()
        logger.info(
            "Actor %s policy replica execution: device=%s, precision=%s",
            self.actor_id,
            execution["torch_device"],
            execution["precision"],
        )

    def _new_policy(self) -> ReplicablePolicy:
        policy = self._replica_learner.policy()
        if not isinstance(policy, ReplicablePolicy):
            raise TypeError("distributed actor requires a ReplicablePolicy")
        return policy

    def _request_base(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": self.fingerprint,
            "actor_id": self.actor_id,
            "session_id": self.session_id,
        }

    @staticmethod
    def _persist_spool_payload(path: Path, payload: bytes) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as destination:
            written = destination.write(payload)
            if written != len(payload):
                raise OSError(f"wrote {written} of {len(payload)} rollout bytes")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        sync_checkpoint_path(path)

    def _actor_seed(self) -> int:
        digest = hashlib.sha256(f"{self.spec.seed}:{self.actor_id}".encode()).digest()
        return int.from_bytes(digest[:4], "big")

    _register = actor_background.register
    _collect = actor_collection.collect
    _reset_environment = actor_collection.reset_environment
    _evaluate = actor_evaluation.evaluate
    _evaluation_policy = actor_evaluation.evaluation_policy
    _evaluate_episode = actor_evaluation.evaluate_episode
    _evaluation_telemetry_failure = actor_evaluation.evaluation_telemetry_failure
    _reset_pipeline = staticmethod(actor_collection.reset_pipeline)
    _reset_policy = staticmethod(actor_collection.reset_policy)
    _snapshot_observation = staticmethod(actor_collection.snapshot_observation)
    _should_flush = actor_collection.should_flush
    _summary = staticmethod(actor_collection.summary)
    _spool = actor_spool.spool
    _wait_for_spool_capacity = actor_spool.wait_for_spool_capacity
    _discard_spooled = actor_spool.discard_spooled
    _current_spool_bytes = actor_spool.current_spool_bytes
    _recover_spool_temporaries = actor_spool.recover_spool_temporaries
    _valid_spool_temporary = actor_spool.valid_spool_temporary
    _scan_spool_bytes = actor_spool.scan_spool_bytes
    _next_sequence = actor_spool.next_sequence
    _sender_loop = actor_background.sender_loop
    _send_spooled_rollouts = actor_background.send_spooled_rollouts
    _policy_loop = actor_background.policy_loop
    _refresh_policy = actor_background.refresh_policy
    _heartbeat_loop = actor_background.heartbeat_loop
    _stop_from_thread = actor_background.stop_from_thread
    _raise_background_failure = actor_background.raise_background_failure
    _external_stop_loop = actor_background.external_stop_loop
    _policy = actor_background.policy


def actor_process_entry(config: ActorProcessRequest) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        ActorRuntime(config).run_forever()
    except BaseException as exc:
        logger.error("Actor process failed: %s: %s", type(exc).__name__, exc)
        raise
