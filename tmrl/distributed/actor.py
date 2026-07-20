"""Continuous rollout actor with disk spooling and atomic policy refresh."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from queue import Empty, Queue
from time import monotonic
from typing import Any

import grpc
from google.protobuf.wrappers_pb2 import BytesValue

from tmrl.core.contracts import ExploratoryPolicy, ReplicablePolicy
from tmrl.core.data import Transition
from tmrl.core.runtime import _instantiate
from tmrl.core.spec import RunSpec
from tmrl.distributed.codec import WireCodec
from tmrl.distributed.protocol import (
    PROTOCOL_VERSION,
    auth_metadata,
    deserialize_message,
    grpc_method,
    run_fingerprint,
    serialize_message,
    transition_to_wire,
)


class _Client:
    def __init__(self, target: str, token: str, codec: WireCodec) -> None:
        options = (
            ("grpc.max_receive_message_length", codec.max_message_bytes),
            ("grpc.max_send_message_length", codec.max_message_bytes),
        )
        self.channel = grpc.insecure_channel(target, options=options)
        self.token = token
        self.codec = codec

    def call(
        self, method: str, value: Mapping[str, Any], *, timeout: float = 10.0
    ) -> Mapping[str, Any]:
        stub = self.channel.unary_unary(
            grpc_method(method),
            request_serializer=serialize_message,
            response_deserializer=deserialize_message,
        )
        response = stub(
            BytesValue(value=self.codec.encode(value)),
            metadata=auth_metadata(self.token),
            timeout=timeout,
        )
        decoded = self.codec.decode(response.value)
        if not isinstance(decoded, Mapping):
            raise ValueError("distributed response must be a mapping")
        return decoded

    def close(self) -> None:
        self.channel.close()


class _PolicyReference:
    def __init__(self, policy: ReplicablePolicy, epsilon: float, version: int) -> None:
        self._lock = threading.Lock()
        self._policy = policy
        self._epsilon = epsilon
        self._version = version

    def get(self) -> tuple[ReplicablePolicy, float, int]:
        with self._lock:
            return self._policy, self._epsilon, self._version

    def replace(self, policy: ReplicablePolicy, epsilon: float, version: int) -> None:
        with self._lock:
            self._policy = policy
            self._epsilon = epsilon
            self._version = version


class ActorRuntime:
    """Drive one environment while networking and learner updates remain asynchronous."""

    def __init__(
        self,
        config_path: Path,
        *,
        target: str,
        actor_id: str | None,
        token: str,
        external_stop: Any | None = None,
    ) -> None:
        self.config_path = config_path.resolve()
        self.base_dir = self.config_path.parent
        self.spec = RunSpec.from_yaml(self.config_path)
        self.target = target
        self.actor_id = actor_id or socket.gethostname()
        self.token = token
        self.external_stop = external_stop
        self.session_id = uuid.uuid4().hex
        self.codec = WireCodec(self.spec.distributed.max_message_bytes)
        self.fingerprint = run_fingerprint(self.spec, self.base_dir)
        self.stop = threading.Event()
        self.stop_reason = "running"
        self.force_refresh = threading.Event()
        # The disk spool is the backpressure boundary. Keeping this queue
        # unbounded prevents a temporarily slow learner from pausing the game.
        self.queue: Queue[Path] = Queue()
        actor_directory = re.sub(r"[^A-Za-z0-9_.-]", "_", self.actor_id)
        self.spool_dir = (
            self.base_dir
            / self.spec.artifacts_dir
            / self.spec.run_id
            / "actors"
            / actor_directory
            / "spool"
        )
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.sequence = self._next_sequence()
        self.client = _Client(target, token, self.codec)
        self._replica_learner: Any = None
        self._policy_ref: _PolicyReference | None = None

    def run_forever(self) -> None:
        pipeline, environment_factory = self._components()
        initial = self._register()
        self._policy_ref = _PolicyReference(self._new_policy(), float(initial["epsilon"]), -1)
        self._refresh_policy()
        for path in sorted(self.spool_dir.glob("*.rollout")):
            self.queue.put(path)
        senders = [
            threading.Thread(
                target=self._sender_loop,
                name=f"tmrl-rollout-sender-{index}",
                daemon=True,
            )
            for index in range(self.spec.distributed.max_inflight_chunks)
        ]
        refresher = threading.Thread(
            target=self._policy_loop, name="tmrl-policy-refresh", daemon=True
        )
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="tmrl-actor-heartbeat", daemon=True
        )
        shutdown = threading.Thread(
            target=self._external_stop_loop, name="tmrl-actor-shutdown", daemon=True
        )
        for sender in senders:
            sender.start()
        refresher.start()
        heartbeat.start()
        shutdown.start()
        environment = environment_factory.create(seed=self._actor_seed())
        try:
            self._collect(environment, pipeline)
        finally:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
            self.stop.set()
            sender_deadline = monotonic() + 10.0
            for sender in senders:
                sender.join(timeout=max(0.0, sender_deadline - monotonic()))
            self.client.close()
            print(f"Actor {self.actor_id} stopped: {self.stop_reason}", flush=True)

    def _components(self) -> tuple[Any, Any]:
        components = self.spec.components
        pipeline = _instantiate(components.feature_pipeline, base_dir=self.base_dir)
        if components.environment is None:
            raise ValueError("distributed actor requires components.environment")
        environment = _instantiate(components.environment, base_dir=self.base_dir)
        model_factory = (
            _instantiate(components.model_factory) if components.model_factory is not None else None
        )
        self._replica_learner = _instantiate(
            components.learner, seed=self._actor_seed(), model_factory=model_factory
        )
        self._replica_learner.setup(
            {
                "seed": self._actor_seed(),
                "run_dir": self.base_dir / self.spec.artifacts_dir / self.spec.run_id,
                "model_factory": model_factory,
            }
        )
        return pipeline, environment

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

    def _register(self) -> Mapping[str, Any]:
        while not self.stop.is_set():
            try:
                return self.client.call("Register", self._request_base())
            except grpc.RpcError as exc:
                if exc.code() in {
                    grpc.StatusCode.UNAUTHENTICATED,
                    grpc.StatusCode.FAILED_PRECONDITION,
                    grpc.StatusCode.INVALID_ARGUMENT,
                    grpc.StatusCode.PERMISSION_DENIED,
                }:
                    raise RuntimeError(f"actor registration rejected: {exc.details()}") from exc
                print(f"Actor {self.actor_id}: waiting for learner at {self.target}...", flush=True)
                self.stop.wait(1.0)
        raise RuntimeError("actor stopped before registering")

    def _collect(self, environment: Any, pipeline: Any) -> None:
        transitions: list[Transition] = []
        summaries: list[dict[str, Any]] = []
        chunk_started = monotonic()
        episode = 0
        while not self.stop.is_set():
            observation, _ = environment.reset(seed=self._actor_seed() + episode)
            total_reward = 0.0
            final_info: Mapping[str, Any] = {}
            episode_id = f"{self.actor_id}/{self.session_id}/{episode:08d}"
            for step in range(self.spec.training.max_episode_steps):
                policy, epsilon, version = self._policy()
                prepared = pipeline.transform_observation(observation)
                action = policy.act(prepared)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                next_prepared = pipeline.transform_observation(next_observation)
                transitions.append(
                    Transition(
                        observation=prepared,
                        action=action,
                        reward=float(reward),
                        next_observation=next_prepared,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info={**dict(info), "policy_version": version, "actor_epsilon": epsilon},
                        episode_id=episode_id,
                        step=step,
                    )
                )
                observation = next_observation
                total_reward += float(reward)
                final_info = info
                if self._should_flush(transitions, chunk_started):
                    self._spool(transitions, summaries, version)
                    transitions, summaries = [], []
                    chunk_started = monotonic()
                if terminated or truncated or self.stop.is_set():
                    break
            summaries.append(self._summary(total_reward, final_info, step + 1))
            episode += 1
            _, _, version = self._policy()
            self._spool(transitions, summaries, version)
            transitions, summaries = [], []
            chunk_started = monotonic()

    def _should_flush(self, transitions: list[Transition], started: float) -> bool:
        return len(transitions) >= self.spec.distributed.rollout_chunk_transitions or (
            bool(transitions) and monotonic() - started >= self.spec.distributed.rollout_flush_s
        )

    def _summary(self, reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
        return {
            "reward": reward,
            "transitions": transitions,
            "progress_pct": float(info.get("progress_pct", 0.0)),
            "progress_m": float(info.get("progress_m", 0.0)),
            "episode_elapsed_s": float(info.get("episode_elapsed_s", 0.0)),
            "race_time_s": float(info.get("race_time_ms", 0.0)) / 1_000.0,
            "termination": str(info.get("termination_reason") or "max_steps"),
        }

    def _spool(
        self,
        transitions: list[Transition],
        summaries: list[dict[str, Any]],
        policy_version: int,
    ) -> None:
        if not transitions and not summaries:
            return
        value = {
            **self._request_base(),
            "sequence": self.sequence,
            "policy_version": min(
                (int(item.info.get("policy_version", policy_version)) for item in transitions),
                default=policy_version,
            ),
            "transitions": [transition_to_wire(item) for item in transitions],
            "episodes": summaries,
        }
        payload = self.codec.encode(value)
        if self._spool_bytes() + len(payload) > self.spec.distributed.spool_max_bytes:
            raise RuntimeError("actor rollout spool reached its configured byte limit")
        path = self.spool_dir / f"{self.sequence:020d}.rollout"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        self.sequence += 1
        self.queue.put(path)

    def _sender_loop(self) -> None:
        while not self.stop.is_set() or not self.queue.empty():
            try:
                path = self.queue.get(timeout=0.2)
            except Empty:
                continue
            while path.exists():
                try:
                    request = self.codec.decode(path.read_bytes())
                    response = self.client.call("Submit", request)
                    if response.get("force_refresh"):
                        self.force_refresh.set()
                    if response.get("stop"):
                        self.stop_reason = "learner requested stop"
                        self.stop.set()
                    if response.get("accepted"):
                        path.unlink(missing_ok=True)
                        continue
                    if self.stop.is_set():
                        break
                    self.force_refresh.set()
                    self.stop.wait(0.1)
                except grpc.RpcError:
                    if self.stop.wait(1.0):
                        break
            self.queue.task_done()

    def _policy_loop(self) -> None:
        refresh_at = monotonic()
        while not self.stop.wait(0.1):
            if not self.force_refresh.is_set() and monotonic() < refresh_at:
                continue
            try:
                self._refresh_policy()
                self.force_refresh.clear()
                refresh_at = monotonic() + self.spec.distributed.policy_refresh_s
            except grpc.RpcError:
                continue

    def _refresh_policy(self) -> None:
        _, _, current = self._policy()
        response = self.client.call("Policy", {**self._request_base(), "current_version": current})
        if response.get("stop"):
            self.stop_reason = "learner requested stop"
            self.stop.set()
        epsilon = float(response["epsilon"])
        version = int(response["policy_version"])
        snapshot = response["snapshot"]
        policy = self._new_policy()
        if snapshot:
            state = self.codec.decode(snapshot)
            if not isinstance(state, Mapping):
                raise ValueError("policy snapshot must decode to a mapping")
            policy.load_state(state)
        else:
            current_policy, _, _ = self._policy()
            policy.load_state(current_policy.export_state())
        if isinstance(policy, ExploratoryPolicy):
            policy.set_exploration_epsilon(epsilon)
        assert self._policy_ref is not None
        self._policy_ref.replace(policy, epsilon, version)

    def _heartbeat_loop(self) -> None:
        while not self.stop.wait(self.spec.distributed.heartbeat_s):
            try:
                _, _, version = self._policy()
                response = self.client.call(
                    "Heartbeat",
                    {
                        **self._request_base(),
                        "policy_version": version,
                        "spool_bytes": self._spool_bytes(),
                    },
                )
                if response.get("stop"):
                    self.stop_reason = "learner requested stop"
                    self.stop.set()
            except grpc.RpcError:
                continue

    def _external_stop_loop(self) -> None:
        if self.external_stop is None:
            return
        self.external_stop.wait()
        self.stop_reason = "local launcher shutdown"
        self.stop.set()

    def _policy(self) -> tuple[ReplicablePolicy, float, int]:
        if self._policy_ref is None:
            raise RuntimeError("actor policy is not initialized")
        return self._policy_ref.get()

    def _spool_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.spool_dir.glob("*.rollout"))

    def _next_sequence(self) -> int:
        existing = [
            int(path.stem) for path in self.spool_dir.glob("*.rollout") if path.stem.isdigit()
        ]
        return max(existing, default=-1) + 1

    def _actor_seed(self) -> int:
        digest = hashlib.sha256(f"{self.spec.seed}:{self.actor_id}".encode()).digest()
        return int.from_bytes(digest[:4], "big")


def actor_process_entry(
    config_path: str,
    target: str,
    actor_id: str | None,
    token: str,
    external_stop: Any | None = None,
) -> None:
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        ActorRuntime(
            Path(config_path),
            target=target,
            actor_id=actor_id,
            token=token,
            external_stop=external_stop,
        ).run_forever()
    except BaseException as exc:
        print(f"Actor process failed: {type(exc).__name__}: {exc}", flush=True)
        raise
