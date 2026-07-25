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

_EPISODE_START_MARGIN_STEPS = 50


class _MarginTracker:
    """Aggregate greedy action gaps reported by a policy across one episode."""

    def __init__(self) -> None:
        self.total = 0.0
        self.minimum = float("inf")
        self.samples = 0
        self.start_total = 0.0
        self.start_samples = 0

    def record(self, policy: Any, step: int) -> None:
        margin = getattr(policy, "last_q_margin", None)
        if margin is None:
            return
        value = float(margin)
        self.total += value
        self.minimum = min(self.minimum, value)
        self.samples += 1
        if step < _EPISODE_START_MARGIN_STEPS:
            self.start_total += value
            self.start_samples += 1

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {"q_margin_mean": 0.0, "q_margin_min": 0.0, "q_margin_start_mean": 0.0}
        return {
            "q_margin_mean": self.total / self.samples,
            "q_margin_min": self.minimum,
            "q_margin_start_mean": (
                self.start_total / self.start_samples if self.start_samples else 0.0
            ),
        }


class _ControlUsageTracker:
    """Aggregate applied-control and step-timing statistics from environment info."""

    def __init__(self) -> None:
        self.gas_total = 0.0
        self.brake_engaged = 0
        self.steer_abs_total = 0.0
        self.race_ms_total = 0.0
        self.samples = 0

    def record(self, info: Mapping[str, Any]) -> None:
        if "control_gas" not in info:
            return
        self.gas_total += float(info["control_gas"])
        self.brake_engaged += int(float(info["control_brake"]) != 0.0)
        self.steer_abs_total += abs(float(info["control_steer"]))
        self.race_ms_total += float(info.get("step_race_time_ms", 0.0))
        self.samples += 1

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "control_gas_fraction": 0.0,
                "control_brake_fraction": 0.0,
                "control_steer_abs_mean": 0.0,
                "step_race_time_ms_mean": 0.0,
            }
        return {
            "control_gas_fraction": self.gas_total / self.samples,
            "control_brake_fraction": self.brake_engaged / self.samples,
            "control_steer_abs_mean": self.steer_abs_total / self.samples,
            "step_race_time_ms_mean": self.race_ms_total / self.samples,
        }


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
        self.evaluate = threading.Event()
        self._evaluation_index = 0
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
        print(
            f"Actor {self.actor_id} (pid={os.getpid()}) registered with learner at "
            f"{self.target}; collecting rollouts (epsilon={float(initial['epsilon']):.4f})",
            flush=True,
        )
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
                print(
                    f"Actor {self.actor_id} (pid={os.getpid()}): learner not ready at "
                    f"{self.target}; retrying...",
                    flush=True,
                )
                self.stop.wait(1.0)
        raise RuntimeError("actor stopped before registering")

    def _collect(self, environment: Any, pipeline: Any) -> None:
        transitions: list[Transition] = []
        summaries: list[dict[str, Any]] = []
        chunk_started = monotonic()
        episode = 0
        while not self.stop.is_set():
            if self.evaluate.is_set():
                self.evaluate.clear()
                self._evaluate(environment, pipeline)
            observation, _ = environment.reset(seed=self._actor_seed() + episode)
            total_reward = 0.0
            time_reward = 0.0
            pbrs_reward = 0.0
            progress_reward = 0.0
            projected_velocity_reward = 0.0
            steering_delta_reward = 0.0
            collision_reward = 0.0
            collision_count = 0
            collision_detected_count = 0
            terminal_reward = 0.0
            velocity_ratio_sum = 0.0
            velocity_ratio_max = 0.0
            final_info: Mapping[str, Any] = {}
            episode_id = f"{self.actor_id}/{self.session_id}/{episode:08d}"
            # One policy snapshot drives the whole episode: a training lap then
            # measures a single policy version instead of a refresh mixture.
            policy, epsilon, version = self._policy()
            margins = _MarginTracker()
            controls = _ControlUsageTracker()
            for step in range(self.spec.training.max_episode_steps):
                prepared = pipeline.transform_observation(observation)
                action = policy.act(prepared)
                margins.record(policy, step)
                next_observation, reward, terminated, truncated, info = environment.step(action)
                controls.record(info)
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
                time_reward += float(info.get("reward_time", 0.0))
                pbrs_reward += float(info.get("reward_pbrs", 0.0))
                progress_reward += float(info.get("reward_progress", 0.0))
                projected_velocity_reward += float(info.get("reward_projected_velocity", 0.0))
                steering_delta_reward += float(info.get("reward_steering_delta", 0.0))
                collision_reward += float(info.get("reward_collision", 0.0))
                collision_count += int(bool(info.get("collision", False)))
                collision_detected_count += int(bool(info.get("collision_detected", False)))
                terminal_reward += float(info.get("reward_terminal", 0.0))
                velocity_ratio = float(info.get("projected_velocity_ratio", 0.0))
                velocity_ratio_sum += velocity_ratio
                velocity_ratio_max = max(velocity_ratio_max, velocity_ratio)
                final_info = info
                if self._should_flush(transitions, chunk_started):
                    self._spool(transitions, summaries, version)
                    transitions, summaries = [], []
                    chunk_started = monotonic()
                if terminated or truncated or self.stop.is_set():
                    break
            summary_info = {
                **dict(final_info),
                "reward_time": time_reward,
                "reward_pbrs": pbrs_reward,
                "reward_progress": progress_reward,
                "reward_projected_velocity": projected_velocity_reward,
                "reward_steering_delta": steering_delta_reward,
                "reward_collision": collision_reward,
                "collision_count": collision_count,
                "collision_detected_count": collision_detected_count,
                "reward_terminal": terminal_reward,
                "projected_velocity_ratio_mean": velocity_ratio_sum / (step + 1),
                "projected_velocity_ratio_max": velocity_ratio_max,
                "actor_epsilon": epsilon,
                "policy_version": version,
                **margins.summary(),
                **controls.summary(),
            }
            summaries.append(self._summary(total_reward, summary_info, step + 1))
            episode += 1
            _, _, version = self._policy()
            self._spool(transitions, summaries, version)
            transitions, summaries = [], []
            chunk_started = monotonic()

    def _evaluate(self, environment: Any, pipeline: Any) -> None:
        suite = getattr(self.spec, "evaluation", None)
        trials = int(getattr(suite, "trials_per_map", 1))
        policy, _, version = self._policy()
        summaries = [
            self._evaluate_episode(environment, pipeline, policy, version) for _ in range(trials)
        ]
        self._spool([], [], version, evaluations=summaries)

    def _evaluate_episode(
        self, environment: Any, pipeline: Any, policy: Any, version: int
    ) -> dict[str, Any]:
        observation, _ = environment.reset(
            seed=self._actor_seed() + 1_000_000 + self._evaluation_index
        )
        total_reward = 0.0
        time_reward = 0.0
        pbrs_reward = 0.0
        progress_reward = 0.0
        projected_velocity_reward = 0.0
        steering_delta_reward = 0.0
        collision_reward = 0.0
        collision_count = 0
        collision_detected_count = 0
        terminal_reward = 0.0
        velocity_ratio_sum = 0.0
        velocity_ratio_max = 0.0
        final_info: Mapping[str, Any] = {}
        margins = _MarginTracker()
        controls = _ControlUsageTracker()
        for _step in range(self.spec.training.max_episode_steps):
            prepared = pipeline.transform_observation(observation)
            action = policy.act(prepared, deterministic=True)
            margins.record(policy, _step)
            observation, reward, terminated, truncated, info = environment.step(action)
            controls.record(info)
            total_reward += float(reward)
            time_reward += float(info.get("reward_time", 0.0))
            pbrs_reward += float(info.get("reward_pbrs", 0.0))
            progress_reward += float(info.get("reward_progress", 0.0))
            projected_velocity_reward += float(info.get("reward_projected_velocity", 0.0))
            steering_delta_reward += float(info.get("reward_steering_delta", 0.0))
            collision_reward += float(info.get("reward_collision", 0.0))
            collision_count += int(bool(info.get("collision", False)))
            collision_detected_count += int(bool(info.get("collision_detected", False)))
            terminal_reward += float(info.get("reward_terminal", 0.0))
            velocity_ratio = float(info.get("projected_velocity_ratio", 0.0))
            velocity_ratio_sum += velocity_ratio
            velocity_ratio_max = max(velocity_ratio_max, velocity_ratio)
            final_info = info
            if terminated or truncated or self.stop.is_set():
                break
        summary = self._summary(
            total_reward,
            {
                **dict(final_info),
                "reward_time": time_reward,
                "reward_pbrs": pbrs_reward,
                "reward_progress": progress_reward,
                "reward_projected_velocity": projected_velocity_reward,
                "reward_steering_delta": steering_delta_reward,
                "reward_collision": collision_reward,
                "collision_count": collision_count,
                "collision_detected_count": collision_detected_count,
                "reward_terminal": terminal_reward,
                "projected_velocity_ratio_mean": velocity_ratio_sum / (_step + 1),
                "projected_velocity_ratio_max": velocity_ratio_max,
                "actor_epsilon": 0.0,
                "policy_version": version,
                **margins.summary(),
                **controls.summary(),
            },
            _step + 1,
        )
        summary["deterministic"] = 1.0
        self._evaluation_index += 1
        return summary

    def _should_flush(self, transitions: list[Transition], started: float) -> bool:
        return len(transitions) >= self.spec.distributed.rollout_chunk_transitions or (
            bool(transitions) and monotonic() - started >= self.spec.distributed.rollout_flush_s
        )

    @staticmethod
    def _summary(reward: float, info: Mapping[str, Any], transitions: int) -> dict[str, Any]:
        termination = str(info.get("termination_reason") or "max_steps")
        finished = termination == "finished"
        race_time_s = float(info.get("race_time_ms", 0.0)) / 1_000.0
        return {
            "return": reward,
            "reward_per_transition": reward / transitions,
            "reward/time": float(info.get("reward_time", 0.0)),
            "reward/pbrs": float(info.get("reward_pbrs", 0.0)),
            "reward/progress": float(info.get("reward_progress", 0.0)),
            "reward/projected_velocity": float(info.get("reward_projected_velocity", 0.0)),
            "reward/steering_delta": float(info.get("reward_steering_delta", 0.0)),
            "reward/collision": float(info.get("reward_collision", 0.0)),
            "reward/terminal": float(info.get("reward_terminal", 0.0)),
            "potential/progress": float(info.get("potential_progress", 0.0)),
            "velocity/projected_mps": float(info.get("projected_velocity_mps", 0.0)),
            "velocity/ratio": float(info.get("projected_velocity_ratio", 0.0)),
            "velocity/ratio_mean": float(info.get("projected_velocity_ratio_mean", 0.0)),
            "velocity/ratio_max": float(info.get("projected_velocity_ratio_max", 0.0)),
            "collision/count": int(info.get("collision_count", 0)),
            "collision/detected_count": int(info.get("collision_detected_count", 0)),
            "q_margin/mean": float(info.get("q_margin_mean", 0.0)),
            "q_margin/min": float(info.get("q_margin_min", 0.0)),
            "q_margin/start_mean": float(info.get("q_margin_start_mean", 0.0)),
            "control/gas_fraction": float(info.get("control_gas_fraction", 0.0)),
            "control/brake_fraction": float(info.get("control_brake_fraction", 0.0)),
            "control/steer_abs_mean": float(info.get("control_steer_abs_mean", 0.0)),
            "timing/step_race_ms_mean": float(info.get("step_race_time_ms_mean", 0.0)),
            "steps": transitions,
            "progress_pct": float(info.get("progress_pct", 0.0)),
            "progress_m": float(info.get("progress_m", 0.0)),
            "duration_s": float(info.get("episode_elapsed_s", 0.0)),
            "race_time_s": race_time_s,
            "finish_time_s": race_time_s if finished else 0.0,
            "finished": float(finished),
            "termination": termination,
            "termination/finished": float(finished),
            "termination/no_progress": float(termination == "no_progress"),
            "termination/slow_progress": float(termination == "slow_progress"),
            "termination/off_track": float(termination == "off_track"),
            "termination/max_steps": float(termination == "max_steps"),
            "exploration_epsilon": float(info.get("actor_epsilon", 0.0)),
            "policy_version": int(info.get("policy_version", 0)),
        }

    def _spool(
        self,
        transitions: list[Transition],
        summaries: list[dict[str, Any]],
        policy_version: int,
        *,
        evaluations: list[dict[str, Any]] | None = None,
    ) -> None:
        if not transitions and not summaries and not evaluations:
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
            "evaluations": evaluations or [],
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
                    if response.get("evaluate"):
                        self.evaluate.set()
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
        total = 0
        for path in self.spool_dir.glob("*.rollout"):
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

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
