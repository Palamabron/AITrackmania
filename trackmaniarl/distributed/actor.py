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
from dataclasses import replace
from math import ceil
from pathlib import Path
from queue import Empty, Queue
from time import monotonic
from typing import Any

import grpc
import numpy as np
import torch
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.builtins import sync_checkpoint_path
from trackmaniarl.core.contracts import ExploratoryPolicy, ReplicablePolicy
from trackmaniarl.core.data import Transition
from trackmaniarl.core.pytree import tree_map
from trackmaniarl.core.runtime import _instantiate
from trackmaniarl.core.spec import ActorExecutionSpec, ComponentSpec, RunSpec
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    auth_metadata,
    deserialize_message,
    grpc_method,
    require_distributed_token,
    run_fingerprint,
    serialize_message,
    transition_to_wire,
)
from trackmaniarl.trackmania.diagnostics import ProgressBinDiagnostics

logger = logging.getLogger(__name__)

_EPISODE_START_MARGIN_STEPS = 50
_TELEMETRY_RESET_ATTEMPTS = 5
_TELEMETRY_RETRY_INITIAL_S = 5.0
_TELEMETRY_RETRY_MAX_S = 60.0
_SPOOL_WAIT_POLL_S = 0.5
_SPOOL_WAIT_WARN_S = 10.0
_RETRYABLE_RPC_CODES = frozenset(
    {
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    }
)


class ActorRuntimeError(RuntimeError):
    """Base class for failures that must terminate an actor process."""


class ActorEnvironmentError(ActorRuntimeError):
    """The actor could not restore its environment connection."""


class ActorBackgroundError(ActorRuntimeError):
    """A required actor background worker failed."""

    def __init__(self, stage: str, cause: BaseException) -> None:
        super().__init__(f"{stage} failed: {type(cause).__name__}: {cause}")
        self.cause = cause


def _is_retryable_rpc_error(exc: grpc.RpcError) -> bool:
    return exc.code() in _RETRYABLE_RPC_CODES


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
    legacy_device = kwargs.pop("device", None)
    if legacy_device is not None:
        execution.setdefault("device", legacy_device)
    execution.update(
        device=override.device,
        precision=override.precision,
        compile=False,
    )
    kwargs["execution"] = execution
    return learner.model_copy(update={"kwargs": kwargs})


def _configure_actor_threads(override: ActorExecutionSpec | None) -> None:
    if override is not None and override.torch_threads is not None:
        torch.set_num_threads(override.torch_threads)


class _InferenceTimingTracker:
    def __init__(self) -> None:
        self.total_s = 0.0
        self.maximum_s = 0.0
        self.samples = 0

    def record(self, duration_s: float) -> None:
        self.total_s += duration_s
        self.maximum_s = max(self.maximum_s, duration_s)
        self.samples += 1

    def summary(self) -> dict[str, float]:
        mean_s = self.total_s / self.samples if self.samples else 0.0
        return {
            "policy_inference_ms_mean": mean_s * 1_000.0,
            "policy_inference_ms_max": self.maximum_s * 1_000.0,
        }


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
    def __init__(self) -> None:
        self.gas_total = 0.0
        self.brake_total = 0.0
        self.brake_taps = 0
        self.steer_abs_total = 0.0
        self.race_ms_total = 0.0
        self.race_ms_values: list[float] = []
        self.samples = 0

    def record(self, info: Mapping[str, Any]) -> None:
        if "control_gas" not in info:
            return
        self.gas_total += float(info["control_gas"])
        self.brake_total += float(info["control_brake"])
        self.brake_taps += int(bool(info.get("control_brake_tap", False)))
        self.steer_abs_total += abs(float(info["control_steer"]))
        race_ms = float(info.get("step_race_time_ms", 0.0))
        self.race_ms_total += race_ms
        self.race_ms_values.append(race_ms)
        self.samples += 1

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "control_gas_fraction": 0.0,
                "control_brake_fraction": 0.0,
                "control_brake_tap_fraction": 0.0,
                "control_steer_abs_mean": 0.0,
                "step_race_time_ms_mean": 0.0,
                "step_race_time_ms_p99": 0.0,
                "step_race_time_ms_max": 0.0,
            }
        ordered = sorted(self.race_ms_values)
        p99_index = ceil(0.99 * len(ordered)) - 1
        return {
            "control_gas_fraction": self.gas_total / self.samples,
            "control_brake_fraction": self.brake_total / self.samples,
            "control_brake_tap_fraction": self.brake_taps / self.samples,
            "control_steer_abs_mean": self.steer_abs_total / self.samples,
            "step_race_time_ms_mean": self.race_ms_total / self.samples,
            "step_race_time_ms_p99": ordered[p99_index],
            "step_race_time_ms_max": ordered[-1],
        }


def _policy_action_count(policy: Any) -> int:
    model = getattr(policy, "model", None)
    count = getattr(model, "action_count", 78)
    if not isinstance(count, int) or count < 2:
        raise ValueError("TrackMania policy must expose at least two actions")
    return count


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
        self.token = require_distributed_token(token)
        self.config_path = config_path.resolve()
        self.base_dir = self.config_path.parent
        self.spec = RunSpec.from_yaml(self.config_path)
        self.target = target
        self.actor_id = actor_id or f"{socket.gethostname()}-{os.getpid()}"
        self.external_stop = external_stop
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
        self._recover_spool_temporaries()
        self._spool_lock = threading.Lock()
        self._spool_bytes_total = self._scan_spool_bytes()
        self.sequence = self._next_sequence()
        self.client = _Client(target, self.token, self.codec)
        self._replica_learner: Any = None
        self._policy_ref: _PolicyReference | None = None
        self._background_failure_lock = threading.Lock()
        self._background_failure: ActorBackgroundError | None = None

    def run_forever(self) -> None:
        pipeline, environment_factory = self._components()
        initial = self._register()
        logger.info(
            "Actor %s (pid=%d) registered with learner at %s; collecting rollouts (epsilon=%.4f)",
            self.actor_id,
            os.getpid(),
            self.target,
            float(initial["epsilon"]),
        )
        self._policy_ref = _PolicyReference(self._new_policy(), float(initial["epsilon"]), -1)
        self._refresh_policy()
        for path in sorted(self.spool_dir.glob("*.rollout")):
            self.queue.put(path)
        senders = [
            threading.Thread(
                target=self._sender_loop,
                name=f"trackmaniarl-rollout-sender-{index}",
                daemon=True,
            )
            for index in range(self.spec.distributed.max_inflight_chunks)
        ]
        refresher = threading.Thread(
            target=self._policy_loop, name="trackmaniarl-policy-refresh", daemon=True
        )
        heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="trackmaniarl-actor-heartbeat", daemon=True
        )
        shutdown = threading.Thread(
            target=self._external_stop_loop, name="trackmaniarl-actor-shutdown", daemon=True
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
            logger.info("Actor %s stopped: %s", self.actor_id, self.stop_reason)
        self._raise_background_failure()

    def _components(self) -> tuple[Any, Any]:
        components = self.spec.components
        _configure_actor_threads(self.spec.distributed.actor_execution)
        pipeline = _instantiate(components.feature_pipeline, base_dir=self.base_dir)
        if components.environment is None:
            raise ValueError("distributed actor requires components.environment")
        environment = _instantiate(components.environment, base_dir=self.base_dir)
        model_factory = (
            _instantiate(components.model_factory) if components.model_factory is not None else None
        )
        learner_spec = _actor_learner_spec(
            components.learner,
            self.spec.distributed.actor_execution,
        )
        self._replica_learner = _instantiate(
            learner_spec,
            seed=self._actor_seed(),
            model_factory=model_factory,
            base_dir=self.base_dir,
        )
        self._replica_learner.setup(
            {
                "seed": self._actor_seed(),
                "run_dir": self.base_dir / self.spec.artifacts_dir / self.spec.run_id,
                "model_factory": model_factory,
            }
        )
        self._log_replica_execution()
        return pipeline, environment

    def _log_replica_execution(self) -> None:
        manifest = getattr(self._replica_learner, "execution_manifest", None)
        if not callable(manifest):
            return
        execution = manifest()
        logger.info(
            "Actor %s policy replica execution: device=%s, precision=%s, compile=%s",
            self.actor_id,
            execution.get("torch_device", execution.get("requested_device", "unknown")),
            execution.get("precision", execution.get("requested_precision", "unknown")),
            execution.get("compile_effective", execution.get("compile_requested", False)),
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

    def _register(self) -> Mapping[str, Any]:
        while not self.stop.is_set():
            try:
                return self.client.call("Register", self._request_base())
            except grpc.RpcError as exc:
                if not _is_retryable_rpc_error(exc):
                    raise RuntimeError(f"actor registration rejected: {exc.details()}") from exc
                logger.info(
                    "Actor %s (pid=%d): learner not ready at %s; retrying...",
                    self.actor_id,
                    os.getpid(),
                    self.target,
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
            observation = self._reset_environment(environment, episode)
            if observation is None:
                break
            self._reset_pipeline(pipeline)
            prepared = pipeline.transform_observation(observation)
            total_reward = 0.0
            time_reward = 0.0
            pbrs_reward = 0.0
            progress_reward = 0.0
            projected_velocity_reward = 0.0
            projected_speed_reward = 0.0
            steering_delta_reward = 0.0
            collision_reward = 0.0
            collision_count = 0
            collision_detected_count = 0
            terminal_reward = 0.0
            time_attack_terminal_reward = 0.0
            pace_reward = 0.0
            velocity_ratio_sum = 0.0
            velocity_ratio_max = 0.0
            final_info: Mapping[str, Any] = {}
            episode_id = f"{self.actor_id}/{self.session_id}/{episode:08d}"
            # One policy snapshot drives the whole episode: a training lap then
            # measures a single policy version instead of a refresh mixture.
            policy, epsilon, version = self._policy()
            self._reset_policy(policy)
            margins = _MarginTracker()
            controls = _ControlUsageTracker()
            inference_timing = _InferenceTimingTracker()
            diagnostics = ProgressBinDiagnostics(_policy_action_count(policy), bin_count=20)
            try:
                for step in range(self.spec.training.max_episode_steps):
                    inference_started = monotonic()
                    sample = getattr(policy, "act_with_info", None)
                    if callable(sample):
                        action, policy_info = sample(prepared)
                    else:
                        action = policy.act(prepared)
                        policy_info = {}
                    inference_timing.record(monotonic() - inference_started)
                    margins.record(policy, step)
                    next_observation, reward, terminated, truncated, info = environment.step(action)
                    if step == self.spec.training.max_episode_steps - 1 and not terminated:
                        truncated = True
                    controls.record(info)
                    diagnostics.record(float(info.get("progress_pct", 0.0)), action, policy, info)
                    next_prepared = pipeline.transform_observation(next_observation)
                    transitions.append(
                        Transition(
                            observation=self._snapshot_observation(prepared),
                            action=action,
                            reward=float(reward),
                            next_observation=self._snapshot_observation(next_prepared),
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            info={
                                **dict(info),
                                "policy_version": version,
                                "actor_epsilon": epsilon,
                                **policy_info,
                            },
                            episode_id=episode_id,
                            step=step,
                        )
                    )
                    prepared = next_prepared
                    total_reward += float(reward)
                    time_reward += float(info.get("reward_time", 0.0))
                    pbrs_reward += float(info.get("reward_pbrs", 0.0))
                    progress_reward += float(info.get("reward_progress", 0.0))
                    projected_velocity_reward += float(info.get("reward_projected_velocity", 0.0))
                    projected_speed_reward += float(info.get("reward_projected_speed", 0.0))
                    steering_delta_reward += float(info.get("reward_steering_delta", 0.0))
                    collision_reward += float(info.get("reward_collision", 0.0))
                    collision_count += int(bool(info.get("collision", False)))
                    collision_detected_count += int(bool(info.get("collision_detected", False)))
                    terminal_reward += float(info.get("reward_terminal", 0.0))
                    time_attack_terminal_reward += float(
                        info.get("reward_time_attack_terminal", 0.0)
                    )
                    pace_reward += float(info.get("reward_pace", 0.0))
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
            except (TimeoutError, ConnectionError) as exc:
                logger.warning(
                    "Actor %s telemetry stalled mid-episode (%s: %s); closing the "
                    "available rollout as a bootstrappable truncation",
                    self.actor_id,
                    type(exc).__name__,
                    exc,
                )
                if transitions:
                    last = transitions[-1]
                    transitions[-1] = replace(
                        last,
                        terminated=False,
                        truncated=True,
                        info={
                            **dict(last.info),
                            "termination_reason": "telemetry_interruption",
                            "telemetry_health": "interrupted",
                        },
                    )
                episode += 1
                _, _, version = self._policy()
                self._spool(transitions, summaries, version)
                transitions, summaries = [], []
                chunk_started = monotonic()
                continue
            summary_info = {
                **dict(final_info),
                "reward_time": time_reward,
                "reward_pbrs": pbrs_reward,
                "reward_progress": progress_reward,
                "reward_projected_velocity": projected_velocity_reward,
                "reward_projected_speed": projected_speed_reward,
                "reward_steering_delta": steering_delta_reward,
                "reward_collision": collision_reward,
                "collision_count": collision_count,
                "collision_detected_count": collision_detected_count,
                "reward_terminal": terminal_reward,
                "reward_time_attack_terminal": time_attack_terminal_reward,
                "reward_pace": pace_reward,
                "projected_velocity_ratio_mean": velocity_ratio_sum / (step + 1),
                "projected_velocity_ratio_max": velocity_ratio_max,
                "actor_epsilon": epsilon,
                "policy_version": version,
                **margins.summary(),
                **controls.summary(),
                **inference_timing.summary(),
                **diagnostics.flat_summary(),
            }
            summaries.append(self._summary(total_reward, summary_info, step + 1))
            episode += 1
            _, _, version = self._policy()
            self._spool(transitions, summaries, version)
            transitions, summaries = [], []
            chunk_started = monotonic()

    def _reset_environment(
        self,
        environment: Any,
        episode: int,
        *,
        attempts: int = _TELEMETRY_RESET_ATTEMPTS,
        stop_on_failure: bool = True,
    ) -> Any:
        if attempts < 1:
            raise ValueError("telemetry reset attempts must be positive")
        delay = _TELEMETRY_RETRY_INITIAL_S
        for attempt in range(attempts):
            try:
                observation, _ = environment.reset(seed=self._actor_seed() + episode)
                return observation
            except (TimeoutError, ConnectionError) as exc:
                if attempt == attempts - 1:
                    reason = (
                        f"telemetry unavailable after {attempts} "
                        f"reset attempts: {type(exc).__name__}: {exc}"
                    )
                    if stop_on_failure:
                        self.stop_reason = reason
                        self.stop.set()
                        raise ActorEnvironmentError(reason) from exc
                    else:
                        logger.warning("Actor %s evaluation %s", self.actor_id, reason)
                    return None
                logger.warning(
                    "Actor %s environment reset failed (%s: %s); retry %d/%d in %.0fs",
                    self.actor_id,
                    type(exc).__name__,
                    exc,
                    attempt + 1,
                    attempts - 1,
                    delay,
                )
                if self.stop.wait(delay):
                    return None
                delay = min(delay * 2.0, _TELEMETRY_RETRY_MAX_S)
        raise AssertionError("unreachable")

    def _evaluate(self, environment: Any, pipeline: Any) -> None:
        suite = getattr(self.spec, "evaluation", None)
        trials = int(getattr(suite, "trials_per_map", 1))
        policy, version = self._evaluation_policy()
        summaries: list[dict[str, Any]] = []
        for trial in range(trials):
            summary = self._evaluate_episode(environment, pipeline, policy, version)
            summaries.append(summary)
            if summary["termination"] == "telemetry_error":
                summaries.extend(
                    self._evaluation_telemetry_failure(version) for _ in range(trial + 1, trials)
                )
                break
        export_state = getattr(policy, "export_state", None)
        snapshot = self.codec.encode(dict(export_state())) if callable(export_state) else None
        self._spool(
            [],
            [],
            version,
            evaluations=summaries,
            evaluation_snapshot=snapshot,
        )

    def _evaluation_policy(self) -> tuple[ReplicablePolicy, int]:
        lock = getattr(self, "_evaluation_request_lock", None)
        if lock is None:
            policy, _, version = self._policy()
            return policy, version
        with lock:
            request = self._evaluation_request
            self._evaluation_request = None
        if request is None:
            policy, _, version = self._policy()
            return policy, version
        snapshot, version = request
        state = self.codec.decode(snapshot)
        if not isinstance(state, Mapping):
            raise ValueError("evaluation policy snapshot must decode to a mapping")
        policy = self._new_policy()
        policy.load_state(state)
        if isinstance(policy, ExploratoryPolicy):
            policy.set_exploration_epsilon(0.0)
        return policy, version

    def _evaluate_episode(
        self, environment: Any, pipeline: Any, policy: Any, version: int
    ) -> dict[str, Any]:
        observation = self._reset_environment(
            environment,
            1_000_000 + self._evaluation_index,
            attempts=1,
            stop_on_failure=False,
        )
        if observation is None:
            return self._evaluation_telemetry_failure(version)
        self._reset_pipeline(pipeline)
        prepared = pipeline.transform_observation(observation)
        self._reset_policy(policy)
        total_reward = 0.0
        time_reward = 0.0
        pbrs_reward = 0.0
        progress_reward = 0.0
        projected_velocity_reward = 0.0
        projected_speed_reward = 0.0
        steering_delta_reward = 0.0
        collision_reward = 0.0
        collision_count = 0
        collision_detected_count = 0
        terminal_reward = 0.0
        time_attack_terminal_reward = 0.0
        pace_reward = 0.0
        velocity_ratio_sum = 0.0
        velocity_ratio_max = 0.0
        final_info: Mapping[str, Any] = {}
        margins = _MarginTracker()
        controls = _ControlUsageTracker()
        inference_timing = _InferenceTimingTracker()
        diagnostics = ProgressBinDiagnostics(_policy_action_count(policy), bin_count=20)
        for _step in range(self.spec.training.max_episode_steps):
            inference_started = monotonic()
            action = policy.act(prepared, deterministic=True)
            inference_timing.record(monotonic() - inference_started)
            margins.record(policy, _step)
            try:
                observation, reward, terminated, truncated, info = environment.step(action)
            except (TimeoutError, ConnectionError) as exc:
                logger.warning(
                    "Actor %s deterministic evaluation telemetry failed (%s: %s)",
                    self.actor_id,
                    type(exc).__name__,
                    exc,
                )
                return self._evaluation_telemetry_failure(version)
            controls.record(info)
            diagnostics.record(float(info.get("progress_pct", 0.0)), action, policy, info)
            prepared = pipeline.transform_observation(observation)
            total_reward += float(reward)
            time_reward += float(info.get("reward_time", 0.0))
            pbrs_reward += float(info.get("reward_pbrs", 0.0))
            progress_reward += float(info.get("reward_progress", 0.0))
            projected_velocity_reward += float(info.get("reward_projected_velocity", 0.0))
            projected_speed_reward += float(info.get("reward_projected_speed", 0.0))
            steering_delta_reward += float(info.get("reward_steering_delta", 0.0))
            collision_reward += float(info.get("reward_collision", 0.0))
            collision_count += int(bool(info.get("collision", False)))
            collision_detected_count += int(bool(info.get("collision_detected", False)))
            terminal_reward += float(info.get("reward_terminal", 0.0))
            time_attack_terminal_reward += float(info.get("reward_time_attack_terminal", 0.0))
            pace_reward += float(info.get("reward_pace", 0.0))
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
                "reward_projected_speed": projected_speed_reward,
                "reward_steering_delta": steering_delta_reward,
                "reward_collision": collision_reward,
                "collision_count": collision_count,
                "collision_detected_count": collision_detected_count,
                "reward_terminal": terminal_reward,
                "reward_time_attack_terminal": time_attack_terminal_reward,
                "reward_pace": pace_reward,
                "projected_velocity_ratio_mean": velocity_ratio_sum / (_step + 1),
                "projected_velocity_ratio_max": velocity_ratio_max,
                "actor_epsilon": 0.0,
                "policy_version": version,
                **margins.summary(),
                **controls.summary(),
                **inference_timing.summary(),
                **diagnostics.flat_summary(),
            },
            _step + 1,
        )
        summary["deterministic"] = 1.0
        self._evaluation_index += 1
        return summary

    def _evaluation_telemetry_failure(self, version: int) -> dict[str, Any]:
        summary = self._summary(
            0.0,
            {
                "termination_reason": "telemetry_error",
                "telemetry_error": 1.0,
                "actor_epsilon": 0.0,
                "policy_version": version,
            },
            1,
        )
        summary["deterministic"] = 1.0
        self._evaluation_index += 1
        return summary

    @staticmethod
    def _reset_pipeline(pipeline: Any) -> None:
        reset = getattr(pipeline, "reset_episode", None)
        if callable(reset):
            reset()

    @staticmethod
    def _reset_policy(policy: Any) -> None:
        reset = getattr(policy, "reset_episode", None)
        if callable(reset):
            reset()

    @staticmethod
    def _snapshot_observation(observation: Any) -> Any:
        def copy_leaf(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return value.clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            return value

        return tree_map(
            copy_leaf,
            observation,
        )

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
            "reward/projected_speed": float(info.get("reward_projected_speed", 0.0)),
            "reward/steering_delta": float(info.get("reward_steering_delta", 0.0)),
            "reward/collision": float(info.get("reward_collision", 0.0)),
            "reward/terminal": float(info.get("reward_terminal", 0.0)),
            "reward/time_attack_terminal": float(info.get("reward_time_attack_terminal", 0.0)),
            "reward/pace": float(info.get("reward_pace", 0.0)),
            "pace/reference_time_s": float(info.get("reference_time_s", 0.0)),
            "pace/time_debt_s": float(info.get("time_debt_s", 0.0)),
            "progress/nearest_distance_m": float(info.get("nearest_distance_m", 0.0)),
            "progress/accepted_delta_m": float(info.get("accepted_progress_delta_m", 0.0)),
            "progress/window_m": float(info.get("window_progress_m", 0.0)),
            "progress/steps_since": float(info.get("steps_since_progress", 0.0)),
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
            "control/brake_tap_fraction": float(info.get("control_brake_tap_fraction", 0.0)),
            "control/steer_abs_mean": float(info.get("control_steer_abs_mean", 0.0)),
            "timing/step_race_ms_mean": float(info.get("step_race_time_ms_mean", 0.0)),
            "timing/step_race_ms_p99": float(info.get("step_race_time_ms_p99", 0.0)),
            "timing/step_race_ms_max": float(info.get("step_race_time_ms_max", 0.0)),
            "timing/policy_inference_ms_mean": float(info.get("policy_inference_ms_mean", 0.0)),
            "timing/policy_inference_ms_max": float(info.get("policy_inference_ms_max", 0.0)),
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
            "termination/time_limit": float(termination == "time_limit"),
            "termination/max_steps": float(termination == "max_steps"),
            "termination/telemetry_error": float(termination == "telemetry_error"),
            "telemetry/error": float(info.get("telemetry_error", 0.0)),
            "exploration_epsilon": float(info.get("actor_epsilon", 0.0)),
            "policy_version": int(info.get("policy_version", 0)),
            **{key: float(value) for key, value in info.items() if key.startswith("progress_bin/")},
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

    def _spool(
        self,
        transitions: list[Transition],
        summaries: list[dict[str, Any]],
        policy_version: int,
        *,
        evaluations: list[dict[str, Any]] | None = None,
        evaluation_snapshot: bytes | None = None,
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
            "evaluation_snapshot": evaluation_snapshot or b"",
        }
        payload = self.codec.encode(value)
        self._wait_for_spool_capacity(len(payload))
        if self.stop.is_set():
            return
        path = self.spool_dir / f"{self.sequence:020d}.rollout"
        self._persist_spool_payload(path, payload)
        with self._spool_lock:
            self._spool_bytes_total += len(payload)
        self.sequence += 1
        self.queue.put(path)

    def _wait_for_spool_capacity(self, payload_bytes: int) -> None:
        limit = self.spec.distributed.spool_max_bytes
        if payload_bytes > limit:
            raise ValueError(f"rollout payload is {payload_bytes} bytes; spool limit is {limit}")
        warned_at = float("-inf")
        while not self.stop.is_set():
            if self._current_spool_bytes() + payload_bytes <= limit:
                return
            if monotonic() - warned_at >= _SPOOL_WAIT_WARN_S:
                logger.warning(
                    "Actor %s rollout spool is full (%d bytes); pausing collection until "
                    "the learner drains it",
                    self.actor_id,
                    self._current_spool_bytes(),
                )
                warned_at = monotonic()
            self.stop.wait(_SPOOL_WAIT_POLL_S)

    def _sender_loop(self) -> None:
        try:
            self._send_spooled_rollouts()
        except Exception as exc:
            self._stop_from_thread("rollout sender", exc)

    def _send_spooled_rollouts(self) -> None:
        while not self.stop.is_set() or not self.queue.empty():
            try:
                path = self.queue.get(timeout=0.2)
            except Empty:
                continue
            while path.exists():
                try:
                    payload = path.read_bytes()
                    request = self.codec.decode(payload)
                    response = self.client.call("Submit", request)
                    if response.get("force_refresh"):
                        self.force_refresh.set()
                    if response.get("evaluate"):
                        snapshot = response.get("evaluation_snapshot", b"")
                        version = int(response.get("evaluation_policy_version", -1))
                        if not isinstance(snapshot, bytes) or not snapshot or version < 0:
                            raise ValueError(
                                "evaluation request requires a policy snapshot/version"
                            )
                        with self._evaluation_request_lock:
                            self._evaluation_request = (snapshot, version)
                        self.evaluate.set()
                    if response.get("stop"):
                        self.stop_reason = "learner requested stop"
                        self.stop.set()
                    if response.get("accepted"):
                        self._discard_spooled(path, len(payload))
                        continue
                    if response.get("reason") == "hard_policy_lag":
                        self._discard_spooled(path, len(payload))
                        self.force_refresh.set()
                        break
                    if self.stop.is_set():
                        break
                    if response.get("reason") != "backpressure":
                        self.force_refresh.set()
                    self.stop.wait(0.1)
                except grpc.RpcError as exc:
                    if not _is_retryable_rpc_error(exc):
                        raise
                    if self.stop.wait(1.0):
                        break
            self.queue.task_done()

    def _discard_spooled(self, path: Path, size: int) -> None:
        path.unlink(missing_ok=True)
        with self._spool_lock:
            self._spool_bytes_total = max(0, self._spool_bytes_total - size)

    def _policy_loop(self) -> None:
        refresh_at = monotonic()
        while not self.stop.wait(0.1):
            if not self.force_refresh.is_set() and monotonic() < refresh_at:
                continue
            try:
                self._refresh_policy()
                self.force_refresh.clear()
                refresh_at = monotonic() + self.spec.distributed.policy_refresh_s
            except grpc.RpcError as exc:
                if _is_retryable_rpc_error(exc):
                    continue
                self._stop_from_thread("policy refresh", exc)
                return
            except Exception as exc:
                self._stop_from_thread("policy refresh", exc)
                return

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
                        "spool_bytes": self._current_spool_bytes(),
                    },
                )
                if response.get("stop"):
                    self.stop_reason = "learner requested stop"
                    self.stop.set()
            except grpc.RpcError as exc:
                if _is_retryable_rpc_error(exc):
                    continue
                self._stop_from_thread("heartbeat", exc)
                return
            except Exception as exc:
                self._stop_from_thread("heartbeat", exc)
                return

    def _stop_from_thread(self, stage: str, exc: BaseException) -> None:
        logger.exception("Actor %s %s failed; stopping the actor", self.actor_id, stage)
        failure = ActorBackgroundError(stage, exc)
        with self._background_failure_lock:
            if self._background_failure is None:
                self._background_failure = failure
                self.stop_reason = str(failure)
        self.stop.set()

    def _raise_background_failure(self) -> None:
        with self._background_failure_lock:
            failure = self._background_failure
        if failure is not None:
            raise failure from failure.cause

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

    def _current_spool_bytes(self) -> int:
        with self._spool_lock:
            return self._spool_bytes_total

    def _recover_spool_temporaries(self) -> None:
        temporaries = sorted(
            (path for path in self.spool_dir.glob("*.tmp") if path.stem.isdigit()),
            key=lambda path: (int(path.stem), path.name),
        )
        occupied = {
            int(path.stem) for path in self.spool_dir.glob("*.rollout") if path.stem.isdigit()
        }
        reserved = occupied | {int(path.stem) for path in temporaries}
        next_sequence = max(reserved, default=-1) + 1
        for temporary in temporaries:
            if not self._valid_spool_temporary(temporary):
                continue
            sequence = int(temporary.stem)
            if sequence in occupied:
                sequence = next_sequence
                next_sequence += 1
            path = self.spool_dir / f"{sequence:020d}.rollout"
            os.replace(temporary, path)
            sync_checkpoint_path(path)
            occupied.add(sequence)
            logger.info("Recovered orphaned actor spool file %s", path.name)

    def _valid_spool_temporary(self, path: Path) -> bool:
        try:
            size = path.stat().st_size
            if size > self.codec.max_message_bytes:
                raise ValueError(
                    f"compressed payload is {size} bytes; limit is {self.codec.max_message_bytes}"
                )
            payload = path.read_bytes()
            value = self.codec.decode(payload)
        except Exception as exc:
            logger.warning(
                "Leaving invalid actor spool temporary %s in place: %s: %s",
                path.name,
                type(exc).__name__,
                exc,
            )
            return False
        if isinstance(value, Mapping):
            return True
        logger.warning("Leaving non-mapping actor spool temporary %s in place", path.name)
        return False

    def _scan_spool_bytes(self) -> int:
        total = 0
        paths = (
            path
            for path in self.spool_dir.iterdir()
            if path.stem.isdigit() and path.suffix in {".rollout", ".tmp"}
        )
        for path in paths:
            try:
                total += path.stat().st_size
            except FileNotFoundError:
                continue
        return total

    def _next_sequence(self) -> int:
        existing = [
            int(path.stem)
            for path in self.spool_dir.iterdir()
            if path.stem.isdigit() and path.suffix in {".rollout", ".tmp"}
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
        logger.error("Actor process failed: %s: %s", type(exc).__name__, exc)
        raise
