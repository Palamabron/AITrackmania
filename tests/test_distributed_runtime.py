from __future__ import annotations

import multiprocessing
import socket
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any, cast

import grpc
import numpy as np
import pytest
import torch
from tmrl.core.builtins import TorchCheckpointCodec
from tmrl.core.data import BatchRequest, Transition
from tmrl.core.replay import InMemoryReplayStore, PrioritizedSampler, UniformSampler, _make_batch
from tmrl.core.runtime import ResolvedRun
from tmrl.core.spec import RunSpec
from tmrl.distributed.actor import ActorRuntime, _Client, _PolicyReference
from tmrl.distributed.codec import WireCodec
from tmrl.distributed.coordinator import Coordinator, _Counters, _MetricAccumulator
from tmrl.distributed.journal import RolloutJournal
from tmrl.distributed.protocol import (
    PROTOCOL_VERSION,
    authenticate,
    require_loopback_bind,
    run_fingerprint,
    transition_to_wire,
)


class _Pipeline:
    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> Mapping[str, Any]:
        return {"reward": np.asarray([item.reward for item in transitions])}


def test_metric_accumulator_averages_windows_and_preserves_maxima() -> None:
    metrics = _MetricAccumulator()
    metrics.add({"loss": 2.0, "debug/gradient_norm_max": 3.0})
    metrics.add({"loss": 4.0, "debug/gradient_norm_max": 2.0})

    output = metrics.flush()

    assert output == {"loss": 3.0, "debug/gradient_norm_max": 3.0}


class _Policy:
    def __init__(self, value: int) -> None:
        self.value = value

    def act(self, observation: Any, *, deterministic: bool = False) -> int:
        return self.value

    def export_state(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.value = int(state["value"])


class _Context:
    def __init__(self, authorization: str) -> None:
        self.authorization = authorization

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return (("authorization", self.authorization),)

    def abort(self, code: grpc.StatusCode, message: str) -> None:
        raise RuntimeError(f"{code.name}: {message}")


class _SlowLearner:
    def __init__(self) -> None:
        self.value = 0

    def setup(self, context: Mapping[str, Any]) -> None:
        del context

    def update(self, batch: Any) -> Mapping[str, float]:
        del batch
        time.sleep(0.01)
        self.value += 1
        return {"loss/fake": 1.0 / self.value}

    def policy(self) -> _Policy:
        return _Policy(self.value)

    def state_dict(self) -> Mapping[str, Any]:
        return {"value": self.value}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.value = int(state["value"])


class _RestoreSpy:
    def __init__(self) -> None:
        self.restored: Mapping[str, Any] | None = None

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.restored = state


class _Logger:
    def __init__(self) -> None:
        self.events: list[str] = []

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del payload, step
        self.events.append(event)

    def close(self) -> None:
        return


def _spawn_probe(queue: Any) -> None:
    queue.put("spawn-ok")


def _environment_probe(queue: Any) -> None:
    import torch

    queue.put((sys.prefix, torch.__version__, torch.version.cuda))


def _transition(actor: str, step: int, reward: float, *, terminal: bool = False) -> Transition:
    return Transition(
        observation=np.asarray([step], dtype=np.float32),
        action=step,
        reward=reward,
        next_observation=np.asarray([step + 1], dtype=np.float32),
        terminated=terminal,
        truncated=False,
        episode_id=f"{actor}/session/episode",
        step=step,
        info={"policy_version": 7, "actor_epsilon": 0.1},
    )


def test_wire_codec_round_trips_tensor_pytree_without_pickle() -> None:
    codec = WireCodec(1024 * 1024)
    value = {
        "tensor": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "array": np.asarray([1, 2, 3], dtype=np.int16),
        "nested": (True, b"safe"),
    }

    decoded = codec.decode(codec.encode(value))

    assert torch.equal(decoded["tensor"], value["tensor"])
    np.testing.assert_array_equal(decoded["array"], value["array"])
    assert decoded["nested"] == (True, b"safe")


def test_wire_codec_rejects_unknown_objects_and_message_overflow() -> None:
    with pytest.raises(TypeError, match="unsupported wire value"):
        WireCodec(1024).encode(object())
    with pytest.raises(ValueError, match="limit"):
        WireCodec(8).encode({"payload": "too large"})


def test_actor_observation_snapshots_do_not_share_tensor_storage() -> None:
    observation = {"lidar": torch.zeros(4, 60), "telemetry": torch.zeros(20)}
    first = ActorRuntime._snapshot_observation(observation)
    second = ActorRuntime._snapshot_observation(observation)

    WireCodec(1024 * 1024).encode({"first": first, "second": second})


def test_wire_codec_supports_concurrent_calls() -> None:
    codec = WireCodec(1024 * 1024)
    value = {"tensor": torch.arange(256), "metadata": ["rollout", 17]}
    failures: list[BaseException] = []

    def round_trip() -> None:
        try:
            for _ in range(100):
                decoded = codec.decode(codec.encode(value))
                assert torch.equal(decoded["tensor"], value["tensor"])
        except BaseException as exc:
            failures.append(exc)

    workers = [threading.Thread(target=round_trip) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert all(not worker.is_alive() for worker in workers)
    assert not failures


def test_rollout_journal_is_idempotent_and_recovers_rows(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    journal = RolloutJournal(path)
    first_id, inserted = journal.append("session", 0, b"first")
    duplicate_id, duplicate_inserted = journal.append("session", 0, b"ignored")
    second_id, second_inserted = journal.append("session", 1, b"second")
    profile = journal.actor_profile("PC-1", 4)
    journal.close()

    reopened = RolloutJournal(path)
    try:
        assert inserted
        assert second_inserted
        assert not duplicate_inserted
        assert duplicate_id == first_id
        assert reopened.rows_after(first_id) == [(second_id, b"second")]
        assert reopened.actor_profile("PC-1", 4) == profile
    finally:
        reopened.close()


def test_authentication_and_run_fingerprint_cover_geometry(tmp_path: Path) -> None:
    authenticate(cast(Any, _Context("Bearer secret")), "secret")
    with pytest.raises(RuntimeError, match="UNAUTHENTICATED"):
        authenticate(cast(Any, _Context("Bearer wrong")), "secret")

    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"geometry-v1")
    config = {
        "api_version": "1.2",
        "run_id": "run-a",
        "components": {
            "learner": {"class_path": "tmrl.core.builtins:NullLearner"},
            "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
            "feature_pipeline": {"class_path": "tmrl.core.builtins:IdentityFeaturePipeline"},
        },
        "evaluation": {
            "name": "map",
            "version": "1",
            "maps": [
                {
                    "id": "map",
                    "map_path": "map.Map.Gbx",
                    "geometry_path": geometry.name,
                    "expected_map_uid": "uid",
                }
            ],
        },
    }
    first = run_fingerprint(RunSpec.model_validate(config), tmp_path)
    config["run_id"] = "run-b"
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) == first
    geometry.write_bytes(b"geometry-v2")
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != first


def test_run_fingerprint_hashes_component_geometry_and_requires_loopback(tmp_path: Path) -> None:
    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"geometry-v1")
    config = {
        "api_version": "1.2",
        "run_id": "run-a",
        "components": {
            "learner": {"class_path": "tmrl.core.builtins:NullLearner"},
            "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
            "feature_pipeline": {
                "class_path": "tmrl.core.builtins:IdentityFeaturePipeline",
                "kwargs": {"geometry_path": geometry.name},
            },
        },
    }

    first = run_fingerprint(RunSpec.model_validate(config), tmp_path)
    geometry.write_bytes(b"geometry-v2")

    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != first
    assert require_loopback_bind("127.0.0.1:8787") == "127.0.0.1:8787"
    with pytest.raises(ValueError, match="loopback"):
        require_loopback_bind("0.0.0.0:8787")


def test_interleaved_actors_build_episode_local_n_step_returns() -> None:
    store = InMemoryReplayStore()
    actor_a_ids = []
    actor_b_ids = []
    for step in range(3):
        actor_a_ids.append(store.append(_transition("a", step, float(step + 1))))
        actor_b_ids.append(
            store.append(_transition("b", step, float(10 + step), terminal=step == 2))
        )

    assert store.n_step_ids(actor_a_ids[0], 3) == actor_a_ids
    assert store.n_step_ids(actor_b_ids[0], 3) == actor_b_ids
    batch = _make_batch(
        store,
        _Pipeline(),
        [actor_a_ids[0], actor_b_ids[0]],
        BatchRequest(batch_size=2, n_step=3, gamma=0.5),
    )

    np.testing.assert_allclose(batch.rewards, [2.75, 18.5])
    np.testing.assert_allclose(batch.bootstrap_discounts, [0.125, 0.0])

    per_batch = PrioritizedSampler(_Pipeline(), seed=3).sample(
        store, BatchRequest(batch_size=4, n_step=3, gamma=0.5)
    )
    expected = {
        actor_a_ids[0]: 2.75,
        actor_b_ids[0]: 18.5,
        actor_b_ids[1]: 17.0,
        actor_b_ids[2]: 12.0,
    }
    for transition_id, reward in zip(
        per_batch.transition_ids, np.asarray(per_batch.rewards), strict=True
    ):
        assert float(reward) == pytest.approx(expected[transition_id])


def test_policy_snapshot_is_identical_and_replacement_is_atomic() -> None:
    codec = WireCodec(1024 * 1024)
    original = {"weight": torch.randn(4, 3), "bias": torch.randn(4)}
    restored = codec.decode(codec.encode(original))
    assert torch.equal(restored["weight"], original["weight"])
    assert torch.equal(restored["bias"], original["bias"])

    reference = _PolicyReference(_Policy(1), epsilon=1.0, version=0)
    before = reference.get()
    reference.replace(_Policy(2), epsilon=0.1, version=7)
    after = reference.get()
    assert (before[0].act(None), before[1:]) == (1, (1.0, 0))
    assert (after[0].act(None), after[1:]) == (2, (0.1, 7))


def test_distributed_epsilon_uses_transition_schedule_and_profile_multiplier(
    tmp_path: Path,
) -> None:
    spec = RunSpec.model_validate(
        {
            "run_id": "epsilon",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": {"warmup_transitions": 10},
            "distributed": {
                "epsilon_profiles": [1.0, 0.5],
                "epsilon_start": 0.6,
                "epsilon_final": 0.2,
                "epsilon_decay_transitions": 100,
            },
        }
    )
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "epsilon",
        learner=_SlowLearner(),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )
    coordinator = Coordinator(
        run,
        bind="127.0.0.1:8787",
        token="secret",
        fingerprint="fingerprint",
    )
    try:
        assert coordinator._epsilon(0) == 1.0
        coordinator.counters.transitions = 60
        assert coordinator._epsilon(0) == pytest.approx(0.4)
        assert coordinator._epsilon(1) == pytest.approx(0.2)
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_new_actor_requests_the_initial_policy_snapshot() -> None:
    codec = WireCodec(1024 * 1024)

    class ReplicaLearner:
        def policy(self) -> _Policy:
            return _Policy(0)

    class Client:
        def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert method == "Policy"
            assert value["current_version"] == -1
            return {
                "stop": False,
                "epsilon": 0.1,
                "policy_version": 0,
                "snapshot": codec.encode({"value": 7}),
            }

    actor = object.__new__(ActorRuntime)
    actor.codec = codec
    actor.client = Client()
    actor._replica_learner = ReplicaLearner()
    actor._policy_ref = _PolicyReference(_Policy(0), epsilon=1.0, version=-1)
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor._request_base = lambda: {"actor_id": "actor"}

    actor._refresh_policy()

    policy, epsilon, version = actor._policy()
    assert (policy.act(None), epsilon, version) == (7, 0.1, 0)


def test_actor_evaluation_is_greedy_and_never_spooled_as_training_data() -> None:
    deterministic_calls: list[bool] = []
    spooled: list[tuple[list[Any], list[Any], int, list[dict[str, Any]]]] = []

    class Policy:
        def act(self, observation: Any, *, deterministic: bool = False) -> int:
            del observation
            deterministic_calls.append(deterministic)
            return 0

    class Environment:
        def __init__(self) -> None:
            self.steps = 0

        def reset(self, *, seed: int) -> tuple[int, dict[str, Any]]:
            assert seed == 1_000_007
            return 0, {}

        def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            assert action == 0
            self.steps += 1
            if self.steps == 1:
                return (
                    1,
                    2.0,
                    False,
                    False,
                    {
                        "reward_time": -0.1,
                        "reward_pbrs": 2.1,
                    },
                )
            return (
                2,
                3.0,
                True,
                False,
                {
                    "termination_reason": "finished",
                    "race_time_ms": 12_500.0,
                    "reward_time": -0.2,
                    "reward_pbrs": 3.2,
                    "reward_terminal": 10.0,
                },
            )

    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(training=SimpleNamespace(max_episode_steps=3))
    actor.stop = threading.Event()
    actor._evaluation_index = 0
    actor._actor_seed = lambda: 7
    actor._policy = lambda: (Policy(), 0.5, 9)
    actor._spool = lambda transitions, episodes, version, *, evaluations=None: spooled.append(
        (transitions, episodes, version, evaluations or [])
    )

    actor._evaluate(Environment(), _Pipeline())

    assert deterministic_calls == [True, True]
    assert spooled[0][0:3] == ([], [], 9)
    assert spooled[0][3][0]["finish_time_s"] == 12.5
    assert spooled[0][3][0]["reward/time"] == pytest.approx(-0.3)
    assert spooled[0][3][0]["reward/pbrs"] == pytest.approx(5.3)
    assert spooled[0][3][0]["reward/terminal"] == 10.0


def test_actor_evaluation_runs_the_configured_trial_count() -> None:
    spooled: list[tuple[list[Any], list[Any], int, list[dict[str, Any]]]] = []

    class Policy:
        def act(self, observation: Any, *, deterministic: bool = False) -> int:
            del observation
            assert deterministic
            return 0

    class Environment:
        def reset(self, *, seed: int) -> tuple[int, dict[str, Any]]:
            del seed
            return 0, {}

        def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
            assert action == 0
            return 0, 1.0, True, False, {"termination_reason": "finished", "race_time_ms": 1_000.0}

    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        training=SimpleNamespace(max_episode_steps=2),
        evaluation=SimpleNamespace(trials_per_map=3),
    )
    actor.stop = threading.Event()
    actor._evaluation_index = 0
    actor._actor_seed = lambda: 7
    actor._policy = lambda: (Policy(), 0.0, 9)
    actor._spool = lambda transitions, episodes, version, *, evaluations=None: spooled.append(
        (transitions, episodes, version, evaluations or [])
    )

    actor._evaluate(Environment(), _Pipeline())

    assert len(spooled[0][3]) == 3
    assert actor._evaluation_index == 3


def test_distributed_ingest_normalizes_source_demo_marker() -> None:
    store = InMemoryReplayStore()
    coordinator = object.__new__(Coordinator)
    coordinator.run = SimpleNamespace(
        replay_store=store,
        spec=SimpleNamespace(
            training=SimpleNamespace(
                warmup_transitions=1,
                updates_per_transition=1.0,
                evaluate_every_episodes=None,
            )
        ),
        logger=_Logger(),
    )
    coordinator.counters = _Counters()
    coordinator._last_ingest_at = time.monotonic()
    coordinator._rollouts = Queue()

    transition = Transition(
        observation=np.asarray([0], dtype=np.float32),
        action=0,
        reward=1.0,
        next_observation=np.asarray([1], dtype=np.float32),
        terminated=True,
        truncated=False,
        info={"source": "demo"},
        episode_id="actor",
        step=0,
    )
    coordinator._ingest(
        {
            "actor_id": "actor",
            "session_id": "session",
            "sequence": 0,
            "policy_version": 0,
            "transitions": [transition_to_wire(transition)],
            "episodes": [],
            "evaluations": [],
        },
        1,
    )

    assert store.get([0])[0].info == {"is_demo": True}


def test_actor_retries_a_rejected_rollout_before_deleting_it(tmp_path: Path) -> None:
    codec = WireCodec(1024 * 1024)
    path = tmp_path / "00000000000000000000.rollout"
    path.write_bytes(codec.encode({"sequence": 0}))

    class Client:
        def __init__(self, stop: threading.Event) -> None:
            self.calls = 0
            self.stop = stop

        def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert method == "Submit"
            assert value["sequence"] == 0
            self.calls += 1
            if self.calls == 1:
                return {"accepted": False, "force_refresh": True, "stop": False}
            self.stop.set()
            return {"accepted": True, "force_refresh": False, "stop": False}

    actor = object.__new__(ActorRuntime)
    actor.stop = threading.Event()
    actor.force_refresh = threading.Event()
    actor.stop_reason = "running"
    actor.queue = Queue()
    actor.queue.put(path)
    actor.codec = codec
    actor.client = Client(actor.stop)

    actor._sender_loop()

    assert actor.client.calls == 2
    assert actor.force_refresh.is_set()
    assert not path.exists()


def test_actor_spool_bytes_ignores_a_rollout_deleted_during_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "00000000000000000000.rollout"
    actor = object.__new__(ActorRuntime)
    actor.spool_dir = tmp_path
    monkeypatch.setattr(Path, "glob", lambda _path, _pattern: iter((missing,)))

    assert actor._spool_bytes() == 0


def test_windows_compatible_spawn_entrypoint() -> None:
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_spawn_probe, args=(queue,))
    process.start()
    process.join(timeout=10)
    try:
        assert process.exitcode == 0
        assert queue.get(timeout=2) == "spawn-ok"
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


def test_cli_spawn_context_inherits_the_active_virtual_environment() -> None:
    from tmrl import cli

    context = cli._spawn_context()
    queue = context.Queue()
    process = context.Process(target=_environment_probe, args=(queue,))
    process.start()
    process.join(timeout=10)
    try:
        assert process.exitcode == 0
        child_prefix, child_torch, child_cuda = queue.get(timeout=2)
        assert child_prefix == sys.prefix
        assert child_torch == torch.__version__
        assert child_cuda == torch.version.cuda
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


def test_coordinator_reset_replay_restores_only_learner_state(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    learner = _SlowLearner()
    replay, sampler = _RestoreSpy(), _RestoreSpy()
    checkpoint = {
        "schema_version": "2.0",
        "learner": {"value": 7},
        "replay_store": {"transitions": ["old"]},
        "sampler": {"priorities": [1.0]},
        "distributed": asdict(_Counters(transitions=42, updates=11)),
    }
    run = SimpleNamespace(
        spec=SimpleNamespace(distributed=SimpleNamespace(max_message_bytes=1024 * 1024)),
        run_dir=tmp_path / "weights-only",
        learner=learner,
        replay_store=replay,
        sampler=sampler,
        checkpoint_codec=SimpleNamespace(load=lambda _: checkpoint),
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{port}",
        token="secret",
        fingerprint="fingerprint",
    )

    try:
        coordinator.restore_checkpoint(tmp_path / "checkpoint.pt", reset_replay=True)
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()

    assert learner.value == 7
    assert replay.restored is None
    assert sampler.restored is None
    assert coordinator.counters == _Counters()


def test_coordinator_recovers_wal_without_a_checkpoint(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    spec = RunSpec.model_validate(
        {
            "api_version": "1.2",
            "run_id": "journal-recovery",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": {
                "total_transitions": 1,
                "batch_size": 1,
                "warmup_transitions": 10,
            },
        }
    )
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "journal-recovery",
        learner=_SlowLearner(),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=_Logger(),
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )
    stop = threading.Event()
    stop.set()
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{port}",
        token="secret",
        fingerprint="fingerprint",
        external_stop=stop,
    )
    payload = coordinator.codec.encode(
        {
            "actor_id": "actor",
            "session_id": "session",
            "sequence": 0,
            "policy_version": 0,
            "transitions": [transition_to_wire(_transition("actor", 0, 1.0, terminal=True))],
            "episodes": [],
        }
    )
    coordinator.journal.append("session", 0, payload)

    result = coordinator.run_forever()

    assert result.transitions == 1
    assert len(run.replay_store) == 1


def test_two_fake_actors_feed_slow_learner_without_data_loss(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    spec = RunSpec.model_validate(
        {
            "api_version": "1.2",
            "run_id": "async-smoke",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "tmrl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "tmrl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": {
                "total_transitions": 16,
                "batch_size": 2,
                "n_step": 2,
                "warmup_transitions": 4,
                "updates_per_transition": 0.25,
                "checkpoint_interval_updates": 100,
            },
            "distributed": {"policy_refresh_s": 0.001},
        }
    )
    pipeline = _Pipeline()
    logger = _Logger()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "async-smoke",
        learner=_SlowLearner(),
        environment_factory=None,
        model_factory=None,
        replay_store=InMemoryReplayStore(),
        sampler=UniformSampler(pipeline, seed=0),
        feature_pipeline=pipeline,
        logger=logger,
        checkpoint_codec=TorchCheckpointCodec(),
        evaluator=None,
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{port}",
        token="secret",
        fingerprint="fingerprint",
    )
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            coordinator.run_forever()
        except BaseException as exc:
            failures.append(exc)

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    codec = WireCodec(spec.distributed.max_message_bytes)
    clients = [_Client(f"127.0.0.1:{port}", "secret", codec) for _ in range(2)]
    for client in clients:
        grpc.channel_ready_future(client.channel).result(timeout=10)

    def send(actor_index: int) -> None:
        actor_id = f"actor-{actor_index}"
        base = {
            "protocol_version": PROTOCOL_VERSION,
            "fingerprint": "fingerprint",
            "actor_id": actor_id,
            "session_id": f"session-{actor_index}",
        }
        clients[actor_index].call("Register", base)
        transitions = [
            transition_to_wire(_transition(actor_id, step, float(step), terminal=step == 7))
            for step in range(8)
        ]
        response = clients[actor_index].call(
            "Submit",
            {
                **base,
                "sequence": 0,
                "policy_version": 0,
                "transitions": transitions,
                "episodes": [],
            },
        )
        assert response["accepted"]

    senders = [threading.Thread(target=send, args=(index,)) for index in range(2)]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join(timeout=10)
    server_thread.join(timeout=10)
    for client in clients:
        client.close()

    assert not server_thread.is_alive()
    assert not failures
    assert coordinator.counters.transitions == 16
    assert len(run.replay_store) == 16
    assert coordinator.counters.updates == 3
    assert coordinator.counters.policy_version >= 1
    assert logger.events.count("distributed/ingest") == 2
