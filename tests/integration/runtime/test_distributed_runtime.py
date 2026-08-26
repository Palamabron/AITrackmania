from __future__ import annotations

import multiprocessing
import os
import socket
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from typing import Any, cast

import grpc
import numpy as np
import pytest
import torch
import zstandard
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.data import BatchRequest, Transition
from trackmaniarl.core.replay import (
    InMemoryReplayStore,
    PrioritizedSampler,
    UniformSampler,
    _make_batch,
)
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import (
    ActorBackgroundError,
    ActorEnvironmentError,
    ActorRuntime,
    _Client,
    _is_retryable_rpc_error,
    _PolicyReference,
    actor_process_entry,
)
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator import (
    Coordinator,
    _AsyncCheckpointWriter,
    _BatchPrefetcher,
    _Counters,
    _MetricAccumulator,
    _PendingRollout,
)
from trackmaniarl.distributed.journal import JournalPayloadConflictError, RolloutJournal
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    authenticate,
    run_fingerprint,
    transition_to_wire,
)

_DISTRIBUTED_TOKEN = "tests-only-distributed-token-0123456789"


class _Pipeline:
    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> Mapping[str, Any]:
        return {"reward": np.asarray([item.reward for item in transitions])}


def test_metric_accumulator_averages_each_metric_over_its_own_observations() -> None:
    metrics = _MetricAccumulator()

    metrics.add({"loss/total": 2.0})
    metrics.add({"loss/total": 4.0, "debug/q_selected_mean": 8.0})
    metrics.add({"debug/td_abs_max": 5.0})
    metrics.add({"debug/td_abs_max": 3.0})

    assert metrics.flush() == {
        "loss/total": 3.0,
        "debug/q_selected_mean": 8.0,
        "debug/td_abs_max": 5.0,
    }
    assert metrics.flush() == {}


def test_online_partial_progress_is_not_mislabeled_as_an_elite_lap() -> None:
    partial = Coordinator._replay_info({"progress_pct": 75.0, "race_time_ms": 27_000.0})
    expert = Coordinator._replay_info(
        {
            "is_demo": True,
            "progress_pct": 75.0,
            "race_time_ms": 27_000.0,
            "sampling/projected_lap_time_s": 36.035,
        }
    )

    assert "sampling/projected_lap_time_s" not in partial
    assert expert == {
        "is_demo": True,
        "sampling/projected_lap_time_s": 36.035,
    }


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


class _RpcFailure(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode) -> None:
        super().__init__()
        self._code = code

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._code.name


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
        self.records: list[tuple[str, Mapping[str, Any]]] = []

    def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
        del step
        self.events.append(event)
        self.records.append((event, dict(payload)))

    def close(self) -> None:
        return


def _spawn_probe(queue: Any) -> None:
    queue.put("spawn-ok")


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


def test_wire_codec_rejects_compressed_payload_above_decompressed_limit() -> None:
    payload = zstandard.ZstdCompressor().compress(b"x" * 2_048)

    with pytest.raises(ValueError, match="decompressed size limit"):
        WireCodec(1_024).decode(payload)


def test_rollout_journal_is_idempotent_and_recovers_rows(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    journal = RolloutJournal(path)
    first_id, inserted = journal.append("session", 0, b"first")
    duplicate_id, duplicate_inserted = journal.append("session", 0, b"first")
    with pytest.raises(JournalPayloadConflictError, match="different payload"):
        journal.append("session", 0, b"ignored")
    second_id, second_inserted = journal.append("session", 1, b"second")
    profile = journal.actor_profile("PC-1", 4)
    journal.close()

    reopened = RolloutJournal(path)
    try:
        assert inserted
        assert second_inserted
        assert not duplicate_inserted
        assert duplicate_id == first_id
        assert list(reopened.rows_after(first_id)) == [(second_id, b"second")]
        assert reopened.actor_profile("PC-1", 4) == profile
        assert reopened.identity == journal.identity
    finally:
        reopened.close()


def test_rollout_receipt_survives_prune_and_rejects_unsafe_rollback(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.sqlite3"
    journal = RolloutJournal(path)
    foreign = RolloutJournal(tmp_path / "foreign.sqlite3")
    try:
        row_id, inserted = journal.append("session", 0, b"payload")
        assert inserted

        journal.prune(row_id)
        journal.close()
        journal = RolloutJournal(path)

        assert journal.pruned_through == row_id
        assert not journal.has_rows()
        assert journal.append("session", 0, b"payload") == (row_id, False)
        with pytest.raises(ValueError, match="predates data already pruned"):
            journal.validate_checkpoint(journal.identity, row_id - 1)
        with pytest.raises(ValueError, match="different rollout journal"):
            foreign.validate_checkpoint(journal.identity, 0)
        with pytest.raises(ValueError, match="ahead of durable WAL history"):
            journal.validate_checkpoint(journal.identity, row_id + 1)
    finally:
        foreign.close()
        journal.close()


def test_concurrent_identical_rollout_has_one_durable_identity(tmp_path: Path) -> None:
    journal = RolloutJournal(tmp_path / "rollouts.sqlite3")
    barrier = threading.Barrier(8)
    results: list[tuple[int, bool]] = []

    def append() -> None:
        barrier.wait(timeout=5.0)
        results.append(journal.append("session", 7, b"same-payload"))

    threads = [threading.Thread(target=append) for _ in range(barrier.parties)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    try:
        assert all(not thread.is_alive() for thread in threads)
        assert len(results) == barrier.parties
        assert len({row_id for row_id, _ in results}) == 1
        assert sum(inserted for _, inserted in results) == 1
        assert len(list(journal.rows_after(0))) == 1
    finally:
        journal.close()


def test_failed_checkpoint_never_advances_the_pruned_frontier(tmp_path: Path) -> None:
    class BrokenCodec:
        def save(self, state: Mapping[str, Any], path: Path) -> None:
            del state
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
            raise OSError("simulated checkpoint crash")

    journal = RolloutJournal(tmp_path / "rollouts.sqlite3")
    row_id, _ = journal.append("session", 0, b"payload")
    writer = _AsyncCheckpointWriter(BrokenCodec())
    try:
        writer.submit({}, tmp_path / "checkpoint.pt", lambda: journal.prune(row_id))
        with pytest.raises(OSError, match="simulated checkpoint crash"):
            writer.wait()

        assert journal.pruned_through == 0
        assert [stored_id for stored_id, _ in journal.rows_after(0)] == [row_id]
    finally:
        writer.close()
        journal.close()


def test_coordinator_emits_checkpoint_failure_without_completion(tmp_path: Path) -> None:
    class BrokenCodec:
        def save(self, state: Mapping[str, Any], path: Path) -> None:
            del state, path
            raise OSError("checkpoint device failed")

    run = replace(
        _resolved_run(
            tmp_path,
            "checkpoint-incident",
            {
                "total_transitions": 10,
                "warmup_transitions": 10,
                "checkpoint_interval_updates": None,
            },
        ),
        checkpoint_codec=cast(Any, BrokenCodec()),
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    try:
        coordinator._checkpoint()
        with pytest.raises(OSError, match="checkpoint device failed"):
            coordinator._checkpoint_writer.wait()

        assert "train/checkpoint" in run.logger.events
        assert "train/checkpoint_failed" in run.logger.events
        assert "train/checkpoint_completed" not in run.logger.events
        assert coordinator.journal.pruned_through == 0
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_async_checkpoint_snapshots_mutable_replay_and_sampler_state(tmp_path: Path) -> None:
    started = threading.Event()
    release = threading.Event()
    saved: dict[str, Any] = {}

    class DelayedCodec:
        def save(self, state: Mapping[str, Any], path: Path) -> None:
            started.set()
            if not release.wait(timeout=5.0):
                raise TimeoutError("checkpoint test did not release the codec")
            saved["state"] = state
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"checkpoint")

    class MutableState:
        def __init__(self, value: str) -> None:
            self.value: dict[str, Any] = {
                "nested": {"value": value},
                "tensor": torch.tensor([1.0]),
            }

        def state_dict(self) -> Mapping[str, Any]:
            return self.value

    replay = MutableState("replay-before")
    sampler = MutableState("sampler-before")
    run = replace(
        _resolved_run(
            tmp_path,
            "checkpoint-snapshot",
            {
                "total_transitions": 10,
                "warmup_transitions": 10,
                "checkpoint_interval_updates": None,
            },
        ),
        replay_store=cast(Any, replay),
        sampler=cast(Any, sampler),
        checkpoint_codec=cast(Any, DelayedCodec()),
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )

    try:
        coordinator._checkpoint()
        assert started.wait(timeout=5.0)
        replay.value["nested"]["value"] = "replay-after"
        replay.value["tensor"].fill_(2.0)
        sampler.value["nested"]["value"] = "sampler-after"
        sampler.value["tensor"].fill_(2.0)
        release.set()
        coordinator._checkpoint_writer.wait()

        checkpoint = saved["state"]
        assert checkpoint["replay_store"]["nested"]["value"] == "replay-before"
        assert torch.equal(checkpoint["replay_store"]["tensor"], torch.tensor([1.0]))
        assert checkpoint["sampler"]["nested"]["value"] == "sampler-before"
        assert torch.equal(checkpoint["sampler"]["tensor"], torch.tensor([1.0]))
    finally:
        release.set()
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_non_overlapped_batch_path_never_prepares_or_speculatively_samples() -> None:
    batch = object()

    class Sampler:
        def __init__(self) -> None:
            self.calls = 0

        def sample(self, store: object, request: BatchRequest) -> Any:
            del store, request
            self.calls += 1
            return batch

    class Learner:
        def __init__(self) -> None:
            self.prepare_calls = 0

        def prepare_batch(self, value: Any) -> Any:
            self.prepare_calls += 1
            return value

    sampler = Sampler()
    learner = Learner()
    prefetcher = _BatchPrefetcher(
        cast(Any, SimpleNamespace(sampler=sampler, learner=learner, replay_store=object()))
    )

    first, _, first_wait = prefetcher.next(BatchRequest(batch_size=1))
    second, _, second_wait = prefetcher.next(BatchRequest(batch_size=1))
    prefetcher.close()

    assert first is batch
    assert second is batch
    assert sampler.calls == 2
    assert learner.prepare_calls == 0
    assert first_wait == second_wait == 0.0


def test_actor_runtime_rejects_short_token_before_loading_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        ActorRuntime(
            tmp_path / "missing.yaml",
            target="127.0.0.1:8787",
            actor_id="actor",
            token="short",
        )


def test_coordinator_rejects_short_token_before_initialization() -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        Coordinator(
            cast(Any, object()),
            bind="127.0.0.1:8787",
            token="short",
            fingerprint="fingerprint",
        )


def test_authentication_and_run_fingerprint_cover_geometry(tmp_path: Path) -> None:
    authenticate(cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")), _DISTRIBUTED_TOKEN)
    with pytest.raises(RuntimeError, match="UNAUTHENTICATED"):
        authenticate(cast(Any, _Context("Bearer wrong")), _DISTRIBUTED_TOKEN)

    geometry = tmp_path / "geometry.npz"
    geometry.write_bytes(b"geometry-v1")
    config = {
        "api_version": "2.0",
        "run_id": "run-a",
        "components": {
            "learner": {"class_path": "trackmaniarl.core.builtins:SmokeLearner"},
            "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
            "feature_pipeline": {
                "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
            },
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


def test_run_fingerprint_hashes_reexported_implementation_and_effective_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "fingerprint_package"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    implementation = package / "implementation.py"
    implementation.write_text(
        "class Component:\n    def __init__(self, width=4):\n        self.width = width\n",
        encoding="utf-8",
    )
    (package / "reexport.py").write_text(
        "from fingerprint_package.implementation import Component\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config = {
        "api_version": "2.0",
        "run_id": "fingerprint",
        "components": {
            "learner": {"class_path": "fingerprint_package.reexport:Component"},
            "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
            "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
            "feature_pipeline": {
                "class_path": "trackmaniarl.core.builtins:IdentityFeaturePipeline"
            },
        },
    }

    implicit_default = run_fingerprint(RunSpec.model_validate(config), tmp_path)
    explicit = deepcopy(config)
    explicit["components"]["learner"]["kwargs"] = {"width": 4}
    assert run_fingerprint(RunSpec.model_validate(explicit), tmp_path) == implicit_default

    changed_parameter = deepcopy(config)
    changed_parameter["components"]["learner"]["kwargs"] = {"width": 5}
    assert run_fingerprint(RunSpec.model_validate(changed_parameter), tmp_path) != implicit_default

    implementation.write_text(
        "class Component:\n"
        "    implementation_version = 2\n"
        "    def __init__(self, width=4):\n"
        "        self.width = width\n",
        encoding="utf-8",
    )
    assert run_fingerprint(RunSpec.model_validate(config), tmp_path) != implicit_default


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


def _ephemeral_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_distributed_epsilon_uses_transition_schedule_and_profile_multiplier(
    tmp_path: Path,
) -> None:
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "epsilon",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
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
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    try:
        assert coordinator._epsilon(0) == pytest.approx(0.6)
        coordinator.counters.transitions = 50
        assert coordinator._epsilon(0) == pytest.approx(0.4)
        assert coordinator._epsilon(1) == pytest.approx(0.2)
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_distributed_epsilon_can_follow_completed_learner_updates(tmp_path: Path) -> None:
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "epsilon-updates",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "distributed": {
                "epsilon_start": 0.6,
                "epsilon_final": 0.2,
                "epsilon_decay_transitions": 10,
                "epsilon_decay_updates": 100,
            },
        }
    )
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "epsilon-updates",
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
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    try:
        coordinator.counters.transitions = 10_000
        coordinator.counters.updates = 50
        assert coordinator._epsilon(0) == pytest.approx(0.4)
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


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
    actor._spool = (
        lambda transitions, episodes, version, *, evaluations=None, evaluation_snapshot=None: (
            spooled.append((transitions, episodes, version, evaluations or []))
        )
    )

    actor._evaluate(Environment(), _Pipeline())

    assert deterministic_calls == [True, True]
    assert spooled[0][0:3] == ([], [], 9)
    assert spooled[0][3][0]["finish_time_s"] == 12.5
    assert spooled[0][3][0]["reward/time"] == pytest.approx(-0.3)
    assert spooled[0][3][0]["reward/pbrs"] == pytest.approx(5.3)
    assert spooled[0][3][0]["reward/terminal"] == 10.0


def test_actor_training_episode_freezes_one_policy_and_reports_action_gaps() -> None:
    policies: list[Any] = []
    spooled: list[tuple[list[Transition], list[dict[str, Any]], int]] = []

    class MarginPolicy:
        def __init__(self, version: int) -> None:
            self.version = version
            self.margins = iter((3.0, 1.0, 2.0))
            self.last_q_margin: float | None = None
            self.calls = 0

        def act(self, observation: Any, *, deterministic: bool = False) -> int:
            del observation, deterministic
            self.calls += 1
            self.last_q_margin = next(self.margins)
            return 0

    versions = iter(range(100))

    def next_policy() -> tuple[Any, float, int]:
        policy = MarginPolicy(next(versions))
        policies.append(policy)
        return policy, 0.1, policy.version

    class Environment:
        def __init__(self, stop: threading.Event) -> None:
            self.stop = stop
            self.episode_steps = 0
            self.total_steps = 0

        def reset(self, *, seed: int) -> tuple[Any, dict[str, Any]]:
            del seed
            self.episode_steps = 0
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            assert action == 0
            self.episode_steps += 1
            self.total_steps += 1
            terminal = self.episode_steps == 3
            if self.total_steps == 6:
                self.stop.set()
            info: dict[str, Any] = {
                "control_gas": 1.0,
                "control_brake": 0.0,
                "control_steer": 0.5,
                "step_race_time_ms": 66.0,
            }
            if terminal:
                info.update({"termination_reason": "finished", "race_time_ms": 1_000.0})
            return np.zeros(1, dtype=np.float32), 1.0, terminal, False, info

    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        training=SimpleNamespace(max_episode_steps=10),
        distributed=SimpleNamespace(rollout_chunk_transitions=128, rollout_flush_s=60.0),
    )
    actor.actor_id = "actor"
    actor.session_id = "session"
    actor.stop = threading.Event()
    actor.evaluate = threading.Event()
    actor._actor_seed = lambda: 7
    actor._policy = next_policy
    actor._spool = lambda transitions, episodes, version, *, evaluations=None: spooled.append(
        (list(transitions), list(episodes), version)
    )

    actor._collect(Environment(actor.stop), _Pipeline())

    acting = [policy for policy in policies if policy.calls]
    assert [policy.calls for policy in acting] == [3, 3]
    episode_versions = [
        {item.info["policy_version"] for item in transitions}
        for transitions, _, _ in spooled
        if transitions
    ]
    assert episode_versions == [{acting[0].version}, {acting[1].version}]
    summary = spooled[0][1][0]
    assert summary["q_margin/mean"] == pytest.approx(2.0)
    assert summary["q_margin/min"] == pytest.approx(1.0)
    assert summary["q_margin/start_mean"] == pytest.approx(2.0)
    assert summary["control/gas_fraction"] == pytest.approx(1.0)
    assert summary["control/brake_fraction"] == 0.0
    assert summary["control/steer_abs_mean"] == pytest.approx(0.5)
    assert summary["timing/step_race_ms_mean"] == pytest.approx(66.0)
    assert summary["timing/step_race_ms_p99"] == pytest.approx(66.0)
    assert summary["timing/step_race_ms_max"] == pytest.approx(66.0)
    assert summary["termination/time_limit"] == 0.0


def test_actor_marks_mid_episode_telemetry_interruption_as_truncated() -> None:
    spooled: list[list[Transition]] = []

    class Policy:
        action_count = 2

        def act(self, observation: Any, *, deterministic: bool = False) -> int:
            del observation, deterministic
            return 0

    class Environment:
        def __init__(self, stop: threading.Event) -> None:
            self.stop = stop
            self.steps = 0

        def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
            del seed
            return np.zeros(1, dtype=np.float32), {}

        def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            assert action == 0
            self.steps += 1
            if self.steps == 1:
                return np.ones(1, dtype=np.float32), 1.0, False, False, {}
            self.stop.set()
            raise TimeoutError("telemetry packet missing")

    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(
        training=SimpleNamespace(max_episode_steps=3),
        distributed=SimpleNamespace(rollout_chunk_transitions=128, rollout_flush_s=60.0),
    )
    actor.actor_id = "actor"
    actor.session_id = "session"
    actor.stop = threading.Event()
    actor.evaluate = threading.Event()
    actor._actor_seed = lambda: 7
    actor._policy = lambda: (Policy(), 0.0, 1)
    actor._spool = lambda transitions, *_args, **_kwargs: spooled.append(list(transitions))

    actor._collect(Environment(actor.stop), _Pipeline())

    transition = next(batch[0] for batch in spooled if batch)
    assert not transition.terminated
    assert transition.truncated
    assert transition.info["termination_reason"] == "telemetry_interruption"


def test_external_stop_does_not_ingest_or_train_a_queued_backlog(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    events: list[str] = []

    class OrderedLearner(_SlowLearner):
        def update(self, batch: Any) -> Mapping[str, float]:
            events.append("update")
            return super().update(batch)

    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "drain-first",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": {
                "total_transitions": 6,
                "batch_size": 1,
                "warmup_transitions": 1,
                "updates_per_transition": 1.0,
                "checkpoint_interval_updates": 1000,
            },
        }
    )
    pipeline = _Pipeline()
    run = ResolvedRun(
        spec=spec,
        run_dir=tmp_path / "drain-first",
        learner=OrderedLearner(),
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
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
        external_stop=stop,
    )
    ingest = coordinator._ingest

    def tracking_ingest(value: Mapping[str, Any], row_id: int) -> None:
        events.append("ingest")
        ingest(value, row_id)

    coordinator._ingest = tracking_ingest
    for chunk in range(3):
        payload = {
            "actor_id": f"actor-{chunk}",
            "session_id": "session",
            "sequence": chunk,
            "policy_version": 0,
            "transitions": [
                transition_to_wire(_transition(f"actor-{chunk}", offset, 1.0, terminal=offset == 1))
                for offset in range(2)
            ],
            "episodes": [],
            "evaluations": [],
        }
        coordinator._rollouts.put(_PendingRollout(payload, chunk + 1, time.monotonic()))

    coordinator.run_forever()

    assert events == []
    assert coordinator.counters.transitions == 0
    assert coordinator.counters.updates == 0
    assert coordinator._rollouts.qsize() == 3


def test_ingest_aggregates_evaluation_batches_and_checkpoints_best(tmp_path: Path) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class RecordingLogger:
        def log(self, event: str, payload: Mapping[str, Any], *, step: int | None = None) -> None:
            del step
            events.append((event, dict(payload)))

        def close(self) -> None:
            return

    checkpoints: list[int] = []
    coordinator = object.__new__(Coordinator)
    coordinator.run = SimpleNamespace(
        replay_store=InMemoryReplayStore(),
        spec=SimpleNamespace(
            distributed=SimpleNamespace(max_update_credit=512),
            training=SimpleNamespace(
                warmup_transitions=1,
                updates_per_transition=1.0,
                evaluate_every_episodes=None,
            ),
        ),
        logger=RecordingLogger(),
    )
    coordinator.counters = _Counters()
    coordinator._last_ingest_at = time.monotonic()
    coordinator._rollouts = Queue()
    coordinator._recovering = False
    coordinator._time_buckets = (40.0, 38.0, 36.0)
    coordinator._best_evaluation = None
    coordinator._checkpoints = []
    coordinator._checkpoint = lambda: (
        checkpoints.append(coordinator.counters.updates),
        tmp_path / "best.pt",
    )[1]
    journal_row = 0

    def evaluation(finish_time_s: float, *, finished: bool) -> dict[str, Any]:
        return {
            "finished": float(finished),
            "finish_time_s": finish_time_s,
            "policy_version": 41,
            "q_margin/start_mean": 0.5,
            "control/gas_fraction": 0.75,
            "control/brake_fraction": 0.2,
            "control/brake_tap_fraction": 0.1,
            "control/steer_abs_mean": 0.6,
            "progress_bin/90_100/action_count": 2.0,
            "progress_bin/90_100/action_entropy": 0.5,
            "progress_bin/90_100/action_coverage": 0.25,
            "progress_bin/90_100/q_margin_mean": 1.5,
            "progress_bin/90_100/q_margin_min": 0.75,
            "progress_bin/90_100/q_max_mean": 3.0,
        }

    def ingest(evaluations: list[dict[str, Any]]) -> None:
        nonlocal journal_row
        journal_row += 1
        coordinator._ingest(
            {
                "actor_id": "actor",
                "session_id": "session",
                "sequence": 0,
                "policy_version": 0,
                "transitions": [],
                "episodes": [],
                "evaluations": evaluations,
            },
            journal_row,
        )

    ingest([evaluation(52.0, finished=True), evaluation(0.0, finished=False)])
    ingest([evaluation(0.0, finished=False), evaluation(0.0, finished=False)])
    ingest([evaluation(50.0, finished=True), evaluation(46.0, finished=True)])

    summaries = [payload for event, payload in events if event == "eval/summary"]
    assert [item["finish_rate"] for item in summaries] == [0.5, 0.0, 1.0]
    assert summaries[0]["finish_time_mean_s"] == pytest.approx(52.0)
    assert summaries[0]["policy_version"] == 41.0
    assert summaries[0]["q_margin_start_mean"] == pytest.approx(0.5)
    assert summaries[0]["control_gas_fraction_mean"] == pytest.approx(0.75)
    assert summaries[0]["control_brake_fraction_mean"] == pytest.approx(0.2)
    assert summaries[0]["control_brake_tap_fraction_mean"] == pytest.approx(0.1)
    assert summaries[0]["control_steer_abs_mean"] == pytest.approx(0.6)
    assert summaries[2]["finish_time_median_s"] == pytest.approx(48.0)
    assert summaries[2]["finish_time_best_s"] == pytest.approx(46.0)
    assert summaries[2]["sub_40_rate"] == 0.0
    assert summaries[0]["progress_bin/90_100/action_count"] == 4.0
    assert summaries[0]["progress_bin/90_100/q_margin_mean"] == 1.5
    progress_events = [payload for event, payload in events if event == "eval/progress_bin"]
    assert progress_events[0]["90_100/q_max_mean"] == 3.0
    assert len(checkpoints) == 1
    best_events = [payload for event, payload in events if event == "eval/best_checkpoint"]
    assert [item["finish_rate"] for item in best_events] == [1.0]
    assert best_events[0]["finish_time_median_s"] == pytest.approx(48.0)


def test_evaluation_stop_requires_consecutive_successful_batches() -> None:
    coordinator = object.__new__(Coordinator)
    coordinator.run = SimpleNamespace(
        spec=SimpleNamespace(
            training=SimpleNamespace(
                evaluation_stop_min_finish_rate=0.9,
                evaluation_stop_median_s=36.0,
                evaluation_stop_consecutive_batches=2,
            )
        ),
        logger=_Logger(),
    )
    coordinator.counters = _Counters(updates=10)
    coordinator._consecutive_evaluation_passes = 0
    coordinator._evaluation_stop_reason = None

    coordinator._record_evaluation_stop({"finish_rate": 0.2, "finish_time_median_s": 58.0})
    coordinator._record_evaluation_stop({"finish_rate": 0.9, "finish_time_median_s": 35.9})
    coordinator._record_evaluation_stop({"finish_rate": 0.8, "finish_time_median_s": 35.0})

    assert coordinator._evaluation_stop_reason is None

    coordinator._record_evaluation_stop({"finish_rate": 0.9, "finish_time_median_s": 36.0})
    coordinator._record_evaluation_stop({"finish_rate": 1.0, "finish_time_median_s": 35.8})

    assert coordinator._evaluation_stop_reason is not None
    assert "evaluation target passed 2 consecutive times" in coordinator._evaluation_stop_reason


def test_actor_recovers_valid_numeric_temporary_without_overwriting_spool(
    tmp_path: Path,
) -> None:
    codec = WireCodec(1024 * 1024)
    existing = tmp_path / "00000000000000000000.rollout"
    orphan = tmp_path / "00000000000000000000.tmp"
    invalid = tmp_path / "00000000000000000002.tmp"
    existing_payload = codec.encode({"sequence": 100})
    orphan_payload = codec.encode({"sequence": 0})
    existing.write_bytes(existing_payload)
    orphan.write_bytes(orphan_payload)
    invalid.write_bytes(b"incomplete")
    actor = object.__new__(ActorRuntime)
    actor.spool_dir = tmp_path
    actor.codec = codec

    actor._recover_spool_temporaries()

    recovered = tmp_path / "00000000000000000003.rollout"
    assert existing.read_bytes() == existing_payload
    assert recovered.read_bytes() == orphan_payload
    assert not orphan.exists()
    assert invalid.read_bytes() == b"incomplete"
    assert actor._scan_spool_bytes() == sum(
        path.stat().st_size for path in (existing, recovered, invalid)
    )
    assert actor._next_sequence() == 4


def test_actor_spool_write_fsyncs_before_replace_and_syncs_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    path = tmp_path / "00000000000000000000.rollout"
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(descriptor: int) -> None:
        events.append("file-fsync")
        real_fsync(descriptor)

    def replace_temporary(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    def sync_directory(replaced: Path) -> None:
        assert replaced == path
        events.append("directory-sync")

    monkeypatch.setattr("trackmaniarl.distributed.actor.os.fsync", fsync)
    monkeypatch.setattr("trackmaniarl.distributed.actor.os.replace", replace_temporary)
    monkeypatch.setattr("trackmaniarl.distributed.actor.sync_checkpoint_path", sync_directory)

    ActorRuntime._persist_spool_payload(path, b"durable-payload")

    assert path.read_bytes() == b"durable-payload"
    assert events == ["file-fsync", "replace", "directory-sync"]


def test_actor_spool_replace_failure_keeps_the_fsynced_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "00000000000000000000.rollout"
    temporary = path.with_suffix(".tmp")

    def fail_replace(source: Path, destination: Path) -> None:
        assert source == temporary
        assert destination == path
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("trackmaniarl.distributed.actor.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated atomic replace failure"):
        ActorRuntime._persist_spool_payload(path, b"unacknowledged")

    assert not path.exists()
    assert temporary.read_bytes() == b"unacknowledged"


def test_actor_rejects_a_rollout_larger_than_the_entire_spool_without_waiting() -> None:
    class Stop:
        def is_set(self) -> bool:
            return False

        def wait(self, timeout: float) -> bool:
            del timeout
            raise AssertionError("oversized rollout must not enter the capacity wait loop")

    actor = object.__new__(ActorRuntime)
    actor.spec = SimpleNamespace(distributed=SimpleNamespace(spool_max_bytes=16))
    actor.stop = cast(Any, Stop())

    with pytest.raises(ValueError, match="rollout payload is 17 bytes; spool limit is 16"):
        actor._wait_for_spool_capacity(17)


@pytest.mark.parametrize(
    ("code", "retryable"),
    [
        (grpc.StatusCode.UNAVAILABLE, True),
        (grpc.StatusCode.DEADLINE_EXCEEDED, True),
        (grpc.StatusCode.UNAUTHENTICATED, False),
        (grpc.StatusCode.PERMISSION_DENIED, False),
        (grpc.StatusCode.FAILED_PRECONDITION, False),
        (grpc.StatusCode.INVALID_ARGUMENT, False),
    ],
)
def test_actor_rpc_retry_classifier(code: grpc.StatusCode, retryable: bool) -> None:
    assert _is_retryable_rpc_error(_RpcFailure(code)) is retryable


def test_actor_sender_retries_transient_rpc_but_preserves_spool_on_permanent_rpc(
    tmp_path: Path,
) -> None:
    class ImmediateStop:
        def __init__(self) -> None:
            self.stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def set(self) -> None:
            self.stopped = True

        def wait(self, timeout: float) -> bool:
            del timeout
            return self.stopped

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert method == "Submit"
            assert value["sequence"] == 0
            self.calls += 1
            code = (
                grpc.StatusCode.UNAVAILABLE if self.calls == 1 else grpc.StatusCode.UNAUTHENTICATED
            )
            raise _RpcFailure(code)

    codec = WireCodec(1024 * 1024)
    path = tmp_path / "00000000000000000000.rollout"
    path.write_bytes(codec.encode({"sequence": 0}))
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = cast(Any, ImmediateStop())
    actor.stop_reason = "running"
    actor.force_refresh = threading.Event()
    actor.queue = Queue()
    actor.queue.put(path)
    actor.codec = codec
    actor.client = cast(Any, Client())
    actor._spool_lock = threading.Lock()
    actor._spool_bytes_total = path.stat().st_size
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None

    actor._sender_loop()

    assert actor.client.calls == 2
    assert path.exists()
    assert actor._spool_bytes_total == path.stat().st_size
    with pytest.raises(ActorBackgroundError, match="rollout sender failed"):
        actor._raise_background_failure()


def test_actor_policy_refresh_retries_a_transient_rpc() -> None:
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor.force_refresh = threading.Event()
    actor.force_refresh.set()
    actor.spec = SimpleNamespace(distributed=SimpleNamespace(policy_refresh_s=60.0))
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None
    calls = 0

    def refresh() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _RpcFailure(grpc.StatusCode.DEADLINE_EXCEEDED)
        actor.stop.set()

    actor._refresh_policy = refresh

    actor._policy_loop()

    assert calls == 2
    assert actor._background_failure is None


def test_actor_heartbeat_treats_a_permanent_rpc_as_a_background_failure() -> None:
    class Client:
        def call(self, method: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert method == "Heartbeat"
            assert value["policy_version"] == 4
            raise _RpcFailure(grpc.StatusCode.PERMISSION_DENIED)

    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor.spec = SimpleNamespace(distributed=SimpleNamespace(heartbeat_s=0.001))
    actor.client = cast(Any, Client())
    actor._policy = lambda: (cast(Any, object()), 0.0, 4)
    actor._spool_lock = threading.Lock()
    actor._spool_bytes_total = 0
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None

    actor._heartbeat_loop()

    assert actor.stop.is_set()
    with pytest.raises(ActorBackgroundError, match="heartbeat failed"):
        actor._raise_background_failure()


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
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.force_refresh = threading.Event()
    actor.stop_reason = "running"
    actor.queue = Queue()
    actor.queue.put(path)
    actor.codec = codec
    actor.client = Client(actor.stop)
    actor._spool_lock = threading.Lock()
    actor._spool_bytes_total = path.stat().st_size

    actor._sender_loop()

    assert actor.client.calls == 2
    assert actor.force_refresh.is_set()
    assert not path.exists()
    assert actor._spool_bytes_total == 0


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
        token=_DISTRIBUTED_TOKEN,
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


def _resolved_run(tmp_path: Path, run_id: str, training: dict[str, Any]) -> ResolvedRun:
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": run_id,
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
                "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
            },
            "training": training,
        }
    )
    pipeline = _Pipeline()
    return ResolvedRun(
        spec=spec,
        run_dir=tmp_path / run_id,
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


def test_concurrent_submit_wakes_do_not_reorder_journal_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _resolved_run(
        tmp_path,
        "ordered-wal",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    committed = threading.Event()
    release = threading.Event()
    original_append = coordinator.journal.append

    def delayed_append(session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        result = original_append(session_id, sequence, payload)
        if sequence == 0:
            committed.set()
            assert release.wait(timeout=2.0)
        return result

    monkeypatch.setattr(coordinator.journal, "append", delayed_append)
    codec = coordinator.codec

    def request(sequence: int) -> BytesValue:
        return BytesValue(
            value=codec.encode(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": "fingerprint",
                    "actor_id": "actor",
                    "session_id": "session",
                    "sequence": sequence,
                    "policy_version": 0,
                    "transitions": [
                        transition_to_wire(
                            _transition("actor", sequence, float(sequence), terminal=sequence == 1)
                        )
                    ],
                    "episodes": [],
                    "evaluations": [],
                }
            )
        )

    failures: list[BaseException] = []

    def submit_first() -> None:
        try:
            coordinator._submit(request(0), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=submit_first)
    first.start()
    assert committed.wait(timeout=2.0)
    response = coordinator._submit(request(1), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
    assert codec.decode(response.value)["accepted"]

    coordinator._drain_rollouts(2)
    release.set()
    first.join(timeout=2.0)

    try:
        assert not failures
        assert not first.is_alive()
        assert [item.action for item in run.replay_store.get([0, 1])] == [0, 1]
        assert coordinator.counters.journal_applied_frontier == 2
    finally:
        release.set()
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_journal_recovery_crosses_internal_batch_boundary(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "large-recovery",
        {"total_transitions": 400, "warmup_transitions": 400, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    for sequence in range(300):
        value = {
            "actor_id": "actor",
            "session_id": "session",
            "sequence": sequence,
            "policy_version": 0,
            "transitions": [
                transition_to_wire(_transition("actor", sequence, 1.0, terminal=sequence == 299))
            ],
            "episodes": [],
            "evaluations": [],
        }
        coordinator.journal.append("session", sequence, coordinator.codec.encode(value))

    try:
        coordinator._recover_journal(0)

        assert len(run.replay_store) == 300
        assert coordinator.counters.transitions == 300
        assert coordinator.counters.journal_applied_frontier == 300
        assert run.logger.events.count("distributed/wal_recovery") == 1
        recovery = dict(run.logger.records)["distributed/wal_recovery"]
        assert recovery["rows"] == 300
        assert recovery["to_frontier"] == 300
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_corrupt_recovery_row_emits_wal_error(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "corrupt-recovery",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    coordinator.journal.append("session", 0, b"not-a-wire-payload")

    try:
        with pytest.raises(ValueError, match="wire payload"):
            coordinator._recover_journal(0)

        incident = dict(run.logger.records)["distributed/wal_error"]
        assert run.logger.events.count("distributed/wal_error") == 1
        assert incident["operation"] == "recovery_decode"
        assert "distributed/wal_recovery" not in run.logger.events
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_checkpoint_prunes_only_applied_frontier_and_recovers_tail(tmp_path: Path) -> None:
    training = {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }
    first_run = _resolved_run(tmp_path, "crash-recovery", training)
    first = Coordinator(
        first_run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    for sequence in range(2):
        value = {
            "actor_id": "actor",
            "session_id": "session",
            "sequence": sequence,
            "policy_version": 0,
            "transitions": [
                transition_to_wire(_transition("actor", sequence, 1.0, terminal=sequence == 1))
            ],
            "episodes": [],
            "evaluations": [],
        }
        first.journal.append("session", sequence, first.codec.encode(value))
    first._drain_rollouts(1)
    checkpoint = first._checkpoint()
    first._checkpoint_writer.wait()

    assert first.journal.pruned_through == 1
    assert [row_id for row_id, _ in first.journal.rows_after(0)] == [2]
    assert first_run.logger.events.count("train/checkpoint_completed") == 1
    first._checkpoint_writer.close()
    first.journal.close()

    resumed_run = _resolved_run(tmp_path, "crash-recovery", training)
    resumed = Coordinator(
        resumed_run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    try:
        resumed.restore_checkpoint(checkpoint)

        assert len(resumed_run.replay_store) == 2
        assert resumed.counters.transitions == 2
        assert resumed.counters.journal_applied_frontier == 2
        assert [item.step for item in resumed_run.replay_store.get([0, 1])] == [0, 1]
    finally:
        resumed._checkpoint_writer.close()
        resumed.journal.close()


def test_full_wakeup_queue_does_not_reject_a_durable_rollout(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "wake-only-queue",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    codec = coordinator.codec
    coordinator._rollouts = Queue(maxsize=1)
    coordinator._rollouts.put(object())
    request = BytesValue(
        value=codec.encode(
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": "fingerprint",
                "actor_id": "actor",
                "session_id": "session",
                "sequence": 0,
                "policy_version": 0,
                "transitions": [],
                "episodes": [],
            }
        )
    )

    try:
        response = coordinator._submit(request, cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
        decoded = codec.decode(response.value)

        assert decoded["accepted"] is True
        assert decoded["duplicate"] is False
        assert coordinator.journal.has_rows()
        assert coordinator._rollouts.qsize() == 1
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


@pytest.mark.parametrize(
    "malformation",
    [
        "transitions_not_list",
        "non_finite_reward",
        "invalid_action",
        "incomplete_episode",
        "invalid_evaluation_snapshot",
    ],
)
def test_malformed_rollout_is_rejected_before_wal_append(tmp_path: Path, malformation: str) -> None:
    run = _resolved_run(
        tmp_path,
        f"invalid-before-wal-{malformation}",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    token = _DISTRIBUTED_TOKEN
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=token,
        fingerprint="fingerprint",
    )
    transition = transition_to_wire(_transition("actor", 0, 1.0))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor",
        "session_id": "session",
        "sequence": 0,
        "policy_version": 0,
        "transitions": [transition],
        "episodes": [],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }
    if malformation == "transitions_not_list":
        payload["transitions"] = {}
    elif malformation == "non_finite_reward":
        transition["reward"] = torch.tensor(float("nan"))
    elif malformation == "invalid_action":
        transition["action"] = b"not-a-numeric-pytree"
    elif malformation == "incomplete_episode":
        payload["transitions"] = []
        payload["episodes"] = [{"finished": True, "finish_time_s": 1.0}]
    else:
        payload["transitions"] = []
        payload["evaluations"] = [{"finished": True, "finish_time_s": 1.0, "policy_version": 1}]
        payload["evaluation_snapshot"] = coordinator.codec.encode(["not", "a", "mapping"])
    request = BytesValue(value=coordinator.codec.encode(payload))

    try:
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            coordinator._submit(request, cast(Any, _Context(f"Bearer {token}")))
        assert not coordinator.journal.has_rows()
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_actor_episode_summary_is_accepted_before_wal_append(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "valid-actor-summary",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    summary = ActorRuntime._summary(
        1.0,
        {"termination_reason": "finished", "race_time_ms": 1_000.0},
        1,
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor",
        "session_id": "session",
        "sequence": 0,
        "policy_version": 0,
        "transitions": [],
        "episodes": [summary],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }
    request = BytesValue(value=coordinator.codec.encode(payload))

    try:
        response = coordinator._submit(
            request,
            cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")),
        )
        assert coordinator.codec.decode(response.value)["accepted"] is True
        assert coordinator.journal.has_rows()
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_evaluation_waits_for_the_first_trained_policy_snapshot(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "evaluation-after-warmup",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    codec = coordinator.codec
    coordinator._evaluation_due.add("actor")

    def request(sequence: int) -> BytesValue:
        return BytesValue(
            value=codec.encode(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": "fingerprint",
                    "actor_id": "actor",
                    "session_id": "session",
                    "sequence": sequence,
                    "policy_version": 0,
                    "transitions": [],
                    "episodes": [],
                    "evaluations": [],
                }
            )
        )

    try:
        before_training = codec.decode(
            coordinator._submit(
                request(0), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
            ).value
        )
        assert before_training["evaluate"] is False
        assert "actor" in coordinator._evaluation_due

        coordinator.counters.policy_version = 1
        coordinator._policy_payload = codec.encode({"model": {}})
        after_training = codec.decode(
            coordinator._submit(
                request(1), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}"))
            ).value
        )
        assert after_training["evaluate"] is True
        assert "actor" not in coordinator._evaluation_due
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_rollout_rejection_and_wal_failure_emit_reasoned_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _resolved_run(
        tmp_path,
        "runtime-incidents",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    codec = coordinator.codec

    def request(sequence: int) -> BytesValue:
        return BytesValue(
            value=codec.encode(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "fingerprint": "fingerprint",
                    "actor_id": "actor",
                    "session_id": "session",
                    "sequence": sequence,
                    "policy_version": 0,
                    "transitions": [transition_to_wire(_transition("actor", sequence, 1.0))],
                    "episodes": [],
                    "evaluations": [],
                }
            )
        )

    coordinator.counters.updates = run.spec.distributed.hard_policy_lag_updates + 1
    rejected = coordinator._submit(request(0), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
    coordinator.counters.updates = 0

    def fail_append(session_id: str, sequence: int, payload: bytes) -> tuple[int, bool]:
        del session_id, sequence, payload
        raise OSError("simulated WAL I/O failure")

    monkeypatch.setattr(coordinator.journal, "append", fail_append)
    try:
        with pytest.raises(OSError, match="simulated WAL I/O failure"):
            coordinator._submit(request(1), cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))

        assert codec.decode(rejected.value)["reason"] == "hard_policy_lag"
        assert run.logger.events.count("distributed/rollout_rejected") == 1
        assert run.logger.events.count("distributed/wal_error") == 1
        incidents = dict(run.logger.records)
        assert incidents["distributed/rollout_rejected"]["reason"] == "hard_policy_lag"
        assert incidents["distributed/wal_error"]["operation"] == "append"
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_stale_evaluation_only_payload_bypasses_hard_policy_lag(tmp_path: Path) -> None:
    run = _resolved_run(
        tmp_path,
        "stale-evaluation",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )
    codec = coordinator.codec
    coordinator.counters.updates = run.spec.distributed.hard_policy_lag_updates + 1
    request = BytesValue(
        value=codec.encode(
            {
                "protocol_version": PROTOCOL_VERSION,
                "fingerprint": "fingerprint",
                "actor_id": "actor",
                "session_id": "session",
                "sequence": 0,
                "policy_version": 0,
                "transitions": [],
                "episodes": [],
                "evaluations": [{"finished": True, "finish_time_s": 36.0}],
            }
        )
    )

    try:
        response = coordinator._submit(request, cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
        decoded = codec.decode(response.value)

        assert decoded["accepted"] is True
        assert decoded["force_refresh"] is True
        assert coordinator.journal.has_rows()
        assert run.logger.events.count("distributed/rollout_rejected") == 0
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()


def test_distributed_run_failure_is_emitted_and_resources_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _resolved_run(
        tmp_path,
        "run-failure",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    coordinator = Coordinator(
        run,
        bind=f"127.0.0.1:{_ephemeral_port()}",
        token=_DISTRIBUTED_TOKEN,
        fingerprint="fingerprint",
    )

    def fail() -> Any:
        raise RuntimeError("simulated learner-loop failure")

    monkeypatch.setattr(coordinator, "_run_forever", fail)

    with pytest.raises(RuntimeError, match="simulated learner-loop failure"):
        coordinator.run_forever()

    assert run.logger.events.count("run/failure") == 1
    failure = dict(run.logger.records)["run/failure"]
    assert failure["phase"] == "distributed_training"
    assert failure["exception_type"] == "RuntimeError"


def test_actor_policy_refresh_failure_stops_the_actor_loudly() -> None:
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor.force_refresh = threading.Event()
    actor.force_refresh.set()
    actor.spec = SimpleNamespace(distributed=SimpleNamespace(policy_refresh_s=60.0))
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None

    def broken_refresh() -> None:
        raise ValueError("policy snapshot must decode to a mapping")

    actor._refresh_policy = broken_refresh

    actor._policy_loop()

    assert actor.stop.is_set()
    assert "policy refresh failed" in actor.stop_reason
    assert "ValueError" in actor.stop_reason
    with pytest.raises(ActorBackgroundError, match="policy refresh failed"):
        actor._raise_background_failure()


def test_actor_reset_exhaustion_raises_a_typed_process_failure() -> None:
    class Environment:
        def reset(self, *, seed: int) -> tuple[Any, dict[str, Any]]:
            del seed
            raise TimeoutError("no telemetry frames")

    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor._actor_seed = lambda: 7

    with pytest.raises(ActorEnvironmentError, match="telemetry unavailable"):
        actor._reset_environment(Environment(), 0, attempts=1)

    assert actor.stop.is_set()
    assert "TimeoutError: no telemetry frames" in actor.stop_reason


def test_actor_process_entry_reraises_typed_failures_for_a_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Runtime:
        def __init__(self, *_: object, **__: object) -> None:
            return None

        def run_forever(self) -> None:
            raise ActorEnvironmentError("telemetry reset failed")

    monkeypatch.setattr("trackmaniarl.distributed.actor.ActorRuntime", Runtime)

    with pytest.raises(ActorEnvironmentError, match="telemetry reset failed"):
        actor_process_entry("run.yaml", "127.0.0.1:8787", "actor", _DISTRIBUTED_TOKEN)


def test_actor_snapshots_own_nested_tensors_and_arrays() -> None:
    tensor = torch.tensor([1.0, 2.0])
    array = np.asarray([3.0, 4.0], dtype=np.float32)
    observation = {"tensor": tensor, "nested": ([array], "immutable")}

    snapshot = ActorRuntime._snapshot_observation(observation)
    tensor[0] = 99.0
    array[0] = 88.0

    assert snapshot["tensor"].tolist() == [1.0, 2.0]
    assert snapshot["nested"][0][0].tolist() == [3.0, 4.0]
    assert snapshot["tensor"].data_ptr() != tensor.data_ptr()
    assert not np.shares_memory(snapshot["nested"][0][0], array)


def test_two_fake_actors_feed_slow_learner_without_data_loss(tmp_path: Path) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    spec = RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": "async-smoke",
            "artifacts_dir": str(tmp_path),
            "components": {
                "learner": {"class_path": "tests.fake:SlowLearner"},
                "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
                "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
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
        token=_DISTRIBUTED_TOKEN,
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
    clients = [_Client(f"127.0.0.1:{port}", _DISTRIBUTED_TOKEN, codec) for _ in range(2)]
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
