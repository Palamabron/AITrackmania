from __future__ import annotations

import threading
import time
from pathlib import Path

import grpc
import pytest

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _Logger,
    _Pipeline,
    _SlowLearner,
    _transition,
    _TransitionSpec,
    _TransitionState,
)
from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor_transport import Client
from trackmaniarl.distributed.codec import WireCodec
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.protocol import PROTOCOL_VERSION, transition_to_wire


def _async_spec(tmp_path: Path) -> RunSpec:
    config = {
        "api_version": "2.0",
        "run_id": "async-smoke",
        "artifacts_dir": str(tmp_path),
        "components": _async_components(),
        "training": _async_training(),
        "distributed": {"policy_refresh_s": 0.001},
    }
    return RunSpec.model_validate(config)


def _async_components() -> dict[str, dict[str, str]]:
    return {
        "learner": {"class_path": "tests.fake:SlowLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
    }


def _async_training() -> dict[str, float | int]:
    return {
        "total_transitions": 16,
        "batch_size": 2,
        "n_step": 2,
        "warmup_transitions": 4,
        "updates_per_transition": 0.25,
        "checkpoint_interval_updates": 100,
    }


def _async_run(tmp_path: Path) -> tuple[ResolvedRun, _Logger]:
    spec = _async_spec(tmp_path)
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
    return run, logger


def _serve(coordinator: Coordinator, failures: list[BaseException]) -> None:
    try:
        coordinator.run_forever()
    except BaseException as exc:
        failures.append(exc)


def _await_port(coordinator: Coordinator, server: threading.Thread) -> int:
    deadline = time.monotonic() + 10.0
    while coordinator._bound_port is None and server.is_alive():
        if time.monotonic() >= deadline:
            pytest.fail("coordinator did not bind its dynamic port")
        time.sleep(0.01)
    return coordinator.bound_port


def _clients(port: int, spec: RunSpec) -> list[Client]:
    codec = WireCodec(spec.distributed.max_message_bytes)
    clients = [Client(f"127.0.0.1:{port}", _DISTRIBUTED_TOKEN, codec) for _ in range(2)]
    for client in clients:
        grpc.channel_ready_future(client.channel).result(timeout=10)
    return clients


def _actor_transitions(actor_id: str) -> list[dict[str, object]]:
    return [
        transition_to_wire(
            _transition(
                _TransitionSpec(
                    actor_id,
                    step,
                    float(step),
                    _TransitionState.TERMINATES if step == 7 else _TransitionState.CONTINUES,
                )
            )
        )
        for step in range(8)
    ]


def _send_actor(client: Client, actor_index: int) -> None:
    actor_id = f"actor-{actor_index}"
    base = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": actor_id,
        "session_id": f"session-{actor_index}",
    }
    client.call("Register", base)
    payload = {**base, "sequence": 0, "policy_version": 0}
    payload.update(
        transitions=_actor_transitions(actor_id),
        episodes=[],
        evaluations=[],
        evaluation_snapshot=b"",
    )
    assert client.call("Submit", payload)["accepted"]


def _run_senders(clients: list[Client]) -> None:
    senders = [
        threading.Thread(target=_send_actor, args=(client, index))
        for index, client in enumerate(clients)
    ]
    for sender in senders:
        sender.start()
    for sender in senders:
        sender.join(timeout=10)


def _assert_result(coordinator: Coordinator, run: ResolvedRun, logger: _Logger) -> None:
    assert isinstance(run.learner, _SlowLearner)
    assert run.learner.manifest_present_at_setup
    assert coordinator.counters.transitions == 16
    assert len(run.replay_store) == 16
    assert coordinator.counters.updates == 3
    assert coordinator.counters.policy_version >= 1
    assert logger.events.count("distributed/ingest") == 2
    registrations = [payload for event, payload in logger.records if event == "actor/registered"]
    assert {payload["run_fingerprint"] for payload in registrations} == {"fingerprint"}


def test_two_fake_actors_feed_slow_learner_without_data_loss(tmp_path: Path) -> None:
    run, logger = _async_run(tmp_path)
    config = CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint")
    coordinator = Coordinator(run, config)
    failures: list[BaseException] = []
    server = threading.Thread(target=_serve, args=(coordinator, failures))
    server.start()
    clients = _clients(_await_port(coordinator, server), run.spec)

    _run_senders(clients)
    server.join(timeout=10)
    for client in clients:
        client.close()

    assert not server.is_alive()
    assert not failures
    _assert_result(coordinator, run, logger)
