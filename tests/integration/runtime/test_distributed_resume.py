from __future__ import annotations

import multiprocessing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import grpc
import pytest
import zstandard
from google.protobuf.wrappers_pb2 import BytesValue

from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _ephemeral_port,
    _resolved_run,
    _RestoreSpy,
    _SlowLearner,
    _spawn_probe,
)
from trackmaniarl.distributed.actor_transport import Client
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_support import _Counters
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig, ReplayRestoreMode
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    auth_metadata,
    deserialize_message,
    grpc_method,
    serialize_message,
)


@dataclass(frozen=True, slots=True)
class _RestoreCase:
    coordinator: Coordinator
    learner: _SlowLearner
    replay: _RestoreSpy
    sampler: _RestoreSpy


@dataclass(frozen=True, slots=True)
class _RestoreParts:
    tmp_path: Path
    learner: _SlowLearner
    replay: _RestoreSpy
    sampler: _RestoreSpy
    checkpoint: dict[str, object]


def _restore_case(tmp_path: Path) -> _RestoreCase:
    learner = _SlowLearner()
    replay, sampler = _RestoreSpy(), _RestoreSpy()
    checkpoint = {
        "schema_version": "2.0",
        "journal_contract_version": 2,
        "journal_id": "pending",
        "run_fingerprint": "fingerprint",
        "learner": {"value": 7},
        "replay_store": {"transitions": ["old"]},
        "sampler": {"priorities": [1.0]},
        "distributed": asdict(_Counters(transitions=42, updates=11)),
        "evaluated_policy_version": None,
    }
    run = _restore_run(_RestoreParts(tmp_path, learner, replay, sampler, checkpoint))
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    coordinator = Coordinator(run, config)
    checkpoint["journal_id"] = coordinator.journal.identity
    return _RestoreCase(coordinator, learner, replay, sampler)


def _restore_run(parts: _RestoreParts) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(
            distributed=SimpleNamespace(
                max_message_bytes=1024 * 1024,
                max_update_credit=512,
            ),
            evaluation=None,
        ),
        run_dir=parts.tmp_path / "weights-only",
        learner=parts.learner,
        replay_store=parts.replay,
        sampler=parts.sampler,
        checkpoint_codec=SimpleNamespace(load=lambda _: parts.checkpoint),
    )


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
    case = _restore_case(tmp_path)

    try:
        case.coordinator.restore_checkpoint(
            tmp_path / "checkpoint.pt", ReplayRestoreMode.LEARNER_ONLY
        )
    finally:
        case.coordinator._close_runtime()

    assert case.learner.value == 7
    assert case.replay.restored is None
    assert case.sampler.restored is None
    assert case.coordinator.counters == _Counters()


def test_coordinator_restores_evaluation_leaders(
    tmp_path: Path,
) -> None:
    case = _restore_case(tmp_path)
    checkpoint = case.coordinator.run.checkpoint_codec.load(tmp_path / "checkpoint.pt")
    checkpoint["distributed"]["best_evaluation_rank"] = (1.0, -38.0, 0.8)
    checkpoint["distributed"]["fastest_evaluation_rank"] = (-36.8, 0.2, -36.8)

    try:
        case.coordinator.restore_checkpoint(tmp_path / "checkpoint.pt")
    finally:
        case.coordinator._close_runtime()

    assert case.coordinator._best_evaluation == (1.0, -38.0, 0.8)
    assert case.coordinator._fastest_evaluation == (-36.8, 0.2, -36.8)


def test_coordinator_accepts_checkpoint_without_evaluation_leaders(tmp_path: Path) -> None:
    case = _restore_case(tmp_path)

    try:
        case.coordinator.restore_checkpoint(tmp_path / "checkpoint.pt")
    finally:
        case.coordinator._close_runtime()

    assert case.coordinator._best_evaluation is None
    assert case.coordinator._fastest_evaluation is None


@pytest.mark.parametrize("invalid_rank", [(1.0, "-38.0", 0.8), (1.0, True, 0.8)])
def test_coordinator_rejects_non_real_evaluation_leader_rank(
    tmp_path: Path, invalid_rank: tuple[object, object, object]
) -> None:
    case = _restore_case(tmp_path)
    checkpoint = case.coordinator.run.checkpoint_codec.load(tmp_path / "checkpoint.pt")
    checkpoint["distributed"]["best_evaluation_rank"] = invalid_rank

    try:
        with pytest.raises(TypeError, match="must contain three numbers"):
            case.coordinator.restore_checkpoint(tmp_path / "checkpoint.pt")
    finally:
        case.coordinator._close_runtime()


def _wire_coordinator(tmp_path: Path) -> Coordinator:
    run = _resolved_run(
        tmp_path,
        "grpc-wire-limits",
        {
            "total_transitions": 10,
            "warmup_transitions": 10,
            "checkpoint_interval_updates": None,
        },
    )
    distributed = run.spec.distributed.model_copy(update={"max_message_bytes": 1_024})
    run = replace(run, spec=run.spec.model_copy(update={"distributed": distributed}))
    config = CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def _start_wire_client(coordinator: Coordinator) -> Client:
    coordinator._start_server()
    assert coordinator.bound_port > 0
    assert Coordinator.bound_port.fset is None
    client = Client(f"127.0.0.1:{coordinator.bound_port}", _DISTRIBUTED_TOKEN, coordinator.codec)
    grpc.channel_ready_future(client.channel).result(timeout=10.0)
    return client


def _heartbeat_rpc(client: Client) -> Any:
    return client.channel.unary_unary(
        grpc_method("Heartbeat"),
        request_serializer=serialize_message,
        response_deserializer=deserialize_message,
    )


def _assert_malformed_request(heartbeat: Any) -> None:
    with pytest.raises(grpc.RpcError) as malformed:
        heartbeat(
            BytesValue(value=b"malformed-wire-payload"),
            metadata=auth_metadata(_DISTRIBUTED_TOKEN),
            timeout=5.0,
        )
    assert malformed.value.code() is grpc.StatusCode.INVALID_ARGUMENT
    assert malformed.value.details() == "distributed request payload is malformed"


def _assert_decompression_limit(heartbeat: Any, coordinator: Coordinator) -> None:
    expanded = b"decompression-bomb-marker" * 128
    compressed = zstandard.ZstdCompressor().compress(expanded)
    assert len(compressed) < coordinator.codec.max_message_bytes < len(expanded)
    with pytest.raises(grpc.RpcError) as exhausted:
        heartbeat(
            BytesValue(value=compressed),
            metadata=auth_metadata(_DISTRIBUTED_TOKEN),
            timeout=5.0,
        )
    assert exhausted.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert exhausted.value.details() == "distributed request exceeds the configured size limit"
    assert "decompression-bomb-marker" not in exhausted.value.details()


def _assert_recovered_heartbeat(client: Client, coordinator: Coordinator) -> None:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor-after-bomb",
        "session_id": "session-after-bomb",
        "policy_version": 0,
        "spool_bytes": 0,
    }
    assert client.call("Heartbeat", payload) == {"stop": False}
    assert coordinator.run.logger.events[-1] == "actor/heartbeat"


def _assert_response_limit(client: Client, coordinator: Coordinator) -> None:
    coordinator._policy_payload = b"response-overflow-marker" * 32
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "actor-after-bomb",
        "session_id": "session-after-bomb",
        "current_version": -1,
    }
    with pytest.raises(grpc.RpcError) as exhausted:
        client.call("Policy", payload, timeout=5.0)
    assert exhausted.value.code() is grpc.StatusCode.RESOURCE_EXHAUSTED
    assert exhausted.value.details() == "distributed response exceeds the configured size limit"
    assert "response-overflow-marker" not in exhausted.value.details()


def test_grpc_maps_wire_errors_and_recovers_after_decompression_bomb(tmp_path: Path) -> None:
    coordinator = _wire_coordinator(tmp_path)
    with pytest.raises(RuntimeError, match="has not bound"):
        _ = coordinator.bound_port
    client = _start_wire_client(coordinator)
    try:
        heartbeat = _heartbeat_rpc(client)
        _assert_malformed_request(heartbeat)
        _assert_decompression_limit(heartbeat, coordinator)
        _assert_recovered_heartbeat(client, coordinator)
        _assert_response_limit(client, coordinator)
    finally:
        client.close()
        coordinator._close_runtime()
