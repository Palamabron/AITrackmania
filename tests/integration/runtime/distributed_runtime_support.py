from __future__ import annotations

import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import grpc
import numpy as np
from google.protobuf.wrappers_pb2 import BytesValue

from trackmaniarl.core.builtins import TorchCheckpointCodec
from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.core.data import Transition
from trackmaniarl.core.replay import InMemoryReplayStore, UniformSampler
from trackmaniarl.core.runtime import ResolvedRun
from trackmaniarl.core.spec import RunSpec
from trackmaniarl.distributed.actor import ActorRuntime
from trackmaniarl.distributed.coordinator import Coordinator
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig
from trackmaniarl.distributed.journal import RolloutJournal
from trackmaniarl.distributed.protocol import (
    PROTOCOL_VERSION,
    transition_to_wire,
)

_DISTRIBUTED_TOKEN = "tests-only-distributed-token-0123456789"


class _Pipeline:
    def transform_observation(self, observation: Any) -> Any:
        return observation

    def collate(self, transitions: list[Transition]) -> Mapping[str, Any]:
        return {"reward": np.asarray([item.reward for item in transitions])}


class _Policy:
    def __init__(self, value: int) -> None:
        self.value = value

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del observation, mode
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


def _append_journal_then_exit(path: str, payload: bytes) -> None:
    journal = RolloutJournal(Path(path))
    journal.append("crashed-session", 0, payload)
    os._exit(23)


def _submit_rollout_then_exit(path: str) -> None:
    run = _crash_run(Path(path))
    coordinator = Coordinator(
        run,
        CoordinatorConfig("127.0.0.1:0", _DISTRIBUTED_TOKEN, "fingerprint"),
    )
    request = _crash_submit_request(coordinator)
    coordinator._submit(request, cast(Any, _Context(f"Bearer {_DISTRIBUTED_TOKEN}")))
    os._exit(25)


def _crash_run(path: Path) -> ResolvedRun:
    training = {
        "total_transitions": 10,
        "warmup_transitions": 10,
        "checkpoint_interval_updates": None,
    }
    return _resolved_run(path, "learner-crash-after-submit", training)


def _crash_submit_request(coordinator: Coordinator) -> BytesValue:
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "fingerprint": "fingerprint",
        "actor_id": "crashed-actor",
        "session_id": "crashed-session",
        "sequence": 0,
        "policy_version": 0,
        "transitions": [transition_to_wire(_transition(_TransitionSpec("crashed-actor", 7, 1.0)))],
        "episodes": [],
        "evaluations": [],
        "evaluation_snapshot": b"",
    }
    return BytesValue(value=coordinator.codec.encode(payload))


def _persist_spool_then_exit(path: str, payload: bytes) -> None:
    ActorRuntime._persist_spool_payload(Path(path), payload)
    os._exit(24)


class _TransitionState(StrEnum):
    CONTINUES = "continues"
    TERMINATES = "terminates"


@dataclass(frozen=True, slots=True)
class _TransitionSpec:
    actor: str
    step: int
    reward: float
    state: _TransitionState = _TransitionState.CONTINUES


def _transition(spec: _TransitionSpec) -> Transition:
    return Transition(
        observation=np.asarray([spec.step], dtype=np.float32),
        action=spec.step,
        reward=spec.reward,
        next_observation=np.asarray([spec.step + 1], dtype=np.float32),
        terminated=spec.state is _TransitionState.TERMINATES,
        truncated=False,
        episode_id=f"{spec.actor}/session/episode",
        step=spec.step,
        info={"policy_version": 7, "actor_epsilon": 0.1},
    )


def _ephemeral_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _resolved_run(tmp_path: Path, run_id: str, training: dict[str, Any]) -> ResolvedRun:
    spec = _run_spec(tmp_path, run_id, training)
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


def _run_spec(tmp_path: Path, run_id: str, training: dict[str, Any]) -> RunSpec:
    return RunSpec.model_validate(
        {
            "api_version": "2.0",
            "run_id": run_id,
            "artifacts_dir": str(tmp_path),
            "components": _component_specs(),
            "training": training,
        }
    )


def _component_specs() -> dict[str, dict[str, str]]:
    return {
        "learner": {"class_path": "tests.fake:SlowLearner"},
        "replay_store": {"class_path": "trackmaniarl.core.replay:InMemoryReplayStore"},
        "sampler": {"class_path": "trackmaniarl.core.replay:UniformSampler"},
        "feature_pipeline": {"class_path": "tests.fake:Pipeline"},
    }
