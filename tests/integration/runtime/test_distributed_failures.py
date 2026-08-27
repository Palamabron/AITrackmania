from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import trackmaniarl.distributed.coordinator_runtime as coordinator_runtime
from tests.integration.runtime.distributed_runtime_support import (
    _DISTRIBUTED_TOKEN,
    _ephemeral_port,
    _resolved_run,
)
from trackmaniarl.distributed.actor import (
    ActorBackgroundError,
    ActorEnvironmentError,
    ActorRuntime,
    actor_process_entry,
)
from trackmaniarl.distributed.actor_requests import ActorProcessRequest, EnvironmentReset
from trackmaniarl.distributed.coordinator import (
    Coordinator,
)
from trackmaniarl.distributed.coordinator_types import CoordinatorConfig


def _failing_coordinator(tmp_path: Path) -> Coordinator:
    run = _resolved_run(
        tmp_path,
        "run-failure",
        {"total_transitions": 10, "warmup_transitions": 10, "checkpoint_interval_updates": None},
    )
    config = CoordinatorConfig(f"127.0.0.1:{_ephemeral_port()}", _DISTRIBUTED_TOKEN, "fingerprint")
    return Coordinator(run, config)


def test_distributed_run_failure_is_emitted_and_resources_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = _failing_coordinator(tmp_path)

    def fail(runtime: Coordinator) -> Any:
        del runtime
        raise RuntimeError("simulated learner-loop failure")

    monkeypatch.setattr(coordinator_runtime, "_run_forever", fail)

    with pytest.raises(RuntimeError, match="simulated learner-loop failure"):
        coordinator.run_forever()

    run = coordinator.run
    assert run.logger.events.count("run/failure") == 1
    failure = dict(run.logger.records)["run/failure"]
    assert failure["phase"] == "distributed_training"
    assert failure["exception_type"] == "RuntimeError"


def _policy_failure_actor() -> ActorRuntime:
    actor = object.__new__(ActorRuntime)
    actor.actor_id = "actor"
    actor.stop = threading.Event()
    actor.stop_reason = "running"
    actor.force_refresh = threading.Event()
    actor.force_refresh.set()
    actor.spec = SimpleNamespace(distributed=SimpleNamespace(policy_refresh_s=60.0))
    actor._background_failure_lock = threading.Lock()
    actor._background_failure = None
    return actor


def test_actor_policy_refresh_failure_stops_the_actor_loudly() -> None:
    actor = _policy_failure_actor()

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
        actor._reset_environment(EnvironmentReset(Environment(), 0, attempts=1))

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

    config = ActorProcessRequest("run.yaml", "127.0.0.1:8787", "actor", _DISTRIBUTED_TOKEN)
    with pytest.raises(ActorEnvironmentError, match="telemetry reset failed"):
        actor_process_entry(config)
