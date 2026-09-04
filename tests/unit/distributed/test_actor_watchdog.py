from __future__ import annotations

import threading
from types import SimpleNamespace

import grpc
import pytest

import trackmaniarl.distributed.actor_background as actor_background
import trackmaniarl.distributed.actor_collection as actor_collection
import trackmaniarl.distributed.actor_watchdog as actor_watchdog
from trackmaniarl.distributed.actor_watchdog import ActorStallError, ProgressWatchdog, stall


class _RetryableRpcError(grpc.RpcError):
    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.UNAVAILABLE


def _stalled_runtime(now: list[float], stops: list[str]) -> SimpleNamespace:
    watchdog = ProgressWatchdog(clock=lambda: now[0])
    watchdog.arm()
    now[0] = 11.0
    return SimpleNamespace(
        watchdog=watchdog,
        stop=threading.Event(),
        spec=SimpleNamespace(distributed=SimpleNamespace(actor_stall_timeout_s=10.0)),
        _stop_from_thread=lambda stage, exc: stops.append(f"{stage}: {exc}"),
    )


def test_watchdog_reports_idle_time_with_injected_clock() -> None:
    now = [100.0]
    watchdog = ProgressWatchdog(clock=lambda: now[0])
    now[0] = 130.0

    assert watchdog.idle_s() is None
    assert stall(watchdog, 20.0) is None
    watchdog.arm()
    now[0] = 160.0
    assert watchdog.idle_s() == 30.0
    assert stall(watchdog, None) is None
    assert stall(watchdog, 60.0) is None
    error = stall(watchdog, 20.0)
    assert isinstance(error, ActorStallError)
    assert error.idle_s == 30.0
    watchdog.touch()
    assert watchdog.idle_s() == 0.0


def test_collection_arms_watchdog_before_prewarm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [5.0]
    watchdog = ProgressWatchdog(clock=lambda: now[0])
    runtime = SimpleNamespace(watchdog=watchdog)

    def observe_armed(context: actor_collection.CollectionContext) -> bool:
        assert context.runtime is runtime
        assert watchdog.idle_s() == 0.0
        return False

    monkeypatch.setattr(actor_collection, "_prewarm_initial_policy", observe_armed)

    actor_collection.collect(runtime, object(), object())


def test_terminate_after_grace_exits_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(actor_watchdog, "_terminate_process", calls.append)

    thread = actor_watchdog.terminate_after_grace(0.01)
    thread.join(2.0)

    assert calls == [3]


def test_heartbeat_stall_requests_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    stops: list[str] = []
    exits: list[float] = []
    monkeypatch.setattr(actor_watchdog, "terminate_after_grace", exits.append)
    runtime = SimpleNamespace(
        watchdog=ProgressWatchdog(clock=lambda: now[0]),
        spec=SimpleNamespace(distributed=SimpleNamespace(actor_stall_timeout_s=10.0)),
        _stop_from_thread=lambda stage, exc: stops.append(f"{stage}: {exc}"),
    )
    runtime.watchdog.arm()

    now[0] = 5.0
    assert actor_background._stop_on_collection_stall(runtime) is False
    now[0] = 11.0
    assert actor_background._stop_on_collection_stall(runtime) is True
    assert stops == ["collection watchdog: actor collection made no progress for 11 s (limit 10 s)"]
    assert exits == [actor_watchdog.STALL_EXIT_GRACE_S]


def test_retryable_heartbeat_failure_still_checks_collection_stall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    stops: list[str] = []
    exits: list[float] = []

    def fail_heartbeat(runtime: object) -> None:
        del runtime
        raise _RetryableRpcError

    monkeypatch.setattr(actor_background, "_send_heartbeat", fail_heartbeat)
    monkeypatch.setattr(actor_watchdog, "terminate_after_grace", exits.append)
    runtime = _stalled_runtime(now, stops)

    assert actor_background._heartbeat_once(runtime) is True

    assert stops == ["collection watchdog: actor collection made no progress for 11 s (limit 10 s)"]
    assert exits == [actor_watchdog.STALL_EXIT_GRACE_S]
