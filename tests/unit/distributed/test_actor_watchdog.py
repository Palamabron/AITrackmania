from __future__ import annotations

from types import SimpleNamespace

import pytest

import trackmaniarl.distributed.actor_background as actor_background
import trackmaniarl.distributed.actor_watchdog as actor_watchdog
from trackmaniarl.distributed.actor_watchdog import ActorStallError, ProgressWatchdog, stall


def test_watchdog_reports_idle_time_with_injected_clock() -> None:
    now = [100.0]
    watchdog = ProgressWatchdog(clock=lambda: now[0])
    now[0] = 130.0

    assert watchdog.idle_s() == 30.0
    assert stall(watchdog, None) is None
    assert stall(watchdog, 60.0) is None
    error = stall(watchdog, 20.0)
    assert isinstance(error, ActorStallError)
    assert error.idle_s == 30.0
    watchdog.touch()
    assert watchdog.idle_s() == 0.0


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

    now[0] = 5.0
    assert actor_background._stop_on_collection_stall(runtime) is False
    now[0] = 11.0
    assert actor_background._stop_on_collection_stall(runtime) is True
    assert stops == ["collection watchdog: actor collection made no progress for 11 s (limit 10 s)"]
    assert exits == [actor_watchdog.STALL_EXIT_GRACE_S]
