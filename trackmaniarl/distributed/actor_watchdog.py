"""Actor-side stall detection for the collection thread.

The heartbeat thread keeps an actor "alive" from the learner's point of view even
when the collection thread is blocked in a game reset or a native input call.
The v104f run idled for 4.4 h that way. The watchdog records the time of the
last environment step; the heartbeat loop compares it with
``distributed.actor_stall_timeout_s`` and, when exceeded, requests a graceful
stop and hard-exits the process if the stop does not complete in time, so the
launcher sees a dead actor instead of a silent one.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from time import monotonic

logger = logging.getLogger(__name__)

STALL_EXIT_GRACE_S = 60.0
_terminate_process: Callable[[int], None] = os._exit


class ActorStallError(RuntimeError):
    def __init__(self, idle_s: float, timeout_s: float) -> None:
        super().__init__(
            f"actor collection made no progress for {idle_s:.0f} s (limit {timeout_s:.0f} s)"
        )
        self.idle_s = idle_s
        self.timeout_s = timeout_s


class ProgressWatchdog:
    """Timestamp of the collection thread's last environment step, readable across threads."""

    def __init__(self, clock: Callable[[], float] = monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._last = clock()

    def touch(self) -> None:
        with self._lock:
            self._last = self._clock()

    def idle_s(self) -> float:
        with self._lock:
            return self._clock() - self._last


def watchdog_of(runtime: object) -> ProgressWatchdog | None:
    """Return the runtime's watchdog; test doubles and legacy runtimes may not carry one."""

    watchdog = getattr(runtime, "watchdog", None)
    return watchdog if isinstance(watchdog, ProgressWatchdog) else None


def touch(runtime: object) -> None:
    watchdog = watchdog_of(runtime)
    if watchdog is not None:
        watchdog.touch()


def stall(watchdog: ProgressWatchdog, timeout_s: float | None) -> ActorStallError | None:
    """Return the stall error when the collection thread exceeded ``timeout_s`` of idleness."""

    if timeout_s is None:
        return None
    idle_s = watchdog.idle_s()
    if idle_s <= timeout_s:
        return None
    return ActorStallError(idle_s, timeout_s)


def terminate_after_grace(grace_s: float) -> threading.Thread:
    """Hard-exit the process if it is still running ``grace_s`` after a stall stop request."""

    def _exit_if_still_running() -> None:
        threading.Event().wait(grace_s)
        logger.error("actor did not stop %.0f s after a collection stall; exiting", grace_s)
        _terminate_process(3)

    thread = threading.Thread(
        target=_exit_if_still_running, name="trackmaniarl-actor-stall-exit", daemon=True
    )
    thread.start()
    return thread
