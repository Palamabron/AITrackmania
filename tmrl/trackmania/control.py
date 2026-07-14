"""Controller backends for TrackMania; optional drivers are imported lazily."""

from __future__ import annotations

from threading import RLock, Timer
from time import sleep
from typing import Protocol, runtime_checkable

import numpy as np

from tmrl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL


@runtime_checkable
class Controller(Protocol):
    def apply(self, action: np.ndarray) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class GamepadController:
    """Virtual XInput controller with an explicit TrackMania respawn action."""

    _RESPAWN_BUTTON = 0x2000  # Xbox B; TrackMania's default respawn binding.

    def __init__(self) -> None:
        try:
            import vgamepad
        except ImportError as exc:
            raise RuntimeError("Install tmrl[trackmania] to use GamepadController") from exc
        self._gamepad = vgamepad.VX360Gamepad()
        self._tap_lock = RLock()
        self._tap_timer: Timer | None = None
        self._tap_generation = 0

    def _apply(self, action: np.ndarray) -> None:
        gas, brake, steer = np.clip(
            np.nan_to_num(action, nan=0.0), [-0.0, 0.0, -1.0], [1.0, 1.0, 1.0]
        )
        self._gamepad.right_trigger_float(float(gas))
        self._gamepad.left_trigger_float(float(brake))
        self._gamepad.left_joystick_float(float(steer), 0.0)
        self._gamepad.update()

    def _cancel_tap_unlocked(self) -> None:
        self._tap_generation += 1
        if self._tap_timer is not None:
            self._tap_timer.cancel()
            self._tap_timer = None

    def _release_tap(self, generation: int, gas: float, steer: float) -> None:
        with self._tap_lock:
            if generation != self._tap_generation:
                return
            self._tap_timer = None
            self._apply(np.asarray([gas, 0.0, steer], dtype=np.float32))

    def apply(self, action: np.ndarray) -> None:
        with self._tap_lock:
            self._cancel_tap_unlocked()
            self._apply(action)

    def apply_discrete(self, action: np.ndarray) -> None:
        """Apply a table action, releasing the brake after the explicit tap interval."""

        control = np.asarray(action, dtype=np.float32).copy()
        if control.shape != (3,):
            raise ValueError("discrete TrackMania control must be [gas, brake, steer]")
        if float(control[1]) == BRAKE_TAP_SENTINEL:
            with self._tap_lock:
                self._cancel_tap_unlocked()
                self._apply(np.asarray([control[0], 1.0, control[2]], dtype=np.float32))
                generation = self._tap_generation
                self._tap_timer = Timer(
                    BRAKE_TAP_DURATION_S,
                    self._release_tap,
                    args=(generation, float(control[0]), float(control[2])),
                )
                self._tap_timer.daemon = True
                self._tap_timer.start()
            return
        self.apply(control)

    def reset(self) -> None:
        """Release controls and request a TrackMania respawn before an episode."""

        with self._tap_lock:
            self._cancel_tap_unlocked()
            self._gamepad.reset()
            self._gamepad.press_button(button=self._RESPAWN_BUTTON)
            self._gamepad.update()
            sleep(0.1)
            self._gamepad.release_button(button=self._RESPAWN_BUTTON)
            self._gamepad.update()

    def close(self) -> None:
        with self._tap_lock:
            self._cancel_tap_unlocked()
            self._gamepad.reset()
            self._gamepad.update()


class RecordingController:
    """Safe controller used by tests and dry diagnostics without game input."""

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def apply(self, action: np.ndarray) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def apply_discrete(self, action: np.ndarray) -> None:
        self.apply(action)

    def reset(self) -> None:
        self.actions.clear()

    def close(self) -> None:
        return None
