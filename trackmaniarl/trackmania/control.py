"""Controller backends for TrackMania; optional drivers are imported lazily."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes
from threading import RLock, Timer
from time import sleep
from typing import Protocol, runtime_checkable

import numpy as np

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouse_data", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("virtual_key", wintypes.WORD),
        ("scan_code", wintypes.WORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("extra_info", ctypes.c_size_t),
    )


class _HardwareInput(ctypes.Structure):
    _fields_ = (
        ("message", wintypes.DWORD),
        ("parameter_low", wintypes.WORD),
        ("parameter_high", wintypes.WORD),
    )


class _InputValue(ctypes.Union):
    _fields_ = (("mouse", _MouseInput), ("keyboard", _KeyboardInput), ("hardware", _HardwareInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = (("kind", wintypes.DWORD), ("value", _InputValue))


def _vgamepad_callback[Callback: Callable[..., object]](callback: Callback) -> Callback:
    """Match vgamepad's unannotated callback contract at runtime."""

    callback.__annotations__.clear()
    return callback


def _focus_trackmania() -> bool:
    if sys.platform != "win32":
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    window = user32.FindWindowW(None, "Trackmania")
    if window:
        user32.SetForegroundWindow(window)
        sleep(0.1)
        return True
    return False


def _confirm_trackmania_finish() -> None:
    if not _focus_trackmania():
        return
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event(0x0D, 0, 0, 0)
    sleep(0.1)
    user32.keybd_event(0x0D, 0, 0x0002, 0)


@runtime_checkable
class Controller(Protocol):
    def apply(self, action: np.ndarray) -> None: ...

    def consume_collision(self) -> bool: ...

    def reset(self) -> None: ...

    def confirm_finish(self) -> None: ...

    def close(self) -> None: ...


class GamepadController:
    """Virtual XInput controller with an explicit TrackMania restart action."""

    _RESTART_BUTTON = 0x2000  # Xbox B; TrackMania's default Give Up binding.

    def __init__(self) -> None:
        try:
            import vgamepad
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[trackmania] to use GamepadController") from exc
        self._gamepad = vgamepad.VX360Gamepad()
        self._tap_lock = RLock()
        self._collision_lock = RLock()
        self._tap_timer: Timer | None = None
        self._tap_generation = 0
        self._collision_detected = False
        register_notification = getattr(self._gamepad, "register_notification", None)
        if callable(register_notification):
            register_notification(callback_function=self._on_vibration)

    @_vgamepad_callback
    def _on_vibration(
        self,
        client: object,
        target: object,
        large_motor: int,
        small_motor: int,
        led_number: int,
        user_data: object,
    ) -> None:
        del client, target, small_motor, led_number, user_data
        if large_motor > 100:
            with self._collision_lock:
                self._collision_detected = True

    def consume_collision(self) -> bool:
        """Return and clear the most recent collision rumble event."""

        with self._collision_lock:
            collision_detected = self._collision_detected
            self._collision_detected = False
        return collision_detected

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
        """Release controls and request a TrackMania restart before an episode."""

        with self._tap_lock:
            self._cancel_tap_unlocked()
            self._gamepad.reset()
            self._gamepad.press_button(button=self._RESTART_BUTTON)
            self._gamepad.update()
            sleep(0.1)
            self._gamepad.release_button(button=self._RESTART_BUTTON)
            self._gamepad.update()
        self.consume_collision()

    def confirm_finish(self) -> None:
        """Confirm TrackMania's personal-record screen with Enter."""

        _confirm_trackmania_finish()

    def close(self) -> None:
        with self._tap_lock:
            self._cancel_tap_unlocked()
            self._gamepad.reset()
            self._gamepad.update()


class KeyboardController:
    """Digital W/S/A/D controller matching keyboard-recorded demonstrations."""

    _GAS = 0x57
    _BRAKE = 0x53
    _LEFT = 0x41
    _RIGHT = 0x44
    _RESTART = 0x2E
    _EXTENDED_KEYS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E})
    _KEY_EXTENDED = 0x0001
    _KEY_UP = 0x0002
    _KEY_SCAN_CODE = 0x0008

    def __init__(self, key_event: Callable[[int, bool], None] | None = None) -> None:
        if key_event is None and sys.platform != "win32":
            raise RuntimeError("KeyboardController requires Windows")
        self._requires_focus = key_event is None
        self._key_event = key_event or self._windows_key_event
        self._pressed: set[int] = set()
        self._lock = RLock()
        self._tap_timer: Timer | None = None

    @classmethod
    def _windows_key_event(cls, key: int, pressed: bool) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        scan_code = int(user32.MapVirtualKeyW(key, 0))
        flags = cls._KEY_SCAN_CODE
        if key in cls._EXTENDED_KEYS:
            flags |= cls._KEY_EXTENDED
        if not pressed:
            flags |= cls._KEY_UP
        event = _Input(
            kind=1,
            keyboard=_KeyboardInput(
                virtual_key=0,
                scan_code=scan_code,
                flags=flags,
                time=0,
                extra_info=0,
            ),
        )
        if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(_Input)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def apply(self, action: np.ndarray) -> None:
        control = np.asarray(action, dtype=np.float32).reshape(-1)
        if control.shape != (3,):
            raise ValueError("keyboard control must be [gas, brake, steer]")
        gas, brake, steer = (float(value) for value in control)
        if not self._binary(gas) or not self._binary(brake) or steer not in {-1.0, 0.0, 1.0}:
            raise ValueError("keyboard control only supports binary gas, brake, and steering")
        target = set()
        if gas > 0.5:
            target.add(self._GAS)
        if brake > 0.5:
            target.add(self._BRAKE)
        if steer > 0.5:
            target.add(self._RIGHT)
        elif steer < -0.5:
            target.add(self._LEFT)
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(target)

    def apply_discrete(self, action: np.ndarray) -> None:
        control = np.asarray(action, dtype=np.float32).copy()
        if control.shape != (3,):
            raise ValueError("keyboard control must be [gas, brake, steer]")
        if float(control[1]) != BRAKE_TAP_SENTINEL:
            self.apply(control)
            return
        control[1] = 1.0
        with self._lock:
            self._cancel_tap_unlocked()
            self._apply_without_timer(control)
            self._tap_timer = Timer(BRAKE_TAP_DURATION_S, self._release_brake)
            self._tap_timer.daemon = True
            self._tap_timer.start()

    def _apply_without_timer(self, control: np.ndarray) -> None:
        gas, brake, steer = (float(value) for value in control)
        target = set()
        if gas > 0.5:
            target.add(self._GAS)
        if brake > 0.5:
            target.add(self._BRAKE)
        if steer > 0.5:
            target.add(self._RIGHT)
        elif steer < -0.5:
            target.add(self._LEFT)
        self._set_pressed(target)

    def _release_brake(self) -> None:
        with self._lock:
            self._tap_timer = None
            target = self._pressed - {self._BRAKE}
            self._set_pressed(target)

    def _cancel_tap_unlocked(self) -> None:
        if self._tap_timer is not None:
            self._tap_timer.cancel()
            self._tap_timer = None

    def _set_pressed(self, target: set[int]) -> None:
        for key in sorted(self._pressed - target):
            self._key_event(key, False)
        for key in sorted(target - self._pressed):
            self._key_event(key, True)
        self._pressed = target

    @staticmethod
    def _binary(value: float) -> bool:
        return value in {0.0, 1.0}

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())
            if self._requires_focus and not _focus_trackmania():
                raise RuntimeError("Trackmania window was not found for keyboard control")
            self._key_event(self._RESTART, True)
            sleep(0.1)
            self._key_event(self._RESTART, False)

    def confirm_finish(self) -> None:
        _confirm_trackmania_finish()

    def close(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())


class RecordingController:
    """Safe controller used by tests and dry diagnostics without game input."""

    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []

    def apply(self, action: np.ndarray) -> None:
        self.actions.append(np.asarray(action, dtype=np.float32).copy())

    def apply_discrete(self, action: np.ndarray) -> None:
        self.apply(action)

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        self.actions.clear()

    def confirm_finish(self) -> None:
        return None

    def close(self) -> None:
        return None
