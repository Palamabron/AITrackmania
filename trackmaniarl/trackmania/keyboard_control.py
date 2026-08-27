"""Windows keyboard control for TrackMania."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum
from threading import RLock, Timer
from time import sleep

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


class _KeyState(Enum):
    PRESSED = "pressed"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class KeyboardKeyEvent:
    key: int
    pressed: bool


@dataclass(frozen=True, slots=True)
class _KeyStroke:
    key: int
    scan_code: int
    state: _KeyState


def _windows_dll(name: str) -> object:
    if sys.platform != "win32":
        raise RuntimeError(f"{name} is only available on Windows")
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise RuntimeError("Windows ctypes support is unavailable")
    return loader(name, use_last_error=True)


def _windows_call(library: object, name: str, *args: object) -> object:
    function = getattr(library, name)
    if not callable(function):
        raise RuntimeError(f"Windows API function is unavailable: {name}")
    return function(*args)


def _windows_int_call(library: object, name: str, *args: object) -> int:
    result = _windows_call(library, name, *args)
    if not isinstance(result, int):
        raise RuntimeError(f"Windows API function returned a non-integer result: {name}")
    return result


def _focus_trackmania() -> bool:
    if sys.platform != "win32":
        return False
    user32 = _windows_dll("user32")
    window = _windows_call(user32, "FindWindowW", None, "Trackmania")
    if not window:
        return False
    _windows_call(user32, "SetForegroundWindow", window)
    sleep(0.1)
    return True


def confirm_trackmania_finish() -> None:
    if not _focus_trackmania():
        return
    user32 = _windows_dll("user32")
    _windows_call(user32, "keybd_event", 0x0D, 0, 0, 0)
    sleep(0.1)
    _windows_call(user32, "keybd_event", 0x0D, 0, 0x0002, 0)


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
    _STEERING_DEADZONE = 0.25

    def __init__(self, key_event: Callable[[KeyboardKeyEvent], None] | None = None) -> None:
        if key_event is None and sys.platform != "win32":
            raise RuntimeError("KeyboardController requires Windows")
        self._requires_focus = key_event is None
        self._key_event = key_event or self._windows_key_event
        self._pressed: set[int] = set()
        self._lock = RLock()
        self._tap_timer: Timer | None = None

    @classmethod
    def _windows_key_event(cls, command: KeyboardKeyEvent) -> None:
        user32 = _windows_dll("user32")
        scan_code = _windows_int_call(user32, "MapVirtualKeyW", command.key, 0)
        state = _KeyState.PRESSED if command.pressed else _KeyState.RELEASED
        event = cls._keyboard_input(_KeyStroke(command.key, scan_code, state))
        sent = _windows_int_call(user32, "SendInput", 1, ctypes.byref(event), ctypes.sizeof(_Input))
        if sent != 1:
            error_code = _windows_int_call(_windows_dll("kernel32"), "GetLastError")
            raise OSError(error_code, "SendInput failed")

    @classmethod
    def _keyboard_input(cls, stroke: _KeyStroke) -> _Input:
        return _Input(
            kind=1,
            keyboard=_KeyboardInput(
                virtual_key=0,
                scan_code=stroke.scan_code,
                flags=cls._keyboard_flags(stroke),
                time=0,
                extra_info=0,
            ),
        )

    @classmethod
    def _keyboard_flags(cls, stroke: _KeyStroke) -> int:
        flags = cls._KEY_SCAN_CODE
        if stroke.key in cls._EXTENDED_KEYS:
            flags |= cls._KEY_EXTENDED
        if stroke.state is _KeyState.RELEASED:
            flags |= cls._KEY_UP
        return flags

    def apply(self, action: np.ndarray) -> None:
        control = self._digital_control(action)
        self._validate_control(control)
        self._apply_target(self._key_target(control))

    def apply_discrete(self, action: np.ndarray) -> None:
        control = np.asarray(action, dtype=np.float32).copy()
        self._validate_control(control)
        if float(control[1]) != BRAKE_TAP_SENTINEL:
            self.apply(control)
            return
        control = self._digital_control_with_brake_tap(control)
        control[1] = 1.0
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(self._key_target(control))
            self._tap_timer = Timer(BRAKE_TAP_DURATION_S, self._release_brake)
            self._tap_timer.daemon = True
            self._tap_timer.start()

    def _apply_target(self, target: set[int]) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(target)

    def _key_target(self, control: np.ndarray) -> set[int]:
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
        return target

    @staticmethod
    def _validate_control(control: np.ndarray) -> None:
        if control.shape != (3,):
            raise ValueError("keyboard control must be [gas, brake, steer]")

    def _release_brake(self) -> None:
        with self._lock:
            self._tap_timer = None
            self._set_pressed(self._pressed - {self._BRAKE})

    def _cancel_tap_unlocked(self) -> None:
        if self._tap_timer is not None:
            self._tap_timer.cancel()
            self._tap_timer = None

    def _set_pressed(self, target: set[int]) -> None:
        for key in sorted(self._pressed - target):
            self._key_event(KeyboardKeyEvent(key, False))
        for key in sorted(target - self._pressed):
            self._key_event(KeyboardKeyEvent(key, True))
        self._pressed = target

    @classmethod
    def _digital_control(cls, action: np.ndarray) -> np.ndarray:
        return cls._quantized_control(action, cls._threshold_brake)

    @classmethod
    def _digital_control_with_brake_tap(cls, action: np.ndarray) -> np.ndarray:
        return cls._quantized_control(action, cls._preserve_brake_tap)

    @classmethod
    def _quantized_control(
        cls, action: np.ndarray, brake_value: Callable[[float], float]
    ) -> np.ndarray:
        control = np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1))
        cls._validate_control(control)
        gas, brake, steer = (float(value) for value in control)
        digital_steer = 0.0
        if abs(steer) >= cls._STEERING_DEADZONE:
            digital_steer = float(np.sign(steer))
        return np.asarray([float(gas > 0.5), brake_value(brake), digital_steer], dtype=np.float32)

    @staticmethod
    def _threshold_brake(brake: float) -> float:
        return float(brake > 0.5)

    @staticmethod
    def _preserve_brake_tap(brake: float) -> float:
        if brake == BRAKE_TAP_SENTINEL:
            return BRAKE_TAP_SENTINEL
        return float(brake > 0.5)

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())
            if self._requires_focus and not _focus_trackmania():
                raise RuntimeError("Trackmania window was not found for keyboard control")
            self._key_event(KeyboardKeyEvent(self._RESTART, True))
            sleep(0.1)
            self._key_event(KeyboardKeyEvent(self._RESTART, False))

    def confirm_finish(self) -> None:
        confirm_trackmania_finish()

    def close(self) -> None:
        with self._lock:
            self._cancel_tap_unlocked()
            self._set_pressed(set())
