"""Controller backends for TrackMania; optional drivers are imported lazily."""

from __future__ import annotations

from inspect import Parameter, Signature
from threading import RLock
from time import sleep
from typing import ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL
from trackmaniarl.trackmania.keyboard_control import (
    KeyboardController as KeyboardController,
)
from trackmaniarl.trackmania.keyboard_control import (
    confirm_trackmania_finish,
    restart_trackmania_editor_validation,
    restart_trackmania_race,
)


@runtime_checkable
class Controller(Protocol):
    def apply(self, action: np.ndarray) -> None: ...

    def apply_discrete(self, action: np.ndarray) -> None: ...

    def consume_collision(self) -> bool: ...

    def reset(self) -> None: ...

    def confirm_finish(self) -> None: ...

    def close(self) -> None: ...


class _VibrationCallback:
    __signature__: ClassVar[Signature] = Signature(
        tuple(
            Parameter(name, Parameter.POSITIONAL_OR_KEYWORD)
            for name in (
                "client",
                "target",
                "large_motor",
                "small_motor",
                "led_number",
                "user_data",
            )
        )
    )

    def __init__(self, controller: GamepadController) -> None:
        self.controller = controller

    def __call__(self, *values: object) -> None:
        if len(values) != 6:
            raise TypeError("vgamepad vibration callback requires six values")
        large_motor = values[2]
        if not isinstance(large_motor, int):
            raise TypeError("vgamepad large-motor value must be an integer")
        self.controller._record_vibration(large_motor)


class GamepadController:
    """Virtual XInput controller with an explicit TrackMania restart action."""

    _RESTART_BUTTON = 0x2000  # Xbox B; TrackMania's default Give Up binding.

    def __init__(
        self,
        *,
        restart_input: Literal["gamepad", "keyboard", "editor_validation"] = "gamepad",
    ) -> None:
        try:
            import vgamepad
        except ImportError as exc:
            raise RuntimeError("Install trackmaniarl[trackmania] to use GamepadController") from exc
        self._gamepad = vgamepad.VX360Gamepad()
        self._tap_lock = RLock()
        self._collision_lock = RLock()
        self._collision_detected = False
        self._restart_input = restart_input
        self._vibration_callback = _VibrationCallback(self)
        self._gamepad.register_notification(callback_function=self._vibration_callback)

    def _record_vibration(self, large_motor: int) -> None:
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

    def apply(self, action: np.ndarray) -> None:
        with self._tap_lock:
            self._apply(action)

    def apply_discrete(self, action: np.ndarray) -> None:
        """Apply a table action, releasing the brake after the explicit tap interval."""

        control = np.asarray(action, dtype=np.float32).copy()
        if control.shape != (3,):
            raise ValueError("discrete TrackMania control must be [gas, brake, steer]")
        if float(control[1]) == BRAKE_TAP_SENTINEL:
            with self._tap_lock:
                self._apply(np.asarray([control[0], 1.0, control[2]], dtype=np.float32))
                sleep(BRAKE_TAP_DURATION_S)
                self._apply(np.asarray([control[0], 0.0, control[2]], dtype=np.float32))
            return
        self.apply(control)

    def reset(self) -> None:
        """Release controls and request a TrackMania restart before an episode."""

        with self._tap_lock:
            self._gamepad.reset()
            self._gamepad.update()
            match self._restart_input:
                case "keyboard":
                    restart_trackmania_race()
                case "editor_validation":
                    restart_trackmania_editor_validation()
                case "gamepad":
                    self._restart_with_gamepad()
        self.consume_collision()

    def _restart_with_gamepad(self) -> None:
        self._gamepad.press_button(button=self._RESTART_BUTTON)
        self._gamepad.update()
        sleep(0.1)
        self._gamepad.release_button(button=self._RESTART_BUTTON)
        self._gamepad.update()

    def confirm_finish(self) -> None:
        """Confirm TrackMania's personal-record screen with Enter."""

        confirm_trackmania_finish()

    def close(self) -> None:
        with self._tap_lock:
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

    def consume_collision(self) -> bool:
        return False

    def reset(self) -> None:
        self.actions.clear()

    def confirm_finish(self) -> None:
        return None

    def close(self) -> None:
        return None
