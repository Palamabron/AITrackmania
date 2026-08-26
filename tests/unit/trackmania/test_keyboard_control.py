from __future__ import annotations

from threading import RLock

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL
from trackmaniarl.trackmania.control import GamepadController, KeyboardController


def test_keyboard_controller_maps_recorded_steering_sign_to_keys() -> None:
    events: list[tuple[int, bool]] = []
    controller = KeyboardController(lambda key, pressed: events.append((key, pressed)))

    controller.apply(np.asarray([1.0, 0.0, -1.0], dtype=np.float32))
    controller.apply(np.asarray([1.0, 0.0, 1.0], dtype=np.float32))
    controller.close()

    assert events == [
        (0x41, True),
        (0x57, True),
        (0x41, False),
        (0x44, True),
        (0x44, False),
        (0x57, False),
    ]


def test_keyboard_brake_tap_uses_the_same_steering_sign() -> None:
    events: list[tuple[int, bool]] = []
    controller = KeyboardController(lambda key, pressed: events.append((key, pressed)))

    controller.apply_discrete(np.asarray([1.0, -1.0, 1.0], dtype=np.float32))
    controller.close()

    assert (0x44, True) in events
    assert (0x41, True) not in events


def test_keyboard_controller_digitizes_analog_model_actions() -> None:
    events: list[tuple[int, bool]] = []
    controller = KeyboardController(lambda key, pressed: events.append((key, pressed)))

    controller.apply(np.asarray([0.8, 0.2, 0.5], dtype=np.float32))
    controller.close()

    assert events == [(0x44, True), (0x57, True), (0x44, False), (0x57, False)]


def test_keyboard_controller_has_no_rumble_collision_signal() -> None:
    controller = KeyboardController(lambda _key, _pressed: None)

    assert controller.consume_collision() is False


def test_gamepad_brake_tap_releases_synchronously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = object.__new__(GamepadController)
    controller._tap_lock = RLock()
    applied: list[np.ndarray] = []
    delays: list[float] = []
    monkeypatch.setattr(controller, "_apply", lambda action: applied.append(action.copy()))
    monkeypatch.setattr("trackmaniarl.trackmania.control.sleep", delays.append)

    controller.apply_discrete(np.asarray([1.0, BRAKE_TAP_SENTINEL, 0.5], dtype=np.float32))

    assert delays == [BRAKE_TAP_DURATION_S]
    assert len(applied) == 2
    assert applied[0] == pytest.approx([1.0, 1.0, 0.5])
    assert applied[1] == pytest.approx([1.0, 0.0, 0.5])
