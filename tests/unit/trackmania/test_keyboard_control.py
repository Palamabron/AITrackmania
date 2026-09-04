from __future__ import annotations

from threading import RLock

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import BRAKE_TAP_DURATION_S, BRAKE_TAP_SENTINEL
from trackmaniarl.trackmania.control import GamepadController, KeyboardController


class _ResetGamepad:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def reset(self) -> None:
        self.calls.append("release")

    def update(self) -> None:
        self.calls.append("update")

    def press_button(self, *, button: int) -> None:
        raise AssertionError(f"unexpected gamepad restart button {button}")


def test_keyboard_controller_maps_recorded_steering_sign_to_keys() -> None:
    events: list[tuple[int, bool]] = []
    controller = KeyboardController(lambda event: events.append((event.key, event.pressed)))

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


@pytest.mark.parametrize(
    ("mode", "function"),
    [
        ("keyboard", "restart_trackmania_race"),
        ("editor_validation", "restart_trackmania_editor_validation"),
    ],
)
def test_gamepad_can_use_keyboard_restart_modes(
    monkeypatch: pytest.MonkeyPatch, mode: str, function: str
) -> None:
    calls: list[str] = []
    controller = object.__new__(GamepadController)
    controller._tap_lock = RLock()
    controller._gamepad = _ResetGamepad(calls)
    controller._restart_input = mode
    monkeypatch.setattr(
        f"trackmaniarl.trackmania.control.{function}",
        lambda: calls.append("delete"),
    )
    monkeypatch.setattr(controller, "consume_collision", lambda: False)

    controller.reset()

    assert calls == ["release", "update", "delete"]
