# http://www.flint.jp/misc/?q=dik&lang=en  key indicator

import platform
import time
from typing import Any

if platform.system() == "Windows":
    import ctypes

    from tmrl.custom.tm.utils.control.mouse import (
        mouse_change_name_replay_tm20,
        mouse_close_replay_window_tm20,
        mouse_save_replay_tm20,
    )

    SendInput = ctypes.windll.user32.SendInput  # type: ignore[attr-defined]

    W = 0x11
    A = 0x1E
    S = 0x1F
    D = 0x20
    DEL = 0xD3
    R = 0x13

    # C struct redefinitions

    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        """ctypes mirror of the Win32 KEYBDINPUT structure used with SendInput."""

        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class HardwareInput(ctypes.Structure):
        """ctypes mirror of the Win32 HARDWAREINPUT structure used with SendInput."""

        _fields_ = [
            ("uMsg", ctypes.c_ulong),
            ("wParamL", ctypes.c_short),
            ("wParamH", ctypes.c_ushort),
        ]

    class MouseInput(ctypes.Structure):
        """ctypes mirror of the Win32 MOUSEINPUT structure used with SendInput."""

        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class InputI(ctypes.Union):
        """ctypes mirror of the Win32 INPUT union (keyboard, mouse, or hardware event)."""

        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]  # noqa: RUF012

    class Input(ctypes.Structure):
        """ctypes mirror of the Win32 INPUT structure passed to SendInput."""

        _fields_ = [("type", ctypes.c_ulong), ("ii", InputI)]

    def press_key(hex_key_code):
        """Send a key-down event via the Win32 SendInput API.

        Uses DirectInput scan codes (dwFlags = KEYEVENTF_SCANCODE = 0x0008).

        Args:
            hex_key_code: DirectInput scan code for the target key
                (e.g. 0x11 for W, 0x1E for A, 0xD3 for Delete).
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # type: ignore[attr-defined]

    def release_key(hex_key_code):
        """Send a key-up event via the Win32 SendInput API.

        Combines KEYEVENTF_SCANCODE (0x0008) with KEYEVENTF_KEYUP (0x0002) in
        dwFlags.

        Args:
            hex_key_code: DirectInput scan code for the target key
                (e.g. 0x11 for W, 0x1E for A, 0xD3 for Delete).
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        ii_.ki = KeyBdInput(0, hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))  # type: ignore[attr-defined]

    def apply_control(action, window_id=None):
        """Map an action string to WASD key press/release events.

        Each letter present in the action string activates its corresponding
        key; absent letters trigger a release.  Mapping: "f" → W (forward),
        "b" → S (backward), "l" → A (left), "r" → D (right).

        Args:
            action: String containing zero or more of "f", "b", "l", "r".
            window_id: Unused on Windows; accepted for API compatibility with
                the Linux implementation.
        """
        if "f" in action:
            press_key(W)
        else:
            release_key(W)
        if "b" in action:
            press_key(S)
        else:
            release_key(S)
        if "l" in action:
            press_key(A)
        else:
            release_key(A)
        if "r" in action:
            press_key(D)
        else:
            release_key(D)

    def keyres():
        """Tap the Delete key (scan code 0xD3) to trigger an in-game respawn."""
        press_key(DEL)
        release_key(DEL)

    def is_del_pressed() -> bool:
        """Non-blocking check: True if Del key is currently pressed."""
        return bool(ctypes.windll.user32.GetAsyncKeyState(0xD3) & 0x8000)  # type: ignore[attr-defined]

    def keysavereplay():  # TODO: debug - verify replay save flow works across TM2020 versions
        """Save the current replay via keyboard and mouse actions.

        Presses R to open the replay menu, waits 1 s for the UI to appear,
        then uses mouse helpers to click into the name field, types the current
        nanosecond timestamp as the replay name, and confirms the save.  Each
        UI step includes a 1 s delay to allow animations to complete.
        """
        import keyboard

        press_key(R)
        time.sleep(0.1)
        release_key(R)
        time.sleep(1.0)
        mouse_change_name_replay_tm20()
        time.sleep(1.0)
        keyboard.write(str(time.time_ns()))
        time.sleep(1.0)
        mouse_save_replay_tm20()
        time.sleep(1.0)
        mouse_close_replay_window_tm20()
        time.sleep(1.0)

elif platform.system() == "Linux":
    import subprocess

    from loguru import logger

    KEY_UP = "Up"
    KEY_DOWN = "Down"
    KEY_RIGHT = "Right"
    KEY_LEFT = "Left"
    KEY_BACKSPACE = "BackSpace"

    process = None

    def execute_command(c):
        """Write a shell command to the persistent bash subprocess.

        Reuses the global subprocess across calls to avoid per-key process
        spawn overhead.  If the process has exited it is transparently recreated
        before the command is sent.

        Args:
            c: Shell command string to execute (must be newline-terminated so
                bash processes it immediately).
        """
        global process
        if process is None or process.poll() is not None:
            logger.debug("(re-)create process")
            process = subprocess.Popen("/bin/bash", stdin=subprocess.PIPE)
        assert process.stdin is not None
        process.stdin.write(c.encode())
        process.stdin.flush()

    def press_key(key):
        """Send a key-down event via xdotool.

        Args:
            key: xdotool key name (e.g. "Up", "Left", "BackSpace").
        """
        c = f"xdotool keydown {key!s}\n"
        execute_command(c)

    def release_key(key):
        """Send a key-up event via xdotool.

        Args:
            key: xdotool key name (e.g. "Up", "Left", "BackSpace").
        """
        c = f"xdotool keyup {key!s}\n"
        execute_command(c)

    def apply_control(action, window_id=None):
        """Map an action string to arrow-key press/release events via xdotool.

        Optionally focuses the target window before sending keys.  Mapping:
        "f" → Up, "b" → Down, "l" → Left, "r" → Right.

        Args:
            action: String containing zero or more of "f", "b", "l", "r".
            window_id: xdotool window ID to focus before sending keys, or None
                to send to the currently focused window.
        """
        if window_id is not None:
            c_focus = f"xdotool windowfocus {window_id!s}"
            execute_command(c_focus)

        if "f" in action:
            press_key(KEY_UP)
        else:
            release_key(KEY_UP)
        if "b" in action:
            press_key(KEY_DOWN)
        else:
            release_key(KEY_DOWN)
        if "l" in action:
            press_key(KEY_LEFT)
        else:
            release_key(KEY_LEFT)
        if "r" in action:
            press_key(KEY_RIGHT)
        else:
            release_key(KEY_RIGHT)

    def keyres():
        """Tap the BackSpace key to trigger an in-game respawn on Linux."""
        press_key(KEY_BACKSPACE)
        release_key(KEY_BACKSPACE)

    def is_del_pressed() -> bool:
        """Non-blocking check: True if Del key is currently pressed (Linux: not implemented)."""
        return False

else:

    def apply_control(action: Any, window_id: Any = None):
        pass

    def keyres():
        pass

    def is_del_pressed() -> bool:
        return False
