"""Platform-specific window management for screen capture and resize.

Provides ``WindowInterface`` on both Windows (via win32gui/win32ui BitBlt) and
Linux (via mss + xdotool). The class is defined conditionally on
``platform.system()``. On platforms other than Windows or Linux no
``WindowInterface`` is defined.

Note:
    Callers that import ``WindowInterface`` by name will receive an
    ``ImportError`` on non-Windows platforms if the Linux branch also fails
    (e.g. xdotool / mss not installed). This is a known limitation tracked
    separately from this module.
"""

import platform

import numpy as np
from loguru import logger

if platform.system() == "Windows":
    import win32con
    import win32gui
    import win32ui

    class WindowInterface:
        """Win32-backed window interface for screenshot capture and resize."""

        def __init__(self, window_name):
            """Locate the named window and compute window-border offsets.

            Polls until the window has a non-zero client area (handles the case where
            the window is minimized at startup). Stores ``w_diff`` and ``h_diff`` (px)
            — typically 16 and 39 px on Windows 10 — needed to convert client-area
            dimensions to full-frame dimensions.

            Args:
                window_name: Exact title of the target window as shown in the taskbar.

            Raises:
                AssertionError: If no window with the given name is found.
            """
            self.window_name = window_name

            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            while True:  # in case the window is reduced
                wr = win32gui.GetWindowRect(hwnd)
                cr = win32gui.GetClientRect(hwnd)
                if cr[2] > 0 and cr[3] > 0:
                    break

            self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]  # (16 on W10)
            self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]  # (39 on W10)

            self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)

            self.x_origin_offset = -self.w_diff // 2
            self.y_origin_offset = 0

        def screenshot(self):
            """Capture the client area of the window as a NumPy array.

            Uses Win32 GDI BitBlt for low-overhead screen capture. Polls until the
            window has a positive client size to avoid zero-dimension arrays.

            Returns:
                uint8 array of shape ``(height, width, 4)`` in BGRA channel order.

            Raises:
                AssertionError: If the window cannot be found.
            """
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."

            while True:  # avoids crashes when the window is reduced
                x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
                w = x1 - x - self.w_diff
                h = y1 - y - self.h_diff
                if w > 0 and h > 0:
                    break
            hdc = win32gui.GetWindowDC(hwnd)
            dc = win32ui.CreateDCFromHandle(hdc)
            memdc = dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(dc, w, h)
            oldbmp = memdc.SelectObject(bitmap)
            memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)
            bits = bitmap.GetBitmapBits(True)
            img = np.frombuffer(bits, dtype="uint8")
            img.shape = (h, w, 4)
            memdc.SelectObject(oldbmp)  # avoids memory leak
            win32gui.DeleteObject(bitmap.GetHandle())
            memdc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hdc)
            return img

        def move_and_resize(self, x=1, y=0, w=None, h=None):
            """Move and resize the window to the requested client-area dimensions.

            Adjusts the requested client-area size by the stored border offsets so the
            actual client area matches ``(w, h)`` exactly.

            Args:
                x: Target left edge of the client area in screen coordinates (px).
                y: Target top edge of the client area in screen coordinates (px).
                w: Target client-area width in px. Falls back to ``WINDOW_WIDTH`` config.
                h: Target client-area height in px. Falls back to ``WINDOW_HEIGHT`` config.

            Raises:
                AssertionError: If the window cannot be found.
            """
            from tmrl.config.constants import WINDOW_HEIGHT, WINDOW_WIDTH

            if w is None:
                w = WINDOW_WIDTH
            if h is None:
                h = WINDOW_HEIGHT
            x += self.x_origin_offset
            y += self.y_origin_offset
            w += self.w_diff
            h += self.h_diff
            hwnd = win32gui.FindWindow(None, self.window_name)
            assert hwnd != 0, f"Could not find a window named {self.window_name}."
            win32gui.MoveWindow(hwnd, x, y, w, h, True)


elif platform.system() == "Linux":
    import subprocess
    import time

    import mss

    def get_window_id(name):
        """Return the xdotool window ID for the first visible window with the given name.

        Args:
            name: Exact window title to search for.

        Returns:
            xdotool window ID string.

        Raises:
            NoSuchWindowError: If no matching visible window is found or xdotool fails.
        """
        try:
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", "."],
                capture_output=True,
                text=True,
                check=True,
            )
            window_ids = result.stdout.strip().split("\n")
            for window_id in window_ids:
                result = subprocess.run(
                    ["xdotool", "getwindowname", window_id],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout.strip() == name:
                    logger.debug(f"detected window {name}, id={window_id}")
                    return window_id

            logger.error(f"failed to find window '{name}'")
            raise NoSuchWindowError(name)

        except subprocess.CalledProcessError as err:
            logger.error(f"process error searching for window '{name}")
            raise NoSuchWindowError(name) from err

    def get_window_geometry(name):
        """
        FIXME: xdotool doesn't agree with MSS, so we use hardcoded offsets instead for now
        """
        try:
            result = subprocess.run(
                ["xdotool", "search", "--name", name, "getwindowgeometry", "--shell"],
                capture_output=True,
                text=True,
                check=True,
            )
            elements = result.stdout.strip().split("\n")
            res_id = None
            res_x = None
            res_y = None
            res_w = None
            res_h = None
            for elt in elements:
                low_elt = elt.lower()
                if low_elt.startswith("window="):
                    res_id = elt[7:]
                elif low_elt.startswith("x="):
                    res_x = int(elt[2:])
                elif low_elt.startswith("y="):
                    res_y = int(elt[2:])
                elif low_elt.startswith("width="):
                    res_w = int(elt[6:])
                elif low_elt.startswith("height="):
                    res_h = int(elt[7:])

            if None in (res_id, res_x, res_y, res_w, res_h):
                geom = (res_id, res_x, res_y, res_w, res_h)
                raise GeometrySearchError(f"Found None in window '{name}' geometry: {geom}")

            return res_id, res_x, res_y, res_w, res_h

        except subprocess.CalledProcessError as e:
            logger.error(f"process error searching for {name} window geometry")
            raise e

    class NoSuchWindowError(Exception):
        """Raised when a named window cannot be found via xdotool."""

        pass

    class GeometrySearchError(Exception):
        """Raised when xdotool cannot retrieve a window's geometry."""

        pass

    class WindowInterface:  # type: ignore[no-redef]
        """Linux window interface for screenshot capture and resize via mss + xdotool."""

        def __init__(self, window_name, linux_x_offset: int = 0, linux_y_offset: int = 0):
            """Initialize the Linux window interface.

            Args:
                window_name: Exact title of the target window.
                linux_x_offset: Horizontal pixel offset applied to every screenshot crop.
                    Compensates for coordinate disagreements between xdotool and mss.
                linux_y_offset: Vertical pixel offset applied to every screenshot crop.
            """
            self.sct = mss.mss()

            self.window_name = window_name
            try:
                self.window_id = get_window_id(window_name)
            except NoSuchWindowError as e:
                logger.error(f"get_window_id failed, is xdotool correctly installed? {e!s}")
                self.window_id = None

            self.w = None
            self.h = None
            self.x = None
            self.y = None
            self.x_offset = linux_x_offset
            self.y_offset = linux_y_offset

            self.process: subprocess.Popen[bytes] | None = None

        def __del__(self):
            """Release the mss screen-capture context."""
            pass
            self.sct.close()

        def execute_command(self, c):
            """Write a shell command to the persistent bash subprocess stdin.

            Lazily spawns a bash process if none is running. Commands are fire-and-forget;
            no stdout/stderr is consumed.

            Args:
                c: Shell command string (must include a trailing newline).
            """
            if self.process is None or self.process.poll() is not None:
                self.process = subprocess.Popen(
                    "/bin/bash",
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            assert self.process.stdin is not None
            self.process.stdin.write(c.encode())
            self.process.stdin.flush()

        def screenshot(self):
            """Capture the client area using mss at the stored window geometry.

            Returns:
                uint8 NumPy array of shape ``(height, width, 4)`` in BGRA channel order.

            Raises:
                AssertionError: If window geometry has not been set via
                    :meth:`move_and_resize`.
            """
            try:
                x, y, w, h = self.x, self.y, self.w, self.h
                assert x is not None
                assert y is not None
                assert w is not None
                assert h is not None
                monitor: dict[str, int] = {
                    "top": int(y + self.y_offset),
                    "left": int(x + self.x_offset),
                    "width": int(w),
                    "height": int(h),
                }
                img = np.array(self.sct.grab(monitor))
                return img

            except subprocess.CalledProcessError as e:
                logger.error("failed to capture screenshot")
                raise e

        def move_and_resize(self, x=0, y=0, w=None, h=None):
            """Reposition and resize the window via xdotool and record the new geometry.

            Sleeps 1 s after issuing the commands to give the window manager time to
            apply the change before the next screenshot reads geometry. Uses a sleep
            rather than ``xdotool --sync`` because ``--sync`` does not reliably return.

            Args:
                x: Target X coordinate in screen coordinates (px).
                y: Target Y coordinate in screen coordinates (px).
                w: Target width in px. Falls back to ``WINDOW_WIDTH`` config.
                h: Target height in px. Falls back to ``WINDOW_HEIGHT`` config.
            """
            from tmrl.config.constants import WINDOW_HEIGHT, WINDOW_WIDTH

            if w is None:
                w = WINDOW_WIDTH
            if h is None:
                h = WINDOW_HEIGHT
            logger.debug(f"prepare {self.window_name} to {w}x{h} @ {x}, {y}")

            try:
                c_focus = f"xdotool windowfocus {self.window_id}\n"
                self.execute_command(c_focus)

                logger.debug(f"move window {self.window_name!s}")
                c_move = f"xdotool windowmove {self.window_id!s} {x!s} {y!s}\n"
                self.execute_command(c_move)

                logger.debug(f"resize window {self.window_name!s}")
                c_resize = f"xdotool windowsize {self.window_id!s} {w!s} {h!s}\n"
                self.execute_command(c_resize)

                self.w = w
                self.h = h
                self.x = x
                self.y = y

                # instead of using xdotool --sync, which doesn't return
                logger.debug("success, let me nap 1s to make sure everything computed")
                time.sleep(1)

                # # retrieve actual position of the window and set offsets
                # geo_id, geo_x, geo_y, geo_w, geo_h = get_window_geometry(self.window_name)
                #
                # if geo_id != self.window_id:
                #     raise GeometrySearchException(f"wrong geo_id: {geo_id} != {self.window_id}")
                # if geo_w != self.w:
                #     raise GeometrySearchException(f"wrong geo_w: {geo_w} != {self.w}")
                # if geo_h != self.h:
                #     raise GeometrySearchException(f"wrong geo_h: {geo_h} != {self.h}")
                #
                # self.x_offset = geo_x - self.x
                # self.y_offset = geo_y - self.y

            except subprocess.CalledProcessError:
                logger.error(f"failed to resize window_id '{self.window_id}'")

            except NoSuchWindowError as e:
                logger.error(f"failed to find window: {e!s}")

            # except GeometrySearchException as e:
            #     logger.error(f"failed to retrieve window geometry: {str(e)}")


def profile_screenshot():
    """Profile screenshot throughput for the "Trackmania" window using pyinstrument.

    Runs 5000 consecutive screenshot calls and prints a flamegraph to stdout.
    Used for local performance benchmarking; not called during normal training.
    """
    from pyinstrument import Profiler

    pro = Profiler()
    window_interface = WindowInterface("Trackmania")
    pro.start()
    for _ in range(5000):
        _ = window_interface.screenshot()
    pro.stop()
    pro.print(show_all=True)


if __name__ == "__main__":
    profile_screenshot()
