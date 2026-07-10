"""Shared helpers for parsing TM2020 telemetry payloads.

Supports three payload layouts:
- 19-float legacy
- 20-float TQC
- 33-float TMRL_GrabData
"""

from tmrl.custom.interfaces.telemetry_indices import (
    TMRL_GRABDATA_FLOAT_COUNT,
    TmrlDataPlugin,
)

_FINISH_IDX_TQC = 9
_FINISH_IDX_LEGACY = 8


def _position_xyz(data: tuple[float, ...]) -> list[float]:
    """Extract the (x, y, z) position from a telemetry payload.

    Handles all three supported payload layouts by length:

    - ``len >= TMRL_GRABDATA_FLOAT_COUNT`` (33-float): reads from
      ``TmrlDataPlugin.POS_X`` and the two following indices.
    - ``len >= 20`` (20-float TQC layout): positions at indices 3, 4, 5.
    - ``len < 20`` (19-float legacy layout): positions at indices 2, 3, 4.

    Args:
        data: Raw telemetry payload tuple from the game plugin.

    Returns:
        Three-element list ``[x, y, z]`` in TM2020 world coordinates (metres).
    """
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        px = int(TmrlDataPlugin.POS_X)
        return [data[px], data[px + 1], data[px + 2]]
    if len(data) >= 20:
        return [data[3], data[4], data[5]]
    return [data[2], data[3], data[4]]


def _is_lap_finished(data: tuple[float, ...]) -> bool:
    """Return True when the lap-finish flag is set in the telemetry payload.

    The finish flag index differs across payload layouts — TQC uses index 9,
    the legacy 19-float layout uses index 8, and the 33-float TMRL_GrabData
    layout exposes the flag via ``TmrlDataPlugin.FINISH_UI_ACTIVE``.

    Args:
        data: Raw telemetry payload tuple from the game plugin.

    Returns:
        True if the finish indicator is active for this frame.
    """
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        return bool(data[int(TmrlDataPlugin.FINISH_UI_ACTIVE)])
    finish_idx = _FINISH_IDX_TQC if len(data) >= 20 else _FINISH_IDX_LEGACY
    return bool(data[finish_idx])
