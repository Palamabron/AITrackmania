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
    """Return [x, y, z] from a telemetry payload."""
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        px = int(TmrlDataPlugin.POS_X)
        return [data[px], data[px + 1], data[px + 2]]
    if len(data) >= 20:
        return [data[3], data[4], data[5]]
    return [data[2], data[3], data[4]]


def _is_lap_finished(data: tuple[float, ...]) -> bool:
    """Return True when the lap-finish flag is set in the telemetry payload."""
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        return bool(data[int(TmrlDataPlugin.FINISH_UI_ACTIVE)])
    finish_idx = _FINISH_IDX_TQC if len(data) >= 20 else _FINISH_IDX_LEGACY
    return bool(data[finish_idx])
