"""Canonical re-export facade for tmrl game interfaces."""

from __future__ import annotations

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.interfaces.base import TrackMania2020InterfaceBase

    __all__ += ["TrackMania2020InterfaceBase"]
except Exception:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Concrete interfaces
# ---------------------------------------------------------------------------
try:
    from tmrl.custom.interfaces.vision import TM2020Interface

    __all__ += ["TM2020Interface"]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.interfaces.car_state import TM2020RLInterface

    __all__ += ["TM2020RLInterface"]
except Exception:  # pragma: no cover
    pass

try:
    from tmrl.custom.interfaces.boundary import (
        TM2020InterfaceBoundary,
        TM2020InterfaceBoundaryImages,
    )

    __all__ += [
        "TM2020InterfaceBoundary",
        "TM2020InterfaceBoundaryImages",
    ]
except Exception:  # pragma: no cover
    pass
