"""Canonical namespace for game interfaces (observation/action adapters).

This package is a re-export facade over ``tmrl.custom.interfaces``.  All
interface classes remain in their original locations — nothing is moved — so
existing imports are unaffected.

An *interface* bridges a real-time game environment (via ``rtgym``) and the
RL agent: it defines how raw sensor data are turned into observations, how
actions are sent to the game, and how resets/done conditions work.

Available interfaces
--------------------
- :class:`TrackMania2020InterfaceBase` — shared abstract base class; subclass
  this when writing a new interface from scratch.
- :class:`TM2020Interface` — camera-only baseline that ingests the 33-float
  GrabData telemetry stream.
- :class:`TM2020RLInterface` — unified RL interface combining vehicle state
  and optional image/lidar observations; the recommended default.
- :class:`TM2020InterfaceBoundary` — boundary-lidar interface backed by
  pre-recorded track boundary files.
- :class:`TM2020InterfaceBoundaryImages` — boundary-lidar interface that also
  attaches a camera frame to each observation.

Plugin extension
----------------
Register a custom interface class so that tmrl can discover it by name::

    from tmrl.registry import INTERFACES

    @INTERFACES.register("my_interface")
    class MyInterface(TrackMania2020InterfaceBase):
        ...

For third-party packages, declare an entry point in your ``pyproject.toml``
(or ``setup.cfg``) under the ``"tmrl.interfaces"`` group::

    [project.entry-points."tmrl.interfaces"]
    my_interface = "mypackage.interfaces:MyInterface"

tmrl will call ``importlib.metadata.entry_points(group="tmrl.interfaces")``
at startup and register each discovered class automatically.

Usage
-----
::

    from tmrl.interfaces import TM2020RLInterface, TM2020InterfaceBoundary
"""

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
