"""Public API for TrackMania 2020 rtgym interfaces.

Import interface classes from this package instead of from individual sub-modules, e.g.::

    from tmrl.custom.interfaces import TM2020Interface, TM2020RLInterface

The concrete implementations live in the following modules:

- :mod:`tmrl.custom.interfaces.base`      - shared abstract base class
- :mod:`tmrl.custom.interfaces.vision`    - camera-only baseline (33-float GrabData)
- :mod:`tmrl.custom.interfaces.car_state` - unified RL interface (:class:`TM2020RLInterface`)
- :mod:`tmrl.custom.interfaces.boundary`  - pre-recorded track-boundary interfaces
"""

from tmrl.custom.interfaces.base import TrackMania2020InterfaceBase
from tmrl.custom.interfaces.boundary import (
    TM2020InterfaceBoundary,
    TM2020InterfaceBoundaryImages,
)
from tmrl.custom.interfaces.car_state import TM2020RLInterface
from tmrl.custom.interfaces.vision import TM2020Interface

__all__ = [
    "TM2020Interface",
    "TM2020InterfaceBoundary",
    "TM2020InterfaceBoundaryImages",
    "TM2020RLInterface",
    "TrackMania2020InterfaceBase",
]
