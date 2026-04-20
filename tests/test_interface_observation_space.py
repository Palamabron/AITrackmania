"""Observation-space (H, W) shape-order tests for TM2020 vision interfaces.

The interfaces now read window dimensions from their constructor arguments (``self.window_width``
/ ``self.window_height``) rather than ``tmrl.config.WINDOW_*``, so tests pass explicit dims
and assert against those to remain hermetic across ``local.yaml`` states.
"""

from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.interfaces.TM2020InterfaceIMPALA import TM2020InterfaceIMPALA

_W = 320
_H = 180


def test_tm2020_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020Interface(
        img_hist_len=3,
        resize_to=None,
        grayscale=True,
        window_width=_W,
        window_height=_H,
    )
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[3]
    assert img_space.shape == (3, _H, _W)


def test_impala_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020InterfaceIMPALA(
        img_hist_len=2,
        resize_to=None,
        grayscale=True,
        window_width=_W,
        window_height=_H,
    )
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[-1]
    assert img_space.shape == (2, _H, _W)
