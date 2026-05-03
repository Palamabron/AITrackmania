"""Smoke coverage for the TrackMania 2020 rtgym interface package.

Each test instantiates an interface and asks it for its observation space without
going through ``initialize()`` (which would require a running TrackMania window +
OpenPlanet socket). This catches regressions in class-level attribute wiring, mis-named
kwargs, broken inheritance and stale re-exports from ``tmrl.custom.interfaces``.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tmrl.config as cfg
from gymnasium import spaces
from tmrl.custom import interfaces as interfaces_pkg
from tmrl.custom.interfaces import (
    TM2020Interface,
    TM2020InterfaceBoundary,
    TM2020InterfaceBoundaryImages,
    TM2020RLInterface,
    TrackMania2020InterfaceBase,
)

_W = 320
_H = 180


def test_tm2020_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020Interface(
        img_hist_len=3,
        resize_to=(_W, _H),
        grayscale=True,
    )
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[3]
    assert img_space.shape == (3, _H, _W)


def test_rl_interface_observation_space_image_tail_shape() -> None:
    interface = TM2020RLInterface(
        img_hist_len=2, resize_to=None, grayscale=True, include_camera_images=True
    )
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[-1]
    assert img_space.shape == (interface.img_hist_len, cfg.IMG_HEIGHT, cfg.IMG_WIDTH)


# -----------------------------------------------------------------------------
# Public API surface: every re-exported symbol must import without error and the
# abstract base must actually be abstract.
# -----------------------------------------------------------------------------


def test_package_all_re_exports_are_resolvable() -> None:
    exported = set(interfaces_pkg.__all__)
    assert exported, "tmrl.custom.interfaces.__all__ must not be empty"
    for name in exported:
        assert hasattr(interfaces_pkg, name), f"{name} listed in __all__ but missing from package"


def test_base_class_is_abstract() -> None:
    assert inspect.isabstract(TrackMania2020InterfaceBase), (
        "TrackMania2020InterfaceBase must be abstract (has @abstractmethod hooks)"
    )
    assert TrackMania2020InterfaceBase.__abstractmethods__, (
        "TrackMania2020InterfaceBase must declare at least one @abstractmethod"
    )
    with pytest.raises(TypeError):
        TrackMania2020InterfaceBase()  # type: ignore[abstract]
    concrete = [
        TM2020Interface,
        TM2020InterfaceBoundary,
        TM2020InterfaceBoundaryImages,
        TM2020RLInterface,
    ]
    for cls in concrete:
        assert issubclass(cls, TrackMania2020InterfaceBase), f"{cls.__name__} must derive from base"


# -----------------------------------------------------------------------------
# get_observation_space smoke tests for every concrete interface.
# -----------------------------------------------------------------------------


def _assert_tuple_of_boxes(obs_space: Any, *, min_arity: int = 1) -> None:
    assert isinstance(obs_space, spaces.Tuple), f"expected Tuple, got {type(obs_space).__name__}"
    assert len(obs_space.spaces) >= min_arity, "observation space has too few components"
    for i, box in enumerate(obs_space.spaces):
        assert isinstance(box, spaces.Box), (
            f"component #{i} is {type(box).__name__}, expected spaces.Box"
        )


_SMALL_IMG_KW: dict[str, Any] = {"resize_to": (16, 16), "grayscale": True}


def _boundary_factory_requires_csv() -> bool:
    return os.path.exists(cfg.BOUNDARY_CSV_LEFT) and os.path.exists(cfg.BOUNDARY_CSV_RIGHT)


@pytest.mark.parametrize(
    ("factory", "expected_min_arity"),
    [
        pytest.param(lambda: TM2020Interface(img_hist_len=2, **_SMALL_IMG_KW), 4, id="vision"),
        pytest.param(
            lambda: TM2020InterfaceBoundaryImages(img_hist_len=2, **_SMALL_IMG_KW),
            4,
            id="boundary-images",
        ),
    ],
)
def test_interface_observation_space_is_tuple_of_boxes(
    factory: Callable[[], Any], expected_min_arity: int
) -> None:
    interface = factory()
    obs_space = interface.get_observation_space()
    _assert_tuple_of_boxes(obs_space, min_arity=expected_min_arity)


@pytest.mark.skipif(
    not _boundary_factory_requires_csv(),
    reason="boundary CSV fixtures missing from output_files/tracks/tmrl-test/",
)
def test_boundary_observation_space_is_tuple_of_boxes() -> None:
    interface = TM2020InterfaceBoundary(img_hist_len=1)
    obs_space = interface.get_observation_space()
    _assert_tuple_of_boxes(obs_space, min_arity=9)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: TM2020RLInterface(img_hist_len=1), id="rl-telemetry"),
        pytest.param(
            lambda: TM2020RLInterface(img_hist_len=1, include_camera_images=True),
            id="rl+camera",
        ),
    ],
)
def test_car_state_family_observation_space(factory: Callable[[], Any]) -> None:
    interface = factory()
    obs_space = interface.get_observation_space()
    _assert_tuple_of_boxes(obs_space, min_arity=1)


# -----------------------------------------------------------------------------
# Regressions for behavior changes introduced by this branch.
# -----------------------------------------------------------------------------


def test_boundary_observed_boundaries_gated_on_record() -> None:
    """Without ``record=True`` the debug trace list must be None so long training
    runs don't silently leak memory as boundary slices accumulate each step."""
    if not _boundary_factory_requires_csv():
        pytest.skip("boundary CSV fixtures missing")
    prod = TM2020InterfaceBoundary(img_hist_len=1, record=False)
    debug = TM2020InterfaceBoundary(img_hist_len=1, record=True)
    assert prod._observed_boundaries is None
    assert debug._observed_boundaries == [[], [], [], [], []]


def test_vision_iqn_discrete_action_space() -> None:
    """IQN default: 13 steer bins x 2 gas x 3 brake = 78 discrete actions."""
    iface = TM2020Interface(img_hist_len=1, discrete_n_steer_bins=13)
    act_space = iface.get_action_space()
    assert isinstance(act_space, spaces.Discrete)
    assert act_space.n == 78


def test_boundary_forwards_discrete_n_steer_bins_to_base() -> None:
    """Registry wiring passes ``discrete_n_steer_bins`` in kwargs; boundary must forward."""
    iface = TM2020InterfaceBoundary(img_hist_len=1, discrete_n_steer_bins=13)
    assert iface.discrete_action_table is not None
    assert len(iface.discrete_action_table) == 78


def test_all_interface_registry_keys_exist() -> None:
    """Every key returned by ``_determine_interface_name`` must be registered in INTERFACES."""
    from tmrl.registry import INTERFACES

    expected_keys = {
        "vision",
        "lidar",
        "lidar_images",
        "tqc",
        "sophy",
        "impala",
    }
    for key in expected_keys:
        assert key in INTERFACES, f"interface key {key!r} missing from INTERFACES registry"


def test_iqn_n_actions_matches_steer_bins() -> None:
    """Pydantic schema must reject mismatched iqn_n_actions / iqn_n_steer_bins."""
    from tmrl.config.schema.algorithm import AlgorithmConfig

    ok = AlgorithmConfig(name="IQN", iqn_n_steer_bins=13, iqn_n_actions=78)
    assert ok.iqn_n_actions == 78

    with pytest.raises(ValueError, match="iqn_n_actions"):
        AlgorithmConfig(name="IQN", iqn_n_steer_bins=13, iqn_n_actions=100)


def test_config_objects_has_no_stale_interface_names() -> None:
    """After renaming ``TM2020InterfaceTrackMap`` -> ``TM2020InterfaceBoundary`` and
    its ``Images`` sibling, the dispatch logic in ``config_objects`` must not reference
    the old class names."""
    import tmrl.config.config_objects as cfg_obj

    src = Path(cfg_obj.__file__).read_text(encoding="utf-8")
    assert "TM2020InterfaceTrackMap" not in src, (
        "stale class name TM2020InterfaceTrackMap in config_objects.py"
    )
    assert "TM2020InterfaceTrackMapImages" not in src, (
        "stale class name TM2020InterfaceTrackMapImages in config_objects.py"
    )
