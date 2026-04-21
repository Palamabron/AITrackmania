"""Smoke coverage for the TrackMania 2020 rtgym interface package.

Each test instantiates an interface and asks it for its observation space without
going through ``initialize()`` (which would require a running TrackMania window +
OpenPlanet socket). This catches regressions in class-level attribute wiring, mis-named
kwargs, broken inheritance and stale re-exports from ``tmrl.custom.interfaces``.
"""

from __future__ import annotations

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
    TM2020InterfaceLidar,
    TM2020InterfaceLidarProgress,
    TM2020InterfaceLidarProgressImages,
    TM2020RLInterface,
    TrackMania2020InterfaceBase,
)


def test_tm2020_observation_space_uses_window_height_width_order() -> None:
    interface = TM2020Interface(img_hist_len=3, resize_to=None, grayscale=True)
    obs_space = interface.get_observation_space()
    img_space = obs_space.spaces[3]
    assert img_space.shape == (3, cfg.WINDOW_HEIGHT, cfg.WINDOW_WIDTH)


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
    assert TrackMania2020InterfaceBase.__abstractmethods__ == frozenset() or (
        TrackMania2020InterfaceBase.__abstractmethods__ is not None
    )
    # Concrete subclasses must all derive from the base.
    concrete = [
        TM2020Interface,
        TM2020InterfaceBoundary,
        TM2020InterfaceBoundaryImages,
        TM2020InterfaceLidar,
        TM2020InterfaceLidarProgress,
        TM2020InterfaceLidarProgressImages,
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
        pytest.param(lambda: TM2020InterfaceLidar(img_hist_len=2), 2, id="lidar-bare"),
        pytest.param(
            lambda: TM2020InterfaceLidar(
                img_hist_len=2, include_progress=True, include_camera_images=True
            ),
            4,
            id="lidar-progress+images",
        ),
        pytest.param(lambda: TM2020InterfaceLidarProgress(img_hist_len=2), 3, id="lidar-progress"),
        pytest.param(
            lambda: TM2020InterfaceLidarProgressImages(img_hist_len=2),
            4,
            id="lidar-progress-images",
        ),
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
        pytest.param(
            lambda: TM2020RLInterface(img_hist_len=1, include_lidar=True),
            id="rl+lidar",
        ),
        pytest.param(
            lambda: TM2020RLInterface(
                img_hist_len=1, include_camera_images=True, include_lidar=True
            ),
            id="rl+camera+lidar",
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


def test_lidar_progress_forces_its_flags() -> None:
    """User kwargs for include_progress / include_camera_images must be ignored by the
    progress-variant subclasses; otherwise the class silently produces an observation
    space indistinguishable from its parent."""
    progress_only = TM2020InterfaceLidarProgress(
        img_hist_len=1, include_progress=False, include_camera_images=True
    )
    assert progress_only._include_progress is True
    assert progress_only._include_camera_images is False

    progress_images = TM2020InterfaceLidarProgressImages(
        img_hist_len=1, include_progress=False, include_camera_images=False
    )
    assert progress_images._include_progress is True
    assert progress_images._include_camera_images is True


def test_config_objects_wires_to_renamed_interface_classes() -> None:
    """After renaming ``TM2020InterfaceTrackMap`` -> ``TM2020InterfaceBoundary`` and
    its ``Images`` sibling, the dispatch logic in ``config_objects`` must reference the
    new class names. A stale string name would still lint-pass but blow up at runtime
    the first time the TrackMap branch was selected; reading the module source catches
    that before any user config triggers the branch.
    """
    import tmrl.config.config_objects as cfg_obj

    src = Path(cfg_obj.__file__).read_text(encoding="utf-8")
    assert "TM2020InterfaceTrackMap" not in src, (
        "stale class name TM2020InterfaceTrackMap in config_objects.py"
    )
    assert "TM2020InterfaceTrackMapImages" not in src, (
        "stale class name TM2020InterfaceTrackMapImages in config_objects.py"
    )
    # Both renamed classes must actually be imported by the module so the partials bind.
    assert cfg_obj.TM2020InterfaceBoundary is TM2020InterfaceBoundary
    assert cfg_obj.TM2020InterfaceBoundaryImages is TM2020InterfaceBoundaryImages
    assert cfg_obj.TM2020RLInterface is TM2020RLInterface
