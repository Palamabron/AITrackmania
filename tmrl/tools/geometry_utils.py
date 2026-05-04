"""Shared geometry helpers used by record_reward, record_track, and related tools."""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


def line(pt1, pt2, dist):
    """Step along the segment ``pt1 -> pt2`` by ``dist`` metres.

    Returns:
        ``(pt, 0.0)`` when a new point was produced, or ``(None, remaining)``
        when the segment was shorter than ``dist`` and ``remaining`` metres
        still need to be walked on the next segment.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        logger.warning(
            "line(): pt1 and pt2 are identical ({} == {}); returning (None, dist). "
            "This may indicate duplicate or [0,0,0] glitch points in the track data.",
            pt1,
            pt2,
        )
        return None, dist
    if norm < dist:
        return None, dist - norm
    vec_unit = vec / norm
    pt = pt1 + vec_unit * dist
    return pt, 0.0


def smooth_points(points, sigma=12):
    """Apply a per-axis Gaussian filter (``sigma`` samples) to (N, 3) ``points``."""
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    smoothed_z = gaussian_filter1d(points[:, 2], sigma)
    return np.column_stack((smoothed_x, smoothed_y, smoothed_z))


def interp_points_with_cubic_spline(sub_array, data_density=3):
    """Cubic-spline interpolate ``sub_array`` (N, 3), upsampled by ``data_density``."""
    if len(sub_array) < 2:
        raise ValueError(
            f"CubicSpline needs at least 2 points, got {len(sub_array)}. "
            "Drive longer before stopping recording."
        )
    n = len(sub_array)
    original_i = np.arange(0, data_density * n, step=data_density)
    new_i = np.arange(0, data_density * n - 1)
    cs = CubicSpline(original_i, sub_array)
    return cs(new_i)
