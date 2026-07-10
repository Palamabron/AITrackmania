"""Track feature extraction helpers for observation construction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def discrete_curvature_xz(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, *, signed: bool = True
) -> float:
    """3-point discrete curvature in the XZ plane.

    Returns signed curvature (positive = leftward) when *signed* is True,
    otherwise the absolute value.  Returns 0.0 when any segment is degenerate.
    """
    v1 = (p1 - p0)[[0, 2]]
    v2 = (p2 - p1)[[0, 2]]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    v1, v2 = v1 / n1, v2 / n2
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    angle = math.acos(dot)
    arc = 0.5 * (n1 + n2)
    if arc < 1e-9:
        return 0.0
    kappa = angle / arc
    if signed:
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        kappa = kappa if cross >= 0 else -kappa
    return float(kappa)


class TrackFeatureProvider:
    """Provides track boundary/lookahead features from RewardFunction state."""

    def __init__(self, reward_ctx: Any) -> None:
        """Args:
        reward_ctx: A :class:`RewardFunction` instance that exposes trajectory data
            (``data``, ``left_track``, ``right_track``, ``cur_idx``,
            ``_cumulative_dist``, etc.). The provider holds a live reference — it
            does not copy — so any updates to ``reward_ctx`` are immediately visible.
        """
        self._ctx = reward_ctx

    def get_n_next_checkpoints_xy(
        self, position: list[float] | np.ndarray, number_of_next_points: int
    ) -> list[float]:
        """Return the next N look-ahead checkpoint XZ offsets, scaled by 10.

        Coordinates are expressed in the XZ plane (TrackMania's horizontal plane; Y is
        vertical). Each offset is multiplied by 10.0 so the values fall in the
        ~[-300, 300] range for typical TM2020 track geometries, keeping them
        commensurate with other observation channels.

        Args:
            position: Current car position as [x, y, z] in meters (world-frame).
            number_of_next_points: Number of look-ahead checkpoints.

        Returns:
            Flat list of ``2 * number_of_next_points`` floats arranged as
            [dx0, dz0, dx1, dz1, …], where each pair is the offset from
            ``position`` to the checkpoint in game units * 10.
        """
        max_idx = len(self._ctx.data) - 1
        next_indices = [
            min(self._ctx.cur_idx + step * self._ctx._checkpoint_stride, max_idx)
            for step in range(1, number_of_next_points + 1)
        ]
        route_to_next_poses = []
        for pos_index in next_indices:
            for axis in (0, -1):
                route_to_next_poses.append(
                    (self._ctx.data[pos_index][axis] - position[axis]) * 10.0
                )
        return route_to_next_poses

    def get_track_info(
        self, position: list[float] | np.ndarray, points_number: int
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Compute track boundary and look-ahead geometry relative to current position.

        Selects ``points_number`` look-ahead indices either by evenly-spaced arc-length
        steps (when ``_ctx._point_spacing_m > 0``) or by fixed index strides.  For each
        index the method returns left-boundary, center-line, and right-boundary XZ
        offsets from ``position`` (in meters), a log-compressed arc-length distance, and
        an optional discrete curvature value.

        Args:
            position: Current car position as [x, y, z] in meters (world-frame).
            points_number: Number of look-ahead track points to compute.

        Returns:
            Five-tuple of lists:
                left_positions: Flat list of 2 * points_number floats (x, z offsets in
                    meters from ``position`` to the left boundary at each look-ahead point).
                center_positions: Same layout for the track center-line.
                right_positions: Same layout for the right boundary.
                curvatures: List of points_number floats; signed discrete XZ curvature
                    at each point (positive = left turn, rad/m). All zeros when
                    ``_ctx._track_curvature_obs`` is False.
                log_distances: List of points_number floats; ``log1p`` of the arc-length
                    distance from the current trajectory index to each look-ahead point
                    (in meters), providing a compressed distance encoding.
        """
        max_idx = (
            min(len(self._ctx.data), len(self._ctx.left_track), len(self._ctx.right_track)) - 1
        )
        if (getattr(self._ctx, "_point_spacing_m", 0) or 0) > 0 and (
            getattr(self._ctx, "_points_number", 0) or 0
        ) > 0:
            next_indices = []
            cur_idx = self._ctx.cur_idx
            cur_dist = self._ctx._cumulative_dist[min(cur_idx, self._ctx.datalen - 1)]
            n_pts = self._ctx._points_number or 0
            for _ in range(n_pts):
                target_dist = cur_dist + self._ctx._point_spacing_m
                while (
                    cur_idx < self._ctx.datalen - 1
                    and self._ctx._cumulative_dist[cur_idx + 1] <= target_dist
                ):
                    cur_idx += 1
                if cur_idx < self._ctx.datalen - 1:
                    cur_idx += 1
                    cur_dist = self._ctx._cumulative_dist[cur_idx]
                else:
                    cur_dist = self._ctx._cumulative_dist[-1] + self._ctx._point_spacing_m
                next_indices.append(min(cur_idx, max_idx))
        else:
            next_indices = [
                self._ctx.cur_idx + step * self._ctx._checkpoint_stride + 1
                for step in range(points_number)
            ]
            for idx in range(len(next_indices)):
                if next_indices[idx] > max_idx:
                    next_indices[idx] = max_idx

        left_positions, center_positions, right_positions = [], [], []
        log_distances: list[float] = []
        base_dist = float(self._ctx._cumulative_dist[min(self._ctx.cur_idx, self._ctx.datalen - 1)])
        for pos_index in next_indices:
            point_dist = float(self._ctx._cumulative_dist[min(pos_index, self._ctx.datalen - 1)])
            delta_dist = max(0.0, point_dist - base_dist)
            log_distances.append(float(math.log1p(delta_dist)))
            for axis in (0, -1):
                left_val = self._ctx.left_track[pos_index][axis]
                right_val = self._ctx.right_track[pos_index][axis]
                center_val = (left_val + right_val) / 2.0
                left_positions.append(left_val - position[axis])
                center_positions.append(center_val - position[axis])
                right_positions.append(right_val - position[axis])

        curvatures = [0.0] * len(next_indices)
        if self._ctx._track_curvature_obs and len(next_indices) > 0:
            for idx, pos_index in enumerate(next_indices):
                i0 = max(0, pos_index - 1)
                i2 = min(self._ctx.datalen - 1, pos_index + 1)
                if i0 >= i2 or self._ctx.datalen < 2:
                    continue
                curvatures[idx] = discrete_curvature_xz(
                    self._ctx.data[i0],
                    self._ctx.data[pos_index],
                    self._ctx.data[i2],
                    signed=True,
                )
        return left_positions, center_positions, right_positions, curvatures, log_distances
