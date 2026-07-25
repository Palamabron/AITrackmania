"""Discrete TrackMania action encoding must round-trip through replay safely."""

from __future__ import annotations

import numpy as np
from tmrl.trackmania.actions import (
    BRAKE_TAP_STEERING_STRIDE,
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
    discrete_indices_to_control_batch,
)


def test_composite_discrete_controls_round_trip_to_their_original_indices() -> None:
    action_count, table = build_brake_tap_action_table(n_steer=5)
    original = np.arange(action_count, dtype=np.int64)
    controls = discrete_indices_to_control_batch(original, table)
    restored = continuous_control_to_discrete_indices_batch(controls, table)
    assert np.array_equal(restored, original)


def test_steering_stride_shifts_exactly_one_steering_bin() -> None:
    action_count, table = build_brake_tap_action_table()
    steer_spacing = 2.0 / (13 - 1)
    for index in (0, 30, action_count - BRAKE_TAP_STEERING_STRIDE - 1):
        gas, brake, steer = table[index]
        neighbor_gas, neighbor_brake, neighbor_steer = table[index + BRAKE_TAP_STEERING_STRIDE]
        assert (gas, brake) == (neighbor_gas, neighbor_brake)
        assert abs(float(neighbor_steer - steer) - steer_spacing) < 1e-6
