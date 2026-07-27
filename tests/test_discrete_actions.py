"""Discrete TrackMania action encoding must round-trip through replay safely."""

from __future__ import annotations

import numpy as np

from tmrl.trackmania.actions import (
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
