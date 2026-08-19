"""Discrete TrackMania action encoding must round-trip through replay safely."""

from __future__ import annotations

import numpy as np

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
    discrete_indices_to_control_batch,
    select_brake_tap_actions,
    select_brake_tap_exploration_weights,
)


def test_composite_discrete_controls_round_trip_to_their_original_indices() -> None:
    action_count, table = build_brake_tap_action_table(n_steer=5)
    original = np.arange(action_count, dtype=np.int64)
    controls = discrete_indices_to_control_batch(original, table)
    restored = continuous_control_to_discrete_indices_batch(controls, table)
    assert np.array_equal(restored, original)


def test_compact_brake_tap_actions_preserve_their_canonical_controls() -> None:
    canonical_count, canonical = build_brake_tap_action_table()
    action_ids = (0, 1, 3, 39, 72, 73, 75)

    compact_count, compact = select_brake_tap_actions(action_ids)

    assert compact_count == len(action_ids)
    assert canonical_count == 78
    assert all(
        np.array_equal(control, canonical[index])
        for index, control in zip(action_ids, compact, strict=True)
    )
    assert np.array_equal(
        select_brake_tap_exploration_weights(action_ids),
        select_brake_tap_exploration_weights(None)[list(action_ids)],
    )
