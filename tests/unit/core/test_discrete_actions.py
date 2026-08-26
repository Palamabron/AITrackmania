"""Discrete TrackMania action encoding must round-trip through replay safely."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trackmaniarl.trackmania.actions import (
    TrackmaniaActionSelector,
    build_brake_tap_action_table,
    continuous_control_to_discrete_indices_batch,
    discrete_indices_to_control_batch,
    select_brake_tap_actions,
    select_brake_tap_exploration_weights,
)


def test_trackmania_action_selector_can_hold_actions_without_recurrence() -> None:
    selector = TrackmaniaActionSelector(
        (3, 39, 75),
        minimum_action_hold_steps=2,
        switch_q_margin=0.1,
    )

    first = selector.select(
        torch.tensor([[3.0, 1.0, 0.0]]),
        torch.tensor([0]),
        deterministic=True,
        epsilon=0.0,
    )
    held = selector.select(
        torch.tensor([[1.0, 3.0, 0.0]]),
        torch.tensor([1]),
        deterministic=True,
        epsilon=0.0,
    )
    switched = selector.select(
        torch.tensor([[1.0, 3.0, 0.0]]),
        torch.tensor([1]),
        deterministic=True,
        epsilon=0.0,
    )

    assert (first.item(), held.item(), switched.item()) == (0, 0, 1)
    selector.reset_episode()
    assert (
        selector.select(
            torch.tensor([[0.0, 3.0, 1.0]]),
            torch.tensor([1]),
            deterministic=True,
            epsilon=0.0,
        ).item()
        == 1
    )


def test_trackmania_action_selector_holds_only_exploratory_actions() -> None:
    torch.manual_seed(7)
    selector = TrackmaniaActionSelector(exploration_hold_steps=4)
    q_values = torch.zeros(1, 78)
    greedy = torch.tensor([39])

    exploratory = selector.select(
        q_values,
        greedy,
        deterministic=False,
        epsilon=1.0,
    )
    held = [selector.select(q_values, greedy, deterministic=False, epsilon=0.0) for _ in range(3)]
    released = selector.select(q_values, greedy, deterministic=False, epsilon=0.0)

    assert all(action.item() == exploratory.item() for action in held)
    assert released.item() == greedy.item()
    selector.reset_episode()
    assert selector._exploration_steps_remaining == 0


def test_zero_q_margin_does_not_suppress_epsilon_exploration() -> None:
    class FixedExplorationSelector(TrackmaniaActionSelector):
        def _exploration_action(self, q_values: torch.Tensor, greedy: torch.Tensor) -> torch.Tensor:
            del q_values
            return greedy.new_tensor([1])

    selector = FixedExplorationSelector(switch_q_margin=0.0)

    selected = selector.select(
        torch.tensor([[10.0, 0.0]]),
        torch.tensor([0]),
        deterministic=False,
        epsilon=1.0,
    )

    assert selected.item() == 1


def test_trackmania_action_selector_can_explore_drive_modes_globally() -> None:
    torch.manual_seed(17)
    q_values = torch.zeros(128, 78)
    greedy = torch.full((128,), 1, dtype=torch.long)
    neighboring = TrackmaniaActionSelector(global_exploration_probability=0.0)
    global_selector = TrackmaniaActionSelector(global_exploration_probability=1.0)

    neighboring_actions = neighboring._exploration_action(q_values, greedy)
    global_actions = global_selector._exploration_action(q_values, greedy)

    assert torch.all(neighboring_actions % 6 == greedy % 6)
    assert torch.any(global_actions % 6 != greedy % 6)


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_trackmania_action_selector_rejects_invalid_global_exploration_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError, match="global exploration probability"):
        TrackmaniaActionSelector(global_exploration_probability=probability)


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
