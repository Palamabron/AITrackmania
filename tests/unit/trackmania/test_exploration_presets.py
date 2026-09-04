from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_SENTINEL,
    TrackmaniaActionSelector,
    _exploration_event_probability,
    build_brake_tap_action_table,
    build_expert_keyboard_exploration_weights,
    select_exploration_weights,
)


def _marginal(weights: np.ndarray, predicate: Callable[[np.ndarray], bool]) -> float:
    _, table = build_brake_tap_action_table()
    mask = np.asarray([predicate(entry) for entry in table])
    return float(weights[mask].sum() / weights.sum())


def test_expert_keyboard_preset_matches_expert_joint_controls() -> None:
    weights = build_expert_keyboard_exploration_weights()

    assert weights.shape == (78,)
    assert _marginal(weights, lambda entry: tuple(entry) == (1.0, 0.0, 0.0)) == pytest.approx(
        0.3489, abs=0.002
    )
    assert _marginal(weights, lambda entry: entry[0] == 1.0 and entry[1] == 0.0) == pytest.approx(
        0.8255, abs=0.002
    )
    assert _marginal(weights, lambda entry: entry[0] == 0.0 and entry[1] == 1.0) == pytest.approx(
        0.0538, abs=0.002
    )
    assert _marginal(weights, lambda entry: entry[0] == 1.0 and entry[1] == 1.0) < 0.001
    assert _marginal(weights, lambda entry: entry[1] == BRAKE_TAP_SENTINEL) < 0.001
    assert _marginal(weights, lambda entry: abs(entry[2]) not in (0.0, 1.0)) < 0.001
    assert _marginal(weights, lambda entry: entry[2] == 1.0) == pytest.approx(0.3657, abs=0.002)
    assert _marginal(weights, lambda entry: entry[2] == -1.0) == pytest.approx(0.2853, abs=0.002)


def test_exploration_hold_preserves_requested_step_fraction() -> None:
    probability = _exploration_event_probability(0.30, hold_steps=6)

    assert probability == pytest.approx(1.0 / 15.0)
    assert 6.0 * probability / (1.0 + 5.0 * probability) == pytest.approx(0.30)


def test_select_exploration_weights_presets_and_subsets() -> None:
    default = select_exploration_weights(None, "throttle_biased")
    expert = select_exploration_weights((0, 39, 75), "expert_keyboard")

    assert default.shape == (78,)
    assert expert.shape == (3,)
    with pytest.raises(ValueError, match="preset"):
        select_exploration_weights(None, "no-such-preset")


def test_selector_accepts_preset_in_config() -> None:
    selector = TrackmaniaActionSelector({"exploration_weights_preset": "expert_keyboard"})

    assert selector.weights.shape == (78,)
    assert selector.weights.sum() > 0
