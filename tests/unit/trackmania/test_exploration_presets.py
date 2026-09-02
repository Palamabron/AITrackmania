from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from trackmaniarl.trackmania.actions import (
    BRAKE_TAP_SENTINEL,
    TrackmaniaActionSelector,
    build_brake_tap_action_table,
    build_expert_keyboard_exploration_weights,
    select_exploration_weights,
)


def _marginal(weights: np.ndarray, predicate: Callable[[np.ndarray], bool]) -> float:
    _, table = build_brake_tap_action_table()
    mask = np.asarray([predicate(entry) for entry in table])
    return float(weights[mask].sum() / weights.sum())


def test_expert_keyboard_preset_matches_expert_marginals() -> None:
    weights = build_expert_keyboard_exploration_weights()

    assert weights.shape == (78,)
    assert _marginal(weights, lambda entry: entry[0] == 1.0) == pytest.approx(0.83, abs=0.01)
    assert _marginal(weights, lambda entry: entry[1] == 1.0) == pytest.approx(0.055, abs=0.01)
    assert _marginal(weights, lambda entry: entry[1] == BRAKE_TAP_SENTINEL) < 0.02
    assert _marginal(weights, lambda entry: abs(entry[2]) in (0.0, 1.0)) > 0.9


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
