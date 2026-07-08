"""Tests for boundary-lidar observation scaling and demo recording action slots."""

from __future__ import annotations

import numpy as np
import pytest
from tmrl.custom.tm.tm_preprocessors import (
    discrete_action_index_scale,
    obs_preprocessor_lidar_act_in_obs,
)


def _raw_boundary_obs(fc: float = 0.0, act_tail: tuple = ()) -> tuple:
    """Raw 9-tuple boundary obs plus optional rtgym action-buffer tail."""
    return (
        np.full(60, 150.0, dtype=np.float32),  # track, meters
        np.array([500.0], dtype=np.float32),  # speed km/h
        np.array([3.0], dtype=np.float32),  # gear
        np.array([5000.0], dtype=np.float32),  # rpm
        np.array([50.0], dtype=np.float32),  # acceleration
        np.array([0.5], dtype=np.float32),  # steering angle
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # slip
        np.array([0.0], dtype=np.float32),  # crash
        np.array([fc], dtype=np.float32),  # failure counter (already [0,1])
        *act_tail,
    )


def test_failure_counter_clipped_not_rescaled():
    """The interface already normalizes fc to [0,1]; the preprocessor must not /15 it."""
    out = obs_preprocessor_lidar_act_in_obs(_raw_boundary_obs(fc=0.5))
    assert float(out[8][0]) == pytest.approx(0.5)
    out = obs_preprocessor_lidar_act_in_obs(_raw_boundary_obs(fc=3.0))
    assert float(out[8][0]) == pytest.approx(1.0)  # clipped


def test_discrete_action_tail_scaled_to_unit_range():
    tail = (np.array(0, dtype=np.int64), np.array(77, dtype=np.int64))
    out = obs_preprocessor_lidar_act_in_obs(_raw_boundary_obs(act_tail=tail))
    assert len(out) == 11
    lo, hi = np.asarray(out[9]), np.asarray(out[10])
    assert lo.dtype == np.float32 and hi.dtype == np.float32
    assert float(lo) == pytest.approx(0.0)
    assert float(hi) == pytest.approx(77.0 * discrete_action_index_scale())
    assert 0.0 <= float(hi) <= 1.0


def test_continuous_action_tail_passthrough():
    """SAC-style (3,) continuous action slots must not be rescaled."""
    tail = (np.array([1.0, 0.0, -0.5], dtype=np.float32),)
    out = obs_preprocessor_lidar_act_in_obs(_raw_boundary_obs(act_tail=tail))
    assert np.array_equal(out[9], tail[0])


def test_record_episode_rewrites_discrete_action_slots():
    from tmrl.tools.recording.record_episode import _rewrite_discrete_action_slots

    scale = discrete_action_index_scale()
    placeholder = np.zeros(3, dtype=np.float32)  # neutral action sent during recording

    def sample(control):
        obs = (
            np.zeros(60, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            placeholder.copy(),  # act slot t-1 (rtgym echo of the placeholder)
            placeholder.copy(),  # act slot t
        )
        return (np.asarray(control, dtype=np.float32), obs, 0.0, False, False, {})

    controls = [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    samples = _rewrite_discrete_action_slots([sample(c) for c in controls], act_buf_len=2)

    import tmrl.config as cfg
    from tmrl.custom.tm.utils.control.discrete import (
        build_brake_tap_action_table,
        continuous_control_to_discrete_indices_batch,
    )

    _, table = build_brake_tap_action_table(n_steer=cfg.IQN_N_STEER_BINS)
    expected_idx = continuous_control_to_discrete_indices_batch(
        np.asarray(controls, dtype=np.float32), table
    )

    for i, s in enumerate(samples):
        obs = s[1]
        newest = float(np.asarray(obs[-1]))
        oldest = float(np.asarray(obs[-2]))
        assert newest == pytest.approx(float(expected_idx[i]) * scale)
        prev = expected_idx[max(0, i - 1)]
        assert oldest == pytest.approx(float(prev) * scale)
        assert np.asarray(obs[-1]).dtype == np.float32
