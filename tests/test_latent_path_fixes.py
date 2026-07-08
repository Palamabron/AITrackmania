"""Regression tests for latent-path fixes: MemoryTMBest column alignment,
world-telemetry failure-counter scaling, and curvature-aware observation spaces."""

from __future__ import annotations

import numpy as np
import pytest
from tmrl.custom.memories.enums import TMBestField, TMBestObsField
from tmrl.custom.memories.tm_best import MemoryTMBest
from tmrl.custom.tm.observation_constants import WorldTelemetryObsIndex as WObs
from tmrl.custom.tm.openplanet_observation_space import build_openplanet_tuple_observation_space
from tmrl.custom.tm.tm_preprocessors import make_world_telemetry_obs_preprocessor
from tmrl.networking import Buffer

# ---------------------------------------------------------------------------
# MemoryTMBest: storage columns must match TMBestField (was off-by-one 13-23)
# ---------------------------------------------------------------------------


def _tm_best_obs(i: float) -> tuple:
    """Raw obs tuple per TMBestObsField; scalar fields encode field_id*1000 + i."""
    o = [None] * 25
    scalar_fields = [
        TMBestObsField.POSITION,
        TMBestObsField.SPEED,
        TMBestObsField.ACCELERATION,
        TMBestObsField.JERK,
        TMBestObsField.RACE_PROGRESS,
        TMBestObsField.INPUT_STEER,
        TMBestObsField.INPUT_GAS_PEDAL,
        TMBestObsField.INPUT_BRAKE,
        TMBestObsField.GEAR,
        TMBestObsField.AIM_YAW,
        TMBestObsField.AIM_PITCH,
        TMBestObsField.SURFACE_ID,
        TMBestObsField.GROUND_CONTACT,
        TMBestObsField.REACTOR_AIR_CONTROL,
        TMBestObsField.CRASHED,
        TMBestObsField.FAILURE_COUNTER,
    ]
    for f in scalar_fields:
        o[f] = float(f) * 1000.0 + i
    # Fields read with [0] indexing (arrays):
    for f in (
        TMBestObsField.STEER_ANGLE,
        TMBestObsField.WHEEL_ROT,
        TMBestObsField.WHEEL_ROT_SPEED,
        TMBestObsField.DAMPER_LEN,
    ):
        o[f] = np.array([float(f) * 1000.0 + i, 0.0], dtype=np.float32)
    # Fields stored as whole arrays:
    for f in (
        TMBestObsField.SLIP_COEF,
        TMBestObsField.REACTOR_GROUND_MODE,
        TMBestObsField.GROUND_DIST,
    ):
        o[f] = np.array([float(f) * 1000.0 + i], dtype=np.float32)
    o[TMBestObsField.CRASHED_LIST] = np.array([0.0, 0.0], dtype=np.float32)
    o[TMBestObsField.IMGS] = np.zeros((8, 8), dtype=np.uint8)
    return tuple(o)


def _filled_tm_best(n: int = 8) -> MemoryTMBest:
    memory = MemoryTMBest(
        memory_size=1000,
        batch_size=2,
        imgs_obs=4,
        act_buf_len=2,
        device="cpu",
        discrete_n_steer_bins=0,
    )
    buf = Buffer()
    for i in range(n):
        buf.append_sample(
            (
                np.array([0.5, 0.0, 0.1], dtype=np.float32),
                _tm_best_obs(float(i)),
                np.float32(1.0),
                False,
                False,
                {},
            )
        )
    memory.append_buffer(buf)
    return memory


def test_tm_best_storage_matches_enum():
    """Every TMBestField column must hold the field it names (no shift)."""
    memory = _filled_tm_best()
    i = 0  # data index 0
    checks = {
        TMBestField.SURFACE_ID: float(TMBestObsField.SURFACE_ID) * 1000.0 + i,
        TMBestField.STEER_ANGLE: float(TMBestObsField.STEER_ANGLE) * 1000.0 + i,
        TMBestField.WHEEL_ROT: float(TMBestObsField.WHEEL_ROT) * 1000.0 + i,
        TMBestField.SLIP_COEF: float(TMBestObsField.SLIP_COEF) * 1000.0 + i,
        TMBestField.GROUND_DIST: float(TMBestObsField.GROUND_DIST) * 1000.0 + i,
        TMBestField.CRASHED: float(TMBestObsField.CRASHED) * 1000.0 + i,
        TMBestField.FAILURE_COUNTER: float(TMBestObsField.FAILURE_COUNTER) * 1000.0 + i,
    }
    for field, expected in checks.items():
        stored = float(np.asarray(memory.data[field][i]).flat[0])
        assert stored == pytest.approx(expected), field.name


def test_tm_best_get_transition_returns_aligned_fields():
    """get_transition must read each enum slot from the matching column."""
    memory = _filled_tm_best()
    last_obs, _act, _rew, _new_obs, _term, _trunc, _info = memory.get_transition(0)
    idx_last = memory.min_samples - 1  # = 3

    # Returned obs layout: TMBestObsField order without CRASHED_LIST.
    def expected(f: TMBestObsField) -> float:
        return float(f) * 1000.0 + idx_last

    assert float(np.asarray(last_obs[11]).flat[0]) == pytest.approx(
        expected(TMBestObsField.SURFACE_ID)
    )
    assert float(np.asarray(last_obs[12]).flat[0]) == pytest.approx(
        expected(TMBestObsField.STEER_ANGLE)
    )
    assert float(np.asarray(last_obs[21]).flat[0]) == pytest.approx(
        expected(TMBestObsField.CRASHED)
    )
    assert float(np.asarray(last_obs[22]).flat[0]) == pytest.approx(
        expected(TMBestObsField.FAILURE_COUNTER)
    )


# ---------------------------------------------------------------------------
# World-telemetry preprocessor: failure counter no longer divided by 15 twice
# ---------------------------------------------------------------------------


def _world_obs(fc: float) -> tuple:
    obs = [np.zeros(1, dtype=np.float32) for _ in range(len(WObs))]
    obs[WObs.TRACK_INFO] = np.zeros(70, dtype=np.float32)
    obs[WObs.STEER_ANGLE] = np.zeros(2, dtype=np.float32)
    obs[WObs.SLIP_COEF] = np.zeros(2, dtype=np.float32)
    obs[WObs.FAILURE_COUNTER] = np.array([fc], dtype=np.float32)
    return tuple(obs)


def test_world_telemetry_failure_counter_not_double_normalized():
    pre = make_world_telemetry_obs_preprocessor(50.0)
    out = pre(_world_obs(0.6))  # env already normalized to [0, 1]
    assert float(out[WObs.FAILURE_COUNTER][0]) == pytest.approx(0.6)
    out = pre(_world_obs(7.0))
    assert float(out[WObs.FAILURE_COUNTER][0]) == pytest.approx(1.0)  # clipped


# ---------------------------------------------------------------------------
# Curvature-aware observation space
# ---------------------------------------------------------------------------


def test_openplanet_space_includes_curvature_box_when_enabled():
    base = build_openplanet_tuple_observation_space(points_number=10)
    with_curv = build_openplanet_tuple_observation_space(points_number=10, track_curvature_obs=True)
    assert len(with_curv.spaces) == len(base.spaces) + 1
    assert with_curv.spaces[-1].shape == (10,)
