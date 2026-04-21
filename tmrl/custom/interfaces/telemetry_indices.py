"""Indices for the OpenPlanet TMRL telemetry plugin (script ``TMRL_GrabData``).

Use :class:`TmrlDataPlugin` to index the ``float`` tuple from
``TM2020OpenPlanetClient.retrieve_data()``.
"""

from __future__ import annotations

from enum import IntEnum

TMRL_GRABDATA_FLOAT_COUNT = 33
"""Default number of ``float`` values per TMRL_GrabData frame (current plugin layout)."""


def tmrl_grabdata_payload_nb_floats(reward_config: dict) -> int:
    """Struct size (float count) for ``TM2020OpenPlanetClient`` when talking to TMRL_GrabData.

    Config keys (in order):

    1. ``TMRL_GRABDATA_NB_FLOATS`` - preferred, names the plugin.
    2. ``TQC_GRAB_NB_FLOATS`` - legacy only (TQC is an algorithm, not the plugin).
    3. Default: :data:`TMRL_GRABDATA_FLOAT_COUNT`.
    """
    return int(
        reward_config.get(
            "TMRL_GRABDATA_NB_FLOATS",
            reward_config.get("TQC_GRAB_NB_FLOATS", TMRL_GRABDATA_FLOAT_COUNT),
        )
    )


class TmrlDataPlugin(IntEnum):
    """Field indices for one frame from the OpenPlanet TMRL data plugin (``TMRL_GrabData``)."""

    # 1. Race stats (4)
    CHECKPOINTS_PASSED = 0
    CURRENT_LAP = 1
    FINISH_UI_ACTIVE = 2
    CURRENT_RACE_TIME = 3
    # 2. Transform (12): position, velocity, dir, up
    POS_X = 4
    POS_Y = 5
    POS_Z = 6
    VEL_X = 7
    VEL_Y = 8
    VEL_Z = 9
    DIR_X = 10
    DIR_Y = 11
    DIR_Z = 12
    UP_X = 13
    UP_Y = 14
    UP_Z = 15
    # 3. Engine (3)
    SPEED_MPS = 16
    ENGINE_RPM = 17
    ENGINE_GEAR = 18
    # 4. Wheels & surfaces (8)
    SLIP_FL = 19
    SLIP_FR = 20
    SLIP_RL = 21
    SLIP_RR = 22
    MAT_FL = 23
    MAT_FR = 24
    MAT_RL = 25
    MAT_RR = 26
    # 5. RL-specific (3)
    WHEELS_SKIDDING_COUNT = 27
    FLYING_DURATION = 28
    ADHERENCE_COEF = 29
    # 6. Inputs (3)
    INPUT_STEER = 30
    INPUT_GAS = 31
    INPUT_BRAKE = 32


def yaw_pitch_from_dir_xyz(dir_xyz) -> tuple[float, float]:
    """Unit forward vector → yaw (XZ) and pitch (Y)."""
    import numpy as np

    d = np.asarray(dir_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(d))
    if norm <= 1e-8:
        return 0.0, 0.0
    d = d / norm
    yaw = float(np.arctan2(d[0], d[2]))
    pitch = float(np.arcsin(np.clip(d[1], -1.0, 1.0)))
    return yaw, pitch
