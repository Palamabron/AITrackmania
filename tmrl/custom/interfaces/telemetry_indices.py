"""Indices for the OpenPlanet TMRL telemetry plugin (script ``TMRL_GrabData``).

Use :class:`TmrlDataPlugin` to index the ``float`` tuple from
``TM2020OpenPlanetClient.retrieve_data()``.
"""

from __future__ import annotations

from collections import namedtuple

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


_TmrlDataPluginBase = namedtuple(
    "TmrlDataPlugin",
    [
        # 1. Race stats (4)
        "CHECKPOINTS_PASSED",
        "CURRENT_LAP",
        "FINISH_UI_ACTIVE",
        "CURRENT_RACE_TIME",
        # 2. Transform (12): position, velocity, dir, up
        "POS_X",
        "POS_Y",
        "POS_Z",
        "VEL_X",
        "VEL_Y",
        "VEL_Z",
        "DIR_X",
        "DIR_Y",
        "DIR_Z",
        "UP_X",
        "UP_Y",
        "UP_Z",
        # 3. Engine (3)
        "SPEED_MPS",
        "ENGINE_RPM",
        "ENGINE_GEAR",
        # 4. Wheels & surfaces (8)
        "SLIP_FL",
        "SLIP_FR",
        "SLIP_RL",
        "SLIP_RR",
        "MAT_FL",
        "MAT_FR",
        "MAT_RL",
        "MAT_RR",
        # 5. RL-specific (3)
        "WHEELS_SKIDDING_COUNT",
        "FLYING_DURATION",
        "ADHERENCE_COEF",
        # 6. Inputs (3)
        "INPUT_STEER",
        "INPUT_GAS",
        "INPUT_BRAKE",
    ],
)

TmrlDataPlugin = _TmrlDataPluginBase(*range(TMRL_GRABDATA_FLOAT_COUNT))
"""Field indices for one frame from the OpenPlanet TMRL data plugin (``TMRL_GrabData``)."""


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
