"""Canonical tuple observation space for OpenPlanet-plugin telemetry from trainer config.

Single place for Box shapes so replay alignment, env, and IQN stay consistent.
Keep in sync with
:meth:`tmrl.custom.interfaces.car_state.TM2020RLInterface.get_observation_space` fields.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def build_openplanet_tuple_observation_space(
    points_number: int | None = None,
    track_curvature_obs: bool = False,
) -> spaces.Tuple:
    """Tuple-of-Box layout for OpenPlanet vector fields (no images, no act-in-obs tail).

    Args:
        points_number: Number of track look-ahead points. Falls back to
            ``MAIN_CONFIG.algorithm.num_track_points`` when ``None``.
        track_curvature_obs: Whether to include a curvature observation box.
    """
    if points_number is None:
        from tmrl.config.loader import MAIN_CONFIG

        points_number = int(MAIN_CONFIG.algorithm.num_track_points)
    n = int(points_number)
    track = spaces.Box(low=-100.0, high=100.0, shape=(7 * n,))
    speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
    acceleration = spaces.Box(low=-100.0, high=100.0, shape=(1,))
    jerk = spaces.Box(low=-10.0, high=10.0, shape=(1,))
    race_progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
    input_steer = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    input_gas_pedal = spaces.Box(low=0.0, high=1.0, shape=(1,))
    input_brake = spaces.Box(low=0.0, high=1.0, shape=(1,))
    gear = spaces.Box(low=0.0, high=6.0, shape=(1,))
    engine_rpm = spaces.Box(low=0.0, high=20000.0, shape=(1,))
    skidding_count = spaces.Box(low=0.0, high=4.0, shape=(1,))
    adherence_coef = spaces.Box(low=0.0, high=2.0, shape=(1,))
    aim_yaw = spaces.Box(low=-4.0, high=4.0, shape=(1,))
    aim_pitch = spaces.Box(low=-1.0, high=1.0, shape=(1,))
    steer_angle = spaces.Box(low=-30.0, high=30.0, shape=(2,))
    slip_coef = spaces.Box(low=0.0, high=1.0, shape=(2,))
    failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))
    spaces_list = [
        track,
        speed,
        acceleration,
        jerk,
        race_progress,
        input_steer,
        input_gas_pedal,
        input_brake,
        gear,
        engine_rpm,
        skidding_count,
        adherence_coef,
        aim_yaw,
        aim_pitch,
        steer_angle,
        slip_coef,
        failure_counter,
    ]
    if track_curvature_obs:
        curvature = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        spaces_list.append(curvature)
    return spaces.Tuple(tuple(spaces_list))
