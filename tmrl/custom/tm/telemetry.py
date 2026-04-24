"""
Telemetry data structure for the environment state.

This NamedTuple encapsulates a single frame of raw telemetry data, including:
race progress (cp, lap), kinematics (speed, pos, acceleration, jerk),
driver inputs (input_steer, input_gas, input_brake), vehicle dynamics
(aim, wheel steer angles, slip coefficients, gear), and event flags
(is_finished, is_crashed_raw).

Use this structured object to access telemetry fields securely by name
instead of raw tuple indexing.
"""

from typing import NamedTuple


class Telemetry(NamedTuple):
    cp: int
    lap: int
    speed: float
    pos_x: float
    pos_y: float
    pos_z: float
    input_steer: float
    input_gas: float
    input_brake: bool
    is_finished: bool
    acceleration: float
    jerk: float
    aim_yaw: float
    aim_pitch: float
    fl_steer_angle: float
    fr_steer_angle: float
    fl_slip_coef: float
    fr_slip_coef: float
    is_crashed_raw: bool
    gear: float
