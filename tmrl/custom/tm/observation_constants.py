"""
Constants for the world-telemetry observation structure (TQC_GrabData plugin).

Observation is a list: [track_info, speed, acceleration, jerk, race_progress,
input_steer, input_gas_pedal, input_brake, gear, aim_yaw, aim_pitch,
steer_angle, slip_coef, failure_counter].
Use these indices instead of magic numbers when indexing observations.
"""

from enum import IntEnum


class WorldTelemetryObsIndex(IntEnum):
    """Indices of observation parts in the world-telemetry interface (total_obs list)."""

    TRACK_INFO = 0
    SPEED = 1
    ACCELERATION = 2
    JERK = 3
    RACE_PROGRESS = 4
    INPUT_STEER = 5
    INPUT_GAS_PEDAL = 6
    INPUT_BRAKE = 7
    GEAR = 8
    AIM_YAW = 9
    AIM_PITCH = 10
    STEER_ANGLE = 11
    SLIP_COEF = 12
    FAILURE_COUNTER = 13
