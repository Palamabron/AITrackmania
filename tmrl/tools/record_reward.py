import os
import pickle
import time

import numpy as np
from loguru import logger
from scipy.interpolate import CubicSpline

import tmrl.config as cfg
from tmrl.custom.interfaces.telemetry_indices import tmrl_grabdata_payload_nb_floats
from tmrl.custom.tm.utils.control_keyboard import keyres
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
from tmrl.tools.geometry_utils import interp_points_with_cubic_spline, line
from tmrl.tools.telemetry import _is_lap_finished, _position_xyz

PATH_REWARD = cfg.REWARD_PATH

# Minimum samples before a trajectory can be built; CubicSpline needs enough knots.
MIN_POSITIONS_FOR_RECORDING = 50
# Arc-length spacing (metres) between reward checkpoints.
REWARD_POINT_SPACING_M = 1.05
# Log a progress message every N collected positions.
_LOG_INTERVAL = 1000


def _reset_env_before_recording() -> None:
    logger.info("Resetting environment before reward recording.")
    keyres()
    time.sleep(max(0.0, float(cfg.SLEEP_TIME_AT_RESET)))


def record_reward_dist(path_reward=PATH_REWARD):
    positions: list[list[float]] = []
    client = TM2020OpenPlanetClient(
        port=9000, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
    )
    _reset_env_before_recording()
    recording_announced = False

    while True:
        data = client.retrieve_data(sleep_if_empty=0.01)
        terminated = _is_lap_finished(data)
        if terminated:
            if len(positions) < MIN_POSITIONS_FOR_RECORDING:
                logger.warning(
                    "Ignoring early lap-finished signal with too few positions "
                    f"({len(positions)}). "
                    f"Need at least {MIN_POSITIONS_FOR_RECORDING}; keep driving."
                )
                continue
            logger.info("Computing reward function checkpoints from captured positions...")
            logger.info(f"Initial number of captured positions: {len(positions)}")
            positions_xyz = np.asarray(positions, dtype=np.float64)

            final_positions = [positions_xyz[0]]
            move_by = REWARD_POINT_SPACING_M
            pt1 = final_positions[-1]
            j = 1
            while j < len(positions_xyz):
                pt2 = positions_xyz[j]
                pt, dst = line(pt1, pt2, move_by)
                if pt is not None:
                    final_positions.append(pt)
                    move_by = REWARD_POINT_SPACING_M
                    pt1 = pt
                else:
                    pt1 = pt2
                    j += 1
                    move_by = dst

            final_stack = np.array(final_positions)
            if len(final_stack) < 2:
                logger.error(
                    f"Not enough distinct positions ({len(final_stack)}) for trajectory. "
                    "Drive further along the track before stopping."
                )
                return
            upsampled_arr = interp_points_with_cubic_spline(final_stack, data_density=3)
            spaced_points = _space_points(upsampled_arr)
            logger.debug(f"final_positions: {final_stack}")
            logger.debug(f"upsampled_arr: {upsampled_arr}")
            logger.debug(f"spaced_points: {spaced_points}")
            logger.info(f"Final number of checkpoints in the reward function: {len(spaced_points)}")

            abs_path = os.path.abspath(path_reward)
            with open(path_reward, "wb") as f:
                pickle.dump(spaced_points, f)
            logger.info(f"Saved reward trajectory to: {abs_path}")
            return

        positions.append(_position_xyz(data))
        if not recording_announced:
            recording_announced = True
            logger.info("Recording started")
            logger.info(
                "Recording reward trajectory: telemetry received and samples are being collected."
            )
        elif len(positions) % _LOG_INTERVAL == 0:
            logger.info(f"Recording in progress: collected {len(positions)} position samples.")


def _space_points(points: np.ndarray) -> np.ndarray:
    """Resample ``points`` by arc length onto ``len(points)`` evenly spaced knots."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)
    cs_z = CubicSpline(cumulative_distances, z)
    new_distances = np.linspace(0, cumulative_distances[-1], len(points))
    new_x = cs_x(new_distances)
    new_y = cs_y(new_distances)
    new_z = cs_z(new_distances)
    return np.column_stack((new_x, new_y, new_z))


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
