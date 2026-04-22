import os
import pickle
import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import tmrl.config as cfg
from tmrl.custom.interfaces.telemetry_indices import (
    TMRL_GRABDATA_FLOAT_COUNT,
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
)
from tmrl.custom.tm.utils.control_keyboard import keyres
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH

# Minimum samples before a trajectory can be built; CubicSpline needs enough knots.
MIN_POSITIONS_FOR_RECORDING = 50


def _is_lap_finished(data: tuple[float, ...]) -> bool:
    """Return finish flag for legacy (19f), TQC (20f) and TMRL_GrabData (33f) payloads.

    Layouts:
    - 19-float legacy:           finish flag at index 8
    - 20-float TQC:              finish flag at index 9 (index 8 is braking)
    - 33-float TMRL_GrabData:    finish flag at ``TmrlDataPlugin.FINISH_UI_ACTIVE`` (2)
    """
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        finish_idx = int(TmrlDataPlugin.FINISH_UI_ACTIVE)
    else:
        finish_idx = 9 if len(data) >= 20 else 8
    return bool(data[finish_idx])


def _position_xyz(data: tuple[float, ...]) -> list[float]:
    """Return [x, y, z] for legacy (19f), TQC (20f) and TMRL_GrabData (33f) payloads."""
    if len(data) >= TMRL_GRABDATA_FLOAT_COUNT:
        px = int(TmrlDataPlugin.POS_X)
        return [data[px], data[px + 1], data[px + 2]]
    if len(data) >= 20:
        return [data[3], data[4], data[5]]
    return [data[2], data[3], data[4]]


def _reset_env_before_recording() -> None:
    logger.info("Resetting environment before reward recording.")
    keyres()
    time.sleep(max(0.0, float(cfg.SLEEP_TIME_AT_RESET)))


def record_reward_dist(path_reward=PATH_REWARD):
    positions = []
    client = TM2020OpenPlanetClient(
        port=9000, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
    )
    _reset_env_before_recording()
    path = path_reward
    recording_announced = False

    is_recording = True
    while True:
        if is_recording:
            data = client.retrieve_data(sleep_if_empty=0.01)
            terminated = _is_lap_finished(data)
            should_stop = terminated
            if should_stop:
                if len(positions) < MIN_POSITIONS_FOR_RECORDING:
                    msg = (
                        "Ignoring early lap-finished signal with too few positions "
                        f"({len(positions)}). "
                        f"Need at least {MIN_POSITIONS_FOR_RECORDING}; keep driving."
                    )
                    logger.warning(msg)
                    continue
                logger.info("Computing reward function checkpoints from captured positions...")
                logger.info(f"Initial number of captured positions: {len(positions)}")
                positions = np.array(positions)

                final_positions = [positions[0]]
                dist_between_points = 1.05
                j = 1
                move_by = dist_between_points
                pt1 = final_positions[-1]
                while j < len(positions):
                    pt2 = positions[j]
                    pt, dst = line(pt1, pt2, move_by)
                    if pt is not None:
                        final_positions.append(pt)
                        move_by = dist_between_points
                        pt1 = pt
                    else:
                        pt1 = pt2
                        j += 1
                        move_by = dst

                final_positions = np.array(final_positions)
                if len(final_positions) < 2:
                    logger.error(
                        f"Not enough distinct positions ({len(final_positions)}) for trajectory. "
                        "Drive further along the track before stopping."
                    )
                    return
                upsampled_arr = interp_points_with_cubic_spline(final_positions, data_density=3)
                spaced_points = space_points(upsampled_arr)
                logger.debug(f"final_positions: {final_positions}")
                logger.debug(f"upsampled_arr: {upsampled_arr}")
                logger.debug(f"spaced_points: {spaced_points}")
                logger.info(
                    f"Final number of checkpoints in the reward function: {len(spaced_points)}"
                )

                abs_path = os.path.abspath(path)
                with open(path, "wb") as f:
                    pickle.dump(spaced_points, f)
                logger.info(f"Saved reward trajectory to: {abs_path}")
                return
            else:
                positions.append(_position_xyz(data))
                if not recording_announced:
                    recording_announced = True
                    logger.info("Recording started")
                    logger.info(
                        "Recording reward trajectory: telemetry received and "
                        "samples are being collected."
                    )
                elif len(positions) % 1000 == 0:
                    logger.info(
                        f"Recording in progress: collected {len(positions)} position samples."
                    )


def space_points(points):
    """Resample ``points`` by arc length onto ``len(points)`` evenly spaced knots.

    Also emits a debug scatter/plot comparing the input and interpolated curves.
    """
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
    new_points = np.column_stack((new_x, new_y, new_z))

    plt.figure(figsize=(30, 20))
    plt.scatter(x, y, label="Input Points", color="blue", marker="o")
    plt.plot(new_x, new_y, label="Output Points (Interpolated)", color="red", marker="x")
    return new_points


def interp_points_with_cubic_spline(sub_array, data_density):
    """Cubic-spline interpolate ``sub_array`` (N, 3), upsampled by ``data_density``."""
    if len(sub_array) < 2:
        raise ValueError(
            f"CubicSpline needs at least 2 points, got {len(sub_array)}. "
            "Drive longer before stopping recording."
        )
    original_x, original_y, original_z = sub_array.T
    original_i = np.arange(0, int(data_density * len(original_x)), step=data_density)
    new_i = np.arange(0, int(data_density * len(original_x) - 1))
    cs_x = CubicSpline(original_i, original_x)
    cs_y = CubicSpline(original_i, original_y)
    cs_z = CubicSpline(original_i, original_z)
    return np.array([cs_x(new_i), cs_y(new_i), cs_z(new_i)]).T


def smooth_points(points, sigma=12):
    """Apply a per-axis Gaussian filter (``sigma`` samples) to (N, 3) ``points``."""
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    smoothed_z = gaussian_filter1d(points[:, 2], sigma)
    return np.column_stack((smoothed_x, smoothed_y, smoothed_z))


def line(pt1, pt2, dist):
    """Step along the segment ``pt1 -> pt2`` by ``dist`` metres.

    Returns:
        ``(pt, 0.0)`` when a new point was produced, or ``(None, remaining)``
        when the segment was shorter than ``dist`` and ``remaining`` metres
        still need to be walked on the next segment.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return None, dist - norm
    vec_unit = vec / norm
    pt = pt1 + vec_unit * dist
    return pt, 0.0


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
