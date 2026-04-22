# standard library imports
import os
import pickle
import time

# third-party imports
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import tmrl.config as cfg
from tmrl.custom.interfaces.telemetry_indices import tmrl_grabdata_payload_nb_floats
from tmrl.custom.tm.utils.control_keyboard import keyres
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH

# Minimum positions required to build trajectory (spline needs enough points)
MIN_POSITIONS_FOR_RECORDING = 50


def _is_lap_finished(data: tuple[float, ...]) -> bool:
    """Return finish flag for both legacy (19f) and TQC (20f) payloads.

    Layouts:
    - 19-float legacy: finish flag at index 8
    - 20-float TQC:    finish flag at index 9 (index 8 is braking)
    """
    finish_idx = 2 if len(data) >= 30 else 9 if len(data) >= 20 else 8
    return bool(data[finish_idx])


def _position_xyz(data: tuple[float, ...]) -> list[float]:
    if len(data) >= 30:
        return [data[4], data[5], data[6]]
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
            data = client.retrieve_data(
                sleep_if_empty=0.01
            )  # we need many points to build a smooth curve
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
                    if pt is not None:  # a point was created
                        final_positions.append(pt)  # add the point to the list
                        move_by = dist_between_points
                        pt1 = pt
                    else:  # we passed pt2 without creating a new point
                        pt1 = pt2
                        j += 1
                        move_by = dst  # remaining distance

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
    # Extract x, y, and z coordinates from the input points
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Calculate the cumulative distance between consecutive points, considering all coordinates
    distances = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
    cumulative_distances = np.cumsum(distances)
    cumulative_distances = np.insert(
        cumulative_distances, 0, 0
    )  # Add a starting point distance of 0

    # Create cubic spline interpolations for x, y, and z
    cs_x = CubicSpline(cumulative_distances, x)
    cs_y = CubicSpline(cumulative_distances, y)
    cs_z = CubicSpline(cumulative_distances, z)

    # Define the desired number of points (same as the input list)
    desired_num_points = len(points)

    # Generate evenly spaced points along the spline with the desired number of points
    new_distances = np.linspace(0, cumulative_distances[-1], desired_num_points)
    new_x = cs_x(new_distances)
    new_y = cs_y(new_distances)
    new_z = cs_z(new_distances)

    # Combine the new x, y, and z coordinates into a 2D array
    new_points = np.column_stack((new_x, new_y, new_z))

    # Plot the input and output lists
    plt.figure(figsize=(30, 20))

    # Input points
    plt.scatter(x, y, label="Input Points", color="blue", marker="o")

    # Output points (interpolated)
    plt.plot(new_x, new_y, label="Output Points (Interpolated)", color="red", marker="x")

    return new_points


def interp_points_with_cubic_spline(sub_array, data_density):
    if len(sub_array) < 2:
        raise ValueError(
            f"CubicSpline needs at least 2 points, got {len(sub_array)}. "
            "Drive longer before stopping recording."
        )
    original_x, original_y, original_z = sub_array.T

    # Calculate the new x-values based on data density (e.g., double the points)
    original_i = np.arange(0, int(data_density * len(original_x)), step=data_density)
    new_i = np.arange(0, int(data_density * len(original_x) - 1))

    # Perform cubic spline interpolation for each vector (x, y, z)
    cs_x = CubicSpline(original_i, original_x)
    cs_y = CubicSpline(original_i, original_y)
    cs_z = CubicSpline(original_i, original_z)

    # Interpolate the y-values for the new_x values for each vector
    new_x_values = cs_x(new_i)
    new_y_values = cs_y(new_i)
    new_z_values = cs_z(new_i)

    # Combine the new x, y, and z values into a single NumPy array
    new_data = np.array([new_x_values, new_y_values, new_z_values])

    # Transpose the new_data array to have x, y, z as rows
    new_data = new_data.T

    return new_data


def smooth_points(points, sigma=12):
    """
    Smooths the given points using a Gaussian filter.

    Args:
        points (np.array): The array of points to be smoothed.
        sigma (int): The standard deviation for the Gaussian kernel.

    Returns:
        np.array: The smoothed array of points.
    """

    # Apply Gaussian filter for each dimension independently
    smoothed_x = gaussian_filter1d(points[:, 0], sigma)
    smoothed_y = gaussian_filter1d(points[:, 1], sigma)
    smoothed_z = gaussian_filter1d(points[:, 2], sigma)

    # Combine the smoothed coordinates back into a single array
    smoothed_points = np.column_stack((smoothed_x, smoothed_y, smoothed_z))

    return smoothed_points


def line(pt1, pt2, dist):
    """
    Creates a point between pt1 and pt2, at distance dist from pt1.

    If dist is too large, returns None and the remaining distance (> 0.0).
    Else, returns the point and 0.0 as remaining distance.
    """
    vec = pt2 - pt1
    norm = np.linalg.norm(vec)
    if norm < dist:
        return (
            None,
            dist - norm,
        )  # we couldn't create a new point but we moved by a distance of norm
    else:
        vec_unit = vec / norm
        pt = pt1 + vec_unit * dist
        return pt, 0.0


if __name__ == "__main__":
    record_reward_dist(path_reward=PATH_REWARD)
