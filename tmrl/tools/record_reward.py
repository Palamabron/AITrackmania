# standard library imports
import pickle
import time

# third-party imports
import keyboard
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

import tmrl.config.config_constants as cfg
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

PATH_REWARD = cfg.REWARD_PATH
DATASET_PATH = cfg.DATASET_PATH


def record_reward_dist(path_reward=PATH_REWARD, use_keyboard=False):
    positions = []
    client = TM2020OpenPlanetClient()
    path = path_reward

    is_recording = False
    if use_keyboard:
        print("Press 'e' to start recording, 'q' to stop")
    while True:
        if not is_recording and (not use_keyboard or keyboard.is_pressed("e")):
            logger.info("start recording")
            is_recording = True
        if is_recording:
            data = client.retrieve_data(
                sleep_if_empty=0.01
            )  # we need many points to build a smooth curve
            terminated = bool(data[8])
            early_stop = use_keyboard and keyboard.is_pressed("q")
            if early_stop or terminated:
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
                upsampled_arr = interp_points_with_cubic_spline(final_positions, data_density=3)
                spaced_points = space_points(upsampled_arr)
                print(f"final_positions: {final_positions}", end="\n\n")
                print(f"upsampled_arr: {upsampled_arr}", end="\n\n")
                print(f"spaced_points: {spaced_points}", end="\n\n")
                logger.info(
                    f"Final number of checkpoints in the reward function: {len(spaced_points)}"
                )

                pickle.dump(spaced_points, open(path, "wb"))
                logger.info("All done")
                return
            else:
                positions.append([data[3], data[4], data[5]])
        else:
            if use_keyboard:
                time.sleep(0.05)  # waiting for user to press E


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
