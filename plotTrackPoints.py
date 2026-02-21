import pickle

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d

import config.config_constants as cfg


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


TRACK_PATH_LEFT = cfg.TRACK_PATH_LEFT
TRACK_PATH_RIGHT = cfg.TRACK_PATH_RIGHT

with open(TRACK_PATH_LEFT, "rb") as f:
    left_track = pickle.load(f)

with open(TRACK_PATH_RIGHT, "rb") as f:
    right_track = pickle.load(f)

# left_track = smooth_points(left_track)
# right_track = smooth_points(right_track)

# print(f"LEFT TRACK: \n {left_track}")
# print(f"RIGHT TRACK: \n {right_track}")

# Plotly interactive 3D scatter plot for both tracks
plotly_fig = go.Figure()
plotly_fig.add_trace(
    go.Scatter3d(
        x=left_track[:, 0],
        y=left_track[:, 1],
        z=left_track[:, 2],
        mode="markers",
        marker=dict(size=5, color="blue", opacity=0.8),
        name="Left Track",
    )
)
plotly_fig.add_trace(
    go.Scatter3d(
        x=right_track[:, 0],
        y=right_track[:, 1],
        z=right_track[:, 2],
        mode="markers",
        marker=dict(size=5, color="red", opacity=0.8),
        name="Right Track",
    )
)
plotly_fig.update_layout(
    title="Interactive 3D Scatter Plot of Left and Right Tracks",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        aspectratio=dict(x=1, y=0.01, z=1),
    ),
)
# Display the interactive plot (will be displayed in a separate browser window)
plotly_fig.show(renderer="browser")
