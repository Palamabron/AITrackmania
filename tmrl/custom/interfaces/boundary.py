"""Track-boundary TrackMania 2020 rtgym interfaces.

The observation packs a 60-float vector of left/right track-boundary points ahead of
the car, sampled from a *pre-recorded* boundary map loaded from disk (CSV or pkl).
This differs from ``lidar``, which casts rays against screen pixels at every step.

- ``TM2020InterfaceBoundary``       - telemetry + pre-recorded track boundaries ahead.
- ``TM2020InterfaceBoundaryImages`` - camera image history + track boundaries + progress.

``TM2020InterfaceBoundary`` reads the CSV-based left/right boundaries; the ``Images``
variant reads per-map pickle dumps produced by ``record_track.py``.
"""

from __future__ import annotations

import os
import pickle

import cv2
import numpy as np
from gymnasium import spaces
from scipy import spatial

import tmrl.config as cfg
from tmrl.config.paths import BOUNDARY_CSV_LEFT, BOUNDARY_CSV_RIGHT
from tmrl.custom.interfaces.lidar import TM2020InterfaceLidar, TM2020InterfaceLidarProgress
from tmrl.custom.interfaces.telemetry_indices import TmrlDataPlugin, yaw_pitch_from_dir_xyz
from tmrl.custom.tm.utils.control_mouse import mouse_save_replay_tm20


def _boundary_ahead(
    left_boundary: np.ndarray,
    right_boundary: np.ndarray,
    car_position,
    look_ahead_distance: int,
    nearby_correction: int,
):
    """Slice the pre-recorded left/right boundaries to get the segments ahead of the car.

    ``left_boundary``/``right_boundary`` are (2, N) arrays of (x, z) points. Returns four
    1-D arrays of length ``look_ahead_distance``: left-x, left-z, right-x, right-z.
    """
    combined_points = left_boundary.T.tolist() + right_boundary.T.tolist()
    tree = spatial.KDTree(combined_points)
    (_, i) = tree.query(car_position)
    if i < len(left_boundary.T):
        i_l_min = i
        j_min = max(i_l_min - nearby_correction, 0)
        j_max = min(i_l_min + nearby_correction, len(left_boundary.T) - 1)
        tree_r = spatial.KDTree(right_boundary.T[j_min:j_max])
        (_, i_r_min) = tree_r.query(left_boundary.T[i_l_min])
        i_r_min = i_r_min + j_min
    else:
        i_r_min = i - len(left_boundary.T)
        j_min = max(i_r_min - nearby_correction, 0)
        j_max = min(i_r_min + nearby_correction, len(right_boundary.T) - 1)
        tree_l = spatial.KDTree(left_boundary.T[j_min:j_max])
        (_, i_l_min) = tree_l.query(right_boundary.T[i_r_min])
        i_l_min = i_l_min + j_min

    i_l_max = i_l_min + look_ahead_distance
    i_r_max = i_r_min + look_ahead_distance

    extra_l = np.full((look_ahead_distance, 2), left_boundary.T[-1])
    left_boundary_extended = np.append(left_boundary.T, extra_l, axis=0).T
    extra_r = np.full((look_ahead_distance, 2), right_boundary.T[-1])
    right_boundary_extended = np.append(right_boundary.T, extra_r, axis=0).T

    l_x = left_boundary_extended[0][i_l_min:i_l_max]
    l_z = left_boundary_extended[1][i_l_min:i_l_max]
    r_x = right_boundary_extended[0][i_r_min:i_r_max]
    r_z = right_boundary_extended[1][i_r_min:i_r_max]
    return l_x, l_z, r_x, r_z


def _to_car_frame(l_x, l_z, r_x, r_z, car_position, yaw: float):
    """Translate to the car's frame and rotate by ``yaw`` so points are car-relative."""
    left = (np.array([l_x, l_z]).T - car_position).T
    right = (np.array([r_x, r_z]).T - car_position).T
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    left_x = left[0] * cos_a - left[1] * sin_a
    left_y = left[0] * sin_a + left[1] * cos_a
    right_x = right[0] * cos_a - right[1] * sin_a
    right_y = right[0] * sin_a + right[1] * cos_a
    return left_x, left_y, right_x, right_y


def _load_boundary_pkl(left_path: str, right_path: str):
    """Load a (N, 3) [x, y, z] pkl pair and return (2, N) arrays of (x, z)."""

    def load(path: str) -> np.ndarray:
        if not os.path.exists(path):
            return np.array([[0.0, 1.0], [0.0, 1.0]])
        with open(path, "rb") as f:
            pts = pickle.load(f)
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = np.expand_dims(pts, 0)
        if pts.shape[1] >= 3:
            return pts[:, [0, 2]].T
        return pts.T

    return load(left_path), load(right_path)


class TM2020InterfaceBoundary(TM2020InterfaceLidar):
    """
    Telemetry + pre-recorded track boundaries ahead of the car.

    Observation: (speed, gear, rpm, track_information (60 floats = 15 left xz + 15 right xz),
    acceleration, steering_angle, slipping_tires, crash, failure_counter).
    """

    def __init__(
        self,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        record: bool = False,
        save_replay: bool = False,
    ):
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replay,
        )
        self.record = record
        self.window_interface = None
        self.lidar = None
        self.last_pos = [0, 0]
        self.index = 0
        self.left_boundary = np.loadtxt(BOUNDARY_CSV_LEFT, delimiter=",")
        self.right_boundary = np.loadtxt(BOUNDARY_CSV_RIGHT, delimiter=",")
        # Never set in production: it grows unbounded (one entry per env step, never drained).
        self._observed_boundaries: list[list[list[float]]] | None = (
            [[], [], [], [], []] if record else None
        )
        self._bd_prev_speed_kmh: float = 0.0

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        track_information = spaces.Box(low=-300, high=300, shape=(60,))
        acceleration = spaces.Box(low=-100, high=100.0, shape=(1,))
        steering_angle = spaces.Box(low=-1, high=1.0, shape=(1,))
        slipping_tires = spaces.Box(low=0.0, high=1, shape=(4,))
        crash = spaces.Box(low=0.0, high=1, shape=(1,))
        failure_counter = spaces.Box(low=0.0, high=15, shape=(1,))
        return spaces.Tuple(
            (
                speed,
                gear,
                rpm,
                track_information,
                acceleration,
                steering_angle,
                slipping_tires,
                crash,
                failure_counter,
            )
        )

    def grab_data(self):
        return self.client.retrieve_data()

    def _track_information_vector(self, car_position, yaw):
        l_x, l_z, r_x, r_z = _boundary_ahead(
            self.left_boundary,
            self.right_boundary,
            car_position,
            look_ahead_distance=15,
            nearby_correction=60,
        )
        l_x, l_z, r_x, r_z = _to_car_frame(l_x, l_z, r_x, r_z, car_position, yaw)
        if self._observed_boundaries is not None:
            self._observed_boundaries[0].append(l_x.tolist())
            self._observed_boundaries[1].append(l_z.tolist())
            self._observed_boundaries[2].append(r_x.tolist())
            self._observed_boundaries[3].append(r_z.tolist())
            self._observed_boundaries[4].append(car_position)
        return np.array(np.append(np.append(l_x, r_x), np.append(l_z, r_z)), dtype="float32")

    def get_obs_rew_terminated_info(self):
        data = self.grab_data()
        dir_xyz = (
            float(data[TmrlDataPlugin.DIR_X]),
            float(data[TmrlDataPlugin.DIR_Y]),
            float(data[TmrlDataPlugin.DIR_Z]),
        )
        yaw, _pitch = yaw_pitch_from_dir_xyz(dir_xyz)
        car_position = [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Z]]
        self.last_pos = car_position

        track_information = self._track_information_vector(car_position, yaw)
        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        speed = np.array([speed_kmh], dtype="float32")
        gear = np.array([data[TmrlDataPlugin.ENGINE_GEAR]], dtype="float32")
        rpm = np.array([data[TmrlDataPlugin.ENGINE_RPM]], dtype="float32")
        acceleration = np.array([speed_kmh - self._bd_prev_speed_kmh], dtype="float32")
        self._bd_prev_speed_kmh = speed_kmh
        steering_angle = np.array([data[TmrlDataPlugin.INPUT_STEER]], dtype="float32")
        slipping_tires = np.array(
            [
                data[TmrlDataPlugin.SLIP_FL],
                data[TmrlDataPlugin.SLIP_FR],
                data[TmrlDataPlugin.SLIP_RL],
                data[TmrlDataPlugin.SLIP_RR],
            ],
            dtype="float32",
        )
        crash = np.array([float(self.is_crashed)], dtype="float32")

        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])
        info = {"end_of_track": end_of_track}

        crash_penalty = -10
        reward = 0
        if bool(self.is_crashed):
            reward += crash_penalty

        if end_of_track:
            reward += self.finish_reward
            terminated = True
            failure_counter = 0
            if self.save_replays:
                mouse_save_replay_tm20()
        else:
            rew, terminated, failure_counter = self.reward_function.compute_reward(
                pos=np.array(
                    [
                        data[TmrlDataPlugin.POS_X],
                        data[TmrlDataPlugin.POS_Y],
                        data[TmrlDataPlugin.POS_Z],
                    ]
                )
            )[:3]
            reward += rew

        obs = [
            speed,
            gear,
            rpm,
            track_information,
            acceleration,
            steering_angle,
            slipping_tires,
            crash,
            float(failure_counter),
        ]
        return obs, np.float32(reward), terminated, info

    def reset(self, seed=None, options=None):
        self.reset_common()
        data = self.grab_data()
        self._bd_prev_speed_kmh = 0.0
        track_information = np.full((60,), 0, dtype="float32")
        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        speed = np.array([speed_kmh], dtype="float32")
        gear = np.array([data[TmrlDataPlugin.ENGINE_GEAR]], dtype="float32")
        rpm = np.array([data[TmrlDataPlugin.ENGINE_RPM]], dtype="float32")
        acceleration = np.array([0.0], dtype="float32")
        steering_angle = np.array([data[TmrlDataPlugin.INPUT_STEER]], dtype="float32")
        slipping_tires = np.array(
            [
                data[TmrlDataPlugin.SLIP_FL],
                data[TmrlDataPlugin.SLIP_FR],
                data[TmrlDataPlugin.SLIP_RL],
                data[TmrlDataPlugin.SLIP_RR],
            ],
            dtype="float32",
        )
        crash = np.array([float(self.is_crashed)], dtype="float32")
        obs = [
            speed,
            gear,
            rpm,
            track_information,
            acceleration,
            steering_angle,
            slipping_tires,
            crash,
            0.0,
        ]
        self.reward_function.reset()
        return obs, {}


class TM2020InterfaceBoundaryImages(TM2020InterfaceLidarProgress):
    """
    Camera images + pre-recorded track boundaries ahead + race progress.

    Observation: (speed, progress, track_information, image_history). Uses per-map pkl
    paths from ``cfg.TRACK_PATH_LEFT`` / ``cfg.TRACK_PATH_RIGHT``.
    """

    def __init__(
        self,
        img_hist_len: int = 4,
        gamepad: bool = False,
        grayscale: bool = True,
        resize_to: tuple | None = None,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
        **kwargs,
    ):
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replays,
            **kwargs,
        )
        self.grayscale = grayscale
        self.resize_to = resize_to or (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
        self.image_hist: list = []
        self.left_boundary, self.right_boundary = _load_boundary_pkl(
            cfg.TRACK_PATH_LEFT, cfg.TRACK_PATH_RIGHT
        )
        self.look_ahead_distance = 15
        self.nearby_correction = 60

    def _grab_speed_track_and_image(self):
        assert self.window_interface is not None
        assert self.client is not None
        raw_img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6], dtype="float32")
        car_position = [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Z]]
        yaw, _p = yaw_pitch_from_dir_xyz(
            (
                float(data[TmrlDataPlugin.DIR_X]),
                float(data[TmrlDataPlugin.DIR_Y]),
                float(data[TmrlDataPlugin.DIR_Z]),
            )
        )
        l_x, l_z, r_x, r_z = _boundary_ahead(
            self.left_boundary,
            self.right_boundary,
            car_position,
            self.look_ahead_distance,
            self.nearby_correction,
        )
        l_x, l_z, r_x, r_z = _to_car_frame(l_x, l_z, r_x, r_z, car_position, yaw)
        track_information = np.array(
            np.append(np.append(l_x, r_x), np.append(l_z, r_z)), dtype="float32"
        )
        w, h = self.resize_to
        img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) if img.ndim == 2 else np.transpose(img, (2, 0, 1))
        return speed, data, track_information, img

    def reset(self, seed=None, options=None):
        self.reset_common()
        speed, _data, track_information, img = self._grab_speed_track_and_image()
        self.image_hist = [img for _ in range(self.img_hist_len)]
        progress = np.array([0], dtype="float32")
        images = np.array(list(self.image_hist), dtype="float32")
        self.reward_function.reset()
        return [speed, progress, track_information, images], {}

    def get_obs_rew_terminated_info(self):
        speed, data, track_information, img = self._grab_speed_track_and_image()
        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array(
                [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]]
            )
        )[:3]
        progress = np.array(
            [self.reward_function.cur_idx / max(1, self.reward_function.datalen)],
            dtype="float32",
        )
        self.image_hist.append(img)
        self.image_hist = self.image_hist[-self.img_hist_len :]
        images = np.array(list(self.image_hist), dtype="float32")
        obs = [speed, progress, track_information, images]
        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        return obs, np.float32(rew), terminated, info

    def get_observation_space(self):
        c = 1 if self.grayscale else 3
        h, w = self.resize_to[1], self.resize_to[0]
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        track_information = spaces.Box(low=-300.0, high=300.0, shape=(60,))
        images = spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, c, h, w))
        return spaces.Tuple((speed, progress, track_information, images))
