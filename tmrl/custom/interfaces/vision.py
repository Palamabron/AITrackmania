"""Vision-based TrackMania 2020 rtgym interface.

Exposes ``TM2020Interface``: observations are (speed, gear, rpm, image_history) where
image history is a stack of game screenshots captured via ``WindowInterface``. This is
the baseline class other interface families (``car_state``, ``boundary``) inherit from.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.base import MPS_TO_KMPH, TrackMania2020InterfaceBase
from tmrl.custom.interfaces.telemetry_indices import (
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
)
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_mouse import mouse_save_replay_tm20
from tmrl.custom.tm.utils.discrete_control import build_brake_tap_action_table
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.registry import INTERFACES

OPENPLANET_PORT = 9000


@INTERFACES.register("vision")
class TM2020Interface(TrackMania2020InterfaceBase):
    """
    Base camera-based interface for TrackMania 2020 via rtgym.

    Handles image history, gamepad/keyboard control, and communication with the
    TrackMania game via OpenPlanet. Subclasses extend the observation space and
    telemetry parsing for specific model families (boundary lidar, unified RL, ...).
    """

    def _build_openplanet_client(self):
        """OpenPlanet GrabData on port 9000.

        Same client as :class:`.car_state.TM2020RLInterface`.
        """
        return TM2020OpenPlanetClient(
            port=OPENPLANET_PORT, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
        )

    def __init__(
        self,
        img_hist_len: int = 4,
        gamepad: bool = True,
        save_replays: bool = False,
        grayscale: bool = True,
        resize_to=(64, 64),
        finish_reward=None,
        constant_penalty=None,
        crash_penalty=None,
        min_nb_steps_before_failure=None,
        record_human: bool = False,
        discrete_n_steer_bins: int = 0,
        **kwargs,
    ):
        self.is_crashed = False
        self.last_time = 0.0
        self.img_hist_len = img_hist_len
        self.img_hist: deque[np.ndarray] | list[np.ndarray] | None = None
        self.img: np.ndarray | None = None
        self.reward_function: RewardFunction | None = None
        self.client: TM2020OpenPlanetClient | None = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface: WindowInterface | None = None
        self.record_human = record_human
        self.small_window: bool | None = None
        self.save_replays = save_replays
        self.grayscale = grayscale
        self.resize_to = resize_to
        self.finish_reward = finish_reward if finish_reward is not None else cfg.END_OF_TRACK_REWARD
        self.constant_penalty = (
            constant_penalty
            if constant_penalty is not None
            else cfg.REWARD_CONFIG.get("CONSTANT_PENALTY", 0.0)
        )
        self.initialized = False
        self.crash_penalty = (
            float(crash_penalty) if crash_penalty is not None else float(cfg.CRASH_PENALTY)
        )
        _default_min_steps = (
            min_nb_steps_before_failure if min_nb_steps_before_failure is not None else 70
        )
        self.min_nb_steps_before_failure = cfg.REWARD_CONFIG.get("MIN_STEPS", _default_min_steps)
        self.crash_cooldown = 0
        self.discrete_action_table: list[np.ndarray] | None = None
        if discrete_n_steer_bins > 0:
            _, self.discrete_action_table = build_brake_tap_action_table(
                n_steer=discrete_n_steer_bins
            )
        self._send_control_logged = False
        self._speed_arr = np.zeros((1,), dtype=np.float32)
        self._gear_arr = np.zeros((1,), dtype=np.float32)
        self._rpm_arr = np.zeros((1,), dtype=np.float32)
        self._last_speed_kmh: float = 0.0

    def initialize(self):
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def grab_data_and_img(self):
        assert self.window_interface is not None
        assert self.client is not None
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img[:, :, ::-1]
        data = self.client.retrieve_data()
        self.img = img
        return data, img

    def _update_telemetry_arrays(self, data) -> None:
        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * MPS_TO_KMPH
        self._speed_arr[0] = speed_kmh
        self._gear_arr[0] = data[TmrlDataPlugin.ENGINE_GEAR]
        self._rpm_arr[0] = data[TmrlDataPlugin.ENGINE_RPM]
        self._last_speed_kmh = speed_kmh

    def reset(self, seed=None, options=None):
        rf = getattr(self, "reward_function", None)
        if (
            rf is not None
            and getattr(rf, "step_counter", 0) > 0
            and not getattr(rf, "_logged_run_this_episode", False)
        ):
            rf.log_model_run(terminated=False, end_of_track=False, truncated=True)
        self.reset_common()
        assert self.reward_function is not None
        data, img = self.grab_data_and_img()
        self._update_telemetry_arrays(data)
        for _ in range(self.img_hist_len):
            self._push_img(img)
        imgs = self._get_img_hist_array()
        obs = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        assert self.reward_function is not None
        data, img = self.grab_data_and_img()
        self._update_telemetry_arrays(data)
        reward, terminated, _failure_counter, _ = self.reward_function.compute_reward(
            pos=np.array(
                [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]]
            ),
            velocity_xyz=(
                float(data[TmrlDataPlugin.VEL_X]),
                float(data[TmrlDataPlugin.VEL_Y]),
                float(data[TmrlDataPlugin.VEL_Z]),
            ),
            dir_xyz=(
                float(data[TmrlDataPlugin.DIR_X]),
                float(data[TmrlDataPlugin.DIR_Y]),
                float(data[TmrlDataPlugin.DIR_Z]),
            ),
            speed=self._last_speed_kmh,
        )
        self._push_img(img)
        imgs = self._get_img_hist_array()
        observation = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            terminated = True
            reward += self.finish_reward
            if self.save_replays:
                mouse_save_replay_tm20(True)

        self.reward_function.log_model_run(terminated=bool(terminated), end_of_track=end_of_track)
        reward_out = np.float32(reward)
        return observation, reward_out, terminated, info

    def get_observation_space(self) -> spaces.Tuple:
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        if self.discrete_action_table is not None:
            return spaces.Discrete(len(self.discrete_action_table))
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def get_default_action(self):
        if self.discrete_action_table is not None:
            return np.array(0, dtype=np.int64)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
