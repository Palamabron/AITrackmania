"""Vision-based TrackMania 2020 rtgym interface.

Exposes ``TM2020Interface``: observations are (speed, gear, rpm, image_history) where
image history is a stack of game screenshots captured via ``WindowInterface``. This is
the baseline class every other interface family (``lidar``, ``car_state``, ``boundary``)
inherits from.
"""

import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.base import TrackMania2020InterfaceBase
from tmrl.custom.interfaces.telemetry_indices import (
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
)
from tmrl.custom.tm.utils.control_mouse import mouse_save_replay_tm20
from tmrl.custom.tm.utils.discrete_control import build_yosh_action_table
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient

CHECK_FORWARD = 500  # allows (and rewards) 50m cuts


class TM2020Interface(TrackMania2020InterfaceBase):
    """
    Base camera-based interface for TrackMania 2020 via rtgym.

    Handles image history, gamepad/keyboard control, and communication with the
    TrackMania game via OpenPlanet. Subclasses extend the observation space and
    telemetry parsing for specific model families (LIDAR, boundary, unified RL, ...).
    """

    def _build_openplanet_client(self):
        """OpenPlanet GrabData on port 9000.

        Same client as :class:`.car_state.TM2020RLInterface`.
        """
        return TM2020OpenPlanetClient(
            port=9000, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
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
        **kwargs,
    ):
        self.is_crashed = None
        self.last_time = None
        self.img_hist_len = img_hist_len
        self.img_hist = None
        self.img = None
        self.reward_function = None
        self.client = None
        self.gamepad = gamepad
        self.j = None
        self.window_interface = None
        self.record_human = record_human
        self.small_window = None
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
            crash_penalty
            if crash_penalty is not None
            else cfg.REWARD_CONFIG.get("CRASH_PENALTY", 10.0)
        )
        _default_min_steps = (
            min_nb_steps_before_failure if min_nb_steps_before_failure is not None else 70
        )
        self.min_nb_steps_before_failure = cfg.REWARD_CONFIG.get("MIN_STEPS", _default_min_steps)
        self.crash_cooldown = 0
        _alg_cfg = cfg.TMRL_CONFIG.get("ALG", {})
        self.discrete_action_table: list[np.ndarray] | None = None
        if _alg_cfg.get("ALGORITHM") in ("IQN", "SDSAC"):
            _n_steer = int(_alg_cfg.get("IQN_N_STEER_BINS", 13))
            _, self.discrete_action_table = build_yosh_action_table(n_steer=_n_steer)
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
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img[:, :, ::-1]
        data = self.client.retrieve_data()
        self.img = img
        return data, img

    def reset(self, seed=None, options=None):
        self.reset_common()
        data, img = self.grab_data_and_img()
        self._speed_arr[0] = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        self._gear_arr[0] = data[TmrlDataPlugin.ENGINE_GEAR]
        self._rpm_arr[0] = data[TmrlDataPlugin.ENGINE_RPM]
        self._last_speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        for _ in range(self.img_hist_len):
            self._push_img(img)
        imgs = self._get_img_hist_array()
        obs = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        data, img = self.grab_data_and_img()
        self._speed_arr[0] = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        self._gear_arr[0] = data[TmrlDataPlugin.ENGINE_GEAR]
        self._rpm_arr[0] = data[TmrlDataPlugin.ENGINE_RPM]
        self._last_speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        reward, terminated, _failure_counter, _ = self.reward_function.compute_reward(
            pos=np.array(
                [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]]
            )
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
        reward = np.float32(reward)
        return observation, reward, terminated, info

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
        return np.array([0.0, 0.0, 0.0], dtype="float32")
