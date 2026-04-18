"""
This module provides the base rtgym interfaces for TrackMania 2020.
It includes classes for handling game input, extracting game state (images, lidar, data),
and computing rewards and termination conditions.
"""

import platform
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
from gymnasium import spaces
from loguru import logger
from rtgym import RealTimeGymInterface

from tmrl.custom.tm.utils.auto_drift import compute_drift_steer, is_auto_drift_action
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import (
    control_gamepad,
    gamepad_close_finish_pop_up_tm20,
    gamepad_reset,
)
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.control_mouse import (
    mouse_close_finish_pop_up_tm20,
    mouse_save_replay_tm20,
)
from tmrl.custom.tm.utils.discrete_control import (
    BRAKE_TAP_DURATION_S,
    build_yosh_action_table,
    discrete_index_to_control,
    is_brake_tap,
)
from tmrl.custom.tm.utils.tools import (
    TM2020OpenPlanetClient,
    openplanet_grab_indices,
    save_ghost,
)
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.registry import INTERFACES


@INTERFACES.register("vision")
class TM2020Interface(RealTimeGymInterface):
    """
    Base API for controlling TrackMania 2020 via rtgym.

    This class handles image history, gamepad/keyboard control, and communication
    with the TrackMania game via OpenPlanet.
    """

    j: Any
    window_interface: WindowInterface | None
    reward_function: RewardFunction | None
    client: TM2020OpenPlanetClient | None
    small_window: bool | None
    img_hist: Any
    is_crashed: bool | None
    last_time: float | None
    img: Any

    def __init__(
        self,
        img_hist_len: int = 4,
        gamepad: bool = True,
        save_replays: bool = False,
        grayscale: bool = True,
        resize_to=(64, 64),
        finish_reward: float = 1.0,
        constant_penalty: float = 0.0,
        crash_penalty: float = 10.0,
        record_human: bool = False,
        reward_path: str | None = None,
        reward_check_forward: int = 500,
        reward_check_backward: int = 10,
        reward_max_stray: float = 50.0,
        sleep_time_at_reset: float = 1.5,
        window_width: int = 256,
        window_height: int = 128,
        discrete_n_steer_bins: int = 0,
        reward_config: dict | None = None,
        is_lidar: bool = False,
        track_path_left: str = "",
        track_path_right: str = "",
        time_step_duration: float = 0.05,
        points_distance: float = 1.0,
        lap_cooldown: int = 0,
        config_file_path: str = "",
        use_wandb: bool = False,
        wandb_project: str = "tmrl",
        wandb_entity: str = "tmrl",
        wandb_run_id: str = "",
        wandb_api_key: str = "",
        wandb_config: dict | None = None,
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
        self.finish_reward = float(finish_reward)
        self.constant_penalty = float(constant_penalty)
        self.initialized = False
        self.crash_penalty = float(crash_penalty)
        self.crash_cooldown = 0
        self.crash_curr = 0
        self.reward_path = reward_path
        self.reward_check_forward = reward_check_forward
        self.reward_check_backward = reward_check_backward
        self.reward_max_stray = reward_max_stray
        self.sleep_time_at_reset = sleep_time_at_reset
        self.window_width = window_width
        self.window_height = window_height
        self._reward_config = reward_config
        self._is_lidar = is_lidar
        self._track_path_left = track_path_left
        self._track_path_right = track_path_right
        self._time_step_duration = time_step_duration
        self._points_distance = points_distance
        self._lap_cooldown = lap_cooldown
        self._config_file_path = config_file_path
        self._use_wandb = use_wandb
        self._wandb_project = wandb_project
        self._wandb_entity = wandb_entity
        self._wandb_run_id = wandb_run_id
        self._wandb_api_key = wandb_api_key
        self._wandb_config = wandb_config
        self.discrete_action_table: list[np.ndarray] | None = None
        if discrete_n_steer_bins > 0:
            _, self.discrete_action_table = build_yosh_action_table(
                n_steer=discrete_n_steer_bins
            )
        self._send_control_logged = False
        self._img_buf: np.ndarray | None = None
        self._img_hist_count = 0
        self._img_hist_cursor = 0
        self._speed_arr = np.zeros((1,), dtype=np.float32)
        self._gear_arr = np.zeros((1,), dtype=np.float32)
        self._rpm_arr = np.zeros((1,), dtype=np.float32)
        self._last_speed_kmh: float = 0.0

    def _push_img(self, img: np.ndarray) -> None:
        """
        Pushes a new image into the history buffer.

        Args:
            img (np.ndarray): The new image to add.
        """
        if self._img_buf is None or self._img_buf.shape[1:] != img.shape:
            self._img_buf = np.zeros((self.img_hist_len, *img.shape), dtype=img.dtype)
            self._img_hist_count = 0
            self._img_hist_cursor = 0
        assert self._img_buf is not None
        self._img_buf[self._img_hist_cursor] = img
        self._img_hist_cursor = (self._img_hist_cursor + 1) % self.img_hist_len
        if self._img_hist_count < self.img_hist_len:
            self._img_hist_count += 1

    def _get_img_hist_array(self) -> np.ndarray:
        """
        Retrieves the current image history as a single NumPy array.

        Returns:
            np.ndarray: Array of history images.
        """
        if self._img_buf is None or self._img_hist_count == 0:
            return np.zeros((self.img_hist_len, 1, 1), dtype=np.uint8)
        if self._img_hist_count < self.img_hist_len:
            # Repeat first frame until buffer is full
            res = np.repeat(self._img_buf[:1], self.img_hist_len, axis=0)
            res[-self._img_hist_count :] = self._img_buf[: self._img_hist_count]
            return res
        if self._img_hist_cursor == 0:
            return self._img_buf.copy()
        idx = (
            np.arange(self.img_hist_len, dtype=np.int64) + self._img_hist_cursor
        ) % self.img_hist_len
        return self._img_buf[idx]

    def initialize_common(self):
        """Initializes the window interface, reward function, and game client."""
        if self.gamepad:
            try:
                import vgamepad as vg

                pad = vg.VX360Gamepad()
                pad.register_notification(callback_function=self.crash_callback)
                self.j = pad
                logger.info("Virtual gamepad (Xbox 360) initialized for control.")
            except OSError as e:
                if "libevdev" in str(e) or "libevdev" in str(e.__cause__ or ""):
                    raise RuntimeError(
                        "Virtual gamepad (vgamepad) requires libevdev on Linux. "
                        "Worker likely in WSL while TrackMania runs on Windows."
                    ) from e
                raise
            except Exception as e:
                err_msg = str(e).lower()
                if platform.system() == "Windows" and (
                    "vigem" in err_msg or "driver" in err_msg or "device" in err_msg
                ):
                    raise RuntimeError(
                        "Virtual gamepad failed on Windows. Install ViGEmBus driver."
                    ) from e
                raise
        else:
            logger.info("Using keyboard for control (VIRTUAL_GAMEPAD=false).")
        self.window_interface = WindowInterface("Trackmania")
        self.window_interface.move_and_resize(
            w=self.window_width, h=self.window_height
        )
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self._img_buf = None
        self._img_hist_count = 0
        self._img_hist_cursor = 0
        self.reward_function = RewardFunction(
            reward_data_path=self.reward_path or "",
            nb_obs_forward=self.reward_check_forward,
            nb_obs_backward=self.reward_check_backward,
            max_dist_from_traj=self.reward_max_stray,
            crash_penalty=self.crash_penalty,
            constant_penalty=self.constant_penalty,
            reward_config=self._reward_config,
            is_lidar=self._is_lidar,
            track_path_left=self._track_path_left,
            track_path_right=self._track_path_right,
            time_step_duration=self._time_step_duration,
            points_distance=self._points_distance,
            lap_cooldown=self._lap_cooldown,
            config_file_path=self._config_file_path,
            use_wandb=self._use_wandb,
            wandb_project=self._wandb_project,
            wandb_entity=self._wandb_entity,
            wandb_run_id=self._wandb_run_id,
            wandb_api_key=self._wandb_api_key,
            wandb_config=self._wandb_config,
        )
        if self.client is None:
            self.client = TM2020OpenPlanetClient()
        self.is_crashed = False
        self.crash_cooldown = 0
        self.crash_curr = self.crash_cooldown

    def crash_callback(self, client, target, large_motor, small_motor, led_number, user_data):
        """Callback for detecting crashes via gamepad vibration."""
        self.is_crashed = large_motor > 100 and self.crash_cooldown <= 0
        if self.is_crashed:
            logger.debug("crashed: True (episode will terminate)")
            self.crash_cooldown = 10

    def initialize(self):
        """Calls initialize_common and sets the interface as initialized."""
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def send_control(self, control):
        """
        Applies the action given by the RL policy.

        Handles three brake modes when using discrete actions:
          - 0.0: no brake
          - 1.0: full brake for the whole tick
          - BRAKE_TAP_SENTINEL: 0.01 s pulse then release

        Args:
            control (np.ndarray): [gas, brake, steer] or discrete index.
        """
        if self.record_human:
            return
        if control is not None and self.discrete_action_table is not None:
            idx = int(np.asarray(control).flat[0])
            control = discrete_index_to_control(idx, self.discrete_action_table)
        if control is not None and is_auto_drift_action(control):
            drift_steer = compute_drift_steer(self._last_speed_kmh)
            control = control.copy()
            control[2] = drift_steer
        if self.gamepad:
            if control is not None:
                if self.j is None:
                    logger.error("Virtual gamepad is None; cannot send control.")
                    return
                c = np.asarray(control, dtype=np.float32).ravel()
                control = c
                if not self._send_control_logged:
                    self._send_control_logged = True
                    gas = float(control[0]) if len(control) > 0 else 0
                    brake = float(control[1]) if len(control) > 1 else 0
                    steer = float(control[2]) if len(control) > 2 else 0
                    logger.info(
                        f"First send_control: gas={gas:.2f} brake={brake:.2f} "
                        f"steer={steer:.2f} (virtual gamepad)"
                    )
                if is_brake_tap(control):
                    tap_ctrl = control.copy()
                    tap_ctrl[1] = 1.0
                    control_gamepad(self.j, tap_ctrl)
                    time.sleep(BRAKE_TAP_DURATION_S)
                    tap_ctrl[1] = 0.0
                    control_gamepad(self.j, tap_ctrl)
                else:
                    control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append("f")
                if control[1] > 0:
                    actions.append("b")
                if control[2] > 0.5:
                    actions.append("r")
                elif control[2] < -0.5:
                    actions.append("l")
                apply_control(actions)

    def grab_data_and_img(self):
        """
        Retrieves a screenshot and game state data.

        Returns:
            tuple: (data, img)
        """
        assert self.window_interface is not None and self.client is not None
        img = self.window_interface.screenshot()[:, :, :3]  # BGR ordering
        if self.resize_to is not None:
            img = cv2.resize(img, self.resize_to)
        img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self.grayscale else img[:, :, ::-1]
        )  # RGB convention
        data = self.client.retrieve_data()
        self.img = img
        return data, img

    def reset_race(self):
        """Resets the race via gamepad or keyboard."""
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        """Common reset logic including control reset and race restart."""
        if not self.initialized:
            self.initialize()
        if self.record_human:
            self.record_human = False
            self.send_control(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            self.record_human = True
        else:
            self.send_control(self.get_default_action())
        self.reset_race()
        time_sleep = (
            max(0, self.sleep_time_at_reset - 0.1)
            if self.gamepad
            else self.sleep_time_at_reset
        )
        time.sleep(time_sleep)

    def reset(self, seed=None, options=None):
        """
        Resets the environment and returns the initial observation.

        Returns:
            tuple: (observation, info)
        """
        self.reset_common()
        data, img = self.grab_data_and_img()
        self._speed_arr[0] = data[0]
        self._gear_arr[0] = data[9]
        self._rpm_arr[0] = data[10]
        self._last_speed_kmh = float(data[0])
        for _ in range(self.img_hist_len):
            self._push_img(img)
        imgs = self._get_img_hist_array()
        obs = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        assert self.reward_function is not None
        self.reward_function.reset()
        return obs, {}

    def close_finish_pop_up_tm20(self):
        """Closes the finish pop-up window in the game."""
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=bool(self.small_window))

    def wait(self):
        """Pauses the interface and waits for the next episode."""
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(1.0)
        self.reset_race()
        time.sleep(0.5)
        self.close_finish_pop_up_tm20()

    def get_obs_rew_terminated_info(self):
        """
        Retrieves the current observation, reward, and termination status.

        Returns:
            tuple: (observation, reward, terminated, info)
        """
        data, img = self.grab_data_and_img()
        assert self.client is not None and self.reward_function is not None
        _si, (_xi, _yi, _zi), _eoti = openplanet_grab_indices(self.client.nb_floats)
        self._speed_arr[0] = data[_si]
        self._last_speed_kmh = float(data[_si])
        if self.client.nb_floats >= 20:
            self._gear_arr[0] = data[18]
            self._rpm_arr[0] = data[10]
        else:
            self._gear_arr[0] = data[9]
            self._rpm_arr[0] = data[10]
        reward, terminated, _failure_counter, _ = self.reward_function.compute_reward(
            pos=np.array([data[_xi], data[_yi], data[_zi]])
        )
        reward_f = float(reward)
        self._push_img(img)
        imgs = self._get_img_hist_array()
        observation = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        end_of_track = bool(data[_eoti])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            terminated = True
            reward_f += self.finish_reward
            if self.save_replays:
                mouse_save_replay_tm20(True)
        reward_out = np.float32(reward_f)
        return observation, reward_out, terminated, info

    def get_observation_space(self) -> spaces.Tuple:
        """Returns the Gymnasium observation space."""
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        gear = spaces.Box(low=0.0, high=6, shape=(1,))
        rpm = spaces.Box(low=0.0, high=np.inf, shape=(1,))
        if self.resize_to is not None:
            w, h = self.resize_to
        else:
            w, h = self.window_width, self.window_height
        if self.grayscale:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w))
        else:
            img = spaces.Box(low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3))
        return spaces.Tuple((speed, gear, rpm, img))

    def get_action_space(self):
        """Returns the Gymnasium action space."""
        if self.discrete_action_table is not None:
            return spaces.Discrete(len(self.discrete_action_table))
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def get_default_action(self):
        """Returns the default action at episode start."""
        if self.discrete_action_table is not None:
            return np.array(0, dtype=np.int64)
        return np.array([0.0, 0.0, 0.0], dtype="float32")
