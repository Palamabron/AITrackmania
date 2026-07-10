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
from tmrl.custom.interfaces.base import (
    MPS_TO_KMPH,
    TrackMania2020InterfaceBase,
    apply_episode_length_guards,
    gate_end_of_track_for_reward,
)
from tmrl.custom.interfaces.telemetry_indices import (
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
    yaw_pitch_from_dir_xyz,
)
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control.discrete import build_brake_tap_action_table
from tmrl.custom.tm.utils.control.mouse import mouse_save_replay_tm20
from tmrl.custom.tm.utils.openplanet_client import TM2020OpenPlanetClient
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
        """Initialize instance variables; deferred hardware setup runs in ``initialize()``.

        Args:
            img_hist_len: Number of consecutive frames to stack in the image observation.
            gamepad: Use virtual Xbox 360 gamepad (vgamepad) when True; keyboard otherwise.
            save_replays: Save a ghost replay to disk after each completed lap.
            grayscale: Convert screenshots to single-channel grayscale when True.
            resize_to: ``(width, height)`` in pixels for screenshot downsampling.
                ``None`` uses ``cfg.WINDOW_WIDTH`` x ``cfg.WINDOW_HEIGHT``.
            finish_reward: Scalar bonus added when the finish UI activates.
                Defaults to ``cfg.END_OF_TRACK_REWARD`` when ``None``.
            constant_penalty: Per-step penalty subtracted from reward (km/h units
                of time cost). Defaults to ``cfg.REWARD_CONFIG['CONSTANT_PENALTY']``.
            crash_penalty: Scalar penalty applied on crash detection. Defaults to
                ``cfg.CRASH_PENALTY`` when ``None``.
            min_nb_steps_before_failure: Minimum steps without progress before the
                episode is terminated. Defaults to 70 when ``None``.
            record_human: When True, ``send_control`` is a no-op (human drives).
            discrete_n_steer_bins: Number of steering bins for the discrete action
                space. 0 disables discrete actions (continuous Box space used).
            **kwargs: Forwarded to the rtgym ``RealTimeGymInterface`` base.
        """
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
        # Legacy ctor arg; finish bonuses come from RewardFunction (reward.end_of_track_reward).
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
        self._steps_since_reset = 0
        self._iface_prev_speed_kmh: float = 0.0
        self._iface_prev_acc_kmh: float = 0.0

    def initialize(self):
        """Run ``initialize_common()`` and mark the interface as ready."""
        self.initialize_common()
        self.small_window = True
        self.initialized = True

    def grab_data_and_img(self):
        """Capture a screenshot and retrieve the current telemetry frame atomically.

        Returns:
            tuple: ``(data, img)`` where ``data`` is the raw telemetry float tuple
                from the OpenPlanet client and ``img`` is a uint8 NumPy array of
                shape ``(H, W)`` (grayscale) or ``(H, W, 3)`` (RGB) after
                optional resize and color conversion.
        """
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
        """Update the preallocated speed/gear/rpm scalar arrays from a telemetry frame.

        Converts speed from m/s to km/h and caches the result in ``_last_speed_kmh``
        for crash detection in the next step.

        Args:
            data: Raw float tuple from ``TM2020OpenPlanetClient.retrieve_data()``.
        """
        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * MPS_TO_KMPH
        self._speed_arr[0] = speed_kmh
        self._gear_arr[0] = data[TmrlDataPlugin.ENGINE_GEAR]
        self._rpm_arr[0] = data[TmrlDataPlugin.ENGINE_RPM]
        self._last_speed_kmh = speed_kmh

    def reset(self, seed=None, options=None):
        """Reset the environment and return the initial observation.

        Flushes any unlogged episode metrics, triggers ``reset_common()``,
        grabs the first telemetry frame and screenshot, pre-fills the image
        history buffer with copies of the initial frame, and resets kinematic
        state.

        Args:
            seed: Unused; accepted for gymnasium API compatibility.
            options: Unused; accepted for gymnasium API compatibility.

        Returns:
            tuple: ``(observation, info)`` where ``observation`` is
                ``[speed (km/h), gear, rpm (float32 arrays), image_history]``
                and ``info`` is an empty dict.
        """
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
        self._steps_since_reset = 0
        self._iface_prev_speed_kmh = 0.0
        self._iface_prev_acc_kmh = 0.0
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """Step the interface: capture telemetry + image, compute reward, return SARS tuple.

        Returns:
            tuple: ``(observation, reward, terminated, info)`` where:
                - ``observation`` = ``[speed, gear, rpm, image_history]`` (float32 arrays).
                - ``reward`` is float32.
                - ``terminated`` is True when the reward function signals failure or
                  finish (subject to minimum-episode-length guards).
                - ``info`` contains ``end_of_track``, ``reward_sum``, ``crashed``,
                  and ``crash_penalty``.
        """
        assert self.reward_function is not None
        data, img = self.grab_data_and_img()
        self._update_telemetry_arrays(data)
        speed_kmh = self._last_speed_kmh
        acc_val = speed_kmh - self._iface_prev_speed_kmh
        jerk_val = acc_val - self._iface_prev_acc_kmh
        self._iface_prev_speed_kmh = speed_kmh
        self._iface_prev_acc_kmh = acc_val
        self._sync_crash_state()
        self.crash_fallback(current_speed=speed_kmh, jerk=jerk_val)

        dir_xyz_t = (
            float(data[TmrlDataPlugin.DIR_X]),
            float(data[TmrlDataPlugin.DIR_Y]),
            float(data[TmrlDataPlugin.DIR_Z]),
        )
        yaw_val, _ = yaw_pitch_from_dir_xyz(dir_xyz_t)
        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])
        end_of_track_for_reward = gate_end_of_track_for_reward(
            self._steps_since_reset + 1, end_of_track
        )

        wheel_slips = [
            float(data[TmrlDataPlugin.SLIP_FL]),
            float(data[TmrlDataPlugin.SLIP_FR]),
            float(data[TmrlDataPlugin.SLIP_RL]),
            float(data[TmrlDataPlugin.SLIP_RR]),
        ]
        surface_materials = [
            int(data[TmrlDataPlugin.MAT_FL]),
            int(data[TmrlDataPlugin.MAT_FR]),
            int(data[TmrlDataPlugin.MAT_RL]),
            int(data[TmrlDataPlugin.MAT_RR]),
        ]

        reward, terminated, _failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=np.array(
                [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]]
            ),
            velocity_xyz=(
                float(data[TmrlDataPlugin.VEL_X]),
                float(data[TmrlDataPlugin.VEL_Y]),
                float(data[TmrlDataPlugin.VEL_Z]),
            ),
            dir_xyz=dir_xyz_t,
            surface_materials=surface_materials,
            wheel_slips=wheel_slips,
            crashed=bool(self.is_crashed),
            speed=speed_kmh,
            end_of_track=end_of_track_for_reward,
            input_brake=float(data[TmrlDataPlugin.INPUT_BRAKE]),
            aim_yaw=float(yaw_val),
            input_steer=float(data[TmrlDataPlugin.INPUT_STEER]),
            gear=float(data[TmrlDataPlugin.ENGINE_GEAR]),
            slip_angle_deg=None,
        )

        self._steps_since_reset += 1
        terminated, eot_accepted = apply_episode_length_guards(
            self._steps_since_reset,
            end_of_track_for_reward,
            terminated,
        )
        if eot_accepted and self.save_replays:
            mouse_save_replay_tm20(True)

        self._push_img(img)
        imgs = self._get_img_hist_array()
        observation = [self._speed_arr.copy(), self._gear_arr.copy(), self._rpm_arr.copy(), imgs]
        info = {
            "end_of_track": end_of_track,
            "reward_sum": reward_sum,
            "crashed": bool(self.is_crashed),
            "crash_penalty": float(self.crash_penalty),
        }

        self.cooldown_control()
        self.reward_function.log_model_run(
            terminated=bool(terminated), end_of_track=end_of_track_for_reward
        )
        reward_out = np.float32(reward)
        return observation, reward_out, terminated, info

    def get_observation_space(self) -> spaces.Tuple:
        """Return the gymnasium Tuple observation space for this interface.

        Returns:
            spaces.Tuple: ``(speed [0, 1000] km/h, gear [0, 6], rpm [0, ∞],
                image_history [0, 255])``. Image shape is
                ``(img_hist_len, H, W)`` for grayscale or ``(img_hist_len, H, W, 3)``
                for RGB, where H x W matches ``resize_to``.
        """
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
        """Return the action space: Discrete when a table is set, else Box[-1, 1]^3.

        Returns:
            spaces.Discrete or spaces.Box: Discrete over the brake-tap action table
                when ``discrete_n_steer_bins > 0``, otherwise a continuous
                ``Box(low=-1, high=1, shape=(3,))`` for ``[gas, brake, steer]``.
        """
        if self.discrete_action_table is not None:
            return spaces.Discrete(len(self.discrete_action_table))
        return spaces.Box(low=-1.0, high=1.0, shape=(3,))

    def get_default_action(self):
        """Return the do-nothing action (index 0 for discrete, zero vector for continuous).

        Returns:
            np.ndarray: Scalar int64 index 0 for discrete action space, or
                float32 array ``[0., 0., 0.]`` for continuous.
        """
        if self.discrete_action_table is not None:
            return np.array(0, dtype=np.int64)
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
