"""Unified TrackMania 2020 RL interface (TMRL_GrabData 33-float telemetry).

A single :class:`TM2020RLInterface` replaces the former Sophy / TQC / IMPALA interface
classes. Telemetry follows the OpenPlanet TMRL_GrabData wire layout described in
:mod:`tmrl.custom.interfaces.telemetry_indices` (33 floats, port 9000).

Constructor flags (also driven from ``tmrl.config`` in ``config_objects``):

- ``include_camera_images`` — append a resized game screenshot history (TQCGRAB-style).
- ``include_lidar`` — append screen-derived LIDAR vectors after the telemetry tuple
  (same 19-ray encoding as :class:`tmrl.custom.interfaces.lidar.TM2020InterfaceLidar`).
"""

from __future__ import annotations

import cv2
import numpy as np
from gymnasium import spaces
from loguru import logger

import tmrl.config as cfg
from tmrl.custom.interfaces.telemetry_indices import (
    TMRL_GRABDATA_FLOAT_COUNT,
    TmrlDataPlugin,
    tmrl_grabdata_payload_nb_floats,
    yaw_pitch_from_dir_xyz,
)
from tmrl.custom.interfaces.vision import TM2020Interface
from tmrl.custom.tm.utils.control_mouse import mouse_save_replay_tm20 as _util_save_replay
from tmrl.custom.tm.utils.tools import Lidar, TM2020OpenPlanetClient

_DEFAULT_MIN_STEPS_END_OF_TRACK = 50


class TM2020RLInterface(TM2020Interface):
    """
    Single RL interface: 33-float GrabData + interpolated track from the reward function,
    optional camera history, optional LIDAR tail.
    """

    def __init__(
        self,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = 160,
        record: bool = False,
        save_replay: bool = False,
        save_replays: bool | None = None,
        grayscale: bool = False,
        resize_to: tuple = (128, 64),
        finish_reward=cfg.END_OF_TRACK_REWARD,
        constant_penalty: float = 0.05,
        crash_penalty=cfg.CRASH_PENALTY,
        checkpoint_reward=cfg.CHECKPOINT_REWARD,
        lap_reward=cfg.LAP_REWARD,
        record_human: bool = False,
        include_camera_images: bool = False,
        include_lidar: bool = False,
        **kwargs,
    ):
        if save_replays is not None:
            save_replay = save_replays
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replay,
            grayscale=grayscale,
            finish_reward=finish_reward,
            resize_to=resize_to,
            constant_penalty=constant_penalty,
            crash_penalty=crash_penalty,
            record_human=record_human,
            **kwargs,
        )
        self.record = record
        self.window_interface = None
        self.cur_lap = 0
        self.cur_checkpoint = 0
        self.lap_reward = lap_reward
        self.checkpoint_reward = checkpoint_reward
        self.points_number = cfg.POINTS_NUMBER
        self.include_camera_images = include_camera_images
        self.include_lidar = include_lidar
        self._prev_speed_for_kinematics: float = 0.0
        self._prev_acc_for_kinematics: float = 0.0
        self._steps_since_reset = 0
        self.lidar: Lidar | None = None
        self._lidar_hist: list[np.ndarray] = []

        _rf = self.reward_function
        if (
            _rf is not None
            and float(getattr(_rf, "_point_spacing_m", 0) or 0) > 0
            and getattr(_rf, "_points_number", None) is not None
        ):
            n_rf_any = getattr(_rf, "_points_number", None)
            n_rf = self.points_number if n_rf_any is None else int(n_rf_any)
            if n_rf != self.points_number:
                logger.warning(
                    "POINTS_NUMBER from constants ({}) != reward spacing lookahead ({}); "
                    "using reward value for observation space.",
                    self.points_number,
                    n_rf,
                )
            self.points_number = n_rf

        self._lidar_rgb_grayscale = grayscale
        self._lidar_rgb_resize = (
            resize_to if resize_to is not None else (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
        )

    def _build_openplanet_client(self):
        return TM2020OpenPlanetClient(
            port=9000, nb_floats=tmrl_grabdata_payload_nb_floats(cfg.REWARD_CONFIG)
        )

    def initialize(self):
        self.initialize_common()
        self.small_window = not self.include_lidar
        if self.include_lidar:
            assert self.window_interface is not None
            self.lidar = Lidar(self.window_interface.screenshot())
            self._lidar_hist = []
        else:
            self.lidar = None
            self._lidar_hist = []
        self.initialized = True

    def get_observation_space(self):
        from tmrl.custom.tm.tqc_observation_space import build_tqc_sophy_tuple_observation_space

        base_spaces = build_tqc_sophy_tuple_observation_space(self.points_number)
        if not self.include_camera_images and not self.include_lidar:
            return base_spaces
        spaces_list = list(base_spaces.spaces)
        if self.include_lidar:
            spaces_list.append(
                spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19), dtype=np.float32)
            )
        if self.include_camera_images:
            w, h = self._lidar_rgb_resize
            if self._lidar_rgb_grayscale:
                spaces_list.append(
                    spaces.Box(
                        low=0.0, high=255.0, shape=(self.img_hist_len, h, w), dtype=np.float32
                    )
                )
            else:
                spaces_list.append(
                    spaces.Box(
                        low=0.0, high=255.0, shape=(self.img_hist_len, h, w, 3), dtype=np.float32
                    )
                )
        return spaces.Tuple(tuple(spaces_list))

    def grab_data(self):
        return self.client.retrieve_data()

    def _capture_and_process_image(self, raw_bgr: np.ndarray | None = None):
        assert self.window_interface is not None
        img = raw_bgr if raw_bgr is not None else self.window_interface.screenshot()[:, :, :3]
        w, h = self._lidar_rgb_resize
        img = cv2.resize(img, (w, h))
        img = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if self._lidar_rgb_grayscale else img[:, :, ::-1]
        )
        return img.astype(np.float32)

    def _track_observation(self, pos, yaw):
        track_result = self.reward_function.get_track_info(pos, self.points_number)
        left_track = track_result[0]
        center_track = track_result[1]
        right_track = track_result[2]
        curvature_list = track_result[3] if len(track_result) == 4 else None

        if bool(cfg.REWARD_CONFIG.get("TRACK_LOCAL_FRAME", False)):
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)

            def _rotate(pairs):
                rotated = []
                for j in range(0, len(pairs), 2):
                    dx, dz = pairs[j], pairs[j + 1]
                    rotated.append(dx * cos_y - dz * sin_y)
                    rotated.append(dx * sin_y + dz * cos_y)
                return rotated

            left_track = _rotate(left_track)
            center_track = _rotate(center_track)
            right_track = _rotate(right_track)

        track_list = left_track + center_track + right_track
        if cfg.OBS_TRACK_SCALE != 1.0:
            track_list = [x / cfg.OBS_TRACK_SCALE for x in track_list]
        return np.array(track_list, dtype="float32"), curvature_list

    def _append_lidar_obs(self, total_obs: list, raw_bgr: np.ndarray | None = None) -> None:
        assert self.lidar is not None
        assert self.window_interface is not None
        img = raw_bgr if raw_bgr is not None else self.window_interface.screenshot()[:, :, :3]
        lidar_vec = self.lidar.lidar_20(img=img, show=False)
        self._lidar_hist.append(lidar_vec)
        self._lidar_hist = self._lidar_hist[-self.img_hist_len :]
        lidars = np.array(list(self._lidar_hist), dtype=np.float32)
        total_obs.append(lidars)

    def get_obs_rew_terminated_info(self):
        data = self.grab_data()
        if len(data) < TMRL_GRABDATA_FLOAT_COUNT:
            raise ValueError(
                f"Expected {TMRL_GRABDATA_FLOAT_COUNT}-float TMRL_GrabData payload, "
                f"got len={len(data)}"
            )

        cur_cp = int(data[TmrlDataPlugin.CHECKPOINTS_PASSED])
        cur_lap = int(data[TmrlDataPlugin.CURRENT_LAP])
        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])

        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        speed = np.array([speed_kmh / cfg.OBS_SPEED_SCALE], dtype="float32")
        pos = np.array(
            [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]],
            dtype="float32",
        )
        velocity_xyz = np.array(
            [data[TmrlDataPlugin.VEL_X], data[TmrlDataPlugin.VEL_Y], data[TmrlDataPlugin.VEL_Z]],
            dtype="float32",
        )
        dir_xyz = np.array(
            [data[TmrlDataPlugin.DIR_X], data[TmrlDataPlugin.DIR_Y], data[TmrlDataPlugin.DIR_Z]],
            dtype="float32",
        )
        yaw_val, pitch_val = yaw_pitch_from_dir_xyz(dir_xyz)
        aim_yaw = np.array([yaw_val], dtype="float32")
        aim_pitch = np.array([pitch_val], dtype="float32")
        input_steer = np.array([data[TmrlDataPlugin.INPUT_STEER]], dtype="float32")
        input_gas_pedal = np.array([data[TmrlDataPlugin.INPUT_GAS]], dtype="float32")
        input_brake = np.array([data[TmrlDataPlugin.INPUT_BRAKE]], dtype="float32")

        acceleration_val = speed_kmh - self._prev_speed_for_kinematics
        jerk_val = acceleration_val - self._prev_acc_for_kinematics
        self._prev_speed_for_kinematics = speed_kmh
        self._prev_acc_for_kinematics = acceleration_val
        self._sync_crash_state()
        self.crash_fallback(current_speed=speed_kmh, jerk=jerk_val)
        crashed_this_step = bool(self.is_crashed)
        acceleration = np.array([acceleration_val], dtype="float32")
        jerk = np.array([jerk_val], dtype="float32")

        steer_angle = np.array([0.0, 0.0], dtype="float32")
        slip_coef = np.array(
            [data[TmrlDataPlugin.SLIP_FL], data[TmrlDataPlugin.SLIP_FR]], dtype="float32"
        )
        gear = np.array([data[TmrlDataPlugin.ENGINE_GEAR] / 5.0], dtype="float32")
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

        rew, terminated, failure_counter, reward_sum = self.reward_function.compute_reward(
            pos=pos,
            velocity_xyz=velocity_xyz,
            dir_xyz=dir_xyz,
            surface_materials=surface_materials,
            wheel_slips=wheel_slips,
            crashed=bool(self.is_crashed),
            speed=speed_kmh,
            next_cp=self.cur_checkpoint < cur_cp,
            next_lap=self.cur_lap < cur_lap,
            end_of_track=end_of_track,
            input_brake=float(data[TmrlDataPlugin.INPUT_BRAKE]),
            aim_yaw=float(yaw_val),
            input_steer=float(data[TmrlDataPlugin.INPUT_STEER]),
            gear=float(data[TmrlDataPlugin.ENGINE_GEAR]),
            slip_angle_deg=None,
        )
        track_yaw = yaw_val

        self._dbg_last_step = {
            "terminated": bool(terminated),
            "end_of_track": bool(end_of_track),
            "speed_kmh": float(speed_kmh),
            "reward_sum": float(reward_sum),
            "step_counter": int(getattr(self.reward_function, "step_counter", -1)),
        }

        race_progress = self.reward_function.compute_race_progress()

        self._steps_since_reset = getattr(self, "_steps_since_reset", 0) + 1
        min_steps_before_finish = max(
            _DEFAULT_MIN_STEPS_END_OF_TRACK,
            cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if end_of_track and self._steps_since_reset >= min_steps_before_finish:
            terminated = True
            failure_counter = 0.0
            if self.save_replays:
                _util_save_replay(True)

        self.reward_function.log_model_run(terminated=terminated, end_of_track=end_of_track)

        track_info_arr, curvature_list = self._track_observation(pos, track_yaw)

        self.cooldown_control()

        race_progress = np.array([race_progress], dtype="float32")
        max_count = max(1.0, getattr(self.reward_function, "_max_no_progress_steps", 200.0))
        failure_counter = np.array([float(failure_counter) / max_count], dtype="float32")
        info = {
            "reward_sum": reward_sum,
            "end_of_track": bool(end_of_track),
            "crashed": crashed_this_step,
            "crash_penalty": float(self.crash_penalty),
        }
        if getattr(self.client, "_last_retrieve_invalid", False):
            terminated = True
            info["telemetry_invalid"] = True
        if getattr(self.client, "_last_retrieve_position_patched", False):
            info["position_patched"] = True

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter,
        ]
        if curvature_list is not None:
            curv = np.clip(np.array(curvature_list, dtype="float32") * 10.0, -1.0, 1.0)
            observation.append(curv)
        total_obs = [track_info_arr, *observation]

        self.cur_checkpoint = cur_cp
        self.cur_lap = cur_lap

        min_guaranteed = int(cfg.REWARD_CONFIG.get("MIN_EPISODE_LENGTH_GUARANTEED", 100))
        min_length = max(
            min_guaranteed,
            2 * cfg.REWARD_CONFIG.get("MIN_STEPS", _DEFAULT_MIN_STEPS_END_OF_TRACK),
        )
        if self._steps_since_reset < min_length:
            terminated = False

        raw_bgr = None
        if self.include_lidar or self.include_camera_images:
            raw_bgr = self.window_interface.screenshot()[:, :, :3]

        if self.include_lidar:
            self._append_lidar_obs(total_obs, raw_bgr=raw_bgr)

        if self.include_camera_images:
            assert raw_bgr is not None
            img = self._capture_and_process_image(raw_bgr=raw_bgr)
            self._push_img(img)
            total_obs.append(self._get_img_hist_array())
        return total_obs, np.float32(rew), terminated, info

    def reset(self, seed=None, options=None):
        if (
            getattr(self, "reward_function", None) is not None
            and getattr(self.reward_function, "step_counter", 0) > 0
            and not getattr(self.reward_function, "_logged_run_this_episode", False)
        ):
            self.reward_function.log_model_run(terminated=True, end_of_track=False)
        self.reset_common()
        self._steps_since_reset = 0
        data = self.grab_data()
        if len(data) < TMRL_GRABDATA_FLOAT_COUNT:
            raise ValueError(
                f"Expected {TMRL_GRABDATA_FLOAT_COUNT}-float TMRL_GrabData payload, "
                f"got len={len(data)}"
            )

        self.cur_lap = 0
        self.cur_checkpoint = 0

        speed_kmh = float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6
        speed = np.array([speed_kmh / cfg.OBS_SPEED_SCALE], dtype="float32")
        pos = np.array(
            [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]],
            dtype="float32",
        )
        input_steer = np.array([data[TmrlDataPlugin.INPUT_STEER]], dtype="float32")
        input_gas_pedal = np.array([data[TmrlDataPlugin.INPUT_GAS]], dtype="float32")
        input_brake = np.array([data[TmrlDataPlugin.INPUT_BRAKE]], dtype="float32")
        acceleration = np.array([0.0], dtype="float32")
        jerk = np.array([0.0], dtype="float32")
        dir_xyz = np.array(
            [data[TmrlDataPlugin.DIR_X], data[TmrlDataPlugin.DIR_Y], data[TmrlDataPlugin.DIR_Z]],
            dtype="float32",
        )
        yaw_val, pitch_val = yaw_pitch_from_dir_xyz(dir_xyz)
        aim_yaw = np.array([yaw_val], dtype="float32")
        aim_pitch = np.array([pitch_val], dtype="float32")
        steer_angle = np.array([0.0, 0.0], dtype="float32")
        slip_coef = np.array(
            [data[TmrlDataPlugin.SLIP_FL], data[TmrlDataPlugin.SLIP_FR]], dtype="float32"
        )
        gear = np.array([data[TmrlDataPlugin.ENGINE_GEAR] / 5.0], dtype="float32")
        track_yaw = yaw_val
        self._prev_speed_for_kinematics = speed_kmh
        self._prev_acc_for_kinematics = 0.0

        failure_counter = np.array([0.0], dtype="float32")
        race_progress = np.array([0.0], dtype="float32")

        track_info_arr, curvature_list = self._track_observation(pos, track_yaw)

        observation = [
            speed,
            acceleration,
            jerk,
            race_progress,
            input_steer,
            input_gas_pedal,
            input_brake,
            gear,
            aim_yaw,
            aim_pitch,
            steer_angle,
            slip_coef,
            failure_counter,
        ]
        if curvature_list is not None:
            curv = np.clip(np.array(curvature_list, dtype="float32") * 10.0, -1.0, 1.0)
            observation.append(curv)
        total_obs = [track_info_arr, *observation]

        self.reward_function.reset()
        info = {"reward_sum": 0.0}

        raw_bgr = None
        if self.include_lidar or self.include_camera_images:
            raw_bgr = self.window_interface.screenshot()[:, :, :3]

        if self.include_lidar:
            assert self.lidar is not None
            img0 = raw_bgr if raw_bgr is not None else self.window_interface.screenshot()[:, :, :3]
            z = self.lidar.lidar_20(img=img0, show=False)
            self._lidar_hist = [
                np.asarray(z, dtype=np.float32).copy() for _ in range(self.img_hist_len)
            ]
            total_obs.append(np.array(self._lidar_hist, dtype=np.float32))

        if self.include_camera_images:
            img = self._capture_and_process_image(raw_bgr=raw_bgr)
            for _ in range(self.img_hist_len):
                self._push_img(img)
            total_obs.append(self._get_img_hist_array())
        return total_obs, info
