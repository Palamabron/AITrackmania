"""Shared TrackMania 2020 rtgym interface mechanics and OpenPlanet client hook."""

from __future__ import annotations

import platform
import threading
import time
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from loguru import logger
from rtgym import RealTimeGymInterface

import tmrl.config as cfg
import tmrl.config.loader as loader
import tmrl.config.paths as cfg_paths
from tmrl.custom.tm.utils.auto_drift import compute_drift_steer, is_auto_drift_action
from tmrl.custom.tm.utils.compute_reward import RewardFunction
from tmrl.custom.tm.utils.control_gamepad import (
    control_gamepad,
    gamepad_close_finish_pop_up_tm20,
    gamepad_reset,
)
from tmrl.custom.tm.utils.control_keyboard import apply_control, keyres
from tmrl.custom.tm.utils.control_mouse import mouse_close_finish_pop_up_tm20
from tmrl.custom.tm.utils.discrete_control import (
    BRAKE_TAP_DURATION_S,
    discrete_index_to_control,
    is_brake_tap,
)
from tmrl.custom.tm.utils.tools import TM2020OpenPlanetClient, save_ghost
from tmrl.custom.tm.utils.window import WindowInterface

MPS_TO_KMPH = 3.6

DEFAULT_MIN_STEPS_END_OF_TRACK = 50

RUMBLE_CRASH_THRESHOLD = 100
CRASH_BASE_SPEED_DROP_KMH = 20.0
CRASH_SPEED_DROP_FACTOR = 0.20
CRASH_JERK_THRESHOLD = -1.45
CRASH_COOLDOWN_STEPS = 10

REPLAY_SAVE_SLEEP_S = 1.0
POST_RACE_SLEEP_S = 0.5
KEYBOARD_STEER_DEADZONE = 0.5


def min_steps_before_finish() -> int:
    """Minimum step count before a finish UI signal counts as a valid finish."""
    return max(
        DEFAULT_MIN_STEPS_END_OF_TRACK,
        int(cfg.REWARD_CONFIG.get("MIN_STEPS", DEFAULT_MIN_STEPS_END_OF_TRACK)),
    )


def gate_end_of_track_for_reward(steps_since_reset: int, end_of_track: bool) -> bool:
    """Return True when finish UI is active and the step threshold is met."""
    return end_of_track and steps_since_reset >= min_steps_before_finish()


def apply_episode_length_guards(
    steps_since_reset: int,
    end_of_track_gated: bool,
    terminated: bool,
) -> tuple[bool, bool]:
    """Enforce minimum episode length after reward computation.

    *end_of_track_gated* must be the output of ``gate_end_of_track_for_reward``
    for the same step count (typically after ``_steps_since_reset += 1``).

    Returns ``(terminated, end_of_track_accepted)``.
    """
    min_guaranteed = int(cfg.REWARD_CONFIG.get("MIN_EPISODE_LENGTH_GUARANTEED", 100))
    min_length = max(
        min_guaranteed,
        2 * int(cfg.REWARD_CONFIG.get("MIN_STEPS", DEFAULT_MIN_STEPS_END_OF_TRACK)),
    )
    too_short = steps_since_reset < min_length
    end_of_track_accepted = end_of_track_gated and not too_short
    if end_of_track_accepted:
        terminated = True

    return terminated, end_of_track_accepted


class TrackMania2020InterfaceBase(RealTimeGymInterface, ABC):
    """Control, window init, reward wiring, image ring buffer, and OpenPlanet client creation."""

    client: TM2020OpenPlanetClient | None = None
    record_human: bool = False
    _send_control_logged: bool = False
    img_hist: deque[np.ndarray] | list[np.ndarray] | None = None
    reward_function: RewardFunction | None = None
    window_interface: WindowInterface | None = None

    # Image ring-buffer state. Initialized in ``initialize_common`` but declared here so
    # mypy can bind ``_img_buf`` at the point of use in ``_push_img`` /
    # ``_get_img_hist_array`` (the base class never writes to it in ``__init__``).
    _img_buf: np.ndarray | None = None
    _img_hist_count: int = 0
    _img_hist_cursor: int = 0

    @abstractmethod
    def reset(self, seed=None, options=None):  # pragma: no cover - abstract
        """Return ``(observation, info)`` and bring the interface to a step-ready state."""

    @abstractmethod
    def get_obs_rew_terminated_info(self):  # pragma: no cover - abstract
        """Return ``(observation, reward, terminated, info)`` for the current frame."""

    @abstractmethod
    def get_observation_space(self):  # pragma: no cover - abstract
        """Return the ``gymnasium.spaces.Tuple`` describing this interface's observation."""

    def _build_openplanet_client(self) -> TM2020OpenPlanetClient:
        """Override for alternate GrabData layouts (e.g. TQC 20-float)."""
        return TM2020OpenPlanetClient()

    def _push_img(self, img: np.ndarray) -> None:
        if self._img_buf is None or self._img_buf.shape[1:] != img.shape:
            self._img_buf = np.zeros((self.img_hist_len, *img.shape), dtype=img.dtype)
            self._img_hist_count = 0
            self._img_hist_cursor = 0
        buf = self._img_buf
        buf[self._img_hist_cursor] = img
        self._img_hist_cursor = (self._img_hist_cursor + 1) % self.img_hist_len
        if self._img_hist_count < self.img_hist_len:
            self._img_hist_count += 1

    def _get_img_hist_array(self) -> np.ndarray:
        buf = self._img_buf
        if buf is None or self._img_hist_count == 0:
            return np.zeros((self.img_hist_len, 1, 1), dtype=np.uint8)
        if self._img_hist_count < self.img_hist_len:
            res: np.ndarray = np.repeat(buf[:1], self.img_hist_len, axis=0)
            res[-self._img_hist_count :] = buf[: self._img_hist_count]
            return res
        if self._img_hist_cursor == 0:
            return buf.copy()
        idx = (
            np.arange(self.img_hist_len, dtype=np.int64) + self._img_hist_cursor
        ) % self.img_hist_len
        return np.asarray(buf[idx])

    def initialize_common(self):
        """Initializes the window interface, reward function, and game client."""
        self._crash_lock = threading.Lock()
        self._brake_tap_lock = threading.Lock()
        self._brake_tap_timer = None
        self._brake_tap_seq = 0
        self._async_rumble_event = False
        if self.gamepad:
            try:
                import vgamepad as vg

                self.j = vg.VX360Gamepad()
                self.j.register_notification(callback_function=self.crash_callback)
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
        self.window_interface.move_and_resize()
        self.last_time = time.time()
        self.img_hist = deque(maxlen=self.img_hist_len)
        self.img = None
        self._img_buf = None
        self._img_hist_count = 0
        self._img_hist_cursor = 0
        _mcfg = loader.MAIN_CONFIG
        wandb_cfg = getattr(_mcfg, "wandb", None)
        use_wandb = bool(getattr(wandb_cfg, "log_from_worker", True))
        self.reward_function = RewardFunction(
            reward_data_path=cfg.REWARD_PATH,
            nb_obs_forward=cfg.REWARD_CONFIG.get("CHECK_FORWARD", 500),
            nb_obs_backward=cfg.REWARD_CONFIG.get("CHECK_BACKWARD", 10),
            max_dist_from_traj=cfg.REWARD_CONFIG.get("MAX_STRAY", 50.0),
            crash_penalty=self.crash_penalty,
            constant_penalty=self.constant_penalty,
            require_track_boundary_pickles=cfg.USE_LIDAR_IMAGES,
            track_path_left=cfg_paths.TRACK_PATH_LEFT,
            track_path_right=cfg_paths.TRACK_PATH_RIGHT,
            reward_config=cfg.REWARD_CONFIG,
            time_step_duration=cfg.RTGYM_TIME_STEP_DURATION,
            points_distance=cfg.POINTS_DISTANCE,
            lap_cooldown=cfg.LAP_COOLDOWN,
            config_file_path=str(loader.LOCAL_OVERRIDE_PATH),
            use_wandb=use_wandb,
            wandb_project=str(_mcfg.wandb.project),
            wandb_entity=str(_mcfg.wandb.entity),
            wandb_run_id=str(_mcfg.run.name),
            wandb_api_key=str(_mcfg.wandb.api_key),
            wandb_config=loader.create_config(),
        )
        if self.client is None:
            self.client = self._build_openplanet_client()
        self.is_crashed = False
        self.crash_cooldown = 0
        self._last_speed_kmh = 0.0

    def crash_callback(self, client, target, large_motor, small_motor, led_number, user_data):
        """Callback for detecting crashes via gamepad vibration (thread-safe)."""
        if large_motor > RUMBLE_CRASH_THRESHOLD:
            with self._crash_lock:
                self._async_rumble_event = True

    def crash_fallback(self, current_speed, jerk):
        """Kinematic fallback for collision detection in case of gamepad rumble malfunction.

        Compares the velocity drop between the previous and current frame against a
        dynamic threshold (flat base drop + relative speed drop). A sharp negative jerk
        is required to distinguish sudden impacts from smooth deceleration.
        """
        last_speed = getattr(self, "_last_speed_kmh", current_speed)
        delta_v = last_speed - current_speed

        dynamic_threshold = CRASH_BASE_SPEED_DROP_KMH + (last_speed * CRASH_SPEED_DROP_FACTOR)

        if delta_v > dynamic_threshold and jerk <= CRASH_JERK_THRESHOLD:
            self.is_crashed = True
            self.crash_cooldown = CRASH_COOLDOWN_STEPS

    def _sync_crash_state(self):
        """Consume the latching flag set by the background gamepad telemetry thread.

        Acquires the crash lock, reads and clears the hardware rumble event,
        then updates the main crash state if not in cooldown.
        """
        with self._crash_lock:
            rumble_triggered = self._async_rumble_event
            self._async_rumble_event = False

        if rumble_triggered and self.crash_cooldown == 0:
            self.is_crashed = True
            self.crash_cooldown = CRASH_COOLDOWN_STEPS

    def cooldown_control(self):
        """Reset the single-frame crash impulse and tick the cooldown counter."""
        self.is_crashed = False
        if self.crash_cooldown > 0:
            self.crash_cooldown -= 1

    @staticmethod
    def get_speed_in_kmph(speed):
        """Convert speed from m/s to km/h."""
        return speed * MPS_TO_KMPH

    def _cancel_brake_tap_release_unlocked(self):
        """Cancel a pending brake-tap release timer (caller must hold ``_brake_tap_lock``)."""
        timer = self._brake_tap_timer
        if timer is not None:
            timer.cancel()
            self._brake_tap_timer = None
        self._brake_tap_seq += 1

    def _cancel_brake_tap_release(self):
        """Cancel a pending brake-tap release timer, if any."""
        with self._brake_tap_lock:
            self._cancel_brake_tap_release_unlocked()

    def _schedule_brake_tap_release_unlocked(self, release_ctrl):
        """Release brake after BRAKE_TAP_DURATION_S unless superseded (caller holds lock)."""
        seq = self._brake_tap_seq

        def _release():
            with self._brake_tap_lock:
                if self._brake_tap_seq != seq:
                    return
                j = self.j
                if j is None:
                    return
                control_gamepad(j, release_ctrl)
                self._brake_tap_timer = None

        timer = threading.Timer(BRAKE_TAP_DURATION_S, _release)
        timer.daemon = True
        self._brake_tap_timer = timer
        timer.start()

    def _schedule_brake_tap_release(self, release_ctrl):
        """Release brake after BRAKE_TAP_DURATION_S unless superseded."""
        with self._brake_tap_lock:
            self._schedule_brake_tap_release_unlocked(release_ctrl)

    def send_control(self, control):
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
                    # Press now, release via a background timer: a blocking
                    # sleep here would stall the rtgym control thread for
                    # BRAKE_TAP_DURATION_S (20% of a 50 ms control period).
                    tap_ctrl = control.copy()
                    tap_ctrl[1] = 1.0
                    release_ctrl = tap_ctrl.copy()
                    release_ctrl[1] = 0.0
                    with self._brake_tap_lock:
                        self._cancel_brake_tap_release_unlocked()
                        control_gamepad(self.j, tap_ctrl)
                        self._schedule_brake_tap_release_unlocked(release_ctrl)
                else:
                    # A newer control supersedes any pending tap release; letting
                    # the stale release fire would overwrite gas/steer state.
                    with self._brake_tap_lock:
                        self._cancel_brake_tap_release_unlocked()
                        control_gamepad(self.j, control)
        else:
            if control is not None:
                actions = []
                if control[0] > 0:
                    actions.append("f")
                if control[1] > 0:
                    actions.append("b")
                if control[2] > KEYBOARD_STEER_DEADZONE:
                    actions.append("r")
                elif control[2] < -KEYBOARD_STEER_DEADZONE:
                    actions.append("l")
                apply_control(actions)

    def reset_race(self):
        if self.gamepad:
            gamepad_reset(self.j)
        else:
            keyres()

    def reset_common(self):
        if not self.initialized:
            self.initialize()
        if self.record_human:
            self.record_human = False
            self.send_control(np.array([0.0, 0.0, 0.0], dtype=np.float32))
            self.record_human = True
        else:
            self.send_control(self.get_default_action())
        self.reset_race()
        self.is_crashed = False
        self.crash_cooldown = 0
        self._last_speed_kmh = 0.0
        with self._crash_lock:
            self._async_rumble_event = False
        time_sleep = (
            max(0, cfg.SLEEP_TIME_AT_RESET - 0.1) if self.gamepad else cfg.SLEEP_TIME_AT_RESET
        )
        time.sleep(time_sleep)

    def close_finish_pop_up_tm20(self):
        if self.gamepad:
            gamepad_close_finish_pop_up_tm20(self.j)
        else:
            mouse_close_finish_pop_up_tm20(small_window=self.small_window)

    def wait(self):
        self.send_control(self.get_default_action())
        if self.save_replays:
            save_ghost()
            time.sleep(REPLAY_SAVE_SLEEP_S)
        self.reset_race()
        time.sleep(POST_RACE_SLEEP_S)
        self.close_finish_pop_up_tm20()
