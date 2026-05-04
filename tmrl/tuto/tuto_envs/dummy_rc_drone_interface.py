"""
RealTimeGym interface used for the TMRL library tutorial.

This environment simulates a dummy RC drone evolving in a bounded 2D world.
It features random delays in control and observation capture.
"""

from __future__ import annotations

from threading import Thread
from typing import Any

import cv2
import gymnasium.spaces as spaces
import numpy as np
from rtgym import DEFAULT_CONFIG_DICT, DummyRCDrone, RealTimeGymInterface


class DummyRCDroneInterface(RealTimeGymInterface):
    def __init__(self):
        self.rc_drone: DummyRCDrone | None = None
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.initialized = False
        self.blank_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        self.rendering_thread = Thread(
            target=self._rendering_thread, args=(), kwargs={}, daemon=True
        )

    def _rendering_thread(self):
        from time import sleep

        while True:
            sleep(0.1)
            self.render()

    def get_observation_space(self):
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        return spaces.Tuple((pos_x_space, pos_y_space, tar_x_space, tar_y_space))

    def get_action_space(self):
        return spaces.Box(low=-2.0, high=2.0, shape=(2,))

    def get_default_action(self):
        return np.array([0.0, 0.0], dtype="float32")

    def send_control(self, control):
        assert self.rc_drone is not None
        vel_x = control[0]
        vel_y = control[1]
        self.rc_drone.send_control(vel_x, vel_y)

    def reset(self, seed=None, options=None):
        if not self.initialized:
            self.rc_drone = DummyRCDrone()
            self.rendering_thread.start()
            self.initialized = True
        assert self.rc_drone is not None
        pos_x, pos_y = self.rc_drone.get_observation()
        self.target[0] = np.random.uniform(-0.5, 0.5)
        self.target[1] = np.random.uniform(-0.5, 0.5)
        return [
            np.array([pos_x], dtype="float32"),
            np.array([pos_y], dtype="float32"),
            np.array([self.target[0]], dtype="float32"),
            np.array([self.target[1]], dtype="float32"),
        ], {}

    def get_obs_rew_terminated_info(self):
        assert self.rc_drone is not None
        pos_x, pos_y = self.rc_drone.get_observation()
        tar_x = self.target[0]
        tar_y = self.target[1]
        obs = [
            np.array([pos_x], dtype="float32"),
            np.array([pos_y], dtype="float32"),
            np.array([tar_x], dtype="float32"),
            np.array([tar_y], dtype="float32"),
        ]
        rew = -np.linalg.norm(np.array([pos_x, pos_y], dtype=np.float32) - self.target)
        terminated = rew > -0.01
        info: dict[str, Any] = {}
        return obs, rew, terminated, info

    def wait(self):
        pass

    def render(self):
        assert self.rc_drone is not None
        image = self.blank_image.copy()
        pos_x, pos_y = self.rc_drone.get_observation()
        image = np.asarray(
            cv2.circle(
                img=image,
                center=(int(float(pos_x) * 200) + 250, int(float(pos_y) * 200) + 250),
                radius=10,
                color=(255, 0, 0),
                thickness=1,
            ),
            dtype=np.uint8,
        )
        image = np.asarray(
            cv2.circle(
                img=image,
                center=(
                    int(float(self.target[0]) * 200) + 250,
                    int(float(self.target[1]) * 200) + 250,
                ),
                radius=5,
                color=(0, 0, 255),
                thickness=-1,
            ),
            dtype=np.uint8,
        )
        cv2.imshow("Dummy RC drone", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


# rtgym configuration dictionary:

DUMMY_RC_DRONE_CONFIG = DEFAULT_CONFIG_DICT.copy()
DUMMY_RC_DRONE_CONFIG["interface"] = DummyRCDroneInterface
DUMMY_RC_DRONE_CONFIG["time_step_duration"] = 0.05
DUMMY_RC_DRONE_CONFIG["start_obs_capture"] = 0.05
DUMMY_RC_DRONE_CONFIG["time_step_timeout_factor"] = 1.0
DUMMY_RC_DRONE_CONFIG["ep_max_length"] = 100
DUMMY_RC_DRONE_CONFIG["act_buf_len"] = 4
DUMMY_RC_DRONE_CONFIG["reset_act_buf"] = False
DUMMY_RC_DRONE_CONFIG["benchmark"] = True
DUMMY_RC_DRONE_CONFIG["benchmark_polyak"] = 0.2
