"""LIDAR-based TrackMania 2020 rtgym interfaces.

Single configurable class `TM2020InterfaceLidar` drives all LIDAR variants via two flags:

- ``include_progress``: add a scalar race progress in [0, 1] to the observation.
- ``include_camera_images``: add a resized camera image history for fusion models.

Thin aliases preserve the legacy names (and registry keys) referenced from
``tmrl.config.config_objects``:

- ``TM2020InterfaceLidarProgress``       - progress only (no camera images).
- ``TM2020InterfaceLidarProgressImages`` - progress + camera images.
"""

from __future__ import annotations

import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.telemetry_indices import TmrlDataPlugin
from tmrl.custom.interfaces.vision import TM2020Interface
from tmrl.custom.tm.utils.tools import Lidar


class TM2020InterfaceLidar(TM2020Interface):
    """
    LIDAR observations from a single game screenshot.

    Optionally appends a race-progress scalar and/or a parallel camera image history
    to the base observation.
    """

    def __init__(
        self,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
        include_progress: bool = False,
        include_camera_images: bool = False,
        grayscale: bool = True,
        resize_to: tuple | None = None,
        **kwargs,
    ):
        super().__init__(
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            save_replays=save_replays,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            **kwargs,
        )
        self._include_progress = include_progress
        self._include_camera_images = include_camera_images
        self._lidar_rgb_grayscale = grayscale
        self._lidar_rgb_resize = resize_to or (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
        self.lidar: Lidar | None = None
        self.image_hist: list = []

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        assert self.window_interface is not None
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def _grab_raw(self):
        assert self.window_interface is not None
        assert self.client is not None
        assert self.lidar is not None
        raw_img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([float(data[TmrlDataPlugin.SPEED_MPS]) * 3.6], dtype="float32")
        lidar = self.lidar.lidar_20(img=raw_img, show=False)
        return raw_img, data, speed, lidar

    def grab_lidar_speed_and_data(self):
        _, data, speed, lidar = self._grab_raw()
        return lidar, speed, data

    def grab_lidar_speed_data_and_image(self):
        raw_img, data, speed, lidar = self._grab_raw()
        w, h = self._lidar_rgb_resize
        img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
        if self._lidar_rgb_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) if img.ndim == 2 else np.transpose(img, (2, 0, 1))
        return lidar, speed, data, img

    def _progress_array(self) -> np.ndarray:
        assert self.reward_function is not None
        return np.array(
            [self.reward_function.cur_idx / max(1, self.reward_function.datalen)],
            dtype="float32",
        )

    def _assemble_obs(self, speed, lidars, images=None, progress=None):
        obs = [speed]
        if progress is not None:
            obs.append(progress)
        obs.append(lidars)
        if images is not None:
            obs.append(images)
        return obs

    def get_observation_space(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        lidars = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19))
        boxes = [speed]
        if self._include_progress:
            boxes.append(spaces.Box(low=0.0, high=1.0, shape=(1,)))
        boxes.append(lidars)
        if self._include_camera_images:
            c = 1 if self._lidar_rgb_grayscale else 3
            h, w = self._lidar_rgb_resize[1], self._lidar_rgb_resize[0]
            boxes.append(spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, c, h, w)))
        return spaces.Tuple(tuple(boxes))

    def reset(self, seed=None, options=None):
        self.reset_common()
        assert self.reward_function is not None
        if self._include_camera_images:
            lidar, speed, _data, img = self.grab_lidar_speed_data_and_image()
            self.img_hist = [lidar for _ in range(self.img_hist_len)]
            self.image_hist = [img for _ in range(self.img_hist_len)]
            lidars = np.array(list(self.img_hist), dtype="float32")
            images = np.array(list(self.image_hist), dtype="float32")
        else:
            lidar, speed, _data = self.grab_lidar_speed_and_data()
            self.img_hist = [lidar for _ in range(self.img_hist_len)]
            lidars = np.array(list(self.img_hist), dtype="float32")
            images = None

        progress = np.array([0], dtype="float32") if self._include_progress else None
        obs = self._assemble_obs(speed, lidars, images, progress)
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        assert self.reward_function is not None
        if self._include_camera_images:
            lidar, speed, data, img = self.grab_lidar_speed_data_and_image()
        else:
            lidar, speed, data = self.grab_lidar_speed_and_data()
            img = None

        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array(
                [data[TmrlDataPlugin.POS_X], data[TmrlDataPlugin.POS_Y], data[TmrlDataPlugin.POS_Z]]
            )
        )[:3]

        self.img_hist.append(lidar)
        self.img_hist = self.img_hist[-self.img_hist_len :]
        lidars = np.array(list(self.img_hist), dtype="float32")

        images = None
        if self._include_camera_images and img is not None:
            self.image_hist.append(img)
            self.image_hist = self.image_hist[-self.img_hist_len :]
            images = np.array(list(self.image_hist), dtype="float32")

        progress = self._progress_array() if self._include_progress else None
        obs = self._assemble_obs(speed, lidars, images, progress)

        end_of_track = bool(data[TmrlDataPlugin.FINISH_UI_ACTIVE])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        return obs, np.float32(rew), terminated, info


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):
    """LIDAR with race-progress scalar (no camera images).

    The ``include_progress`` and ``include_camera_images`` flags are forced by the
    subclass and cannot be overridden; pass them to ``TM2020InterfaceLidar`` directly
    if you need a different combination.
    """

    def __init__(self, **kwargs):
        kwargs.pop("include_progress", None)
        kwargs.pop("include_camera_images", None)
        super().__init__(include_progress=True, include_camera_images=False, **kwargs)


class TM2020InterfaceLidarProgressImages(TM2020InterfaceLidar):
    """LIDAR + progress + camera images from the same screenshot (fusion models).

    The ``include_progress`` and ``include_camera_images`` flags are forced by the
    subclass and cannot be overridden; pass them to ``TM2020InterfaceLidar`` directly
    if you need a different combination.
    """

    def __init__(self, img_hist_len: int = 4, **kwargs):
        kwargs.pop("include_progress", None)
        kwargs.pop("include_camera_images", None)
        super().__init__(
            img_hist_len=img_hist_len,
            include_progress=True,
            include_camera_images=True,
            **kwargs,
        )
