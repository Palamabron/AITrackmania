import cv2
import numpy as np
from gymnasium import spaces

import tmrl.config as cfg
from tmrl.custom.interfaces.TM2020Interface import TM2020Interface
from tmrl.custom.tm.utils.tools import Lidar


class TM2020InterfaceLidarConfigurable(TM2020Interface):
    """
    LIDAR observations from a single game screenshot, with optional race progress
    and optional parallel camera image history for fusion models.
    """

    def __init__(
        self,
        *,
        include_progress: bool,
        include_camera_images: bool,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
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

    def grab_lidar_speed_and_data(self):
        assert self.window_interface is not None
        assert self.client is not None
        assert self.lidar is not None
        img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        lidar = self.lidar.lidar_20(img=img, show=False)
        return lidar, speed, data

    def grab_lidar_speed_data_and_image(self):
        assert self.window_interface is not None
        assert self.client is not None
        assert self.lidar is not None
        raw_img = self.window_interface.screenshot()[:, :, :3]
        data = self.client.retrieve_data()
        speed = np.array([data[0]], dtype="float32")
        lidar = self.lidar.lidar_20(img=raw_img, show=False)
        w, h = self._lidar_rgb_resize
        img = cv2.resize(raw_img, (w, h), interpolation=cv2.INTER_AREA)
        if self._lidar_rgb_grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) if img.ndim == 2 else np.transpose(img, (2, 0, 1))
        return lidar, speed, data, img

    def initialize(self):
        super().initialize_common()
        self.small_window = False
        assert self.window_interface is not None
        self.lidar = Lidar(self.window_interface.screenshot())
        self.initialized = True

    def _observation_space_lidar_only(self):
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        lidars = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(self.img_hist_len, 19),
        )
        if not self._include_progress:
            return spaces.Tuple((speed, lidars))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        return spaces.Tuple((speed, progress, lidars))

    def get_observation_space(self):
        if not self._include_camera_images:
            return self._observation_space_lidar_only()
        c = 1 if self._lidar_rgb_grayscale else 3
        h, w = self._lidar_rgb_resize[1], self._lidar_rgb_resize[0]
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        lidars = spaces.Box(low=0.0, high=np.inf, shape=(self.img_hist_len, 19))
        images = spaces.Box(low=0.0, high=1.0, shape=(self.img_hist_len, c, h, w))
        if self._include_progress:
            progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
            return spaces.Tuple((speed, progress, lidars, images))
        return spaces.Tuple((speed, lidars, images))

    def reset(self, seed=None, options=None):
        self.reset_common()
        assert self.reward_function is not None
        if self._include_camera_images:
            lidar, speed, _data, img = self.grab_lidar_speed_data_and_image()
            self.img_hist = [lidar for _ in range(self.img_hist_len)]
            self.image_hist = [img for _ in range(self.img_hist_len)]
            lidars = np.array(list(self.img_hist), dtype="float32")
            images = np.array(list(self.image_hist), dtype="float32")
            if self._include_progress:
                progress = np.array([0], dtype="float32")
                obs = [speed, progress, lidars, images]
            else:
                obs = [speed, lidars, images]
        else:
            img, speed, _data = self.grab_lidar_speed_and_data()
            for _ in range(self.img_hist_len):
                self.img_hist.append(img)
            imgs = np.array(list(self.img_hist), dtype="float32")
            if self._include_progress:
                progress = np.array([0], dtype="float32")
                obs = [speed, progress, imgs]
            else:
                obs = [speed, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        assert self.reward_function is not None
        if self._include_camera_images:
            lidar, speed, data, img = self.grab_lidar_speed_data_and_image()
            rew, terminated, _failure_counter = self.reward_function.compute_reward(
                pos=np.array([data[2], data[3], data[4]])
            )[:3]
            if self._include_progress:
                progress = np.array(
                    [self.reward_function.cur_idx / max(1, self.reward_function.datalen)],
                    dtype="float32",
                )
            self.img_hist.append(lidar)
            self.img_hist = self.img_hist[-self.img_hist_len :]
            self.image_hist.append(img)
            self.image_hist = self.image_hist[-self.img_hist_len :]
            lidars = np.array(list(self.img_hist), dtype="float32")
            images = np.array(list(self.image_hist), dtype="float32")
            if self._include_progress:
                obs = [speed, progress, lidars, images]
            else:
                obs = [speed, lidars, images]
        else:
            img, speed, data = self.grab_lidar_speed_and_data()
            rew, terminated, _failure_counter = self.reward_function.compute_reward(
                pos=np.array([data[2], data[3], data[4]])
            )[:3]
            self.img_hist.append(img)
            imgs = np.array(list(self.img_hist), dtype="float32")
            if self._include_progress:
                progress = np.array(
                    [self.reward_function.cur_idx / self.reward_function.datalen],
                    dtype="float32",
                )
                obs = [speed, progress, imgs]
            else:
                obs = [speed, imgs]
        end_of_track = bool(data[8])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew_out = np.float32(rew)
        return obs, rew_out, terminated, info


class TM2020InterfaceLidar(TM2020InterfaceLidarConfigurable):
    def __init__(
        self,
        img_hist_len: int = 1,
        gamepad: bool = False,
        min_nb_steps_before_failure: int | float = int(20 * 3.5),
        save_replays: bool = False,
        **kwargs,
    ):
        super().__init__(
            include_progress=False,
            include_camera_images=False,
            img_hist_len=img_hist_len,
            gamepad=gamepad,
            min_nb_steps_before_failure=min_nb_steps_before_failure,
            save_replays=save_replays,
            **kwargs,
        )
