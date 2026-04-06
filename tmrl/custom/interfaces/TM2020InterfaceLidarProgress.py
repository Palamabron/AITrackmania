import numpy as np
from gymnasium import spaces

from tmrl.custom.interfaces.TM2020InterfaceLidar import TM2020InterfaceLidar
from tmrl.custom.tm.utils.tools import openplanet_grab_indices


class TM2020InterfaceLidarProgress(TM2020InterfaceLidar):
    def reset(self, seed=None, options=None):
        """
        obs must be a list of numpy arrays
        """
        self.reset_common()
        img, speed, _data = self.grab_lidar_speed_and_data()
        for _ in range(self.img_hist_len):
            self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        progress = np.array([0], dtype="float32")
        obs = [speed, progress, imgs]
        self.reward_function.reset()
        return obs, {}

    def get_obs_rew_terminated_info(self):
        """
        returns the observation, the reward, and a terminated signal for end of episode
        obs must be a list of numpy arrays
        """
        img, speed, data = self.grab_lidar_speed_and_data()
        _, (_xi, _yi, _zi), _eoti = openplanet_grab_indices(self.client.nb_floats)
        rew, terminated, _failure_counter = self.reward_function.compute_reward(
            pos=np.array([data[_xi], data[_yi], data[_zi]])
        )[:3]
        progress = np.array(
            [self.reward_function.cur_idx / self.reward_function.datalen], dtype="float32"
        )
        self.img_hist.append(img)
        imgs = np.array(list(self.img_hist), dtype="float32")
        obs = [speed, progress, imgs]
        end_of_track = bool(data[_eoti])
        info = {"end_of_track": end_of_track}
        if end_of_track:
            rew += self.finish_reward
            terminated = True
        rew = np.float32(rew)
        return obs, rew, terminated, info

    def get_observation_space(self):
        """
        must be a Tuple
        """
        speed = spaces.Box(low=0.0, high=1000.0, shape=(1,))
        progress = spaces.Box(low=0.0, high=1.0, shape=(1,))
        imgs = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(
                self.img_hist_len,
                19,
            ),
        )  # lidars
        return spaces.Tuple((speed, progress, imgs))
