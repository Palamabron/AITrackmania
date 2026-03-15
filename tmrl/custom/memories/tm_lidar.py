"""TrackMania memory implementations for LIDAR-based observations.

This module provides replay memory classes for LIDAR sensor data,
supporting both basic LIDAR and LIDAR with progress information.
"""

import random

import numpy as np

from tmrl.custom.memories.base import MemoryTM, last_true_in_list, replace_hist_before_eoe


class MemoryTMLidar(MemoryTM):
    """Replay memory for TrackMania with LIDAR observations.

    Stores and samples transitions with speed and LIDAR data,
    handling temporal sequences and episode boundaries.
    """

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling.

        CAUTION: item is the first index of the images in the OLD observation's history.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:
                item += 1
            elif item == self.__len__() - 1:
                item -= 1
            elif random.random() < 0.5:
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # Handle reset transitions
        last_eoes = self.data[4][idx_now - self.min_samples : idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(
                hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1
            )
            replace_hist_before_eoe(
                hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset
            )
            replace_hist_before_eoe(
                hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1
            )
            replace_hist_before_eoe(
                hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset
            )

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[7][idx_now]
        truncated = self.data[8][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence for a transition.

        Args:
            item: Starting index for loading images.

        Returns:
            Stacked array of images.
        """
        res = self.data[3][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res)

    def load_acts(self, item: int):
        """Load action sequence for a transition.

        Args:
            item: Starting index for loading actions.

        Returns:
            Array of actions.
        """
        res = self.data[1][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Args:
            buffer: Buffer containing (act, obs, rew, terminated, truncated, info) samples.

        Returns:
            Self for method chaining.
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][1] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[3] for b in buffer.memory]  # terminated
        d8 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            for i, d in enumerate([d0, d1, d2, d3, d4, d5, d6, d7, d8]):
                self.data[i] += d
        else:
            self.data = [d0, d1, d2, d3, d4, d5, d6, d7, d8]

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(9):
                self.data[i] = self.data[i][to_trim:]

        return self


class MemoryTMLidarProgress(MemoryTM):
    """Replay memory for TrackMania with LIDAR and progress observations.

    Similar to MemoryTMLidar but also stores race progress information.
    """

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        if self.data[4][item + self.min_samples - 1]:
            if item == 0:
                item += 1
            elif item == self.__len__() - 1:
                item -= 1
            elif random.random() < 0.5:
                item += 1
            else:
                item -= 1

        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # Handle reset transitions
        last_eoes = self.data[4][idx_now - self.min_samples : idx_now]
        last_eoe_idx = last_true_in_list(last_eoes)

        assert last_eoe_idx is None or last_eoes[last_eoe_idx], f"last_eoe_idx:{last_eoe_idx}"

        if last_eoe_idx is not None:
            replace_hist_before_eoe(
                hist=new_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset - 1
            )
            replace_hist_before_eoe(
                hist=last_act_buf, eoe_idx_in_hist=last_eoe_idx - self.start_acts_offset
            )
            replace_hist_before_eoe(
                hist=imgs_new_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset - 1
            )
            replace_hist_before_eoe(
                hist=imgs_last_obs, eoe_idx_in_hist=last_eoe_idx - self.start_imgs_offset
            )

        imgs_new_obs = np.ndarray.flatten(imgs_new_obs)
        imgs_last_obs = np.ndarray.flatten(imgs_last_obs)

        last_obs = (self.data[2][idx_last], self.data[7][idx_last], imgs_last_obs, *last_act_buf)
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[5][idx_now])
        new_obs = (self.data[2][idx_now], self.data[7][idx_now], imgs_new_obs, *new_act_buf)
        terminated = self.data[8][idx_now]
        truncated = self.data[9][idx_now]
        info = self.data[6][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence for a transition.

        Args:
            item: Starting index for loading images.

        Returns:
            Stacked array of LIDAR images.
        """
        res = self.data[3][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res)

    def load_acts(self, item: int):
        """Load action sequence for a transition.

        Args:
            item: Starting index for loading actions.

        Returns:
            Array of actions.
        """
        res = self.data[1][
            (item + self.start_acts_offset) : (item + self.start_acts_offset + self.act_buf_len + 1)
        ]
        return res

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Args:
            buffer: Buffer containing (act, obs, rew, terminated, truncated, info) samples.

        Returns:
            Self for method chaining.
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0

        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]  # indexes
        d1 = [b[0] for b in buffer.memory]  # actions
        d2 = [b[1][0] for b in buffer.memory]  # speeds
        d3 = [b[1][2] for b in buffer.memory]  # lidar
        d4 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d5 = [b[2] for b in buffer.memory]  # rewards
        d6 = [b[5] for b in buffer.memory]  # infos
        d7 = [b[1][1] for b in buffer.memory]  # progress
        d8 = [b[3] for b in buffer.memory]  # terminated
        d9 = [b[4] for b in buffer.memory]  # truncated

        if self.__len__() > 0:
            for i, d in enumerate([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]):
                self.data[i] += d
        else:
            self.data = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9]

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(10):
                self.data[i] = self.data[i][to_trim:]

        return self


class MemoryTMLidarProgressImages(MemoryTM):
    """Replay memory for (speed, progress, lidar, images) observations.

    One transition equals two steps. Stores full observations without
    temporal stacking.
    """

    def get_transition(self, item: int):
        """Get a single transition.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        if self.data[6][item + 1]:
            if item == 0:
                item += 1
            elif item == self.__len__() - 1:
                item -= 1
            elif random.random() < 0.5:
                item += 1
            else:
                item -= 1

        idx_last = item
        idx_now = item + 1

        last_obs = (
            self.data[2][idx_last],
            self.data[3][idx_last],
            self.data[4][idx_last],
            self.data[5][idx_last],
        )
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[7][idx_now])
        new_obs = (
            self.data[2][idx_now],
            self.data[3][idx_now],
            self.data[4][idx_now],
            self.data[5][idx_now],
        )
        terminated = self.data[9][idx_now]
        truncated = self.data[10][idx_now]
        info = self.data[8][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Args:
            buffer: Buffer containing samples.

        Returns:
            Self for method chaining.
        """
        first_data_idx = self.data[0][-1] + 1 if self.__len__() > 0 else 0
        d0 = [first_data_idx + i for i, _ in enumerate(buffer.memory)]
        d1 = [b[0] for b in buffer.memory]
        d2 = [b[1][0] for b in buffer.memory]
        d3 = [b[1][1] for b in buffer.memory]
        d4 = [b[1][2] for b in buffer.memory]
        d5 = [b[1][3] for b in buffer.memory]
        d6 = [b[3] or b[4] for b in buffer.memory]
        d7 = [b[2] for b in buffer.memory]
        d8 = [b[5] for b in buffer.memory]
        d9 = [b[3] for b in buffer.memory]
        d10 = [b[4] for b in buffer.memory]

        if self.__len__() > 0:
            for i, d in enumerate([d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]):
                self.data[i] += d
        else:
            self.data = [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10]

        to_trim = len(self.data[0]) - self.memory_size
        if to_trim > 0:
            for i in range(len(self.data)):
                self.data[i] = self.data[i][to_trim:]
        return self
