"""R2D2-based memory implementations for TrackMania.

This module provides memory classes based on R2D2Memory for
recurrent reinforcement learning algorithms.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from tmrl.custom.memories.base import last_true_in_list, replace_hist_before_eoe
from tmrl.memory import R2D2Memory


class MemoryR2D2(R2D2Memory):
    """R2D2-style replay memory with full telemetry and images.

    This memory is designed for recurrent off-policy algorithms,
    storing full episode sequences with all telemetry fields.
    """

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        """Initialize MemoryR2D2.

        Args:
            memory_size: Maximum size of the memory buffer.
            batch_size: Size of batches used during training.
            dataset_path: Path to the dataset.
            imgs_obs: Number of observed images.
            act_buf_len: Length of the action buffer.
            nb_steps: Number of steps for multi-step learning.
            sample_preprocessor: A callable function for sample preprocessing.
            crc_debug: Flag indicating whether to debug CRC.
            device: Device where the memory is stored ("cpu" or "cuda").
        """
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def __len__(self) -> int:
        """Return the number of valid transitions in memory.

        Returns:
            Number of transitions available for sampling.
        """
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    def get_transition(self, item: int):
        """Get a single transition with proper episode boundary handling.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]
        new_act_buf = acts[1:]

        imgs = self.load_imgs(item)
        imgs_last_obs = imgs[:-1]
        imgs_new_obs = imgs[1:]

        # Handle reset transitions
        last_eoes = self.data[17][idx_now - self.min_samples : idx_now]
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

        last_obs = (
            self.data[2][idx_last],  # checkpoints
            self.data[3][idx_last],  # speeds
            self.data[4][idx_last],  # accelerations
            self.data[5][idx_last],  # jerks
            self.data[6][idx_last],  # race_progress
            self.data[7][idx_last],  # input_steer
            self.data[8][idx_last],  # input_gas_pedal
            self.data[9][idx_last],  # input_brake
            self.data[10][idx_last],  # gear
            self.data[11][idx_last],  # aim_yaw
            self.data[12][idx_last],  # aim_pitch
            self.data[13][idx_last],  # steer_angle
            self.data[14][idx_last],  # slip_coef
            self.data[15][idx_last],  # failure counter
            *last_act_buf,
        )
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[18][idx_now])
        new_obs = (
            self.data[2][idx_now],
            self.data[3][idx_now],
            self.data[4][idx_now],
            self.data[5][idx_now],
            self.data[6][idx_now],
            self.data[7][idx_now],
            self.data[8][idx_now],
            self.data[9][idx_now],
            self.data[10][idx_now],
            self.data[11][idx_now],
            self.data[12][idx_now],
            self.data[13][idx_now],
            self.data[14][idx_now],
            self.data[15][idx_now],
            *new_act_buf,
        )
        terminated = self.data[20][idx_now]
        truncated = self.data[21][idx_now]
        info = self.data[19][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence for a transition.

        Args:
            item: Starting index for loading images.

        Returns:
            Stacked array of images normalized to [0, 1].
        """
        res = self.data[16][
            (item + self.start_imgs_offset) : (item + self.start_imgs_offset + self.imgs_obs + 1)
        ]
        return np.stack(res).astype(np.float32) / 256.0

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
        d2 = [np.array(b[1][0]) for b in buffer.memory]  # checkpoints
        d3 = [np.array(b[1][1]) for b in buffer.memory]  # speeds
        d4 = [np.array(b[1][2]) for b in buffer.memory]  # accelerations
        d5 = [np.array(b[1][3]) for b in buffer.memory]  # jerks
        d6 = [np.array(b[1][4]) for b in buffer.memory]  # race_progress
        d7 = [np.array(b[1][5]) for b in buffer.memory]  # input_steer
        d8 = [np.array(b[1][6]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array(b[1][7]) for b in buffer.memory]  # input_brake
        d10 = [np.array(b[1][8]) for b in buffer.memory]  # gear
        d11 = [np.array(b[1][9]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array(b[1][10]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array(b[1][11]) for b in buffer.memory]  # steer_angle
        d14 = [np.array(b[1][12]) for b in buffer.memory]  # slip_coef
        d15 = [np.array(b[1][13]) for b in buffer.memory]  # failure counter
        d16 = [np.array(b[1][14]) for b in buffer.memory]  # imgs
        d17 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d18 = [b[2] for b in buffer.memory]  # rewards
        d19 = [b[5] for b in buffer.memory]  # infos
        d20 = [b[3] for b in buffer.memory]  # terminated
        d21 = [b[4] for b in buffer.memory]  # truncated

        data_fields = [
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
            d15,
            d16,
            d17,
            d18,
            d19,
            d20,
            d21,
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(22):
                self.data[i] = self.data[i][to_trim:]

        return self


class MemoryR2D2woImages(R2D2Memory):
    """R2D2-style replay memory without images.

    This memory is designed for recurrent off-policy algorithms
    that use telemetry data only, without camera images.
    """

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        """Initialize MemoryR2D2woImages.

        Args:
            memory_size: Maximum size of the memory buffer.
            batch_size: Size of batches used during training.
            dataset_path: Path to the dataset.
            imgs_obs: Number of observed images (kept for API compatibility).
            act_buf_len: Length of the action buffer.
            nb_steps: Number of steps for multi-step learning.
            sample_preprocessor: A callable function for sample preprocessing.
            crc_debug: Flag indicating whether to debug CRC.
            device: Device where the memory is stored ("cpu" or "cuda").
        """
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def _obs_end(self) -> int:
        """Index (exclusive) of the last observation column in self.data.

        Layout: [indexes, actions, obs_0 ... obs_N-1, eoes, rewards, infos,
        terminated, truncated].  The 5 trailing fields are always present, so
        obs columns span indices 2 .. len(self.data) - 5.
        """
        return len(self.data) - 5

    def get_transition(self, item: int):
        """Get a single transition.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        obs_end = self._obs_end()
        last_obs = tuple(self.data[i][idx_last] for i in range(2, obs_end))
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[obs_end + 1][idx_now])
        new_obs = tuple(self.data[i][idx_now] for i in range(2, obs_end))
        terminated = self.data[obs_end + 3][idx_now]
        truncated = self.data[obs_end + 4][idx_now]
        info = self.data[obs_end + 2][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence (placeholder for API compatibility).

        Args:
            item: Starting index for loading.

        Returns:
            Empty stack (images not used in this memory).
        """
        res = self.data[2][
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

        n_obs = len(buffer.memory[0][1])

        data_fields = []
        data_fields.append([first_data_idx + i for i, _ in enumerate(buffer.memory)])  # indexes
        data_fields.append([b[0] for b in buffer.memory])  # actions
        for j in range(n_obs):
            data_fields.append([np.array(b[1][j]) for b in buffer.memory])
        data_fields.append([b[3] or b[4] for b in buffer.memory])  # eoes
        data_fields.append([b[2] for b in buffer.memory])  # rewards
        data_fields.append([b[5] for b in buffer.memory])  # infos
        data_fields.append([b[3] for b in buffer.memory])  # terminated
        data_fields.append([b[4] for b in buffer.memory])  # truncated

        if self.__len__() > 0 and len(self.data) == len(data_fields):
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            if self.__len__() > 0 and len(self.data) != len(data_fields):
                from loguru import logger
                logger.warning(
                    "Memory column count changed ({} -> {}); resetting buffer.",
                    len(self.data),
                    len(data_fields),
                )
            self.data = list(data_fields)

        self.rewards_index = len(self.data) - 3

        n_fields = len(self.data)
        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(n_fields):
                self.data[i] = self.data[i][to_trim:]

        return self


class MemoryR2D2Sophy(R2D2Memory):
    """R2D2-style replay memory for Sophy interface.

    Similar to MemoryR2D2woImages but designed specifically for
    the Sophy driving model telemetry format.
    """

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 2,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        """Initialize MemoryR2D2Sophy.

        Args:
            memory_size: Maximum size of the memory buffer.
            batch_size: Size of batches used during training.
            dataset_path: Path to the dataset.
            imgs_obs: Number of observed images (kept for API compatibility).
            act_buf_len: Length of the action buffer.
            nb_steps: Number of steps for multi-step learning.
            sample_preprocessor: A callable function for sample preprocessing.
            crc_debug: Flag indicating whether to debug CRC.
            device: Device where the memory is stored ("cpu" or "cuda").
        """
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)

        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def get_transition(self, item: int):
        """Get a single transition.

        Args:
            item: Starting index for the transition.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        idx_last = item + self.min_samples - 1
        idx_now = item + self.min_samples

        acts = self.load_acts(item)
        last_act_buf = acts[:-1]

        last_obs = (
            self.data[2][idx_last],
            self.data[3][idx_last],
            self.data[4][idx_last],
            self.data[5][idx_last],
            self.data[6][idx_last],
            self.data[7][idx_last],
            self.data[8][idx_last],
            self.data[9][idx_last],
            self.data[10][idx_last],
            self.data[11][idx_last],
            self.data[12][idx_last],
            self.data[13][idx_last],
            self.data[14][idx_last],
            self.data[15][idx_last],
            *last_act_buf,
        )
        new_act = self.data[1][idx_now]
        rew = np.float32(self.data[17][idx_now])
        new_obs = (
            self.data[2][idx_now],
            self.data[3][idx_now],
            self.data[4][idx_now],
            self.data[5][idx_now],
            self.data[6][idx_now],
            self.data[7][idx_now],
            self.data[8][idx_now],
            self.data[9][idx_now],
            self.data[10][idx_now],
            self.data[11][idx_now],
            self.data[12][idx_now],
            self.data[13][idx_now],
            self.data[14][idx_now],
            self.data[15][idx_now],
            *acts[1:],
        )
        terminated = self.data[19][idx_now]
        truncated = self.data[20][idx_now]
        info = self.data[18][idx_now]
        return last_obs, new_act, rew, new_obs, terminated, truncated, info

    def load_imgs(self, item: int):
        """Load image sequence (placeholder for API compatibility).

        Args:
            item: Starting index for loading.

        Returns:
            Empty stack (images not used in this memory).
        """
        res = self.data[2][
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
        d2 = [np.array(b[1][0]) for b in buffer.memory]  # track info
        d3 = [np.array(b[1][1]) for b in buffer.memory]  # speeds
        d4 = [np.array(b[1][2]) for b in buffer.memory]  # accelerations
        d5 = [np.array(b[1][3]) for b in buffer.memory]  # jerks
        d6 = [np.array(b[1][4]) for b in buffer.memory]  # race_progress
        d7 = [np.array(b[1][5]) for b in buffer.memory]  # input_steer
        d8 = [np.array(b[1][6]) for b in buffer.memory]  # input_gas_pedal
        d9 = [np.array(b[1][7]) for b in buffer.memory]  # input_brake
        d10 = [np.array(b[1][8]) for b in buffer.memory]  # gear
        d11 = [np.array(b[1][9]) for b in buffer.memory]  # aim_yaw
        d12 = [np.array(b[1][10]) for b in buffer.memory]  # aim_pitch
        d13 = [np.array(b[1][11]) for b in buffer.memory]  # steer_angle
        d14 = [np.array(b[1][12]) for b in buffer.memory]  # slip_coef
        d15 = [np.array(b[1][13]) for b in buffer.memory]  # failure counter
        d16 = [b[3] or b[4] for b in buffer.memory]  # eoes
        d17 = [b[2] for b in buffer.memory]  # rewards
        d18 = [b[5] for b in buffer.memory]  # infos
        d19 = [b[3] for b in buffer.memory]  # terminated
        d20 = [b[4] for b in buffer.memory]  # truncated

        data_fields = [
            d0,
            d1,
            d2,
            d3,
            d4,
            d5,
            d6,
            d7,
            d8,
            d9,
            d10,
            d11,
            d12,
            d13,
            d14,
            d15,
            d16,
            d17,
            d18,
            d19,
            d20,
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = self.__len__() - self.memory_size
        if to_trim > 0:
            for i in range(21):
                self.data[i] = self.data[i][to_trim:]

        return self
