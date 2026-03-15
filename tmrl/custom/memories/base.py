"""Base memory classes for TrackMania reinforcement learning.

This module provides base classes for experience replay memories
used in TrackMania RL training.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from tmrl.memory import TorchMemory


def last_true_in_list(li: list[bool]) -> int | None:
    """Find the index of the last True value in a list.

    Args:
        li: List of boolean values.

    Returns:
        Index of the last True value, or None if no True value exists.
    """
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist: list, eoe_idx_in_hist: int) -> None:
    """Pad history before the End Of Episode (EOE) index.

    Previous entries in hist are padded with copies of the first element
    occurring after EOE.

    Args:
        hist: History list to modify in place.
        eoe_idx_in_hist: Index of the end of episode in the history.

    Raises:
        AssertionError: If eoe_idx_in_hist is beyond the last index.
    """
    last_idx = len(hist) - 1
    assert eoe_idx_in_hist <= last_idx, (
        f"replace_hist_before_eoe: eoe_idx_in_hist:{eoe_idx_in_hist}, last_idx:{last_idx}"
    )
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


class GenericTorchMemory(TorchMemory):
    """Generic torch-based memory for simple replay buffer scenarios.

    This memory implementation stores transitions without complex history
    management, suitable for simple off-policy algorithms.
    """

    def __init__(
        self,
        memory_size: int = 1_000_000,
        batch_size: int = 1,
        dataset_path: str = "",
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        """Initialize GenericTorchMemory.

        Args:
            memory_size: Maximum number of transitions to store.
            batch_size: Number of samples per batch.
            dataset_path: Path to a saved dataset (if loading existing data).
            nb_steps: Number of steps for multi-step learning.
            sample_preprocessor: Optional function to preprocess samples.
            crc_debug: Enable CRC debugging for data integrity checks.
            device: Device to store tensors on ("cpu" or "cuda").
        """
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Args:
            buffer: Buffer containing samples (act, obs, rew, terminated, truncated, info).
        """
        d0 = [b[0] for b in buffer.memory]  # actions
        d1 = [b[1] for b in buffer.memory]  # observations
        d2 = [b[2] for b in buffer.memory]  # rewards
        d3 = [b[3] for b in buffer.memory]  # terminated
        d4 = [b[4] for b in buffer.memory]  # truncated
        d5 = [b[5] for b in buffer.memory]  # info
        d6 = [b[3] or b[4] for b in buffer.memory]  # done

        if self.__len__() > 0:
            self.data[0] += d0
            self.data[1] += d1
            self.data[2] += d2
            self.data[3] += d3
            self.data[4] += d4
            self.data[5] += d5
            self.data[6] += d6
        else:
            self.data.append(d0)
            self.data.append(d1)
            self.data.append(d2)
            self.data.append(d3)
            self.data.append(d4)
            self.data.append(d5)
            self.data.append(d6)

        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            for i in range(7):
                self.data[i] = self.data[i][to_trim:]

    def __len__(self) -> int:
        """Return the number of valid transitions in memory.

        Returns:
            Number of transitions available for sampling.
        """
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        return max(0, res)

    def get_transition(self, item: int):
        """Get a single transition from the memory.

        Args:
            item: Index of the transition to retrieve.

        Returns:
            Tuple of (last_obs, new_act, rew, new_obs, terminated, truncated, info).
        """
        while self.data[6][item]:
            item = np.random.randint(0, self.__len__() - 1)

        idx_last = item
        idx_now = item + 1

        last_obs = self.data[1][idx_last]
        new_act = self.data[0][idx_now]
        rew = self.data[2][idx_now]
        new_obs = self.data[1][idx_now]
        terminated = self.data[3][idx_now]
        truncated = self.data[4][idx_now]
        info = self.data[5][idx_now]

        return last_obs, new_act, rew, new_obs, terminated, truncated, info


class MemoryTM(TorchMemory):
    """Base class for TrackMania replay memories with temporal structure.

    This class provides common functionality for memories that need to handle
    sequences of images and action histories, with proper episode boundary handling.

    Attributes:
        imgs_obs: Number of consecutive images/observations to stack.
        act_buf_len: Length of action history buffer.
        min_samples: Minimum samples needed for a valid transition.
        start_imgs_offset: Offset for image loading.
        start_acts_offset: Offset for action loading.
    """

    def __init__(
        self,
        memory_size: int | None = None,
        batch_size: int | None = None,
        dataset_path: str = "",
        imgs_obs: int = 4,
        act_buf_len: int = 1,
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
    ):
        """Initialize MemoryTM.

        Args:
            memory_size: Maximum size of the memory buffer.
            batch_size: Size of batches used during training.
            dataset_path: Path to the dataset.
            imgs_obs: Number of observed images to stack.
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

    def append_buffer(self, buffer):
        """Append a buffer of samples - must be implemented by subclasses.

        Args:
            buffer: Buffer containing samples to append.

        Raises:
            NotImplementedError: Always, as subclasses must implement this.
        """
        raise NotImplementedError

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
        """Get a single transition - must be implemented by subclasses.

        Args:
            item: Index of the transition to retrieve.

        Raises:
            NotImplementedError: Always, as subclasses must implement this.
        """
        raise NotImplementedError
