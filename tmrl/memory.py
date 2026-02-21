# standard library imports
import os
import pickle
import random
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from random import randint
from typing import Any

# third-party imports
import numpy as np
from loguru import logger

# local imports
import tmrl.config.config_constants as cfg
from tmrl.util import collate_torch

__docformat__ = "google"


def check_samples_crc(
    original_po,
    original_a,
    original_o,
    original_r,
    original_d,
    original_t,
    rebuilt_po,
    rebuilt_a,
    rebuilt_o,
    rebuilt_r,
    rebuilt_d,
    rebuilt_t,
    debug_ts,
    debug_ts_res,
):
    ts_msg = f"Time step: {debug_ts}, since reset: {debug_ts_res}"
    assert original_po is None or str(original_po) == str(rebuilt_po), (
        f"previous observations don't match:\noriginal:\n{original_po}\n!= rebuilt:\n"
        f"{rebuilt_po}\n{ts_msg}"
    )
    assert str(original_a) == str(rebuilt_a), (
        f"actions don't match:\noriginal:\n{original_a}\n!= rebuilt:\n{rebuilt_a}\n{ts_msg}"
    )
    assert str(original_o) == str(rebuilt_o), (
        f"observations don't match:\noriginal:\n{original_o}\n!= rebuilt:\n{rebuilt_o}\n{ts_msg}"
    )
    assert str(original_r) == str(rebuilt_r), (
        f"rewards don't match:\noriginal:\n{original_r}\n!= rebuilt:\n{rebuilt_r}\n{ts_msg}"
    )
    assert str(original_d) == str(rebuilt_d), (
        f"terminated don't match:\noriginal:\n{original_d}\n!= rebuilt:\n{rebuilt_d}\n{ts_msg}"
    )
    assert str(original_t) == str(rebuilt_t), (
        f"truncated don't match:\noriginal:\n{original_t}\n!= rebuilt:\n{rebuilt_t}\n{ts_msg}"
    )
    original_crc = zlib.crc32(
        str.encode(str((original_a, original_o, original_r, original_d, original_t)))
    )
    crc = zlib.crc32(str.encode(str((rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t))))
    assert crc == original_crc, (
        f"CRC failed: new crc:{crc} != old crc:{original_crc}. "
        "Pipeline corrupted or crc_debug False. "
        f"original:\n{(original_a, original_o, original_r, original_d)}\n!= rebuilt:\n"
        f"{(rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d)}\n{ts_msg}"
    )
    print(f"DEBUG: CRC check passed. Time step: {debug_ts}, since reset: {debug_ts_res}")


class Memory(ABC):
    """
    Interface implementing the replay buffer.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(
        self,
        device,
        nb_steps,
        sample_preprocessor: Callable[..., Any] | None = None,
        memory_size=1000000,
        batch_size=256,
        dataset_path="",
        crc_debug=False,
    ):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        self.nb_steps = nb_steps
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0
        self.average_reward = 0
        self.debug = False

        # init memory
        self.path = Path(dataset_path)
        logger.debug(f"Memory self.path:{self.path}")
        if os.path.isfile(self.path / "data.pkl"):
            with open(self.path / "data.pkl", "rb") as f:
                self.data = list(pickle.load(f))
        else:
            logger.info("no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            # self.data = self.data[-self.memory_size:]
            logger.warning(
                f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})"
            )
        # random.seed(cfg.SEED)

    def __iter__(self):
        for _ in range(self.nb_steps):
            yield self.sample()

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Must append a Buffer object to the memory.

        Args:
            buffer (tmrl.networking.Buffer): the buffer of samples to append.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Must return the length of the memory.

        Returns:
            int: the maximum `item` argument of `get_transition`

        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Must return a transition.

        `info` is required in each sample for CRC debugging (the 'crc' key is used).

        Args:
            item (int): the index where to sample

        Returns:
            Tuple: (prev_obs, prev_act, rew, obs, terminated, truncated, info)
        """
        raise NotImplementedError

    @abstractmethod
    def collate(self, batch, device):
        """
        Must collate `batch` onto `device`.

        `batch` is a list of training samples.
        The length of `batch` is `batch_size`.
        Each sample is `(prev_obs, new_act, rew, new_obs, terminated, truncated)`.
        These samples must be collated into 6 tensors of batch dimension `batch_size`.
        These tensors should be collated onto the device indicated by the `device` argument.
        Then, your implementation must return a single tuple containing these 6 tensors.

        Args:
            batch (list): list of `(prev_obs, new_act, rew, new_obs, terminated, truncated)` tuples
            device: device onto which the list needs to be collated into batches `batch_size`

        Returns:
            Tuple of tensors:
            (prev_obs_tens, new_act_tens, rew_tens, new_obs_tens, terminated_tens, truncated_tens)
            collated on device `device`, each of batch dimension `batch_size`
        """
        raise NotImplementedError

    def sample(self):
        indices = self.sample_indices()
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)
        return batch

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d, t = info["crc_sample"]
            debug_ts, debug_ts_res = info["crc_sample_ts"]
            check_samples_crc(
                po,
                a,
                o,
                r,
                d,
                t,
                prev_obs,
                new_act,
                new_obs,
                rew,
                terminated,
                truncated,
                debug_ts,
                debug_ts_res,
            )
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(
                prev_obs, new_act, rew, new_obs, terminated, truncated
            )
        terminated = np.float32(terminated)  # we don't want bool tensors
        truncated = np.float32(truncated)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, terminated, truncated

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batch_size))


class TorchMemory(Memory, ABC):
    """
    Partial implementation of the `Memory` class collating samples into batched torch tensors.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(
        self,
        device,
        nb_steps,
        sample_preprocessor: Callable[..., Any] | None = None,
        memory_size=1000000,
        batch_size=256,
        dataset_path="",
        crc_debug=False,
    ):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
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

    def collate(self, batch, device):
        return collate_torch(batch, device)


class R2D2Memory(Memory, ABC):
    """
    Partial implementation of the `Memory` class collating samples into batched torch tensors.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """

    def __init__(
        self,
        device,
        nb_steps,
        sample_preprocessor: Callable[..., Any] | None = None,
        memory_size=1000000,
        batch_size=256,
        dataset_path="",
        crc_debug=False,
        # info_index=21
    ):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
        """
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
            # info_index=info_index
        )
        self.rewards_index = 19 if cfg.USE_IMAGES else 18
        self.previous_episode = 0
        self.end_episodes_indices: list[int] = []
        self.chosen_episode = 0
        self.burn_ins = (20, 40)
        self.isNewEpisode = True
        self.chosen_burn_in = 0
        self.reward_sums: list[dict[str, Any]] = []
        self.indices: list[int] = []
        self.cur_idx = 0
        self.batch_size = batch_size
        self.rewind = cfg.TMRL_CONFIG["ALG"]["R2D2_REWIND"]
        assert 0.1 <= self.rewind <= 0.9, "R2D2 REWIND CONST SHOULD BE BETWEEN 0.1 AND 0.9"

    def collate(self, batch, device):
        """
        Method in Memory and its subclasses.
        Used to collate a batch of data onto a specified device.
        Calls an external function collate_torch and returns its result.
        """
        return collate_torch(batch, device)

    @staticmethod
    def find_zero_rewards_indices(reward_sums):
        """
        Finds indices where reward sum transitions from non-zero to zero.
        Returns a list of those indices.
        """
        zero_rewards_indices = []
        prev_reward_sum = None

        for i, entry in enumerate(reward_sums):
            reward_sum = entry["reward_sum"]
            if prev_reward_sum is not None and reward_sum == 0.0 and prev_reward_sum != 0.0:
                zero_rewards_indices.append(i - 1)

            prev_reward_sum = reward_sum

        return zero_rewards_indices

    @staticmethod
    def normalize_list(input_list):
        """
        Normalizes a list of values between 0 and 1.
        Handles cases where the range of values is zero to prevent division by zero.
        """
        # Find the minimum and maximum values in the list
        min_val = min(input_list)
        max_val = max(input_list)

        # Check if the range is zero to avoid division by zero
        if min_val == max_val:
            return [0.0] * len(input_list)

        # Normalize each element in the list
        normalized_list = [(x - min_val) / (max_val - min_val) for x in input_list]

        return normalized_list

    def sample_indices(self):
        """
        Generates indices for sampling from the memory based on various conditions.
        Logic involves selecting indices based on episode lengths, rewards, and episode transitions.
        """
        self.end_episodes_indices = self.find_zero_rewards_indices(self.data[self.rewards_index])
        self.reward_sums = [
            self.data[self.rewards_index][index]["reward_sum"]
            for index in self.end_episodes_indices
        ]
        batch_size = self.batch_size

        if len(self.end_episodes_indices) == 0:
            if self.cur_idx == 0:
                self.cur_idx += int(batch_size * self.rewind)

                result = tuple(range(0, self.cur_idx))
                return result
            else:
                if self.cur_idx + batch_size < len(self):
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    self.cur_idx += int(batch_size * self.rewind)
                    return result
                else:
                    result = tuple(range(len(self) - batch_size, len(self) - 1))
                    self.cur_idx = 0
                    return result
        else:
            if self.isNewEpisode:
                if len(self.reward_sums) == 1:
                    self.chosen_episode = self.end_episodes_indices[0]
                    self.previous_episode = 0
                else:
                    if sum(self.reward_sums) > 0:
                        self.chosen_episode = random.choices(
                            self.end_episodes_indices, weights=self.reward_sums, k=1
                        )[0]
                    else:
                        self.chosen_episode = random.choices(
                            self.end_episodes_indices,
                            weights=self.normalize_list(self.reward_sums),
                            k=1,
                        )[0]
                    previous_episode_index = (
                        sorted(self.end_episodes_indices).index(self.chosen_episode) - 1
                    )
                    if previous_episode_index < 0:
                        self.previous_episode = 0
                    else:
                        self.previous_episode = self.end_episodes_indices[previous_episode_index]

                episode_length = self.chosen_episode - self.previous_episode
                self.chosen_burn_in = random.randint(self.burn_ins[0], self.burn_ins[1])

                if episode_length <= batch_size + self.chosen_burn_in:
                    range_length = min(batch_size, episode_length - self.chosen_burn_in)

                    # Calculate the start index ensuring it doesn't go below 0
                    start_idx = max(self.previous_episode, self.chosen_episode - range_length)

                    # Calculate the end index ensuring it doesn't exceed the chosen episode
                    end_idx = min(self.chosen_episode - 1, start_idx + range_length)

                    # Adjust the start index if the range is shorter than batch_size
                    if end_idx - start_idx < batch_size:
                        start_idx = max(0, end_idx - batch_size)

                    result = tuple(range(start_idx, end_idx))

                    return result
                else:
                    if self.previous_episode < 0:
                        self.previous_episode = 0

                    self.cur_idx = self.previous_episode + self.chosen_burn_in
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))

                    self.cur_idx += batch_size
                    self.isNewEpisode = False
                    return result
            else:
                self.cur_idx -= int(batch_size * self.rewind)

                if self.cur_idx + batch_size >= self.chosen_episode:
                    self.isNewEpisode = True

                    result = tuple(range(self.chosen_episode - batch_size, self.chosen_episode))
                    self.cur_idx = self.chosen_episode
                    return result
                else:
                    self.isNewEpisode = False
                    result = tuple(range(self.cur_idx, self.cur_idx + batch_size))
                    self.cur_idx += batch_size
                    return result

    def __len__(self):
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        if res < 0:
            return 0
        else:
            return res

    def sample(self):
        """
        Samples data from the memory using the generated indices from sample_indices.
        Collates the sampled data into a batch using the collate method and returns it.
        """
        indices = self.sample_indices()
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)
        return batch


def load_and_print_pickle_file(
    path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl",
):  # r"D:\data2020"
    """
    Loads and prints content from a pickle file specified by the path.
    Reads the pickle file and displays the number of samples along with their content.
    """
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")
    print("full data:")
    for i, d in enumerate(data):
        print(f"[{i}]: {d}")


if __name__ == "__main__":
    load_and_print_pickle_file()
