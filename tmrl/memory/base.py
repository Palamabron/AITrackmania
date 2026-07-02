"""Abstract base class for TMRL replay memories."""

import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from tmrl.memory._crc import check_samples_crc


class Memory(ABC):
    """Replay buffer interface for TMRL.

    Subclasses must implement append_buffer and __len__, and optionally
    get_transition. When overriding __init__, call super().__init__ and
    accept at least the same arguments as this base class.
    """

    def __init__(
        self,
        device: Any,
        nb_steps: int,
        sample_preprocessor: Callable[..., Any] | None = None,
        memory_size: int = 1000000,
        batch_size: int = 256,
        dataset_path: str = "",
        crc_debug: bool = False,
        n_step_return: int = 1,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            device: Device to collate output tensors to.
            nb_steps: Number of steps per training round.
            sample_preprocessor: Optional data augmentation applied to sampled batches.
            memory_size: Maximum number of transitions in the circular buffer.
            batch_size: Batch size for sampled tensors.
            dataset_path: Path to optional offline dataset to preload.
            crc_debug: If True, run CRC checks on compressed samples (debugging).
            n_step_return: Number of steps for n-step TD returns. When > 1, sample_indices()
                           ensures that n consecutive transitions are available for each sample.
        """
        self.nb_steps = nb_steps
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug
        self.n_step_return = n_step_return

        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0.0
        self.stat_train_steps = 0
        self.stat_test_finish_time = 0.0
        self.stat_test_finished_count = 0
        self.stat_test_competition_eliminated = False
        self.stat_test_competition_crashes = 0
        self.average_reward = 0
        self.debug = False

        self.path = Path(dataset_path)
        logger.debug(f"Memory self.path:{self.path}")
        if os.path.isfile(self.path / "data.pkl"):
            with open(self.path / "data.pkl", "rb") as f:
                self.data = list(pickle.load(f))
        else:
            logger.info("no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            logger.warning(
                f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})"
            )

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
        if len(indices) == 0:
            raise RuntimeError(
                f"Cannot sample batch: replay has {len(self)} transition(s) but "
                f"n_step_return={self.n_step_return} requires more data (or buffer is empty)."
            )
        batch = [self[idx] for idx in indices]
        batch = self.collate(batch, self.device)
        return batch

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.stat_test_finish_time = getattr(buffer, "stat_test_finish_time", 0.0)
            self.stat_test_finished_count = getattr(buffer, "stat_test_finished_count", 0)
            self.stat_test_competition_eliminated = getattr(
                buffer, "stat_test_competition_eliminated", False
            )
            self.stat_test_competition_crashes = getattr(buffer, "stat_test_competition_crashes", 0)
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
        terminated = np.float32(terminated)
        truncated = np.float32(truncated)
        # Only pass tensor-serializable keys; skip strings (demo_source, demo_run_id).
        # n_step_effective is the per-sample n-step window length set by memories
        # implementing memory-side n-step returns (1 for plain 1-step transitions).
        info_raw = dict(info) if isinstance(info, dict) else {}
        info: dict[str, Any] = {"is_demo": bool(info_raw.get("is_demo", False))}
        if "n_step_effective" in info_raw:
            info["n_step_effective"] = int(info_raw["n_step_effective"])
        return prev_obs, new_act, rew, new_obs, terminated, truncated, info

    def sample_indices(self):
        length = len(self)
        if length <= 0:
            return ()
        # When n_step_return > 1, ensure we can fetch n consecutive transitions per sample.
        # Sample from [0, length - n_step_return] so indices [i..i+n-1] are all valid.
        if self.n_step_return > 1:
            max_start_idx = length - self.n_step_return
            if max_start_idx <= 0:
                return ()  # Not enough data for n-step returns
            return np.random.randint(0, max_start_idx, size=self.batch_size, dtype=np.int64)
        return np.random.randint(0, length, size=self.batch_size, dtype=np.int64)
