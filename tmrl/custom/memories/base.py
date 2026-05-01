"""Base memory classes for TrackMania reinforcement learning."""

from collections.abc import Callable
from typing import Any

import numpy as np

from tmrl.custom.memories.enums import BufferField, GenericField
from tmrl.custom.memories.utils import configure_discrete_steer_bins
from tmrl.memory import TorchMemory
from tmrl.registry import MEMORIES


def last_true_in_list(li: list[bool]) -> int | None:
    """Find the index of the last True value in a list."""
    for i in reversed(range(len(li))):
        if li[i]:
            return i
    return None


def replace_hist_before_eoe(hist: list, eoe_idx_in_hist: int) -> None:
    """Pad history before the End Of Episode (EOE) index."""
    last_idx = len(hist) - 1
    if eoe_idx_in_hist > last_idx:
        raise ValueError(
            f"replace_hist_before_eoe: eoe_idx_in_hist ({eoe_idx_in_hist}) > last_idx ({last_idx})"
        )
    if 0 <= eoe_idx_in_hist < last_idx:
        for i in reversed(range(len(hist))):
            if i <= eoe_idx_in_hist:
                hist[i] = hist[i + 1]


@MEMORIES.register("generic")
class GenericTorchMemory(TorchMemory):
    """Generic torch-based memory for simple replay buffer scenarios."""

    def __init__(
        self,
        memory_size: int = 1_000_000,
        batch_size: int = 1,
        dataset_path: str = "",
        nb_steps: int = 1,
        sample_preprocessor: Callable[..., Any] | None = None,
        crc_debug: bool = False,
        device: str = "cpu",
        discrete_n_steer_bins: int = 0,
    ):
        configure_discrete_steer_bins(discrete_n_steer_bins)
        super().__init__(
            memory_size=memory_size,
            batch_size=batch_size,
            dataset_path=dataset_path,
            nb_steps=nb_steps,
            sample_preprocessor=sample_preprocessor,
            crc_debug=crc_debug,
            device=device,
        )

    def append_buffer(self, buffer: Any) -> None:
        """Append a buffer of samples to the memory."""
        bf = BufferField
        data_fields = [
            [b[bf.ACTION] for b in buffer.memory],
            [b[bf.OBSERVATION] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.TERMINATED] for b in buffer.memory],
            [b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = int(self.__len__() - self.memory_size)
        if to_trim > 0:
            for i in range(len(data_fields)):
                self.data[i] = self.data[i][to_trim:]

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - 1
        return max(0, res)

    def clear(self) -> None:
        """Remove all transitions from the memory."""
        self.data = []

    def get_transition(self, item: int) -> tuple:
        """Get a single transition from the memory."""
        field = GenericField

        # Bounded retries to avoid excessive loops on large buffers
        max_retries = min(100, max(10, self.__len__()))
        for attempt in range(max_retries):
            if not self.data[field.DONE][item]:
                break
            item = np.random.randint(0, self.__len__() - 1)
        else:
            # Provide detailed error message for debugging
            done_count = sum(self.data[field.DONE])
            raise RuntimeError(
                f"Failed to sample non-terminal transition after {max_retries} attempts. "
                f"Buffer has {done_count}/{self.__len__()} done=True transitions. "
                f"This suggests a data quality issue or environment that always terminates immediately."
            )

        idx_last = item
        idx_now = item + 1

        return (
            self.data[field.OBSERVATIONS][idx_last],
            self.data[field.ACTIONS][idx_now],
            self.data[field.REWARDS][idx_now],
            self.data[field.OBSERVATIONS][idx_now],
            self.data[field.TERMINATED][idx_now],
            self.data[field.TRUNCATED][idx_now],
            self.data[field.INFO][idx_now],
        )


@MEMORIES.register("tm_base")
class MemoryTM(TorchMemory):
    """Base class for TrackMania replay memories with temporal structure."""

    #: Index into ``self.data`` for per-step ``info`` dicts
    #: (subclasses must set if demo mixing applies).
    info_field_index: int | None = None

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
        discrete_n_steer_bins: int = 0,
        demo_min_batch_fraction: float = 0.0,
        demo_max_batch_fraction: float = 1.0,
    ):
        configure_discrete_steer_bins(discrete_n_steer_bins)
        self.imgs_obs = imgs_obs
        self.act_buf_len = act_buf_len
        self.min_samples = max(self.imgs_obs, self.act_buf_len)
        self.start_imgs_offset = max(0, self.min_samples - self.imgs_obs)
        self.start_acts_offset = max(0, self.min_samples - self.act_buf_len)
        self.demo_min_batch_fraction = max(0.0, min(1.0, float(demo_min_batch_fraction)))
        self.demo_max_batch_fraction = max(0.0, min(1.0, float(demo_max_batch_fraction)))
        if self.demo_max_batch_fraction < self.demo_min_batch_fraction:
            self.demo_max_batch_fraction = self.demo_min_batch_fraction
        self.last_sample_demo_fraction = 0.0
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
        """Append a buffer of samples - must be implemented by subclasses."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of valid transitions in memory."""
        if len(self.data) == 0:
            return 0
        res = len(self.data[0]) - self.min_samples - 1
        return max(0, res)

    @staticmethod
    def _is_demo_info_entry(info_entry: Any) -> bool:
        if not isinstance(info_entry, dict):
            return False
        return bool(info_entry.get("is_demo", False))

    def _info_field_index(self) -> int | None:
        idx = self.info_field_index
        if idx is None:
            return None
        idx = int(idx)
        if len(self.data) == 0 or idx < 0 or idx >= len(self.data):
            return None
        return idx

    def _item_is_demo(self, item: int) -> bool:
        info_field_index = self._info_field_index()
        if info_field_index is None:
            return False
        idx_now = item + self.min_samples
        info_stream = self.data[info_field_index]
        if idx_now < 0 or idx_now >= len(info_stream):
            return False
        return self._is_demo_info_entry(info_stream[idx_now])

    def _set_last_sample_demo_fraction(self, indices) -> None:
        if len(indices) == 0:
            self.last_sample_demo_fraction = 0.0
            return
        demo_count = sum(1 for idx in indices if self._item_is_demo(int(idx)))
        self.last_sample_demo_fraction = float(demo_count) / float(len(indices))

    def sample_indices(self):
        """Sample transitions, optionally enforcing demo floor/cap for TM memories."""
        length = len(self)
        if length <= 0:
            self.last_sample_demo_fraction = 0.0
            return ()

        demo_min = self.demo_min_batch_fraction
        demo_max = self.demo_max_batch_fraction
        info_field_index = self._info_field_index()
        if info_field_index is None or (demo_min <= 0.0 and demo_max >= 1.0):
            result = np.random.randint(0, length, size=self.batch_size, dtype=np.int64)
            self._set_last_sample_demo_fraction(result)
            return result

        batch_size = int(self.batch_size)
        result = np.random.randint(0, length, size=batch_size, dtype=np.int64)
        demo_positions = [pos for pos, idx in enumerate(result) if self._item_is_demo(int(idx))]
        non_demo_positions = [
            pos for pos, idx in enumerate(result) if not self._item_is_demo(int(idx))
        ]
        demo_items = [idx for idx in range(length) if self._item_is_demo(idx)]
        non_demo_items = [idx for idx in range(length) if not self._item_is_demo(idx)]
        if not demo_items or not non_demo_items:
            self._set_last_sample_demo_fraction(result)
            return result

        min_demo = int(np.ceil(demo_min * batch_size))
        max_demo = int(np.floor(demo_max * batch_size))
        max_demo = max(min_demo, min(max_demo, batch_size))

        if len(demo_positions) < min_demo and non_demo_positions:
            need = min(min_demo - len(demo_positions), len(non_demo_positions))
            replace_positions = np.random.choice(non_demo_positions, size=need, replace=False)
            replacements = np.random.choice(
                demo_items,
                size=need,
                replace=len(demo_items) < need,
            )
            result[replace_positions] = replacements
            demo_positions = [pos for pos, idx in enumerate(result) if self._item_is_demo(int(idx))]

        if len(demo_positions) > max_demo and non_demo_items:
            excess = len(demo_positions) - max_demo
            replace_positions = np.random.choice(demo_positions, size=excess, replace=False)
            replacements = np.random.choice(
                non_demo_items,
                size=excess,
                replace=len(non_demo_items) < excess,
            )
            result[replace_positions] = replacements

        self._set_last_sample_demo_fraction(result)
        return result

    def get_transition(self, item: int):
        """Get a single transition - must be implemented by subclasses."""
        raise NotImplementedError
