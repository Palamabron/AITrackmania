"""Replay memory for boundary lidar + camera fusion: (speed, progress, track, images)."""

import random

import numpy as np

from tmrl.custom.memories._internal.enums import (
    BufferField,
    TMLidarImagesField,
    TMLidarImagesObsField,
)
from tmrl.custom.memories._internal.sampling_utils import canonical_replay_action_vector
from tmrl.custom.memories.base import MemoryTM
from tmrl.registry import MEMORIES


@MEMORIES.register("lidar_images")
class MemoryTMLidarImages(MemoryTM):
    """Replay memory for (speed, progress, track_geometry, images) observations."""

    supports_nstep: bool = True
    info_field_index = TMLidarImagesField.INFOS

    def get_transition(self, item: int):
        """Retrieve a single transition for the lidar+images observation modality.

        When the entry at ``item + 1`` is an EOE marker, the item index is
        randomly shifted by ±1 to avoid returning a terminal step.  Since this
        memory has no multi-frame history (``min_samples = 1``), no history
        padding is needed.

        Args:
            item: Transition item index in ``[0, len(self))``.

        Returns:
            tuple: ``(prev_obs, new_act, rew, new_obs, terminated, truncated, info)``
                where each observation is ``(speeds, progress, track, images)``.
        """
        f = TMLidarImagesField

        if self.data[f.EOES][item + 1]:
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

        return (
            (
                self.data[f.SPEEDS][idx_last],
                self.data[f.PROGRESS][idx_last],
                self.data[f.TRACK][idx_last],
                self.data[f.IMAGES][idx_last],
            ),
            self.data[f.ACTIONS][idx_now],
            np.float32(self.data[f.REWARDS][idx_now]),
            (
                self.data[f.SPEEDS][idx_now],
                self.data[f.PROGRESS][idx_now],
                self.data[f.TRACK][idx_now],
                self.data[f.IMAGES][idx_now],
            ),
            self.data[f.TERMINATED][idx_now],
            self.data[f.TRUNCATED][idx_now],
            self.data[f.INFOS][idx_now],
        )

    def append_buffer(self, buffer):
        """Append a buffer of samples to the memory.

        Extracts all ``TMLidarImagesField`` columns (indexes, actions, speeds,
        progress, track, images, eoes, rewards, infos, terminated, truncated)
        from ``buffer.memory``.  Appends to existing data columns or initialises
        ``self.data`` when the buffer is empty, and trims oldest entries to
        respect ``memory_size``.

        Args:
            buffer: :class:`~tmrl.networking.Buffer` of transitions from a worker.

        Returns:
            MemoryTMLidarImages: ``self`` (for chaining).
        """
        f = TMLidarImagesField
        first_data_idx = self.data[f.INDEXES][-1] + 1 if self.__len__() > 0 else 0
        bf = BufferField
        o = TMLidarImagesObsField

        data_fields = [
            [first_data_idx + i for i, _ in enumerate(buffer.memory)],
            [
                canonical_replay_action_vector(b[bf.ACTION], self.discrete_n_steer_bins)
                for b in buffer.memory
            ],
            [b[bf.OBSERVATION][o.SPEEDS] for b in buffer.memory],
            [b[bf.OBSERVATION][o.PROGRESS] for b in buffer.memory],
            [b[bf.OBSERVATION][o.TRACK] for b in buffer.memory],
            [b[bf.OBSERVATION][o.IMAGES] for b in buffer.memory],
            [b[bf.TERMINATED] or b[bf.TRUNCATED] for b in buffer.memory],
            [b[bf.REWARD] for b in buffer.memory],
            [b[bf.INFO] for b in buffer.memory],
            [b[bf.TERMINATED] for b in buffer.memory],
            [b[bf.TRUNCATED] for b in buffer.memory],
        ]

        if self.__len__() > 0:
            for i, d in enumerate(data_fields):
                self.data[i] += d
        else:
            self.data = list(data_fields)

        to_trim = len(self.data[f.INDEXES]) - self.memory_size
        if to_trim > 0:
            for i in range(len(data_fields)):
                self.data[i] = self.data[i][to_trim:]

        return self
