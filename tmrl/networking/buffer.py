"""In-memory replay buffer shared between Server, RolloutWorker, and Trainer."""

import contextlib
import threading
from collections.abc import Generator
from typing import Any

from loguru import logger

import tmrl.config as cfg


class Buffer:
    """In-memory buffer of transition samples for the Server, RolloutWorker, and Trainer.

    Samples are tuples: (act, new_obs, rew, terminated, truncated, info).

    Intended for single-threaded use (one Buffer per worker or per trainer side). If the same
    Buffer instance is ever shared across threads, construct with thread_safe=True so that
    append_sample, clip_to_maxlen, and __iadd__ are guarded by a lock.
    """

    def __init__(self, maxlen=cfg.BUFFERS_MAXLEN, thread_safe: bool = False):
        """Initialize an empty buffer with optional max length.

        Args:
            maxlen: Maximum number of samples to keep; older samples are dropped when exceeded.
            thread_safe: If True, use a lock around memory updates (for future multi-threaded use).
        """
        self.memory: list[Any] = []
        self.stat_train_return = 0.0
        self.stat_test_return = 0.0
        self.stat_train_steps = 0
        self.stat_test_steps = 0.0
        self.stat_test_finish_time = 0.0
        self.stat_test_finished_track = False
        self.stat_test_finished_count = 0
        self.stat_test_competition_eliminated = False
        self.stat_test_competition_crashes = 0
        self.maxlen = maxlen
        self._lock = threading.RLock() if thread_safe else None

    @contextlib.contextmanager
    def _guarded(self) -> Generator[None, None, None]:
        """Acquire the buffer lock if thread_safe, else no-op. Uses RLock so nesting is safe."""
        if self._lock is not None:
            self._lock.acquire()
        try:
            yield
        finally:
            if self._lock is not None:
                self._lock.release()

    def clip_to_maxlen(self):
        """Drop the oldest samples until ``len(memory) <= maxlen``; warns if any are dropped."""
        with self._guarded():
            lenmem = len(self.memory)
            if lenmem > self.maxlen:
                dropped = lenmem - self.maxlen
                logger.warning(
                    "Buffer overflow: discarding {} oldest samples (kept {} / max {})",
                    dropped,
                    self.maxlen,
                    self.maxlen,
                )
                self.memory = self.memory[dropped:]

    def append_sample(self, sample):
        """Append a sample ``(act, new_obs, rew, terminated, truncated, info)`` to the buffer."""
        with self._guarded():
            self.memory.append(sample)
            self.clip_to_maxlen()

    def clear(self):
        """Clear the buffer but keep train and test return stats."""
        with self._guarded():
            self.memory = []

    def apply_speed_bonus(self, speed_scale: float) -> None:
        """Spread a time/speed bonus over all rewards in this episode (in-place), K/T² formula.

        Each step gets speed_scale / T² so the total bonus is speed_scale / T;
        faster episodes (smaller T) get a higher total and every step carries the signal.
        Avoids a terminal spike that harms TQC convergence and long-horizon credit assignment.

        Call this after collecting a full episode, before sending the buffer.
        No-op if speed_scale <= 0 or buffer is empty.

        Args:
            speed_scale: Typically TIME_BONUS_SCALE * REWARD_SCALE
                (rewards in buffer are already scaled).
        """
        if speed_scale <= 0 or len(self.memory) == 0:
            return
        with self._guarded():
            num_steps = len(self.memory)
            bonus_per_step = speed_scale / (num_steps * num_steps)
            total_bonus = bonus_per_step * num_steps
            new_memory = []
            old_total = 0.0
            for _i, sample in enumerate(self.memory):
                act, obs, rew, term, trunc, info = sample
                old_total += rew
                new_rew = rew + bonus_per_step
                new_info = dict(info) if isinstance(info, dict) else info
                new_memory.append((act, obs, new_rew, term, trunc, new_info))
            new_total = old_total + total_bonus
            if new_memory and isinstance(new_memory[-1][5], dict):
                new_memory[-1][5]["reward_sum"] = new_total
            self.memory = new_memory
            self.stat_train_return = new_total

    def __len__(self):
        """Return the number of samples currently held in the buffer."""
        return len(self.memory)

    def __iadd__(self, other):
        """Merge *other* into this buffer, appending its samples and overwriting episode stats.

        Samples from *other* are appended after the current samples. If the combined
        length exceeds ``maxlen``, the oldest samples are discarded. Episode statistics
        (return, steps, finish time, etc.) are taken from *other*, overwriting the
        corresponding fields on ``self``.

        Args:
            other (Buffer): Buffer whose samples and stats will be merged in.

        Returns:
            Buffer: ``self``, for use with the ``+=`` operator.
        """
        with other._guarded():
            other_memory = list(other.memory)
        with self._guarded():
            self.memory += other_memory
            self.clip_to_maxlen()
            self.stat_train_return = other.stat_train_return
            self.stat_test_return = other.stat_test_return
            self.stat_train_steps = other.stat_train_steps
            self.stat_test_steps = other.stat_test_steps
            self.stat_test_finish_time = getattr(other, "stat_test_finish_time", 0.0)
            self.stat_test_finished_track = getattr(other, "stat_test_finished_track", False)
            self.stat_test_finished_count = getattr(other, "stat_test_finished_count", 0)
            self.stat_test_competition_eliminated = getattr(
                other, "stat_test_competition_eliminated", False
            )
            self.stat_test_competition_crashes = getattr(other, "stat_test_competition_crashes", 0)
            return self
