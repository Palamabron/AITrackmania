"""Private data structures used by replay sampling."""

from __future__ import annotations

import numpy as np


class _FenwickTree:
    """Fixed-capacity prefix-sum tree for O(log N) proportional replay draws."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.values = np.zeros(size + 1, dtype=np.float64)
        self.leaves = np.zeros(size, dtype=np.float32)

    def set(self, index: int, value: float) -> None:
        delta = value - self.leaves[index]
        self.leaves[index] = value
        index += 1
        while index <= self.size:
            self.values[index] += delta
            index += index & -index

    @property
    def total(self) -> float:
        total = 0.0
        index = self.size
        while index:
            total += float(self.values[index])
            index -= index & -index
        return total

    def find(self, target: float) -> int:
        """Return the zero-based leaf containing a target in ``[0, total)``."""

        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            candidate = index + bit
            if candidate <= self.size and self.values[candidate] <= target:
                index = candidate
                target -= self.values[candidate]
            bit >>= 1
        return min(index, self.size - 1)
