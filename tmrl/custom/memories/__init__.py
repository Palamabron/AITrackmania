"""Memory implementations for TrackMania reinforcement learning.

This package provides replay memory implementations for various
TrackMania RL interfaces and algorithms.
"""

# Base classes and utilities
from tmrl.custom.memories.base import (
    GenericTorchMemory,
    MemoryTM,
    last_true_in_list,
    replace_hist_before_eoe,
)

# Sample compressors
from tmrl.custom.memories.compressors import (
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_progress,
    get_local_buffer_sample_lidar_progress_images,
    get_local_buffer_sample_mobilenet,
    get_local_buffer_sample_tm20_imgs,
)

# R2D2-based memories
from tmrl.custom.memories.r2d2 import (
    MemoryR2D2,
    MemoryR2D2Sophy,
    MemoryR2D2woImages,
)

# TrackMania memories
from tmrl.custom.memories.tm_best import MemoryTMBest
from tmrl.custom.memories.tm_full import MemoryTMFull
from tmrl.custom.memories.tm_lidar import (
    MemoryTMLidar,
    MemoryTMLidarProgress,
    MemoryTMLidarProgressImages,
)

__all__ = [
    # Base
    "GenericTorchMemory",
    "MemoryTM",
    "last_true_in_list",
    "replace_hist_before_eoe",
    # Compressors
    "get_local_buffer_sample_lidar",
    "get_local_buffer_sample_lidar_progress",
    "get_local_buffer_sample_lidar_progress_images",
    "get_local_buffer_sample_mobilenet",
    "get_local_buffer_sample_tm20_imgs",
    # R2D2
    "MemoryR2D2",
    "MemoryR2D2woImages",
    "MemoryR2D2Sophy",
    # TM memories
    "MemoryTMBest",
    "MemoryTMFull",
    "MemoryTMLidar",
    "MemoryTMLidarProgress",
    "MemoryTMLidarProgressImages",
]
