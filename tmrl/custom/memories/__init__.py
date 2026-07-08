"""Memory implementations for TrackMania reinforcement learning."""

from tmrl.custom.memories.base import (
    GenericTorchMemory,
    MemoryTM,
    last_true_in_list,
    replace_hist_before_eoe,
)
from tmrl.custom.memories.compressors import (
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_images,
    get_local_buffer_sample_mobilenet,
    get_local_buffer_sample_tm20_imgs,
)
from tmrl.custom.memories.enums import (
    BufferField,
    GenericField,
    R2D2Field,
    R2D2ObsField,
    R2D2SophyField,
    R2D2SophyObsField,
    R2D2woImagesTrailingField,
    TMBestField,
    TMBestObsField,
    TMFullField,
    TMFullObsField,
    TMLidarImagesField,
    TMLidarImagesObsField,
)
from tmrl.custom.memories.r2d2 import (
    MemoryR2D2,
    MemoryR2D2Sophy,
    MemoryR2D2woImages,
)
from tmrl.custom.memories.tm_best import MemoryTMBest
from tmrl.custom.memories.tm_full import MemoryTMFull
from tmrl.custom.memories.tm_lidar_images import MemoryTMLidarImages
from tmrl.custom.memories.utils import (
    ACTION_STEER_INDEX,
    fog_recency_resample,
)

__all__ = [
    "ACTION_STEER_INDEX",
    "BufferField",
    "GenericField",
    "GenericTorchMemory",
    "MemoryR2D2",
    "MemoryR2D2Sophy",
    "MemoryR2D2woImages",
    "MemoryTM",
    "MemoryTMBest",
    "MemoryTMFull",
    "MemoryTMLidarImages",
    "R2D2Field",
    "R2D2ObsField",
    "R2D2SophyField",
    "R2D2SophyObsField",
    "R2D2woImagesTrailingField",
    "TMBestField",
    "TMBestObsField",
    "TMFullField",
    "TMFullObsField",
    "TMLidarImagesField",
    "TMLidarImagesObsField",
    "fog_recency_resample",
    "get_local_buffer_sample_lidar",
    "get_local_buffer_sample_lidar_images",
    "get_local_buffer_sample_mobilenet",
    "get_local_buffer_sample_tm20_imgs",
    "last_true_in_list",
    "replace_hist_before_eoe",
]
