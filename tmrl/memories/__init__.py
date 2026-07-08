"""Canonical namespace for replay memory implementations.

This package is a re-export facade over ``tmrl.custom.memories``.  All memory
classes remain in their original locations — nothing is moved — so existing
imports are unaffected.

Available memories
------------------
- :class:`MemoryTM` — abstract base for all TM-specific memories.
- :class:`MemoryTMBest` — stores only the best lap transitions.
- :class:`MemoryTMFull` — full-history memory (all transitions from all laps).
- :class:`MemoryTMLidarImages` — lidar + image hybrid memory.
- :class:`MemoryR2D2` — recurrent R2D2-style memory.
- :class:`MemoryR2D2Sophy` — R2D2 memory for Sophy hybrid observations.
- :class:`MemoryR2D2woImages` — R2D2 memory without image branch.
- :class:`GenericTorchMemory` — generic non-TM memory for custom environments.

Plugin extension
----------------
Register a custom memory class so that tmrl can discover it by name::

    from tmrl.registry import MEMORIES

    @MEMORIES.register("my_memory")
    class MyMemory(GenericTorchMemory):
        ...

For third-party packages, declare an entry point in your ``pyproject.toml``
(or ``setup.cfg``) under the ``"tmrl.memories"`` group::

    [project.entry-points."tmrl.memories"]
    my_memory = "mypackage.memories:MyMemory"

tmrl will call ``importlib.metadata.entry_points(group="tmrl.memories")`` at
startup and register each discovered class automatically.

Usage
-----
::

    from tmrl.memories import MemoryTMBest, MemoryTMFull
    from tmrl.memories import MemoryR2D2, GenericTorchMemory
"""

from tmrl.custom.memories import (
    ACTION_STEER_INDEX,
    BufferField,
    GenericField,
    GenericTorchMemory,
    MemoryR2D2,
    MemoryR2D2Sophy,
    MemoryR2D2woImages,
    MemoryTM,
    MemoryTMBest,
    MemoryTMFull,
    MemoryTMLidarImages,
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
    _hflip_action,
    _hflip_discrete_action,
    _is_discrete_action,
    fog_recency_resample,
    get_local_buffer_sample_lidar,
    get_local_buffer_sample_lidar_images,
    get_local_buffer_sample_mobilenet,
    get_local_buffer_sample_tm20_imgs,
    last_true_in_list,
    replace_hist_before_eoe,
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
    "_hflip_action",
    "_hflip_discrete_action",
    "_is_discrete_action",
    "fog_recency_resample",
    "get_local_buffer_sample_lidar",
    "get_local_buffer_sample_lidar_images",
    "get_local_buffer_sample_mobilenet",
    "get_local_buffer_sample_tm20_imgs",
    "last_true_in_list",
    "replace_hist_before_eoe",
]
