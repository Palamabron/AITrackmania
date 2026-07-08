"""TrackMania 2020-specific utilities for tmrl.

This package re-exports the canonical public symbols for working with
the TrackMania 2020 OpenPlanet plugin interface: observation structures,
telemetry data types, observation-space builders, and pre-processors.

The implementation lives under ``tmrl.custom.tm``; this namespace is the
stable, documented API.

Key symbols
-----------
Telemetry structure
    :class:`Telemetry`
        Named-tuple encapsulating one frame of raw telemetry (race
        progress, kinematics, inputs, dynamics, event flags).

Observation indexing
    :class:`WorldTelemetryObsIndex`
        ``IntEnum`` for indexing into the world-telemetry observation
        list without magic numbers.

Observation space
    :func:`build_openplanet_tuple_observation_space`
        Build the canonical ``gymnasium.spaces.Tuple`` layout matching
        the OpenPlanet plugin field order.

Pre-processors (for RolloutWorker ``obs_preprocessor`` config)
    ``obs_preprocessor_tm_act_in_obs``
    ``obs_preprocessor_lidar_act_in_obs``
    ``obs_preprocessor_world_telemetry_act_in_obs``
    ``make_world_telemetry_obs_preprocessor``
"""

from tmrl.custom.tm.observation_constants import WorldTelemetryObsIndex
from tmrl.custom.tm.openplanet_observation_space import (
    build_openplanet_tuple_observation_space,
)
from tmrl.custom.tm.telemetry import Telemetry
from tmrl.custom.tm.tm_preprocessors import (
    make_world_telemetry_obs_preprocessor,
    obs_preprocessor_lidar_act_in_obs,
    obs_preprocessor_tm_act_in_obs,
    obs_preprocessor_world_telemetry_act_in_obs,
)

__all__ = [
    # Telemetry
    "Telemetry",
    # Observation indexing
    "WorldTelemetryObsIndex",
    # Observation space builder
    "build_openplanet_tuple_observation_space",
    # Pre-processors
    "make_world_telemetry_obs_preprocessor",
    "obs_preprocessor_lidar_act_in_obs",
    "obs_preprocessor_tm_act_in_obs",
    "obs_preprocessor_world_telemetry_act_in_obs",
]
