"""Canonical re-export facade for TrackMania 2020 utilities."""

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
