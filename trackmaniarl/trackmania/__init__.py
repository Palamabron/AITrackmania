"""Trackmania adapter contracts, collection, and action encodings."""

from trackmaniarl.trackmania.actions import (
    build_brake_tap_action_table,
    build_discrete_to_continuous,
)
from trackmaniarl.trackmania.assets import record_boundary, record_trajectory
from trackmaniarl.trackmania.baseline import TelemetryTqcModelFactory
from trackmaniarl.trackmania.collector import (
    CollectionResult,
    TrackmaniaCollector,
    TrackmaniaEnvironment,
)
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.features import LidarFeaturePipeline, TelemetryFeaturePipeline
from trackmaniarl.trackmania.iqn import LidarIqnModelFactory

__all__ = [
    "CollectionResult",
    "LidarFeaturePipeline",
    "LidarIqnModelFactory",
    "OpenPlanetEnvironmentFactory",
    "TelemetryFeaturePipeline",
    "TelemetryTqcModelFactory",
    "TrackmaniaCollector",
    "TrackmaniaEnvironment",
    "TrackmaniaEnvironmentConfig",
    "TrackmaniaEvaluator",
    "build_brake_tap_action_table",
    "build_discrete_to_continuous",
    "record_boundary",
    "record_trajectory",
]
