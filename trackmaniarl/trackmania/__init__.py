"""Trackmania adapter contracts and action encodings."""

from trackmaniarl.core.collector import CollectionResult, Environment, EpisodeCollector
from trackmaniarl.trackmania.actions import build_brake_tap_action_table
from trackmaniarl.trackmania.assets import record_boundary, record_trajectory
from trackmaniarl.trackmania.baseline import TelemetryTqcModelFactory
from trackmaniarl.trackmania.encoders import LidarSensorEncoder
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.features import LidarFeaturePipeline, TelemetryFeaturePipeline

__all__ = [
    "CollectionResult",
    "Environment",
    "EpisodeCollector",
    "LidarFeaturePipeline",
    "LidarSensorEncoder",
    "OpenPlanetEnvironmentFactory",
    "TelemetryFeaturePipeline",
    "TelemetryTqcModelFactory",
    "TrackmaniaEnvironmentConfig",
    "TrackmaniaEvaluator",
    "build_brake_tap_action_table",
    "record_boundary",
    "record_trajectory",
]
