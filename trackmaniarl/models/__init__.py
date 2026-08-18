"""Composable TrackmaniaRL 1.0 neural-network building blocks."""

from trackmaniarl.models.actors import CategoricalActor, GaussianActor
from trackmaniarl.models.backbones import (
    HypersphericalLinear,
    SimbaV2Backbone,
    SimbaV2Block,
    project_hyperspherical_weights,
)
from trackmaniarl.models.critics import ContinuousQCritic, DiscreteQuantileNetwork, QuantileCritic
from trackmaniarl.models.encoders import ObservationEncoder, TrackGeometryEncoder

__all__ = [
    "CategoricalActor",
    "ContinuousQCritic",
    "DiscreteQuantileNetwork",
    "GaussianActor",
    "HypersphericalLinear",
    "ObservationEncoder",
    "QuantileCritic",
    "SimbaV2Backbone",
    "SimbaV2Block",
    "TrackGeometryEncoder",
    "project_hyperspherical_weights",
]
