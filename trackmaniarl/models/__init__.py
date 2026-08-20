"""Composable TrackmaniaRL neural-network building blocks."""

from trackmaniarl.models.actors import CategoricalActor, GaussianActor
from trackmaniarl.models.backbones import (
    HypersphericalLinear,
    SimbaV2Backbone,
    SimbaV2Block,
    project_hyperspherical_weights,
)
from trackmaniarl.models.composite import CompositeValueModel, FrameBatchAdapter
from trackmaniarl.models.critics import (
    ContinuousQCritic,
    ContinuousValueCritic,
    DiscreteQuantileNetwork,
    QuantileCritic,
)
from trackmaniarl.models.encoders import (
    ObservationEncoder,
    TemporalMambaTrackGeometryEncoder,
    TrackGeometryEncoder,
)
from trackmaniarl.models.factory import CompositeValueModelFactory

__all__ = [
    "CategoricalActor",
    "CompositeValueModel",
    "CompositeValueModelFactory",
    "ContinuousQCritic",
    "ContinuousValueCritic",
    "DiscreteQuantileNetwork",
    "FrameBatchAdapter",
    "GaussianActor",
    "HypersphericalLinear",
    "ObservationEncoder",
    "QuantileCritic",
    "SimbaV2Backbone",
    "SimbaV2Block",
    "TemporalMambaTrackGeometryEncoder",
    "TrackGeometryEncoder",
    "project_hyperspherical_weights",
]
