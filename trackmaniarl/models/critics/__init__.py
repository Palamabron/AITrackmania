"""Value and quantile critics for TrackmaniaRL learners."""

from trackmaniarl.models.critics.value import (
    ContinuousQCritic,
    ContinuousValueCritic,
    DiscreteQuantileNetwork,
    QuantileCritic,
)

__all__ = [
    "ContinuousQCritic",
    "ContinuousValueCritic",
    "DiscreteQuantileNetwork",
    "QuantileCritic",
]
