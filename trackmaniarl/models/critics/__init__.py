"""Value and quantile critics for TrackmaniaRL learners."""

from trackmaniarl.models.critics.value import (
    ContinuousQCritic,
    DiscreteQuantileNetwork,
    QuantileCritic,
)

__all__ = ["ContinuousQCritic", "DiscreteQuantileNetwork", "QuantileCritic"]
