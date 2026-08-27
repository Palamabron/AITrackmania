"""Value and quantile critics for TrackmaniaRL learners."""

from trackmaniarl.models.critics.value import (
    ContinuousQCritic,
    ContinuousValueCritic,
    QuantileCritic,
    QuantileCriticConfig,
)

__all__ = [
    "ContinuousQCritic",
    "ContinuousValueCritic",
    "QuantileCritic",
    "QuantileCriticConfig",
]
