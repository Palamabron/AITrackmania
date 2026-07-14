"""Value and quantile critics for TMRL learners."""

from tmrl.models.critics.value import ContinuousQCritic, DiscreteQuantileNetwork, QuantileCritic

__all__ = ["ContinuousQCritic", "DiscreteQuantileNetwork", "QuantileCritic"]
