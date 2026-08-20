"""Value-distribution strategies for scalar and quantile learners."""

from trackmaniarl.models.strategies.fixed_quantiles import FixedQuantileStrategy
from trackmaniarl.models.strategies.learned_fractions import LearnedFractionStrategy
from trackmaniarl.models.strategies.random_quantiles import RandomQuantileStrategy
from trackmaniarl.models.strategies.scalar import ScalarValueStrategy

__all__ = [
    "FixedQuantileStrategy",
    "LearnedFractionStrategy",
    "RandomQuantileStrategy",
    "ScalarValueStrategy",
]
