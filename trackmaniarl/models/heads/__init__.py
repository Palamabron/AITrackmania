"""Composable value and actor heads."""

from trackmaniarl.models.heads.actor import ContinuousActorHead
from trackmaniarl.models.heads.critic import ContinuousCriticHead
from trackmaniarl.models.heads.fixed_quantile import FixedQuantileHead
from trackmaniarl.models.heads.implicit_quantile import ImplicitQuantileHead
from trackmaniarl.models.heads.scalar_q import ScalarQHead

__all__ = [
    "ContinuousActorHead",
    "ContinuousCriticHead",
    "FixedQuantileHead",
    "ImplicitQuantileHead",
    "ScalarQHead",
]
