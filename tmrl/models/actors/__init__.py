"""Policy heads for continuous and discrete control."""

from tmrl.models.actors.continuous import GaussianActor
from tmrl.models.actors.discrete import CategoricalActor

__all__ = ["CategoricalActor", "GaussianActor"]
