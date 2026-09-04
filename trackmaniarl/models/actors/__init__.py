"""Policy heads for continuous and discrete control."""

from trackmaniarl.models.actors.continuous import (
    GaussianActor,
    GaussianActorConfig,
    PpoGaussianActor,
)
from trackmaniarl.models.actors.discrete import CategoricalActor

__all__ = ["CategoricalActor", "GaussianActor", "GaussianActorConfig", "PpoGaussianActor"]
