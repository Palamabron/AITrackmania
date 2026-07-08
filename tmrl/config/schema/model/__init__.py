"""Model schemas split per preset, selected via `model.type` discriminator."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from tmrl.config.schema.model.hybrid_models import (
    SophyActorCriticModelConfig,
    SophyResidualActorCriticModelConfig,
)
from tmrl.config.schema.model.image_models import (
    EffnetActorCriticModelConfig,
    VanillaCnnActorCriticModelConfig,
    VanillaColorCnnActorCriticModelConfig,
)
from tmrl.config.schema.model.vector_models import (
    MlpActorCriticModelConfig,
    RedqMlpActorCriticModelConfig,
    ResidualMlpActorCriticModelConfig,
    RnnActorCriticModelConfig,
)

type ModelConfig = Annotated[
    (
        VanillaCnnActorCriticModelConfig
        | VanillaColorCnnActorCriticModelConfig
        | SophyActorCriticModelConfig
        | SophyResidualActorCriticModelConfig
        | MlpActorCriticModelConfig
        | ResidualMlpActorCriticModelConfig
        | RedqMlpActorCriticModelConfig
        | RnnActorCriticModelConfig
        | EffnetActorCriticModelConfig
    ),
    Field(discriminator="type"),
]

__all__ = [
    "EffnetActorCriticModelConfig",
    "MlpActorCriticModelConfig",
    "ModelConfig",
    "RedqMlpActorCriticModelConfig",
    "ResidualMlpActorCriticModelConfig",
    "RnnActorCriticModelConfig",
    "SophyActorCriticModelConfig",
    "SophyResidualActorCriticModelConfig",
    "VanillaCnnActorCriticModelConfig",
    "VanillaColorCnnActorCriticModelConfig",
]
