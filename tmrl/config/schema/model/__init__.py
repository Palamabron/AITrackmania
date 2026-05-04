"""Model schemas split per preset, selected via `model.type` discriminator."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from tmrl.config.schema.model.effnet_actor_critic import EffnetActorCriticModelConfig
from tmrl.config.schema.model.mlp_actor_critic import MlpActorCriticModelConfig
from tmrl.config.schema.model.redq_mlp_actor_critic import RedqMlpActorCriticModelConfig
from tmrl.config.schema.model.residual_mlp_actor_critic import ResidualMlpActorCriticModelConfig
from tmrl.config.schema.model.rnn_actor_critic import RnnActorCriticModelConfig
from tmrl.config.schema.model.sophy_actor_critic import SophyActorCriticModelConfig
from tmrl.config.schema.model.sophy_residual_actor_critic import SophyResidualActorCriticModelConfig
from tmrl.config.schema.model.vanilla_cnn_actor_critic import VanillaCnnActorCriticModelConfig
from tmrl.config.schema.model.vanilla_color_cnn_actor_critic import (
    VanillaColorCnnActorCriticModelConfig,
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
