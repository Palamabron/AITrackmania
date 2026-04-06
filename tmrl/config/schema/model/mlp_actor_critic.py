from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class MlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=mlp_actor_critic` preset."""

    type: Literal["mlp_actor_critic"] = Field(default="mlp_actor_critic")
