from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class ResidualMlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=residual_mlp_actor_critic` preset."""

    type: Literal["residual_mlp_actor_critic"] = Field(default="residual_mlp_actor_critic")
