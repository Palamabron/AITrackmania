from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class SophyResidualActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=sophy_residual_actor_critic` preset."""

    type: Literal["sophy_residual_actor_critic"] = Field(default="sophy_residual_actor_critic")
