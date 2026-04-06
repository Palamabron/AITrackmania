from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class SophyActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=sophy_actor_critic` preset."""

    type: Literal["sophy_actor_critic"] = Field(default="sophy_actor_critic")
