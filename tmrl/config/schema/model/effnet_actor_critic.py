from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class EffnetActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=effnet_actor_critic` preset."""

    type: Literal["effnet_actor_critic"] = Field(default="effnet_actor_critic")
