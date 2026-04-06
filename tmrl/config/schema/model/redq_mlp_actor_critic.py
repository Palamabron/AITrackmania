from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class RedqMlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=redq_mlp_actor_critic` preset."""

    type: Literal["redq_mlp_actor_critic"] = Field(default="redq_mlp_actor_critic")
