from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class VanillaCnnActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=vanilla_cnn_actor_critic` preset."""

    discrete_action_compatible: ClassVar[bool] = False

    type: Literal["vanilla_cnn_actor_critic"] = Field(default="vanilla_cnn_actor_critic")
