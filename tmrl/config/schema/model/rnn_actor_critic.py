from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class RnnActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=rnn_actor_critic` preset."""

    type: Literal["rnn_actor_critic"] = Field(default="rnn_actor_critic")
