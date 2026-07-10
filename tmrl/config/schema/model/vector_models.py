"""Model config schemas for vector-input (MLP / residual MLP / RNN) actor-critic presets."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tmrl.config.schema.model.base import BaseModelConfig


class MlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=mlp_actor_critic` preset."""

    type: Literal["mlp_actor_critic"] = Field(default="mlp_actor_critic")


class ResidualMlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=residual_mlp_actor_critic` preset."""

    type: Literal["residual_mlp_actor_critic"] = Field(default="residual_mlp_actor_critic")


class RedqMlpActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=redq_mlp_actor_critic` preset."""

    type: Literal["redq_mlp_actor_critic"] = Field(default="redq_mlp_actor_critic")


class RnnActorCriticModelConfig(BaseModelConfig):
    """Schema for `model=rnn_actor_critic` preset."""

    type: Literal["rnn_actor_critic"] = Field(default="rnn_actor_critic")
