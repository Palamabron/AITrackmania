"""Reference continuous TQC baseline for OpenPlanet telemetry observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.actors import GaussianActor, GaussianActorConfig, PpoGaussianActor
from trackmaniarl.models.critics import (
    ContinuousValueCritic,
    QuantileCritic,
    QuantileCriticConfig,
)
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


@dataclass(frozen=True, slots=True)
class TelemetryTqcConfig:
    input_dim: int = DEFAULT_TELEMETRY_FIELD_COUNT
    action_dim: int = 3
    hidden_dim: int = 256
    quantiles: int = 25
    critics: int = 5

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        return cls(**dict(values))


def _encoder(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())


class TelemetryTqcModel(nn.Module):
    """Five-critic TQC bundle matching the original ensemble formulation."""

    def __init__(self, config: TelemetryTqcConfig | None = None) -> None:
        super().__init__()
        config = config or TelemetryTqcConfig()
        _validate_tqc_shape(config)
        self.actor = _tqc_actor(config.input_dim, config.action_dim, config.hidden_dim)
        self.critics = _tqc_critics(config)


def _tqc_actor(input_dim: int, action_dim: int, hidden_dim: int) -> GaussianActor:
    return GaussianActor(
        _encoder(input_dim, hidden_dim),
        GaussianActorConfig(
            hidden_dim,
            action_dim,
            action_low=(0.0, 0.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
        ),
    )


def _tqc_critics(shape: TelemetryTqcConfig) -> nn.ModuleList:
    return nn.ModuleList(
        [
            QuantileCritic(
                _encoder(shape.input_dim, shape.hidden_dim),
                QuantileCriticConfig(shape.hidden_dim, shape.action_dim, shape.quantiles),
            )
            for _ in range(shape.critics)
        ]
    )


def _validate_tqc_shape(shape: TelemetryTqcConfig) -> None:
    if shape.critics < 2 or shape.quantiles < 2:
        raise ValueError("TQC requires at least two critics and quantiles")
    if shape.action_dim != 3:
        raise ValueError("TelemetryTqcModel requires Trackmania's three control dimensions")


class TelemetryTqcModelFactory:
    model_contract = ModelContract.CONTINUOUS_QUANTILE_ACTOR_CRITIC

    def __init__(self, config: TelemetryTqcConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            config = TelemetryTqcConfig()
        elif not isinstance(config, TelemetryTqcConfig):
            config = TelemetryTqcConfig.from_mapping(config)
        self.config = config

    def build(self) -> TelemetryTqcModel:
        return TelemetryTqcModel(self.config)


class TelemetryPpoModel(nn.Module):
    """PPO actor-value bundle with native Trackmania control bounds."""

    def __init__(
        self,
        input_dim: int = DEFAULT_TELEMETRY_FIELD_COUNT,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.actor = PpoGaussianActor(
            _encoder(input_dim, hidden_dim),
            GaussianActorConfig(
                hidden_dim,
                3,
                action_low=(0.0, 0.0, -1.0),
                action_high=(1.0, 1.0, 1.0),
            ),
        )
        self.value = ContinuousValueCritic(_encoder(input_dim, hidden_dim), hidden_dim)
        _initialize_value(self.value)


class TelemetryPpoModelFactory:
    model_contract = ModelContract.CONTINUOUS_ACTOR_VALUE

    def __init__(
        self,
        input_dim: int = DEFAULT_TELEMETRY_FIELD_COUNT,
        hidden_dim: int = 256,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def build(self) -> TelemetryPpoModel:
        return TelemetryPpoModel(self.input_dim, self.hidden_dim)


def _initialize_value(value: ContinuousValueCritic) -> None:
    for module in value.encoder.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, 2**0.5)
            nn.init.zeros_(module.bias)
    nn.init.orthogonal_(value.value.weight, 1.0)
    nn.init.zeros_(value.value.bias)
