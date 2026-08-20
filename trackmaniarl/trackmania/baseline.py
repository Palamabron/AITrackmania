"""Reference continuous TQC baseline for OpenPlanet telemetry observations."""

from __future__ import annotations

from torch import nn

from trackmaniarl.core.contracts import ModelContract
from trackmaniarl.models.actors import GaussianActor, PpoGaussianActor
from trackmaniarl.models.critics import ContinuousValueCritic, QuantileCritic
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


def _encoder(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())


class TelemetryTqcModel(nn.Module):
    """Five-critic TQC bundle matching the original ensemble formulation."""

    def __init__(
        self,
        input_dim: int = DEFAULT_TELEMETRY_FIELD_COUNT,
        action_dim: int = 3,
        hidden_dim: int = 256,
        quantiles: int = 25,
        critics: int = 5,
    ) -> None:
        super().__init__()
        if critics < 2 or quantiles < 2:
            raise ValueError("TQC requires at least two critics and quantiles")
        self.actor = GaussianActor(_encoder(input_dim, hidden_dim), hidden_dim, action_dim)
        self.critics = nn.ModuleList(
            [
                QuantileCritic(_encoder(input_dim, hidden_dim), hidden_dim, action_dim, quantiles)
                for _ in range(critics)
            ]
        )


class TelemetryTqcModelFactory:
    model_contract = ModelContract.CONTINUOUS_QUANTILE_ACTOR_CRITIC

    def __init__(
        self,
        input_dim: int = DEFAULT_TELEMETRY_FIELD_COUNT,
        action_dim: int = 3,
        hidden_dim: int = 256,
        quantiles: int = 25,
        critics: int = 5,
    ) -> None:
        self.input_dim, self.action_dim, self.hidden_dim = input_dim, action_dim, hidden_dim
        self.quantiles, self.critics = quantiles, critics

    def build(self) -> TelemetryTqcModel:
        return TelemetryTqcModel(
            self.input_dim, self.action_dim, self.hidden_dim, self.quantiles, self.critics
        )


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
            hidden_dim,
            3,
            action_low=(0.0, 0.0, -1.0),
            action_high=(1.0, 1.0, 1.0),
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
