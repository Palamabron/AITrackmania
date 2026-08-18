"""Reference continuous TQC baseline for OpenPlanet telemetry observations."""

from __future__ import annotations

from torch import nn

from trackmaniarl.models.actors import GaussianActor
from trackmaniarl.models.critics import QuantileCritic
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
