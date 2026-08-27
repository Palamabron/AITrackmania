"""Behavior-cloning model and inference policy entry points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, TypedDict, Unpack, cast

import torch
from torch import nn

from trackmaniarl.core.contracts import ModelContract, PolicyMode
from trackmaniarl.models.composite import BatchLayout, FrameBatchAdapter
from trackmaniarl.models.temporal import GruTemporalCore, IdentityTemporalCore
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.encoders import (
    LidarSensorConfig,
    LidarSensorEncoder,
    LidarSimbaSensorEncoder,
)


class ModelOptions(TypedDict):
    action_ids: tuple[int, ...]
    telemetry_dim: NotRequired[int]
    history_length: NotRequired[int]
    spatial_bins: NotRequired[int]
    burn_in: NotRequired[int]
    lidar_channels: NotRequired[int]
    telemetry_group_dims: NotRequired[tuple[int, ...] | None]
    encoder_hidden_dim: NotRequired[int]
    encoder_output_dim: NotRequired[int]
    previous_action_conditioning: NotRequired[bool]
    previous_action_embedding_dim: NotRequired[int]
    minimum_action_hold_steps: NotRequired[int]
    switch_logit_margin: NotRequired[float]
    masked_telemetry_indices: NotRequired[tuple[int, ...]]
    simba_backbone: NotRequired[Mapping[str, Any] | None]
    factorized_action_head: NotRequired[bool]


@dataclass(frozen=True, slots=True)
class _ModelConfiguration:
    action_ids: tuple[int, ...]
    telemetry_dim: int = 26
    history_length: int = 1
    spatial_bins: int = 12
    burn_in: int = 0
    lidar_channels: int = 4
    telemetry_group_dims: tuple[int, ...] | None = None
    encoder_hidden_dim: int = 192
    encoder_output_dim: int = 256
    previous_action_conditioning: bool = False
    previous_action_embedding_dim: int = 16
    minimum_action_hold_steps: int = 1
    switch_logit_margin: float = 0.0
    masked_telemetry_indices: tuple[int, ...] = ()
    simba_backbone: Mapping[str, Any] | None = None
    factorized_action_head: bool = False

    @classmethod
    def from_options(cls, options: ModelOptions) -> _ModelConfiguration:
        return cls(**options)

    def as_options(self) -> ModelOptions:
        return cast(ModelOptions, asdict(self))


class _FactorizedActionHead(nn.Module):
    def __init__(self, input_dim: int, action_ids: tuple[int, ...]) -> None:
        super().__init__()
        canonical = torch.tensor(action_ids, dtype=torch.long)
        self.steering = nn.Linear(input_dim, 13)
        self.drive_mode = nn.Linear(input_dim, 6)
        self.steering_indices: torch.Tensor
        self.drive_mode_indices: torch.Tensor
        self.register_buffer("steering_indices", canonical // 6, persistent=False)
        self.register_buffer("drive_mode_indices", canonical % 6, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        steering = self.steering(features).index_select(-1, self.steering_indices)
        drive_mode = self.drive_mode(features).index_select(-1, self.drive_mode_indices)
        return cast(torch.Tensor, steering + drive_mode)


class LidarBehaviorCloningModel(nn.Module):
    """Categorical policy over an explicit compact action set and frame history."""

    def __init__(self, **options: Unpack[ModelOptions]) -> None:
        super().__init__()
        configuration = _ModelConfiguration.from_options(options)
        _validate_model_configuration(configuration)
        self.action_ids = tuple(configuration.action_ids)
        select_brake_tap_actions(self.action_ids)
        self.action_count = len(self.action_ids)
        self.previous_action_conditioning = configuration.previous_action_conditioning
        self.previous_action_start = self.action_count
        self.minimum_action_hold_steps = configuration.minimum_action_hold_steps
        self.switch_logit_margin = configuration.switch_logit_margin
        self.feedforward_history = configuration.simba_backbone is not None
        self.encoder, self.temporal, self.burn_in = _encoder_stack(configuration)
        self.previous_action_embedding, self.head = _policy_head(
            configuration, self.encoder.output_dim, self.action_ids
        )

    def initial_policy_state(self, device: torch.device) -> Any:
        return self.temporal.initial_state(1, device)

    def policy_logits(
        self, observation: Mapping[str, torch.Tensor], state: Any
    ) -> tuple[torch.Tensor, Any]:
        if not self.feedforward_history and observation["lidar"].ndim == 4:
            return self(observation), state
        frames = {key: observation[key] for key in ("lidar", "lidar_mask", "telemetry")}
        if self.feedforward_history:
            features = self.encoder(frames)
        else:
            batch = FrameBatchAdapter.flatten(frames, BatchLayout.FRAMES)
            features = self.encoder(cast(Any, batch.frames))
        encoded, next_state = self.temporal.step(features, state)
        return self._logits(encoded, observation), next_state

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        frames = {key: observation[key] for key in ("lidar", "lidar_mask", "telemetry")}
        if self.feedforward_history:
            return self._logits(self.encoder(frames), observation)
        sequence = observation["lidar"].ndim == 4
        layout = BatchLayout.SEQUENCE if sequence else BatchLayout.FRAMES
        batch = FrameBatchAdapter.flatten(frames, layout)
        features = batch.restore(self.encoder(cast(Any, batch.frames)))
        encoded = self.temporal.unroll(features, self.burn_in)[:, -1]
        return self._logits(encoded, observation)

    def _logits(
        self, encoded: torch.Tensor, observation: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.previous_action_embedding is not None:
            previous_action = observation.get("previous_action")
            if previous_action is None:
                raise ValueError("previous_action is required by this behavior-cloning model")
            if previous_action.ndim == 2:
                previous_action = previous_action[:, -1]
            encoded = torch.cat(
                (encoded, self.previous_action_embedding(previous_action.long())), dim=-1
            )
        return cast(torch.Tensor, self.head(encoded))


class LidarBehaviorCloningModelFactory:
    model_contract = ModelContract.CATEGORICAL_POLICY

    def __init__(self, **options: Unpack[ModelOptions]) -> None:
        configuration = _ModelConfiguration.from_options(options)
        self.configuration = configuration
        _assign_factory_attributes(self, configuration)

    def build(self) -> LidarBehaviorCloningModel:
        return LidarBehaviorCloningModel(**self.configuration.as_options())


class BehaviorCloningPolicy:
    def __init__(self, model: LidarBehaviorCloningModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.previous_action = model.previous_action_start
        self.action_hold_steps = 0
        self.temporal_state = model.initial_policy_state(device)

    def reset_episode(self) -> None:
        self.previous_action = self.model.previous_action_start
        self.action_hold_steps = 0
        self.temporal_state = self.model.initial_policy_state(self.device)

    def act(
        self,
        observation: Mapping[str, torch.Tensor],
        mode: PolicyMode = PolicyMode.ONLINE,
    ) -> int:
        del mode
        batched = {key: value.to(self.device).unsqueeze(0) for key, value in observation.items()}
        if self.model.previous_action_conditioning:
            batched["previous_action"] = torch.tensor(
                [self.previous_action], device=self.device, dtype=torch.long
            )
        with torch.inference_mode():
            logits, self.temporal_state = self.model.policy_logits(batched, self.temporal_state)
            logits = logits.squeeze(0)
        return self._select_action(logits)

    def _select_action(self, logits: torch.Tensor) -> int:
        action = int(logits.argmax().item())
        if self.previous_action < self.model.action_count and action != self.previous_action:
            switch_margin = float(logits[action] - logits[self.previous_action])
            if (
                self.action_hold_steps < self.model.minimum_action_hold_steps
                or switch_margin < self.model.switch_logit_margin
            ):
                action = self.previous_action
        self.action_hold_steps = self.action_hold_steps + 1 if action == self.previous_action else 1
        self.previous_action = action
        return action


def _validate_model_configuration(configuration: _ModelConfiguration) -> None:
    if configuration.previous_action_embedding_dim < 1:
        raise ValueError("previous-action policy dimensions must be positive")
    if configuration.minimum_action_hold_steps < 1:
        raise ValueError("previous-action policy dimensions must be positive")
    if configuration.switch_logit_margin < 0.0:
        raise ValueError("switch_logit_margin must be non-negative")
    if configuration.simba_backbone is not None and configuration.burn_in:
        raise ValueError("feed-forward Simba history does not use burn-in")


def _sensor_configuration(configuration: _ModelConfiguration) -> dict[str, Any]:
    return {
        "telemetry_dim": configuration.telemetry_dim,
        "spatial_bins": configuration.spatial_bins,
        "lidar_channels": configuration.lidar_channels,
        "telemetry_group_dims": configuration.telemetry_group_dims,
        "hidden_dim": configuration.encoder_hidden_dim,
        "output_dim": configuration.encoder_output_dim,
        "masked_telemetry_indices": configuration.masked_telemetry_indices,
    }


def _encoder_stack(
    configuration: _ModelConfiguration,
) -> tuple[
    LidarSensorEncoder | LidarSimbaSensorEncoder, IdentityTemporalCore | GruTemporalCore, int
]:
    sensor = _sensor_configuration(configuration)
    if configuration.simba_backbone is not None:
        feedforward_encoder = LidarSimbaSensorEncoder(
            sensor, configuration.simba_backbone, configuration.history_length
        )
        return feedforward_encoder, IdentityTemporalCore(feedforward_encoder.output_dim), 0
    recurrent_encoder = LidarSensorEncoder(LidarSensorConfig.from_mapping(sensor))
    temporal = _temporal_core(configuration)
    burn_in = configuration.burn_in if configuration.history_length > 1 else 0
    return recurrent_encoder, temporal, burn_in


def _temporal_core(configuration: _ModelConfiguration) -> IdentityTemporalCore | GruTemporalCore:
    if configuration.history_length == 1:
        return IdentityTemporalCore(configuration.encoder_output_dim)
    return GruTemporalCore(configuration.encoder_output_dim, configuration.encoder_output_dim)


def _policy_head(
    configuration: _ModelConfiguration, encoder_output_dim: int, action_ids: tuple[int, ...]
) -> tuple[nn.Embedding | None, nn.Linear | _FactorizedActionHead]:
    embedding = _previous_action_embedding(configuration, len(action_ids))
    input_dim = encoder_output_dim + (
        configuration.previous_action_embedding_dim if embedding is not None else 0
    )
    if configuration.factorized_action_head:
        return embedding, _FactorizedActionHead(input_dim, action_ids)
    return embedding, nn.Linear(input_dim, len(action_ids))


def _previous_action_embedding(
    configuration: _ModelConfiguration, action_count: int
) -> nn.Embedding | None:
    if not configuration.previous_action_conditioning:
        return None
    return nn.Embedding(action_count + 1, configuration.previous_action_embedding_dim)


def _assign_factory_attributes(
    factory: LidarBehaviorCloningModelFactory, configuration: _ModelConfiguration
) -> None:
    for name, value in asdict(configuration).items():
        setattr(factory, name, value)


__all__ = [
    "BehaviorCloningPolicy",
    "LidarBehaviorCloningModel",
    "LidarBehaviorCloningModelFactory",
]
