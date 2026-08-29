"""Opt-in graph encoders and IQN components for TrackMania."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition
from trackmaniarl.models.backbones import SimbaV2Backbone
from trackmaniarl.models.contracts import ValueRepresentation, ValueSupport
from trackmaniarl.models.track_graphs import TrackGraphTransformer, TrackNeighborGraph
from trackmaniarl.trackmania.geometry import BoundaryGeometry


def _mask_control_labels(physics: torch.Tensor) -> torch.Tensor:
    masked = physics.clone()
    masked[..., 4:7] = 0.0
    masked[..., 10:12] = 0.0
    masked[..., -1] = 0.0
    return masked


def _graph_observation(
    observation: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    track = observation["track"].float()
    physics = observation["physics"].float()
    if track.ndim != 3 or track.shape[1:] != (3, 88):
        raise ValueError("track observation must have shape (batch, 3, 88)")
    if physics.shape != (track.shape[0], 60):
        raise ValueError("physics observation must have shape (batch, 60)")
    return track, _mask_control_labels(physics)


class TrackGnnSimbaEncoder(nn.Module):
    """Neighbor GNN and physics encoder followed by a SimbaV2 backbone."""

    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.track_conv = nn.Sequential(
            TrackNeighborGraph(), nn.Linear(128, 192), nn.LayerNorm(192), nn.SiLU()
        )
        self.physics_proj = nn.Sequential(nn.Linear(60, 192), nn.LayerNorm(192), nn.SiLU())
        self.layernorm_joint = nn.LayerNorm(384)
        self.backbone = SimbaV2Backbone(384, 192, block_count=4, expansion=4)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        track, physics = _graph_observation(observation)
        joint = torch.cat((self.track_conv(track), self.physics_proj(physics)), dim=-1)
        return cast(torch.Tensor, self.backbone(self.layernorm_joint(joint)))


class TrackGtnSimbaEncoder(nn.Module):
    """Graph-transformer and physics encoder followed by a SimbaV2 backbone."""

    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.track_encoder = nn.Sequential(
            TrackGraphTransformer(), nn.Linear(128, 192), nn.LayerNorm(192), nn.SiLU()
        )
        self.physics_projection = nn.Sequential(nn.Linear(60, 192), nn.LayerNorm(192), nn.SiLU())
        self.joint_norm = nn.LayerNorm(384)
        self.backbone = SimbaV2Backbone(384, 192, block_count=4, expansion=4)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        track, physics = _graph_observation(observation)
        joint = torch.cat((self.track_encoder(track), self.physics_projection(physics)), dim=-1)
        return cast(torch.Tensor, self.backbone(self.joint_norm(joint)))


class DuelingImplicitQuantileHead(nn.Module):
    representation = ValueRepresentation.IMPLICIT_QUANTILE
    feature_dim = 192
    action_count = 78

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("i_pi", torch.arange(1, 65, dtype=torch.float32) * torch.pi)
        self.quantile_linear = nn.Linear(64, 192)
        self.value_stream = nn.Sequential(nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(192, 192), nn.SiLU(), nn.Linear(192, 78))

    def evaluate_all(self, features: torch.Tensor, support: ValueSupport) -> torch.Tensor:
        combined = self._combined(features, support.points)
        value = self.value_stream(combined)
        advantage = self.advantage_stream(combined)
        return cast(torch.Tensor, value + advantage - advantage.mean(dim=-1, keepdim=True))

    def evaluate_actions(
        self, features: torch.Tensor, support: ValueSupport, actions: torch.Tensor
    ) -> torch.Tensor:
        combined = self._combined(features, support.points)
        value = self.value_stream(combined).squeeze(-1)
        selected, mean_advantage = self._advantage_statistics(combined, actions)
        return cast(torch.Tensor, value + selected - mean_advantage)

    def _advantage_statistics(
        self, combined: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = cast(nn.SiLU, self.advantage_stream[1])(
            cast(nn.Linear, self.advantage_stream[0])(combined)
        )
        output_layer = cast(nn.Linear, self.advantage_stream[2])
        selected_weight = output_layer.weight[actions].unsqueeze(-2)
        if output_layer.bias is None:
            raise RuntimeError("dueling advantage output requires a bias")
        selected_bias = output_layer.bias[actions].unsqueeze(-1)
        selected = (hidden * selected_weight).sum(dim=-1) + selected_bias
        mean_advantage = F.linear(
            hidden,
            output_layer.weight.mean(dim=0).unsqueeze(0),
            output_layer.bias.mean().reshape(1),
        )
        return selected, mean_advantage.squeeze(-1)

    def _combined(self, features: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.shape[:-1] != features.shape[:-1]:
            raise ValueError("quantile support must share feature leading dimensions")
        cosine = torch.cos(points.unsqueeze(-1) * cast(torch.Tensor, self.i_pi))
        return features.unsqueeze(-2) * F.relu(self.quantile_linear(cosine))


class BoundaryGraphFeaturePipeline:
    """Build boundary-graph and physics features from 33-field telemetry."""

    def __init__(
        self,
        geometry_path: str | Path,
        expected_map_uid: str,
        base_dir: str | Path = ".",
    ) -> None:
        path = self._resolve_path(geometry_path, base_dir)
        self.expected_map_uid = expected_map_uid
        self._install_geometry(BoundaryGeometry(path, expected_map_uid=expected_map_uid))
        self.observation_space = self._build_observation_space()
        self._collator = GymnasiumObservationCollator(self.observation_space)
        self.reset_episode()

    @staticmethod
    def _build_observation_space() -> spaces.Dict:
        physics_low = np.full(60, -1_000.0, dtype=np.float32)
        physics_high = np.full(60, 1_000.0, dtype=np.float32)
        physics_low[-1], physics_high[-1] = 0.0, 77.0
        return spaces.Dict(
            {
                "physics": spaces.Box(physics_low, physics_high, dtype=np.float32),
                "track": spaces.Box(-500.0, 500.0, (3, 88), dtype=np.float32),
            }
        )

    def set_evaluation_map(self, map_spec: Any) -> None:
        if map_spec.expected_map_uid != self.expected_map_uid:
            raise ValueError("evaluation map does not match feature geometry")
        geometry = BoundaryGeometry(
            map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
        )
        self._install_geometry(geometry)
        self.reset_episode()

    def reset_episode(self) -> None:
        self._progress_index = 0
        self._last_time_ms: float | None = None
        self._last_acceleration = 0.0
        self._last_speed_mps = 0.0

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, Mapping):
            return self._validate_prepared(observation)
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("graph features require finite 33-field telemetry")
        position = values[[4, 5, 6]]
        nearest = self._nearest(position)
        track, curvature = self._track_features(position, values[[10, 12]], nearest)
        physics = self._physics(values, nearest, curvature)
        return {"physics": torch.from_numpy(physics), "track": torch.from_numpy(track)}

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))

    def synthetic_observation(self) -> np.ndarray:
        values = np.zeros(33, dtype=np.float32)
        values[4:7] = self._reward_center[0]
        direction = self._reward_center[1] - self._reward_center[0]
        values[10:13] = direction / max(float(np.linalg.norm(direction)), 1.0e-6)
        return values

    def _validate_prepared(self, observation: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        if set(observation) != {"physics", "track"}:
            raise ValueError("prepared observation requires physics and track")
        result = {
            "physics": torch.as_tensor(observation["physics"], dtype=torch.float32),
            "track": torch.as_tensor(observation["track"], dtype=torch.float32),
        }
        if result["physics"].shape != (60,) or result["track"].shape != (3, 88):
            raise ValueError("prepared observation has invalid shape")
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError("prepared observation contains non-finite values")
        result["physics"] = _mask_control_labels(result["physics"])
        return result

    def _nearest(self, position: np.ndarray) -> int:
        line = self._reward_center
        start = max(0, self._progress_index - 10)
        stop = min(len(line), self._progress_index + 101)
        local = np.linalg.norm(line[start:stop] - position, axis=1)
        self._progress_index = start + int(np.argmin(local))
        return self._progress_index

    def _track_features(
        self, position: np.ndarray, direction_xz: np.ndarray, nearest: int
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = self._lookahead_indices(nearest)
        heading = direction_xz / max(float(np.linalg.norm(direction_xz)), 1e-6)
        rotation = np.array(((heading[1], -heading[0]), (heading[0], heading[1])), dtype=np.float32)
        channels: list[np.ndarray] = []
        for line in (self._left, self._center, self._right):
            relative = line[indices][:, [0, 2]] - position[[0, 2]]
            channels.append((relative @ rotation.T).reshape(-1))
        track = np.asarray(channels, dtype=np.float32).reshape(3, 88)
        return track, self._curvature(indices)

    def _lookahead_indices(self, nearest: int) -> np.ndarray:
        targets = self._distance[nearest] + 2.5 * np.arange(1, 45, dtype=np.float32)
        indices = np.searchsorted(self._distance, targets).clip(0, len(self._distance) - 1)
        return np.asarray(indices, dtype=np.int64)

    def _curvature(self, indices: np.ndarray) -> np.ndarray:
        center = self._center
        before = center[np.maximum(indices - 1, 0)][:, [0, 2]]
        current = center[indices][:, [0, 2]]
        after = center[np.minimum(indices + 1, len(center) - 1)][:, [0, 2]]
        first, second = current - before, after - current
        cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
        dot = np.sum(first * second, axis=1)
        curvature = np.arctan2(cross, dot) / np.maximum(
            0.5 * (np.linalg.norm(first, axis=1) + np.linalg.norm(second, axis=1)), 1e-6
        )
        return np.asarray(np.clip(curvature * 10.0, -1.0, 1.0), dtype=np.float32)

    def _physics(self, values: np.ndarray, nearest: int, curvature: np.ndarray) -> np.ndarray:
        derivatives = self._motion_derivatives(values)
        base = self._physics_base(values, nearest, derivatives)
        masked_action = np.zeros(1, dtype=np.float32)
        return np.concatenate((base, curvature.astype(np.float32), masked_action))

    def _motion_derivatives(self, values: np.ndarray) -> tuple[float, float]:
        time_ms, speed_mps = float(values[3]), float(values[16])
        frame_scale = (
            0.0
            if self._last_time_ms is None
            else min(1.0, 10.0 / max(time_ms - self._last_time_ms, 1.0))
        )
        acceleration = (speed_mps - self._last_speed_mps) * frame_scale
        jerk = acceleration - self._last_acceleration
        self._last_time_ms, self._last_speed_mps, self._last_acceleration = (
            time_ms,
            speed_mps,
            acceleration,
        )
        return acceleration, jerk

    def _physics_base(
        self, values: np.ndarray, nearest: int, derivatives: tuple[float, float]
    ) -> np.ndarray:
        acceleration, jerk = derivatives
        speed_mps = float(values[16])
        yaw = float(np.arctan2(values[10], values[12]))
        pitch = float(np.arcsin(np.clip(values[11], -1.0, 1.0)))
        progress = nearest / max(len(self._reward_center) - 1, 1)
        motion = np.array(
            [speed_mps * 3.6, acceleration, jerk, progress, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        return np.concatenate((motion, self._vehicle_features(values, (yaw, pitch))))

    @staticmethod
    def _vehicle_features(values: np.ndarray, orientation: tuple[float, float]) -> np.ndarray:
        yaw, pitch = orientation
        return np.array(
            [
                values[18] / 5.0,
                yaw,
                pitch,
                0.0,
                0.0,
                np.clip(values[19], 0.0, 1.0),
                np.clip(values[20], 0.0, 1.0),
                0.0,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _cumulative_distance(line: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (np.zeros(1), np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1)))
        )

    @staticmethod
    def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else Path(base_dir) / candidate

    def _install_geometry(self, geometry: BoundaryGeometry) -> None:
        recorded = geometry.recorded_count
        self._reward_center = geometry.reward_center
        self._left = geometry.left[:recorded]
        self._center = geometry.center[:recorded]
        self._right = geometry.right[:recorded]
        self._distance = self._cumulative_distance(self._reward_center)
