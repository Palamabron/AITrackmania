"""Configuration and immutable assets for lidar feature pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from gymnasium import spaces

from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.pace import (
    PaceDemonstrationRequest,
    ReferencePaceProfile,
    demonstration_guidance_line,
)


@dataclass(frozen=True, slots=True)
class LidarFeatureConfig:
    geometry_path: str | Path
    expected_map_uid: str | None = None
    samples_per_side: int = 60
    max_distance_m: float = 300.0
    history_length: int = 1
    include_track_relative: bool = False
    include_control_inputs: bool = True
    mask_current_control_inputs: bool = False
    local_velocity_features: bool = False
    use_racing_line: bool = False
    max_speed_mps: float = 80.0
    velocity_to_mps_scale: float = 0.001
    max_time_delta_s: float = 1.0
    limit_progress_by_kinematics: bool = False
    nearest_forward_points: int = 128
    nearest_backward_points: int = 10
    pace_reference_path: str | Path | None = None
    pace_debt_clip_s: float = 10.0
    reference_speed_offsets_m: tuple[float, ...] = (0.0, 20.0, 40.0, 80.0)
    include_racing_line_channels: bool = False
    include_finish_channels: bool = False
    include_dynamics: bool = False
    include_goal_features: bool = False
    base_dir: str | Path = "."

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Self:
        return cls(**dict(values))


@dataclass(frozen=True, slots=True)
class LidarFeatureAssets:
    geometry: BoundaryGeometry
    pace_profile: ReferencePaceProfile | None
    expert_guidance_line: np.ndarray | None
    reference_line: np.ndarray


@dataclass(frozen=True, slots=True)
class LidarFeatureLayout:
    lidar_channels: int
    telemetry_dim: int
    observation_space: spaces.Dict


def validate_feature_config(config: LidarFeatureConfig) -> None:
    _validate_counts(config)
    _validate_scales(config)
    if config.mask_current_control_inputs and not config.include_control_inputs:
        raise ValueError("current control masking requires control inputs")


def _validate_counts(config: LidarFeatureConfig) -> None:
    if config.samples_per_side < 2:
        raise ValueError("samples_per_side must be at least two")
    if config.history_length < 1:
        raise ValueError("history_length must be positive")
    if config.nearest_forward_points < 1 or config.nearest_backward_points < 0:
        raise ValueError("distance and speed scales must be positive")


def _validate_scales(config: LidarFeatureConfig) -> None:
    positive = (
        config.max_distance_m,
        config.max_speed_mps,
        config.velocity_to_mps_scale,
        config.max_time_delta_s,
        config.pace_debt_clip_s,
    )
    if any(value <= 0.0 for value in positive):
        raise ValueError("distance and speed scales must be positive")
    if any(offset < 0.0 for offset in config.reference_speed_offsets_m):
        raise ValueError("distance and speed scales must be positive")


def load_feature_assets(config: LidarFeatureConfig) -> LidarFeatureAssets:
    geometry_path = _resolve_path(config.geometry_path, config.base_dir)
    geometry = BoundaryGeometry(geometry_path, expected_map_uid=config.expected_map_uid)
    reference_line = geometry.racing_line if config.use_racing_line else geometry.reward_center
    pace_profile, guidance = _load_pace_assets(config, geometry, reference_line)
    return LidarFeatureAssets(geometry, pace_profile, guidance, reference_line)


def _load_pace_assets(
    config: LidarFeatureConfig,
    geometry: BoundaryGeometry,
    reference_line: np.ndarray,
) -> tuple[ReferencePaceProfile | None, np.ndarray | None]:
    if config.pace_reference_path is None:
        return None, None
    pace_path = _resolve_path(config.pace_reference_path, config.base_dir)
    request = PaceDemonstrationRequest(pace_path, geometry, reference_line)
    pace = ReferencePaceProfile.from_demonstration(request)
    if not config.include_racing_line_channels:
        return pace, None
    guidance = demonstration_guidance_line(pace_path, geometry, reference_line)
    return pace, guidance


def _resolve_path(path: str | Path, base_dir: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (Path(base_dir) / resolved).resolve()


def build_feature_layout(
    config: LidarFeatureConfig, telemetry_field_count: int, pace_feature_count: int
) -> LidarFeatureLayout:
    lidar_channels = 4 + 2 * int(config.include_racing_line_channels)
    lidar_channels += 2 * int(config.include_finish_channels)
    telemetry_dim = _telemetry_dimension(config, telemetry_field_count, pace_feature_count)
    observation_space = _observation_space(config, lidar_channels, telemetry_dim)
    return LidarFeatureLayout(lidar_channels, telemetry_dim, observation_space)


def _telemetry_dimension(
    config: LidarFeatureConfig, field_count: int, pace_feature_count: int
) -> int:
    control_count = 3 if config.include_control_inputs else 0
    track_count = 6 if config.include_track_relative else 0
    return (
        field_count
        - 3
        + control_count
        + track_count
        + pace_feature_count
        + 4 * int(config.include_dynamics)
        + 14 * int(config.include_goal_features)
    )


def _observation_space(
    config: LidarFeatureConfig, lidar_channels: int, telemetry_dim: int
) -> spaces.Dict:
    lidar_shape = _history_shape(config.history_length, (lidar_channels, config.samples_per_side))
    mask_shape = _history_shape(config.history_length, (config.samples_per_side,))
    telemetry_shape = _history_shape(config.history_length, (telemetry_dim,))
    return spaces.Dict(
        {
            "lidar": spaces.Box(-1.0, 1.0, shape=lidar_shape, dtype=np.float32),
            "lidar_mask": spaces.Box(0.0, 1.0, shape=mask_shape, dtype=np.float32),
            "telemetry": spaces.Box(-1.0, 1.0, shape=telemetry_shape, dtype=np.float32),
        }
    )


def _history_shape(history_length: int, frame_shape: tuple[int, ...]) -> tuple[int, ...]:
    if history_length == 1:
        return frame_shape
    return (history_length, *frame_shape)
