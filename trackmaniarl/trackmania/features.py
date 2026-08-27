"""Feature pipeline for first-party telemetry baselines."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

import trackmaniarl.trackmania.lidar_geometry_features as lidar_geometry_features
import trackmaniarl.trackmania.lidar_observation as lidar_observation
from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.lidar_feature_setup import (
    LidarFeatureAssets,
    LidarFeatureConfig,
    build_feature_layout,
    load_feature_assets,
    validate_feature_config,
)
from trackmaniarl.trackmania.pace import ReferencePaceProfile
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


class TelemetryFeaturePipeline:
    def transform_observation(self, observation: Any) -> torch.Tensor:
        value = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        if value.numel() != DEFAULT_TELEMETRY_FIELD_COUNT:
            raise ValueError(
                "telemetry requires exactly "
                f"{DEFAULT_TELEMETRY_FIELD_COUNT} fields, got {value.numel()}"
            )
        if not torch.isfinite(value).all():
            raise ValueError("telemetry observation contains non-finite values")
        return value

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return {
            "_trackmaniarl_batch_collated": True,
            "observations": torch.stack(
                [self.transform_observation(item.observation) for item in transitions]
            ),
            "actions": torch.stack(
                [torch.as_tensor(item.action, dtype=torch.float32) for item in transitions]
            ),
            "rewards": torch.tensor([item.reward for item in transitions], dtype=torch.float32),
            "next_observations": torch.stack(
                [self.transform_observation(item.next_observation) for item in transitions]
            ),
            "terminated": torch.tensor([item.terminated for item in transitions]),
            "truncated": torch.tensor([item.truncated for item in transitions]),
        }

    def synthetic_observation(self) -> torch.Tensor:
        return torch.zeros(DEFAULT_TELEMETRY_FIELD_COUNT, dtype=torch.float32)


class LidarFeaturePipeline:
    """33-field GrabData source schema projected to telemetry and paired boundary lookahead."""

    schema_version = "5"
    source_fields = (
        "checkpoint",
        "lap",
        "finished",
        "race_time_ms",
        "position_x",
        "position_y",
        "position_z",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "direction_x",
        "direction_y",
        "direction_z",
        "up_x",
        "up_y",
        "up_z",
        "speed",
        "rpm",
        "gear",
        "front_left_slip",
        "front_right_slip",
        "rear_left_slip",
        "rear_right_slip",
        "front_left_surface",
        "front_right_surface",
        "rear_left_surface",
        "rear_right_surface",
        "wheels_skidding_count",
        "flying_duration",
        "adherence",
        "input_steer",
        "input_gas",
        "input_brake",
    )
    telemetry_fields = (
        "checkpoint",
        "lap",
        "finished",
        "race_time_ms",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "speed",
        "rpm",
        "gear",
        "front_left_slip",
        "front_right_slip",
        "rear_left_slip",
        "rear_right_slip",
        "wheels_skidding_count",
        "flying_duration",
        "adherence",
        "input_steer",
        "input_gas",
        "input_brake",
    )
    track_relative_fields = (
        "centerline_progress",
        "lateral_error",
        "heading_sin",
        "heading_cos",
        "projected_velocity",
        "lateral_velocity",
    )
    telemetry_dim = len(telemetry_fields)

    def __init__(self, config: LidarFeatureConfig | Mapping[str, Any]) -> None:
        if not isinstance(config, LidarFeatureConfig):
            config = LidarFeatureConfig.from_mapping(config)
        validate_feature_config(config)
        self._set_core_config(config)
        self._set_feature_flags(config)
        self._set_assets(load_feature_assets(config), config)
        self._initialize_geometry_state()
        self._initialize_episode_state()
        self._initialize_layout(config)

    def _set_core_config(self, config: LidarFeatureConfig) -> None:
        self.samples_per_side = config.samples_per_side
        self.max_distance_m = config.max_distance_m
        self.history_length = config.history_length
        self.max_speed_mps = config.max_speed_mps
        self.velocity_to_mps_scale = config.velocity_to_mps_scale
        self.max_time_delta_s = config.max_time_delta_s
        self.limit_progress_by_kinematics = config.limit_progress_by_kinematics
        self.nearest_forward_points = config.nearest_forward_points
        self.nearest_backward_points = config.nearest_backward_points

    def _set_feature_flags(self, config: LidarFeatureConfig) -> None:
        self.include_track_relative = config.include_track_relative
        self.include_control_inputs = config.include_control_inputs
        self.mask_current_control_inputs = config.mask_current_control_inputs
        self.local_velocity_features = config.local_velocity_features
        self.use_racing_line = config.use_racing_line
        self.include_racing_line_channels = config.include_racing_line_channels
        self.include_finish_channels = config.include_finish_channels
        self.include_dynamics = config.include_dynamics
        self.include_goal_features = config.include_goal_features

    def _set_assets(self, assets: LidarFeatureAssets, config: LidarFeatureConfig) -> None:
        self.geometry = assets.geometry
        self.pace_profile: ReferencePaceProfile | None = assets.pace_profile
        self._expert_guidance_line: np.ndarray | None = assets.expert_guidance_line
        self._reference_line = assets.reference_line
        self.pace_debt_clip_s = config.pace_debt_clip_s
        self.reference_speed_offsets_m = config.reference_speed_offsets_m
        if self.pace_profile is None:
            self.reference_speed_offsets_m = ()

    def _initialize_geometry_state(self) -> None:
        self._reference_cumulative_distance = self._line_cumulative_distance(self._reference_line)
        self._lookahead_line = self._guidance_lookahead_line()
        self._cumulative_distance = self._geometry_cumulative_distance()

    def _initialize_episode_state(self) -> None:
        self._progress_index = 0
        self._last_race_time_ms: float | None = None
        self._last_heading = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self._last_dynamic_race_time_ms: float | None = None
        self._last_dynamic_velocity = np.zeros(2, dtype=np.float32)
        self._last_dynamic_yaw: float | None = None

    def _initialize_layout(self, config: LidarFeatureConfig) -> None:
        pace_count = 0 if self.pace_profile is None else 1 + len(self.reference_speed_offsets_m)
        layout = build_feature_layout(config, len(self.telemetry_fields), pace_count)
        self.lidar_channels = layout.lidar_channels
        self.telemetry_dim = layout.telemetry_dim
        self._history: deque[dict[str, torch.Tensor]] = deque(maxlen=self.history_length)
        self.observation_space = layout.observation_space
        self._collator = GymnasiumObservationCollator(self.observation_space)

    def set_evaluation_map(self, map_spec: Any) -> None:
        """Switch the immutable source asset before evaluating a different declared map."""

        geometry = BoundaryGeometry(
            map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
        )
        if self.pace_profile is not None and geometry.sha256 != self.geometry.sha256:
            raise ValueError("pace reference profiles are only valid for their configured map")
        self.geometry = geometry
        self._reference_line = (
            self.geometry.racing_line if self.use_racing_line else self.geometry.reward_center
        )
        self._reference_cumulative_distance = self._line_cumulative_distance(self._reference_line)
        self._lookahead_line = self._guidance_lookahead_line()
        self._cumulative_distance = self._geometry_cumulative_distance()
        self.reset_episode()

    def reset_episode(self) -> None:
        self._history.clear()
        self._progress_index = 0
        self._last_race_time_ms = None
        self._last_heading = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self._last_dynamic_race_time_ms = None
        self._last_dynamic_velocity.fill(0.0)
        self._last_dynamic_yaw = None

    def _telemetry(self, observation: Any) -> np.ndarray:
        return lidar_geometry_features.telemetry(self, observation)

    def _scale_telemetry(self, values: np.ndarray) -> np.ndarray:
        return lidar_geometry_features.scale_telemetry(self, values)

    def _local_lidar(self, values: np.ndarray, nearest: int) -> tuple[np.ndarray, np.ndarray]:
        return lidar_geometry_features.local_lidar(self, values, nearest)

    def _geometry_cumulative_distance(self) -> np.ndarray:
        return lidar_geometry_features.geometry_cumulative_distance(self)

    @staticmethod
    def _line_cumulative_distance(points: np.ndarray) -> np.ndarray:
        return lidar_geometry_features.line_cumulative_distance(points)

    def _guidance_lookahead_line(self) -> np.ndarray:
        return lidar_geometry_features.guidance_lookahead_line(self)

    def _nearest_progress_index(self, position: np.ndarray, race_time_ms: float) -> int:
        return lidar_geometry_features.nearest_progress_index(self, position, race_time_ms)

    @staticmethod
    def _unit_horizontal(vector: np.ndarray) -> np.ndarray:
        return lidar_geometry_features.unit_horizontal(vector)

    def _horizontal_heading(self, direction: np.ndarray) -> np.ndarray:
        return lidar_geometry_features.horizontal_heading(self, direction)

    def _track_relative(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        return lidar_geometry_features.track_relative(self, values, geometry_index)

    def _pace_features(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        return lidar_geometry_features.pace_features(self, values, geometry_index)

    def _dynamic_features(self, values: np.ndarray) -> np.ndarray:
        return lidar_geometry_features.dynamic_features(self, values)

    def _goal_features(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        return lidar_geometry_features.goal_features(self, values, geometry_index)

    def _frame(self, values: np.ndarray) -> dict[str, torch.Tensor]:
        return lidar_observation.frame(self, values)

    def _stack_history(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return lidar_observation.stack_history(self, frame)

    def _mask_current_controls(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return lidar_observation.mask_current_controls(self, observation)

    def _prepared_shapes(self) -> dict[str, tuple[int, ...]]:
        return lidar_observation.prepared_shapes(self)

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, dict):
            required = {"lidar", "lidar_mask", "telemetry"}
            if set(observation) != required:
                raise ValueError(f"prepared lidar observation keys must be {sorted(required)}")
            prepared = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in observation.items()
            }
            shapes = self._prepared_shapes()
            if any(prepared[key].shape != shape for key, shape in shapes.items()) or not all(
                torch.isfinite(value).all() for value in prepared.values()
            ):
                raise ValueError(
                    "prepared lidar observation has invalid shape or non-finite values"
                )
            return self._mask_current_controls(prepared)
        values = self._telemetry(observation)
        return self._stack_history(self._frame(values))

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))

    def synthetic_observation(self) -> np.ndarray:
        values = np.zeros(len(self.source_fields), dtype=np.float32)
        values[12] = 1.0
        return values
