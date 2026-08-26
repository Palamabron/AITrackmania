"""Feature pipeline for first-party telemetry baselines."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from trackmaniarl.builtins.features import GymnasiumObservationCollator
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.pace import ReferencePaceProfile, demonstration_guidance_line
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


class TelemetryFeaturePipeline:
    def __init__(self, field_count: int = DEFAULT_TELEMETRY_FIELD_COUNT) -> None:
        self.field_count = field_count

    def transform_observation(self, observation: Any) -> torch.Tensor:
        value = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        if value.numel() != self.field_count:
            raise ValueError(
                f"telemetry requires exactly {self.field_count} fields, got {value.numel()}"
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
        return torch.zeros(self.field_count, dtype=torch.float32)


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

    def __init__(
        self,
        geometry_path: str | Path,
        *,
        expected_map_uid: str | None = None,
        samples_per_side: int = 60,
        max_distance_m: float = 300.0,
        history_length: int = 1,
        include_track_relative: bool = False,
        include_control_inputs: bool = True,
        mask_current_control_inputs: bool = False,
        local_velocity_features: bool = False,
        use_racing_line: bool = False,
        max_speed_mps: float = 80.0,
        velocity_to_mps_scale: float = 0.001,
        max_time_delta_s: float = 1.0,
        limit_progress_by_kinematics: bool = False,
        nearest_forward_points: int = 128,
        nearest_backward_points: int = 10,
        pace_reference_path: str | Path | None = None,
        pace_debt_clip_s: float = 10.0,
        reference_speed_offsets_m: tuple[float, ...] = (0.0, 20.0, 40.0, 80.0),
        include_racing_line_channels: bool = False,
        include_finish_channels: bool = False,
        include_dynamics: bool = False,
        include_goal_features: bool = False,
        base_dir: str | Path = ".",
    ) -> None:
        if samples_per_side < 2:
            raise ValueError("samples_per_side must be at least two")
        if (
            max_distance_m <= 0.0
            or max_speed_mps <= 0.0
            or velocity_to_mps_scale <= 0.0
            or max_time_delta_s <= 0.0
            or nearest_forward_points < 1
            or nearest_backward_points < 0
            or pace_debt_clip_s <= 0.0
            or any(offset < 0.0 for offset in reference_speed_offsets_m)
        ):
            raise ValueError("distance and speed scales must be positive")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if mask_current_control_inputs and not include_control_inputs:
            raise ValueError("current control masking requires control inputs")
        path = Path(geometry_path)
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        self.geometry = BoundaryGeometry(path, expected_map_uid=expected_map_uid)
        self.samples_per_side = samples_per_side
        self.max_distance_m = max_distance_m
        self.history_length = history_length
        self.include_track_relative = include_track_relative
        self.include_control_inputs = include_control_inputs
        self.mask_current_control_inputs = mask_current_control_inputs
        self.local_velocity_features = local_velocity_features
        self.use_racing_line = use_racing_line
        self.max_speed_mps = max_speed_mps
        self.velocity_to_mps_scale = velocity_to_mps_scale
        self.max_time_delta_s = max_time_delta_s
        self.limit_progress_by_kinematics = limit_progress_by_kinematics
        self.nearest_forward_points = nearest_forward_points
        self.nearest_backward_points = nearest_backward_points
        self.pace_debt_clip_s = pace_debt_clip_s
        self.reference_speed_offsets_m = tuple(reference_speed_offsets_m)
        self.include_racing_line_channels = include_racing_line_channels
        self.include_finish_channels = include_finish_channels
        self.include_dynamics = include_dynamics
        self.include_goal_features = include_goal_features
        self.pace_profile: ReferencePaceProfile | None = None
        self._expert_guidance_line: np.ndarray | None = None
        if pace_reference_path is not None:
            pace_path = Path(pace_reference_path)
            if not pace_path.is_absolute():
                pace_path = (Path(base_dir) / pace_path).resolve()
            reference = (
                self.geometry.racing_line if use_racing_line else self.geometry.reward_center
            )
            self.pace_profile = ReferencePaceProfile.from_demonstration(
                pace_path, self.geometry, reference
            )
            if include_racing_line_channels:
                self._expert_guidance_line = demonstration_guidance_line(
                    pace_path, self.geometry, reference
                )
        if self.reference_speed_offsets_m and self.pace_profile is None:
            self.reference_speed_offsets_m = ()
        self._reference_line = (
            self.geometry.racing_line if use_racing_line else self.geometry.reward_center
        )
        self._reference_cumulative_distance = self._line_cumulative_distance(self._reference_line)
        self._lookahead_line = self._guidance_lookahead_line()
        self._cumulative_distance = self._geometry_cumulative_distance()
        self._progress_index = 0
        self._last_race_time_ms: float | None = None
        self._last_heading = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        self._last_dynamic_race_time_ms: float | None = None
        self._last_dynamic_velocity = np.zeros(2, dtype=np.float32)
        self._last_dynamic_yaw: float | None = None
        self.lidar_channels = (
            4 + 2 * int(include_racing_line_channels) + 2 * int(include_finish_channels)
        )
        control_input_count = 3 if include_control_inputs else 0
        self.telemetry_dim = (
            len(self.telemetry_fields)
            - 3
            + control_input_count
            + (len(self.track_relative_fields) if include_track_relative else 0)
            + (1 + len(self.reference_speed_offsets_m) if self.pace_profile is not None else 0)
            + 4 * int(include_dynamics)
            + 14 * int(include_goal_features)
        )
        self._history: deque[dict[str, torch.Tensor]] = deque(maxlen=history_length)
        lidar_shape = (
            (self.lidar_channels, self.samples_per_side)
            if history_length == 1
            else (history_length, self.lidar_channels, self.samples_per_side)
        )
        mask_shape = (
            (self.samples_per_side,)
            if history_length == 1
            else (history_length, self.samples_per_side)
        )
        telemetry_shape = (
            (self.telemetry_dim,) if history_length == 1 else (history_length, self.telemetry_dim)
        )
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    -1.0,
                    1.0,
                    shape=lidar_shape,
                    dtype=np.float32,
                ),
                "lidar_mask": spaces.Box(
                    0.0,
                    1.0,
                    shape=mask_shape,
                    dtype=np.float32,
                ),
                "telemetry": spaces.Box(
                    -1.0,
                    1.0,
                    shape=telemetry_shape,
                    dtype=np.float32,
                ),
            }
        )
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
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.shape != (len(self.source_fields),):
            raise ValueError(
                f"lidar source schema {self.schema_version} requires 33 fields, got {values.size}"
            )
        if not np.isfinite(values).all():
            raise ValueError("lidar telemetry contains non-finite values")
        return values

    def _scale_telemetry(self, values: np.ndarray) -> np.ndarray:
        # Stable 20-field projection from the supported 33-field GrabData packet.
        # Velocity and speed use the configured native unit scale normalized by
        # the top speed so they span [-1, 1] instead of collapsing toward zero.
        velocity_scale = self.velocity_to_mps_scale / self.max_speed_mps
        if self.local_velocity_features:
            forward = self._horizontal_heading(values[10:13])
            right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
            velocity = values[7:10]
            velocity_features = (
                float(np.dot(velocity, forward)),
                float(velocity[1]),
                float(np.dot(velocity, right)),
            )
        else:
            velocity_features = (values[7], values[8], values[9])
        selected = [
            values[0] / 100.0,
            values[1] / 10.0,
            values[2],
            values[3] / 60_000.0,
            velocity_features[0] * velocity_scale,
            velocity_features[1] * velocity_scale,
            velocity_features[2] * velocity_scale,
            values[16] * velocity_scale,
            values[17] / 10_000.0,
            values[18] / 6.0,
            values[19],
            values[20],
            values[21],
            values[22],
            values[27] / 4.0,
            values[28] / 1_000.0,
            values[29],
        ]
        if self.include_control_inputs:
            selected.extend((values[30], values[31], values[32]))
        return np.clip(np.asarray(selected, dtype=np.float32), -1.0, 1.0)

    def _local_lidar(self, values: np.ndarray, nearest: int) -> tuple[np.ndarray, np.ndarray]:
        position = values[4:7]
        forward = self._horizontal_heading(values[10:13])
        # Keep the legacy car-frame convention: lateral/right first, then
        # longitudinal/forward.  For the OpenPlanet X/Z coordinates the right
        # vector is (forward_z, -forward_x), not its opposite.
        right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
        indices = nearest + np.arange(1, self.samples_per_side + 1)
        valid = indices < len(self.geometry.center)
        indices = np.clip(indices, 0, len(self.geometry.center) - 1)
        left_relative = self.geometry.left[indices] - position
        right_relative = self.geometry.right[indices] - position
        local = (
            np.stack(
                (
                    left_relative @ right,
                    left_relative @ forward,
                    right_relative @ right,
                    right_relative @ forward,
                ),
                axis=0,
            )
            / self.max_distance_m
        )
        channels = [local]
        if self.include_racing_line_channels:
            line_relative = self._lookahead_line[indices] - position
            channels.append(
                np.stack(
                    (
                        line_relative @ right,
                        line_relative @ forward,
                    ),
                    axis=0,
                )
                / self.max_distance_m
            )
        if self.include_finish_channels:
            finish_index = self.geometry.recorded_count - 1
            finish_marker = np.exp(
                -0.5 * np.square((indices.astype(np.float32) - finish_index) / 2.0)
            )
            virtual_marker = (indices >= self.geometry.recorded_count).astype(np.float32)
            channels.append(np.stack((finish_marker, virtual_marker), axis=0))
        local = np.concatenate(channels, axis=0)
        mask = valid.astype(np.float32)
        local *= mask[None, :]
        return np.clip(local, -1.0, 1.0).astype(np.float32), mask

    def _geometry_cumulative_distance(self) -> np.ndarray:
        return self._line_cumulative_distance(self.geometry.center)

    @staticmethod
    def _line_cumulative_distance(points: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.empty(len(points), dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(distances, out=cumulative[1:])
        return cumulative

    def _guidance_lookahead_line(self) -> np.ndarray:
        guidance = (
            self._expert_guidance_line
            if self._expert_guidance_line is not None
            else self._reference_line
        )
        if len(guidance) != self.geometry.recorded_count:
            raise ValueError("guidance line length must match recorded geometry")
        return np.concatenate((guidance, self.geometry.center[self.geometry.recorded_count :]))

    def _nearest_progress_index(self, position: np.ndarray, race_time_ms: float) -> int:
        start = max(0, self._progress_index - self.nearest_backward_points)
        stop = min(
            len(self.geometry.center),
            self._progress_index + self.nearest_forward_points + 1,
        )
        distances = np.sum((self.geometry.center[start:stop] - position) ** 2, axis=1)
        nearest = start + int(np.argmin(distances))
        if self.limit_progress_by_kinematics and self._last_race_time_ms is not None:
            elapsed_s = max(0.0, (race_time_ms - self._last_race_time_ms) / 1_000.0)
            distance_limit = float(
                self._cumulative_distance[self._progress_index]
            ) + self.max_speed_mps * min(elapsed_s, self.max_time_delta_s)
            reachable = (
                int(np.searchsorted(self._cumulative_distance, distance_limit, side="right")) - 1
            )
            nearest = min(nearest, max(reachable, self._progress_index))
        self._last_race_time_ms = race_time_ms
        self._progress_index = max(self._progress_index, nearest)
        return self._progress_index

    @staticmethod
    def _unit_horizontal(vector: np.ndarray) -> np.ndarray:
        horizontal = np.asarray(vector, dtype=np.float32).copy()
        horizontal[1] = 0.0
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1e-5:
            raise ValueError("track-relative vector has no horizontal component")
        return horizontal / norm

    def _horizontal_heading(self, direction: np.ndarray) -> np.ndarray:
        """Project the car heading onto the track plane, holding the last valid one
        through vertical moments (loops, wallrides, airborne flips)."""

        horizontal = np.asarray(direction, dtype=np.float32).copy()
        horizontal[1] = 0.0
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1e-5:
            return self._last_heading
        heading = horizontal / norm
        self._last_heading = heading
        return heading

    def _track_relative(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        center = self.geometry.racing_line if self.use_racing_line else self.geometry.reward_center
        position = values[4:7]
        index = min(geometry_index, len(center) - 1)
        before = max(0, index - 1)
        after = min(len(center) - 1, index + 1)
        tangent = self._unit_horizontal(center[after] - center[before])
        right = np.asarray([tangent[2], 0.0, -tangent[0]], dtype=np.float32)
        forward = self._horizontal_heading(values[10:13])
        velocity = values[7:10] * self.velocity_to_mps_scale
        half_width = max(
            0.5 * float(np.linalg.norm(self.geometry.left[index] - self.geometry.right[index])),
            1.0,
        )
        progress = index / max(1, self.geometry.recorded_count - 1)
        relative = np.asarray(
            [
                progress,
                float(np.dot(position - center[index], right)) / half_width,
                float(np.dot(forward, right)),
                float(np.dot(forward, tangent)),
                float(np.dot(velocity, tangent)) / self.max_speed_mps,
                float(np.dot(velocity, right)) / self.max_speed_mps,
            ],
            dtype=np.float32,
        )
        return np.clip(relative, -1.0, 1.0)

    def _pace_features(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        assert self.pace_profile is not None
        index = min(geometry_index, len(self._reference_line) - 1)
        reference_time_s = self.pace_profile.time_at_index(index)
        time_debt_s = float(values[3]) / 1_000.0 - reference_time_s
        features = [np.clip(time_debt_s / self.pace_debt_clip_s, -1.0, 1.0)]
        distance = self._reference_cumulative_distance[index]
        for offset in self.reference_speed_offsets_m:
            target = min(distance + offset, self._reference_cumulative_distance[-1])
            future = int(np.searchsorted(self._reference_cumulative_distance, target, side="left"))
            features.append(
                np.clip(
                    self.pace_profile.speed_at_index(future) / self.max_speed_mps,
                    0.0,
                    1.0,
                )
            )
        return np.asarray(features, dtype=np.float32)

    def _dynamic_features(self, values: np.ndarray) -> np.ndarray:
        race_time_ms = float(values[3])
        forward = self._horizontal_heading(values[10:13])
        right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
        velocity = values[7:10] * self.velocity_to_mps_scale
        local_velocity = np.asarray(
            [np.dot(velocity, forward), np.dot(velocity, right)], dtype=np.float32
        )
        yaw = float(np.arctan2(forward[2], forward[0]))
        previous_time = self._last_dynamic_race_time_ms
        elapsed_s = (
            0.0 if previous_time is None else max(0.0, (race_time_ms - previous_time) / 1_000.0)
        )
        if elapsed_s <= 1e-4 or elapsed_s > self.max_time_delta_s:
            features = np.zeros(4, dtype=np.float32)
        else:
            acceleration = (local_velocity - self._last_dynamic_velocity) / elapsed_s
            previous_yaw = self._last_dynamic_yaw if self._last_dynamic_yaw is not None else yaw
            yaw_delta = float(np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw)))
            features = np.asarray(
                [
                    np.clip(elapsed_s / 0.05, 0.0, 1.0),
                    np.clip(yaw_delta / elapsed_s / 4.0, -1.0, 1.0),
                    np.clip(acceleration[0] / 40.0, -1.0, 1.0),
                    np.clip(acceleration[1] / 40.0, -1.0, 1.0),
                ],
                dtype=np.float32,
            )
        self._last_dynamic_race_time_ms = race_time_ms
        self._last_dynamic_velocity = local_velocity
        self._last_dynamic_yaw = yaw
        return features

    def _goal_features(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        finish_index = self.geometry.recorded_count - 1
        center = self.geometry.center[finish_index]
        left = self.geometry.left[finish_index]
        right_edge = self.geometry.right[finish_index]
        position = values[4:7]
        forward = self._horizontal_heading(values[10:13])
        right = np.asarray([forward[2], 0.0, -forward[0]], dtype=np.float32)
        relative = center - position
        left_relative = left - position
        right_relative = right_edge - position
        lateral = float(np.dot(relative, right))
        longitudinal = float(np.dot(relative, forward))
        distance = max(float(np.linalg.norm(relative)), 1e-6)
        tangent = self._unit_horizontal(self._reference_line[-1] - self._reference_line[-2])
        index = min(geometry_index, finish_index)
        remaining = max(
            0.0,
            self._reference_cumulative_distance[-1] - self._reference_cumulative_distance[index],
        )
        return np.asarray(
            [
                np.clip(lateral / self.max_distance_m, -1.0, 1.0),
                np.clip(longitudinal / self.max_distance_m, -1.0, 1.0),
                np.clip(float(np.dot(left_relative, right)) / self.max_distance_m, -1.0, 1.0),
                np.clip(float(np.dot(left_relative, forward)) / self.max_distance_m, -1.0, 1.0),
                np.clip(float(np.dot(right_relative, right)) / self.max_distance_m, -1.0, 1.0),
                np.clip(float(np.dot(right_relative, forward)) / self.max_distance_m, -1.0, 1.0),
                np.clip(
                    np.log1p(distance) / np.log1p(self._reference_cumulative_distance[-1]),
                    0.0,
                    1.0,
                ),
                np.clip(lateral / distance, -1.0, 1.0),
                np.clip(longitudinal / distance, -1.0, 1.0),
                np.clip(float(np.dot(tangent, right)), -1.0, 1.0),
                np.clip(float(np.dot(tangent, forward)), -1.0, 1.0),
                np.clip(0.5 * float(np.linalg.norm(left - right_edge)) / 20.0, 0.0, 1.0),
                np.clip(remaining / self.max_distance_m, 0.0, 1.0),
                float(remaining <= self.max_distance_m),
            ],
            dtype=np.float32,
        )

    def _frame(self, values: np.ndarray) -> dict[str, torch.Tensor]:
        nearest = self._nearest_progress_index(values[4:7], float(values[3]))
        lidar, mask = self._local_lidar(values, nearest)
        telemetry = self._scale_telemetry(values)
        if self.include_track_relative:
            telemetry = np.concatenate((telemetry, self._track_relative(values, nearest)))
        if self.pace_profile is not None:
            telemetry = np.concatenate((telemetry, self._pace_features(values, nearest)))
        if self.include_dynamics:
            telemetry = np.concatenate((telemetry, self._dynamic_features(values)))
        if self.include_goal_features:
            telemetry = np.concatenate((telemetry, self._goal_features(values, nearest)))
        return {
            "lidar": torch.from_numpy(lidar),
            "lidar_mask": torch.from_numpy(mask),
            "telemetry": torch.from_numpy(telemetry),
        }

    def _stack_history(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.history_length == 1:
            return self._mask_current_controls(frame)
        self._history.append(frame)
        frames = [self._history[0]] * (self.history_length - len(self._history)) + list(
            self._history
        )
        stacked = {
            key: torch.stack([item[key] for item in frames])
            for key in ("lidar", "lidar_mask", "telemetry")
        }
        return self._mask_current_controls(stacked)

    def _mask_current_controls(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not self.mask_current_control_inputs:
            return observation
        prepared = dict(observation)
        telemetry = observation["telemetry"].clone()
        if self.history_length == 1:
            telemetry[17:20] = 0.0
        else:
            telemetry[-1, 17:20] = 0.0
        prepared["telemetry"] = telemetry
        return prepared

    def _prepared_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.history_length == 1:
            return {
                "lidar": (self.lidar_channels, self.samples_per_side),
                "lidar_mask": (self.samples_per_side,),
                "telemetry": (self.telemetry_dim,),
            }
        return {
            "lidar": (self.history_length, self.lidar_channels, self.samples_per_side),
            "lidar_mask": (self.history_length, self.samples_per_side),
            "telemetry": (self.history_length, self.telemetry_dim),
        }

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
