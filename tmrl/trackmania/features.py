"""Feature pipeline for first-party telemetry baselines."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from tmrl.builtins.features import GymnasiumObservationCollator
from tmrl.core.data import Transition
from tmrl.trackmania.geometry import BoundaryGeometry
from tmrl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT


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
            "_tmrl_batch_collated": True,
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

    schema_version = "2"
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
        use_racing_line: bool = False,
        max_speed_mps: float = 80.0,
        velocity_to_mps_scale: float = 0.001,
        nearest_forward_points: int = 128,
        nearest_backward_points: int = 10,
        base_dir: str | Path = ".",
    ) -> None:
        if samples_per_side < 2:
            raise ValueError("samples_per_side must be at least two")
        if (
            max_distance_m <= 0.0
            or max_speed_mps <= 0.0
            or velocity_to_mps_scale <= 0.0
            or nearest_forward_points < 1
            or nearest_backward_points < 0
        ):
            raise ValueError("distance and speed scales must be positive")
        if history_length < 1:
            raise ValueError("history_length must be positive")
        path = Path(geometry_path)
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        self.geometry = BoundaryGeometry(path, expected_map_uid=expected_map_uid)
        self.samples_per_side = samples_per_side
        self.max_distance_m = max_distance_m
        self.history_length = history_length
        self.include_track_relative = include_track_relative
        self.use_racing_line = use_racing_line
        self.max_speed_mps = max_speed_mps
        self.velocity_to_mps_scale = velocity_to_mps_scale
        self.nearest_forward_points = nearest_forward_points
        self.nearest_backward_points = nearest_backward_points
        self._progress_index = 0
        self.telemetry_dim = len(self.telemetry_fields) + (
            len(self.track_relative_fields) if include_track_relative else 0
        )
        self._history: deque[dict[str, torch.Tensor]] = deque(maxlen=history_length)
        lidar_shape = (
            (4, self.samples_per_side)
            if history_length == 1
            else (history_length, 4, self.samples_per_side)
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

        self.geometry = BoundaryGeometry(
            map_spec.geometry_path, expected_map_uid=map_spec.expected_map_uid
        )
        self.reset_episode()

    def reset_episode(self) -> None:
        self._history.clear()
        self._progress_index = 0

    def _telemetry(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.shape != (len(self.source_fields),):
            raise ValueError(
                f"lidar source schema {self.schema_version} requires 33 fields, got {values.size}"
            )
        if not np.isfinite(values).all():
            raise ValueError("lidar telemetry contains non-finite values")
        return values

    @staticmethod
    def _scale_telemetry(values: np.ndarray) -> np.ndarray:
        # Stable 20-field projection from the supported 33-field GrabData packet.
        selected = np.asarray(
            [
                values[0] / 100.0,
                values[1] / 10.0,
                values[2],
                values[3] / 60_000.0,
                values[7] / 1_000.0,
                values[8] / 1_000.0,
                values[9] / 1_000.0,
                values[16] / 1_000.0,
                values[17] / 10_000.0,
                values[18] / 6.0,
                values[19],
                values[20],
                values[21],
                values[22],
                values[27] / 4.0,
                values[28] / 1_000.0,
                values[29],
                values[30],
                values[31],
                values[32],
            ],
            dtype=np.float32,
        )
        return np.clip(selected, -1.0, 1.0)

    def _local_lidar(self, values: np.ndarray, nearest: int) -> tuple[np.ndarray, np.ndarray]:
        position = values[4:7]
        forward = values[10:13].copy()
        forward[1] = 0.0
        length = float(np.linalg.norm(forward))
        if length <= 1e-5:
            raise ValueError("vehicle direction has no horizontal component")
        forward /= length
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
        mask = valid.astype(np.float32)
        local *= mask[None, :]
        return np.clip(local, -1.0, 1.0).astype(np.float32), mask

    def _nearest_progress_index(self, position: np.ndarray) -> int:
        start = max(0, self._progress_index - self.nearest_backward_points)
        stop = min(
            len(self.geometry.center),
            self._progress_index + self.nearest_forward_points + 1,
        )
        distances = np.sum((self.geometry.center[start:stop] - position) ** 2, axis=1)
        nearest = start + int(np.argmin(distances))
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

    def _track_relative(self, values: np.ndarray, geometry_index: int) -> np.ndarray:
        center = self.geometry.racing_line if self.use_racing_line else self.geometry.reward_center
        position = values[4:7]
        index = min(geometry_index, len(center) - 1)
        before = max(0, index - 1)
        after = min(len(center) - 1, index + 1)
        tangent = self._unit_horizontal(center[after] - center[before])
        right = np.asarray([tangent[2], 0.0, -tangent[0]], dtype=np.float32)
        forward = self._unit_horizontal(values[10:13])
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

    def _frame(self, values: np.ndarray) -> dict[str, torch.Tensor]:
        nearest = self._nearest_progress_index(values[4:7])
        lidar, mask = self._local_lidar(values, nearest)
        telemetry = self._scale_telemetry(values)
        if self.include_track_relative:
            telemetry = np.concatenate((telemetry, self._track_relative(values, nearest)))
        return {
            "lidar": torch.from_numpy(lidar),
            "lidar_mask": torch.from_numpy(mask),
            "telemetry": torch.from_numpy(telemetry),
        }

    def _stack_history(self, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.history_length == 1:
            return frame
        self._history.append(frame)
        frames = [self._history[0]] * (self.history_length - len(self._history)) + list(
            self._history
        )
        return {
            key: torch.stack([item[key] for item in frames])
            for key in ("lidar", "lidar_mask", "telemetry")
        }

    def _prepared_shapes(self) -> dict[str, tuple[int, ...]]:
        if self.history_length == 1:
            return {
                "lidar": (4, self.samples_per_side),
                "lidar_mask": (self.samples_per_side,),
                "telemetry": (self.telemetry_dim,),
            }
        return {
            "lidar": (self.history_length, 4, self.samples_per_side),
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
            return prepared
        values = self._telemetry(observation)
        return self._stack_history(self._frame(values))

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))

    def synthetic_observation(self) -> np.ndarray:
        values = np.zeros(len(self.source_fields), dtype=np.float32)
        values[12] = 1.0
        return values
