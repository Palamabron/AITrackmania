"""Feature pipeline for first-party telemetry baselines."""

from __future__ import annotations

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
    telemetry_dim = len(telemetry_fields)

    def __init__(
        self,
        geometry_path: str | Path,
        *,
        expected_map_uid: str | None = None,
        samples_per_side: int = 60,
        max_distance_m: float = 300.0,
        base_dir: str | Path = ".",
    ) -> None:
        if samples_per_side < 2:
            raise ValueError("samples_per_side must be at least two")
        if max_distance_m <= 0.0:
            raise ValueError("max_distance_m must be positive")
        path = Path(geometry_path)
        if not path.is_absolute():
            path = (Path(base_dir) / path).resolve()
        self.geometry = BoundaryGeometry(path, expected_map_uid=expected_map_uid)
        self.samples_per_side = samples_per_side
        self.max_distance_m = max_distance_m
        self.observation_space = spaces.Dict(
            {
                "lidar": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(4, self.samples_per_side),
                    dtype=np.float32,
                ),
                "lidar_mask": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.samples_per_side,),
                    dtype=np.float32,
                ),
                "telemetry": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(self.telemetry_dim,),
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

    def _local_lidar(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = values[4:7]
        nearest = int(np.argmin(np.sum((self.geometry.center - position) ** 2, axis=1)))
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

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, dict):
            required = {"lidar", "lidar_mask", "telemetry"}
            if set(observation) != required:
                raise ValueError(f"prepared lidar observation keys must be {sorted(required)}")
            prepared = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in observation.items()
            }
            if (
                prepared["lidar"].shape != (4, self.samples_per_side)
                or prepared["lidar_mask"].shape != (self.samples_per_side,)
                or prepared["telemetry"].shape != (self.telemetry_dim,)
                or not all(torch.isfinite(value).all() for value in prepared.values())
            ):
                raise ValueError(
                    "prepared lidar observation has invalid shape or non-finite values"
                )
            return prepared
        values = self._telemetry(observation)
        lidar, mask = self._local_lidar(values)
        return {
            "lidar": torch.from_numpy(lidar),
            "lidar_mask": torch.from_numpy(mask),
            "telemetry": torch.from_numpy(self._scale_telemetry(values)),
        }

    def collate(self, transitions: list[Transition]) -> dict[str, Any]:
        return dict(self._collator.collate_transitions(transitions))

    def synthetic_observation(self) -> np.ndarray:
        values = np.zeros(len(self.source_fields), dtype=np.float32)
        values[12] = 1.0
        return values
