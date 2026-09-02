"""Second-generation boundary-graph features: body-frame kinematics and input echo.

Opt-in replacements for :class:`BoundaryGraphFeaturePipeline` and
:class:`TrackGnnSimbaEncoder` that keep the observation shapes (``physics`` (60,),
``track`` (3, 88)) so the dueling IQN head and existing checkpoint layouts are
unchanged, but:

* fill the seven constant-zero physics slots of v1 with the quantities the v104f
  policy could not observe (forward/lateral body-frame velocity, yaw rate, the
  game's input echo of the previous control, skidding wheel count);
* replace the raw global yaw (which wraps at +-pi) with the sine/cosine of the
  heading relative to the local track tangent;
* express every scalar in O(1) units (m/s / 100, rad/s / 3, m/s^2 / 50) instead
  of mixing km/h, metres and [-1, 1] curvature in one LayerNorm;
* keep the asset's virtual finish extension for the lookahead (v1 discarded it, so
  the last ~110 m of every lap saw all 44 points stacked on the finish station);
* stop masking the previous control (v1 zeroed it in both the pipeline and the
  encoder) and drop the LayerNorm in front of the SimbaV2 backbone so its shift
  channel keeps the input magnitude.

Enable with::

    feature_pipeline:
      class_path: trackmaniarl.experiments.graph_iqn_v2:BoundaryGraphFeaturePipelineV2
    model_factory.kwargs.encoder:
      class_path: trackmaniarl.experiments.graph_iqn_v2:TrackGnnSimbaEncoderV2
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from trackmaniarl.experiments.graph_iqn import BoundaryGraphFeaturePipeline
from trackmaniarl.models.backbones import SimbaV2Backbone
from trackmaniarl.models.track_graphs import TrackNeighborGraph
from trackmaniarl.trackmania.geometry import BoundaryGeometry

SPEED_SCALE_MPS = 100.0
LATERAL_SCALE_MPS = 20.0
YAW_RATE_SCALE_RAD_S = 3.0
ACCELERATION_SCALE_MPS2 = 50.0
TRACK_SCALE_M = 50.0
CURVATURE_COUNT = 44

PHYSICS_V2_LAYOUT: tuple[str, ...] = (
    "speed",
    "forward_velocity",
    "lateral_velocity",
    "progress",
    "yaw_rate",
    "acceleration",
    "heading_sin",
    "heading_cos",
    "gear",
    "pitch",
    "input_gas",
    "input_brake",
    "front_left_slip",
    "front_right_slip",
    "input_steer",
    *(f"curvature_{index}" for index in range(CURVATURE_COUNT)),
    "skidding_wheels",
)
assert len(PHYSICS_V2_LAYOUT) == 60


def _wrap_angle(value: float) -> float:
    return float((value + np.pi) % (2.0 * np.pi) - np.pi)


def _unit_xz(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=np.float32) / max(float(np.linalg.norm(vector)), 1.0e-6)


def _control_scalars(values: np.ndarray) -> list[float]:
    return [
        float(np.clip(values[31], 0.0, 1.0)),
        float(np.clip(values[32], 0.0, 1.0)),
        float(np.clip(values[19], 0.0, 1.0)),
        float(np.clip(values[20], 0.0, 1.0)),
        float(np.clip(values[30], -1.0, 1.0)),
    ]


class BoundaryGraphFeaturePipelineV2(BoundaryGraphFeaturePipeline):
    """33-field telemetry -> normalised body-frame physics + boundary lookahead."""

    def reset_episode(self) -> None:
        super().reset_episode()
        self._last_yaw: float | None = None

    def _install_geometry(self, geometry: BoundaryGeometry) -> None:
        super()._install_geometry(geometry)
        # Keep the asset's virtual finish extension for the lookahead so the last
        # ~110 m do not collapse onto the finish station; the nearest search and
        # progress stay on the recorded part (reward_center == center[:recorded]).
        self._left, self._center, self._right = geometry.left, geometry.center, geometry.right
        self._reward_distance = self._distance
        self._distance = self._cumulative_distance(geometry.center)

    def _lookahead_indices(self, nearest: int) -> np.ndarray:
        targets = self._reward_distance[nearest] + 2.5 * np.arange(1, 45, dtype=np.float32)
        indices = np.searchsorted(self._distance, targets).clip(0, len(self._distance) - 1)
        return np.asarray(indices, dtype=np.int64)

    def transform_observation(self, observation: Any) -> dict[str, torch.Tensor]:
        if isinstance(observation, Mapping):
            return self._validate_prepared(observation)
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("graph features require finite 33-field telemetry")
        position = values[[4, 5, 6]]
        nearest = self._nearest(position)
        track, curvature = self._track_features(position, values[[10, 12]], nearest)
        physics = self._physics_v2(values, nearest, curvature)
        return {"physics": torch.from_numpy(physics), "track": torch.from_numpy(track)}

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
        return result

    def _physics_v2(self, values: np.ndarray, nearest: int, curvature: np.ndarray) -> np.ndarray:
        heading = _unit_xz(values[[10, 12]])
        scalars = np.array(
            [
                *self._motion_scalars(values, heading, nearest),
                *self._orientation_scalars(values, nearest, heading),
                *_control_scalars(values),
            ],
            dtype=np.float32,
        )
        skidding = np.array([np.clip(values[27], 0.0, 4.0) / 4.0], dtype=np.float32)
        return np.concatenate((scalars, curvature.astype(np.float32), skidding))

    def _motion_scalars(self, values: np.ndarray, heading: np.ndarray, nearest: int) -> list[float]:
        speed_mps = float(values[16])
        yaw = float(np.arctan2(heading[0], heading[1]))
        yaw_rate, acceleration = self._rates(float(values[3]), speed_mps, yaw)
        velocity_xz = values[[7, 9]]
        lateral = float(velocity_xz[0] * heading[1] - velocity_xz[1] * heading[0])
        return [
            speed_mps / SPEED_SCALE_MPS,
            float(velocity_xz @ heading) / SPEED_SCALE_MPS,
            float(np.clip(lateral / LATERAL_SCALE_MPS, -1.0, 1.0)),
            nearest / max(len(self._reward_center) - 1, 1),
            float(np.clip(yaw_rate / YAW_RATE_SCALE_RAD_S, -1.0, 1.0)),
            float(np.clip(acceleration / ACCELERATION_SCALE_MPS2, -1.0, 1.0)),
        ]

    def _orientation_scalars(
        self, values: np.ndarray, nearest: int, heading: np.ndarray
    ) -> list[float]:
        tangent = self._tangent_xz(nearest)
        return [
            float(heading[0] * tangent[1] - heading[1] * tangent[0]),
            float(heading @ tangent),
            float(values[18]) / 5.0,
            float(np.arcsin(np.clip(values[11], -1.0, 1.0))),
        ]

    def _rates(self, time_ms: float, speed_mps: float, yaw: float) -> tuple[float, float]:
        if self._last_time_ms is None or self._last_yaw is None:
            yaw_rate, acceleration = 0.0, 0.0
        else:
            dt_s = max(time_ms - self._last_time_ms, 1.0) / 1000.0
            yaw_rate = _wrap_angle(yaw - self._last_yaw) / dt_s
            acceleration = (speed_mps - self._last_speed_mps) / dt_s
        self._last_time_ms, self._last_speed_mps, self._last_yaw = time_ms, speed_mps, yaw
        return yaw_rate, acceleration

    def _tangent_xz(self, nearest: int) -> np.ndarray:
        center = self._center
        before = center[max(nearest - 1, 0)][[0, 2]]
        after = center[min(nearest + 1, len(center) - 1)][[0, 2]]
        return _unit_xz(after - before)


class TrackGnnSimbaEncoderV2(nn.Module):
    """Neighbour GNN + physics MLP -> SimbaV2, without control masking or joint LayerNorm."""

    output_dim = 192

    def __init__(self) -> None:
        super().__init__()
        self.track_conv = nn.Sequential(
            TrackNeighborGraph(), nn.Linear(128, 192), nn.LayerNorm(192), nn.SiLU()
        )
        self.physics_proj = nn.Sequential(nn.Linear(60, 192), nn.LayerNorm(192), nn.SiLU())
        self.backbone = SimbaV2Backbone(384, 192, block_count=4, expansion=4)

    def forward(self, observation: Mapping[str, torch.Tensor]) -> torch.Tensor:
        track = observation["track"].float()
        physics = observation["physics"].float()
        if track.ndim != 3 or track.shape[1:] != (3, 88):
            raise ValueError("track observation must have shape (batch, 3, 88)")
        if physics.shape != (track.shape[0], 60):
            raise ValueError("physics observation must have shape (batch, 60)")
        joint = torch.cat(
            (self.track_conv(track / TRACK_SCALE_M), self.physics_proj(physics)), dim=-1
        )
        return cast(torch.Tensor, self.backbone(joint))
