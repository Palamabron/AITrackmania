"""Prepared observation assembly and history handling for lidar features."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def frame(pipeline: Any, values: np.ndarray) -> dict[str, torch.Tensor]:
    nearest = pipeline._nearest_progress_index(values[4:7], float(values[3]))
    lidar, mask = pipeline._local_lidar(values, nearest)
    telemetry = pipeline._scale_telemetry(values)
    if pipeline.include_track_relative:
        telemetry = np.concatenate((telemetry, pipeline._track_relative(values, nearest)))
    if pipeline.pace_profile is not None:
        telemetry = np.concatenate((telemetry, pipeline._pace_features(values, nearest)))
    if pipeline.include_dynamics:
        telemetry = np.concatenate((telemetry, pipeline._dynamic_features(values)))
    if pipeline.include_goal_features:
        telemetry = np.concatenate((telemetry, pipeline._goal_features(values, nearest)))
    return {
        "lidar": torch.from_numpy(lidar),
        "lidar_mask": torch.from_numpy(mask),
        "telemetry": torch.from_numpy(telemetry),
    }


def stack_history(pipeline: Any, frame: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if pipeline.history_length == 1:
        return mask_current_controls(pipeline, frame)
    pipeline._history.append(frame)
    frames = [pipeline._history[0]] * (pipeline.history_length - len(pipeline._history)) + list(
        pipeline._history
    )
    stacked = {
        key: torch.stack([item[key] for item in frames])
        for key in ("lidar", "lidar_mask", "telemetry")
    }
    return mask_current_controls(pipeline, stacked)


def mask_current_controls(
    pipeline: Any, observation: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if not pipeline.mask_current_control_inputs:
        return observation
    prepared = dict(observation)
    telemetry = observation["telemetry"].clone()
    if pipeline.history_length == 1:
        telemetry[17:20] = 0.0
    else:
        telemetry[-1, 17:20] = 0.0
    prepared["telemetry"] = telemetry
    return prepared


def prepared_shapes(pipeline: Any) -> dict[str, tuple[int, ...]]:
    if pipeline.history_length == 1:
        return {
            "lidar": (pipeline.lidar_channels, pipeline.samples_per_side),
            "lidar_mask": (pipeline.samples_per_side,),
            "telemetry": (pipeline.telemetry_dim,),
        }
    return {
        "lidar": (pipeline.history_length, pipeline.lidar_channels, pipeline.samples_per_side),
        "lidar_mask": (pipeline.history_length, pipeline.samples_per_side),
        "telemetry": (pipeline.history_length, pipeline.telemetry_dim),
    }
