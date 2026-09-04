"""Open-loop demonstration replay policy."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trackmaniarl.core.contracts import PolicyMode
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    load_demonstration,
    resample_demonstration,
)


@dataclass(frozen=True, slots=True)
class ReplaySamplingConfig:
    decision_interval_ms: float | None
    action_lead_ms: float = 0.0
    aggregate_controls: bool = False


@dataclass(frozen=True, slots=True)
class ReplayReference:
    race_times_ms: np.ndarray
    actions: np.ndarray
    action_ids: tuple[int, ...] | None = None
    action_offset_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class ReplayPathRequest:
    path: str | Path
    action_ids: tuple[int, ...] | None = None
    action_offset_ms: float = 0.0
    sampling: ReplaySamplingConfig = ReplaySamplingConfig(None)


class DemonstrationReplayPolicy:
    """Replays the recorded expert command selected by the current race timer."""

    def __init__(self, reference: ReplayReference) -> None:
        race_times_ms, actions = reference.race_times_ms, reference.actions
        if race_times_ms.shape != actions.shape or race_times_ms.ndim != 1:
            raise ValueError("demonstration replay requires one time stamp per action")
        if len(actions) < 1 or np.any(np.diff(race_times_ms) <= 0.0):
            raise ValueError("demonstration replay time stamps must increase")
        if not np.isfinite(reference.action_offset_ms):
            raise ValueError("demonstration action offset must be finite")
        selected_ids = tuple(range(78)) if reference.action_ids is None else reference.action_ids
        self.race_times_ms = (
            race_times_ms.astype(np.float32, copy=True) + reference.action_offset_ms
        )
        self.actions = self._mapped_actions(actions, selected_ids)
        self.action_count = len(selected_ids)

    @staticmethod
    def _mapped_actions(actions: np.ndarray, selected_ids: tuple[int, ...]) -> np.ndarray:
        mapping = {action: index for index, action in enumerate(selected_ids)}
        missing = sorted({int(action) for action in actions} - mapping.keys())
        if missing:
            raise ValueError(f"demonstration actions are outside selected action IDs: {missing}")
        return np.asarray([mapping[int(action)] for action in actions], dtype=np.int64)

    @classmethod
    def from_path(cls, request: ReplayPathRequest) -> DemonstrationReplayPolicy:
        demonstration = load_demonstration(request.path)
        frames, actions = cls._resampled(demonstration, request.sampling)
        reference = ReplayReference(
            frames[:-1, 3], actions, request.action_ids, request.action_offset_ms
        )
        return cls(reference)

    @staticmethod
    def _resampled(
        demonstration: Demonstration, sampling: ReplaySamplingConfig
    ) -> tuple[np.ndarray, np.ndarray]:
        return resample_demonstration(
            DemonstrationResamplingRequest(
                demonstration,
                sampling.decision_interval_ms,
                DemonstrationResamplingConfig(sampling.action_lead_ms, sampling.aggregate_controls),
            )
        )

    def reset_episode(self) -> None:
        return None

    def act(self, observation: Any, mode: PolicyMode = PolicyMode.ONLINE) -> int:
        del mode
        race_time_ms = self._race_time_ms(observation)
        index = int(np.searchsorted(self.race_times_ms, race_time_ms, side="right") - 1)
        return int(self.actions[np.clip(index, 0, len(self.actions) - 1)])

    @staticmethod
    def _race_time_ms(observation: Any) -> float:
        if isinstance(observation, Mapping):
            if "raw_telemetry" in observation:
                return DemonstrationReplayPolicy._raw_race_time_ms(observation["raw_telemetry"])
            if "telemetry" not in observation:
                raise TypeError("demonstration replay requires telemetry")
            telemetry = observation["telemetry"]
            if isinstance(telemetry, torch.Tensor):
                telemetry = telemetry.detach().float().cpu().numpy()
            values = np.asarray(telemetry, dtype=np.float32)
            if values.shape[-1] < 4:
                raise ValueError("demonstration replay requires race-time telemetry")
            return float(values.reshape(-1, values.shape[-1])[-1, 3] * 60_000.0)

        return DemonstrationReplayPolicy._raw_race_time_ms(observation)

    @staticmethod
    def _raw_race_time_ms(telemetry: Any) -> float:
        if isinstance(telemetry, torch.Tensor):
            telemetry = telemetry.detach().float().cpu().numpy()
        values = np.asarray(telemetry, dtype=np.float32).reshape(-1)
        if values.shape != (33,) or not np.isfinite(values).all():
            raise ValueError("raw demonstration replay requires one finite 33-field frame")
        return float(values[3])
