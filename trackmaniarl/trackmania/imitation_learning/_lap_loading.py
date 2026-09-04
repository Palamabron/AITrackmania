from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationResamplingConfig,
    DemonstrationResamplingRequest,
    load_demonstration,
    resample_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.imitation_learning._data_types import (
    ELITE_LAP_WEIGHT_TEMPERATURE_S,
    INTERVENTION_KEY,
    MINIMUM_LAP_WEIGHT,
    SAMPLE_WEIGHT_KEY,
    STATE_ERROR_KEY,
    STUDENT_ACTION_KEY,
    BehaviorCloningLap,
)


@dataclass(frozen=True, slots=True)
class LapLoadRequest:
    paths: Sequence[Path]
    pipeline: FeaturePipeline
    action_ids: tuple[int, ...]
    expected_action_repeat_frames: int | None = None
    expected_decision_interval_ms: float | None = None
    action_lead_ms: float = 0.0
    aggregate_controls: bool = False
    previous_action_conditioning: bool = False


@dataclass(frozen=True, slots=True)
class _LapLoadSettings:
    expected_action_repeat_frames: int | None = None
    expected_decision_interval_ms: float | None = None
    action_lead_ms: float = 0.0
    aggregate_controls: bool = False
    previous_action_conditioning: bool = False


@dataclass(frozen=True, slots=True)
class _LapContext:
    pipeline: FeaturePipeline
    action_ids: tuple[int, ...]
    mapping: dict[int, int]
    settings: _LapLoadSettings
    best_finish_time_s: float


@dataclass(frozen=True, slots=True)
class _LapSource:
    path: Path
    demonstration: Demonstration


@dataclass(frozen=True, slots=True)
class _ResampledLap:
    source: _LapSource
    frames: np.ndarray
    actions: np.ndarray


@dataclass(frozen=True, slots=True)
class _ObservationRequest:
    frame: np.ndarray
    previous_action: int
    weight: float


def load_behavior_cloning_laps(request: LapLoadRequest) -> list[BehaviorCloningLap]:
    """Convert full demonstration laps into compact supervised examples."""

    sources = [_LapSource(path, load_demonstration(path)) for path in request.paths]
    context = _lap_context(request, sources)
    laps = [_load_lap(context, source) for source in sources]
    if len(laps) < 3:
        raise ValueError("behavior cloning requires at least three complete demonstration laps")
    return laps


def _lap_context(request: LapLoadRequest, sources: list[_LapSource]) -> _LapContext:
    settings = _LapLoadSettings(
        request.expected_action_repeat_frames,
        request.expected_decision_interval_ms,
        request.action_lead_ms,
        request.aggregate_controls,
        request.previous_action_conditioning,
    )
    best = min(source.demonstration.finish_time_s for source in sources)
    mapping = {action: index for index, action in enumerate(request.action_ids)}
    return _LapContext(request.pipeline, request.action_ids, mapping, settings, best)


def _load_lap(context: _LapContext, source: _LapSource) -> BehaviorCloningLap:
    _validate_demonstration_contract(context, source)
    validate_recording_quality(source.demonstration)
    frames, actions = _resample_lap(context.settings, source.demonstration)
    _reset_pipeline(context.pipeline)
    resampled = _ResampledLap(source, frames, actions)
    weight = _lap_weight(source.demonstration, context.best_finish_time_s)
    observations, labels = _prepare_samples(context, resampled, weight)
    return BehaviorCloningLap(
        tuple(observations),
        torch.tensor(labels, dtype=torch.long),
        quality_weight=weight,
        source_id=str(source.path.resolve()),
    )


def _resample_lap(
    settings: _LapLoadSettings, demonstration: Demonstration
) -> tuple[np.ndarray, np.ndarray]:
    return resample_demonstration(
        DemonstrationResamplingRequest(
            demonstration,
            settings.expected_decision_interval_ms,
            DemonstrationResamplingConfig(settings.action_lead_ms, settings.aggregate_controls),
        )
    )


def _reset_pipeline(pipeline: FeaturePipeline) -> None:
    reset = getattr(pipeline, "reset_episode", None)
    if callable(reset):
        reset()


def _lap_weight(demonstration: Demonstration, best_finish_time_s: float) -> float:
    relative = (best_finish_time_s - demonstration.finish_time_s) / ELITE_LAP_WEIGHT_TEMPERATURE_S
    return float(np.clip(np.exp(relative), MINIMUM_LAP_WEIGHT, 1.0))


def _prepare_samples(
    context: _LapContext, lap: _ResampledLap, weight: float
) -> tuple[list[Mapping[str, torch.Tensor]], list[int]]:
    observations: list[Mapping[str, torch.Tensor]] = []
    labels: list[int] = []
    previous_action = len(context.action_ids)
    for frame, action in zip(lap.frames[:-1], lap.actions, strict=True):
        label = _compact_label(context, lap.source, int(action))
        request = _ObservationRequest(frame, previous_action, weight)
        observations.append(_prepare_observation(context, request))
        labels.append(label)
        previous_action = label
    return observations, labels


def _compact_label(context: _LapContext, source: _LapSource, action: int) -> int:
    if action not in context.mapping:
        raise ValueError(f"demo {source.path} contains action {action} outside compact action IDs")
    return context.mapping[action]


def _prepare_observation(
    context: _LapContext, request: _ObservationRequest
) -> dict[str, torch.Tensor]:
    observation = context.pipeline.transform_observation(request.frame)
    if not isinstance(observation, Mapping):
        raise TypeError("behavior cloning requires mapping lidar observations")
    prepared = {key: value.detach().clone() for key, value in observation.items()}
    prepared["expert_previous_action"] = torch.tensor(request.previous_action, dtype=torch.long)
    _attach_default_recovery_metadata(prepared, len(context.action_ids))
    prepared[SAMPLE_WEIGHT_KEY] = torch.tensor(request.weight, dtype=torch.float32)
    if context.settings.previous_action_conditioning:
        prepared["previous_action"] = torch.tensor(request.previous_action, dtype=torch.long)
    return prepared


def _attach_default_recovery_metadata(
    observation: dict[str, torch.Tensor], action_count: int
) -> None:
    observation[SAMPLE_WEIGHT_KEY] = torch.tensor(1.0)
    observation[STUDENT_ACTION_KEY] = torch.tensor(action_count, dtype=torch.long)
    observation[INTERVENTION_KEY] = torch.tensor(False)
    observation[STATE_ERROR_KEY] = torch.tensor(0.0)


def _validate_demonstration_contract(context: _LapContext, source: _LapSource) -> None:
    _validate_geometry(context.pipeline, source)
    _validate_timing(context.settings, source)


def _validate_geometry(pipeline: FeaturePipeline, source: _LapSource) -> None:
    geometry = getattr(pipeline, "geometry", None)
    if geometry is None:
        return
    demonstration = source.demonstration
    if demonstration.map_uid != geometry.map_uid:
        raise ValueError(
            f"demo {source.path} map UID {demonstration.map_uid!r} does not match "
            f"feature geometry {geometry.map_uid!r}"
        )
    if demonstration.geometry_sha256 != geometry.sha256:
        raise ValueError(f"demo {source.path} was recorded against a different geometry asset")


def _validate_timing(settings: _LapLoadSettings, source: _LapSource) -> None:
    interval = settings.expected_decision_interval_ms
    if interval is not None:
        _validate_decision_interval(source, interval)
        return
    expected_repeat = settings.expected_action_repeat_frames
    if expected_repeat is not None and source.demonstration.action_repeat_frames != expected_repeat:
        raise ValueError(
            f"demo {source.path} action repeat {source.demonstration.action_repeat_frames} "
            f"does not match environment action repeat {expected_repeat}"
        )


def _validate_decision_interval(source: _LapSource, expected_interval_ms: float) -> None:
    recorded = source.demonstration.decision_interval_ms
    if recorded is None or np.isclose(recorded, expected_interval_ms, rtol=0.0, atol=0.05):
        return
    raise ValueError(
        f"demo {source.path} decision interval {recorded:g}ms does not match "
        f"environment decision interval {expected_interval_ms:g}ms"
    )
