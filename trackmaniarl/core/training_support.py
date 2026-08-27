"""Shared result and state helpers for local training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Counters returned by a completed bounded training run."""

    episodes: int
    transitions: int
    updates: int
    checkpoints: tuple[Path, ...]
    evaluation: Mapping[str, float] | None


@dataclass(slots=True)
class TrainingCounters:
    transitions: int = 0
    updates: int = 0
    episodes: int = 0
    next_episode_index: int = 0
    fractional_updates: float = 0.0

    def as_mapping(self) -> dict[str, int | float]:
        return {
            "transitions": self.transitions,
            "updates": self.updates,
            "episodes": self.episodes,
            "fractional_updates": self.fractional_updates,
            "next_episode_index": self.next_episode_index,
        }


def episode_metrics(result: Any) -> dict[str, float]:
    telemetry = result.artifact.telemetry
    if not telemetry:
        return {
            "progress_pct": 0.0,
            "progress_m": 0.0,
            "episode_elapsed_s": 0.0,
            "race_time_s": 0.0,
            **_episode_telemetry_metrics(()),
        }
    final = telemetry[-1]
    return {
        "progress_pct": float(final.get("progress_pct", 0.0)),
        "progress_m": float(final.get("progress_m", 0.0)),
        "episode_elapsed_s": float(final.get("episode_elapsed_s", 0.0)),
        "race_time_s": float(final.get("race_time_ms", 0.0)) / 1_000.0,
        **_episode_telemetry_metrics(telemetry),
    }


def _episode_telemetry_metrics(
    telemetry: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not telemetry:
        return _empty_telemetry_metrics()
    controller = [float(item.get("controller_apply_ms", 0.0)) for item in telemetry]
    wait = [float(item.get("telemetry_wait_ms", 0.0)) for item in telemetry]
    skipped = [float(item.get("telemetry_skipped_frames", 0.0)) for item in telemetry]
    return {
        **_latency_metrics(controller, wait),
        "telemetry_skipped_frames_total": sum(skipped),
        "telemetry_skipped_frames_mean": sum(skipped) / len(skipped),
        "telemetry_skipped_frames_max": max(skipped),
        "telemetry_steps_with_skipped_frames_fraction": sum(value > 0 for value in skipped)
        / len(skipped),
    }


def _empty_telemetry_metrics() -> dict[str, float]:
    return {
        "controller_apply_ms_mean": 0.0,
        "controller_apply_ms_max": 0.0,
        "telemetry_wait_ms_mean": 0.0,
        "telemetry_wait_ms_max": 0.0,
        "telemetry_skipped_frames_total": 0.0,
        "telemetry_skipped_frames_mean": 0.0,
        "telemetry_skipped_frames_max": 0.0,
        "telemetry_steps_with_skipped_frames_fraction": 0.0,
    }


def _latency_metrics(controller: list[float], wait: list[float]) -> dict[str, float]:
    return {
        "controller_apply_ms_mean": sum(controller) / len(controller),
        "controller_apply_ms_max": max(controller),
        "telemetry_wait_ms_mean": sum(wait) / len(wait),
        "telemetry_wait_ms_max": max(wait),
    }


def _state_dict(component: object) -> Mapping[str, object]:
    method = getattr(component, "state_dict", None)
    if not callable(method):
        raise TypeError(f"{type(component).__name__} has no state_dict()")
    state = method()
    if not isinstance(state, Mapping):
        raise TypeError(f"{type(component).__name__}.state_dict() must return a mapping")
    return state


def _load_state_dict(component: object, state: object) -> None:
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint component state must be a mapping")
    method = getattr(component, "load_state_dict", None)
    if not callable(method):
        raise TypeError(
            f"{type(component).__name__} cannot resume because it has no load_state_dict()"
        )
    method(state)
