"""Sample-efficient control-schedule search for a fixed TrackMania map."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np

from trackmaniarl.core.contracts import Policy
from trackmaniarl.trackmania.demonstrations import load_demonstration
from trackmaniarl.trackmania.guidance import TrajectoryTrackingDemonstrationPolicy

SCHEDULE_FORMAT = "trackmaniarl-trajectory-schedule-v1"


@dataclass(frozen=True, slots=True)
class SlowControlWindow:
    """A contiguous interval where the expert releases gas or applies brake."""

    first_segment: int
    stop_segment: int


@dataclass(frozen=True, slots=True)
class TrajectorySchedule:
    """Run-length encoded expert controls with optimizable switch boundaries."""

    boundaries: np.ndarray
    segment_controls: np.ndarray
    boundary_offsets: np.ndarray

    def __post_init__(self) -> None:
        boundaries = np.asarray(self.boundaries, dtype=np.int64).copy()
        controls = np.asarray(self.segment_controls, dtype=np.float32).copy()
        offsets = np.asarray(self.boundary_offsets, dtype=np.int64).copy()
        if boundaries.ndim != 1 or len(boundaries) < 2:
            raise ValueError("trajectory schedule requires at least one control segment")
        segment_count = len(boundaries) - 1
        if controls.shape != (segment_count, 3):
            raise ValueError("trajectory schedule controls must have shape (segments, 3)")
        if offsets.shape != (max(0, segment_count - 1),):
            raise ValueError("trajectory schedule requires one offset per internal boundary")
        if boundaries[0] != 0 or np.any(np.diff(boundaries) <= 0):
            raise ValueError("trajectory schedule boundaries must start at zero and increase")
        if not np.isfinite(controls).all():
            raise ValueError("trajectory schedule controls must be finite")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "segment_controls", controls)
        object.__setattr__(self, "boundary_offsets", offsets)
        self.effective_boundaries()

    @classmethod
    def from_controls(cls, controls: np.ndarray) -> TrajectorySchedule:
        values = np.asarray(controls, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3 or not len(values):
            raise ValueError("trajectory controls must have shape (steps, 3)")
        changes = np.flatnonzero(np.any(values[1:] != values[:-1], axis=1)) + 1
        boundaries = np.concatenate(([0], changes, [len(values)])).astype(np.int64)
        return cls(
            boundaries,
            values[boundaries[:-1]],
            np.zeros(max(0, len(boundaries) - 2), dtype=np.int64),
        )

    @property
    def step_count(self) -> int:
        return int(self.boundaries[-1])

    def effective_boundaries(self) -> np.ndarray:
        effective = self.boundaries.copy()
        effective[1:-1] += self.boundary_offsets
        if np.any(np.diff(effective) <= 0):
            raise ValueError("trajectory boundary offsets collapse a control segment")
        return effective

    def materialize(self) -> np.ndarray:
        durations = np.diff(self.effective_boundaries())
        controls = np.repeat(self.segment_controls, durations, axis=0)
        if controls.shape != (self.step_count, 3):
            raise AssertionError("trajectory schedule changed its total duration")
        return controls

    def source_controls(self) -> np.ndarray:
        return np.repeat(self.segment_controls, np.diff(self.boundaries), axis=0)

    def slow_windows(self, *, minimum_ticks: int = 3) -> tuple[SlowControlWindow, ...]:
        if minimum_ticks < 1:
            raise ValueError("minimum_ticks must be positive")
        slow = (self.segment_controls[:, 0] < 0.5) | (self.segment_controls[:, 1] > 0.5)
        boundaries = self.effective_boundaries()
        windows: list[SlowControlWindow] = []
        first = 0
        while first < len(slow):
            if not slow[first]:
                first += 1
                continue
            stop = first + 1
            while stop < len(slow) and slow[stop]:
                stop += 1
            if int(boundaries[stop] - boundaries[first]) >= minimum_ticks:
                windows.append(SlowControlWindow(first, stop))
            first = stop
        return tuple(windows)

    def shorten(
        self,
        window: SlowControlWindow,
        side: Literal["start", "end"],
        ticks: int,
    ) -> TrajectorySchedule:
        if ticks < 1:
            raise ValueError("trajectory shortening must use a positive tick count")
        segment_count = len(self.segment_controls)
        if not 0 <= window.first_segment < window.stop_segment <= segment_count:
            raise ValueError("slow-control window lies outside the trajectory schedule")
        boundary = window.first_segment if side == "start" else window.stop_segment
        if boundary in {0, segment_count}:
            raise ValueError("the first or last control window cannot be shortened on this side")
        offsets = self.boundary_offsets.copy()
        offsets[boundary - 1] += ticks if side == "start" else -ticks
        return replace(self, boundary_offsets=offsets)

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".npz":
            target = target.with_suffix(".npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.stem}.tmp.npz")
        np.savez_compressed(
            temporary,
            format=np.asarray(SCHEDULE_FORMAT),
            boundaries=self.boundaries,
            segment_controls=self.segment_controls,
            boundary_offsets=self.boundary_offsets,
        )
        os.replace(temporary, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> TrajectorySchedule:
        with np.load(path, allow_pickle=False) as data:
            if str(data["format"].item()) != SCHEDULE_FORMAT:
                raise ValueError("unsupported trajectory schedule format")
            return cls(
                boundaries=np.asarray(data["boundaries"], dtype=np.int64),
                segment_controls=np.asarray(data["segment_controls"], dtype=np.float32),
                boundary_offsets=np.asarray(data["boundary_offsets"], dtype=np.int64),
            )


@dataclass(frozen=True, slots=True)
class TrajectoryTrackerConfig:
    action_lead_steps: int = 1
    action_lead_ms: float | None = None
    lateral_gain: float = 0.8
    heading_gain: float = 4.0
    lateral_velocity_gain: float = 0.03
    steering_threshold: float = 0.35
    steering_release_threshold: float = 0.15
    preview_ms: float = 0.0
    minimum_correction_steps: int = 4
    reversal_neutral_steps: int = 2


def build_scheduled_policy(
    demonstration_path: str | Path,
    schedule: TrajectorySchedule,
    config: TrajectoryTrackerConfig | None = None,
) -> TrajectoryTrackingDemonstrationPolicy:
    demonstration = load_demonstration(demonstration_path)
    if schedule.step_count != len(demonstration.controls):
        raise ValueError("trajectory schedule and demonstration lengths do not match")
    if not np.array_equal(schedule.source_controls(), demonstration.controls):
        raise ValueError("trajectory schedule was created from a different demonstration")
    selected = config or TrajectoryTrackerConfig()
    return TrajectoryTrackingDemonstrationPolicy(
        demonstration.frames[:-1],
        schedule.materialize(),
        action_lead_steps=selected.action_lead_steps,
        action_lead_ms=selected.action_lead_ms,
        lateral_gain=selected.lateral_gain,
        heading_gain=selected.heading_gain,
        lateral_velocity_gain=selected.lateral_velocity_gain,
        steering_threshold=selected.steering_threshold,
        steering_release_threshold=selected.steering_release_threshold,
        preview_ms=selected.preview_ms,
        minimum_correction_steps=selected.minimum_correction_steps,
        reversal_neutral_steps=selected.reversal_neutral_steps,
    )


def run_trajectory_trial(
    environment: Any,
    policy: Policy,
    max_episode_steps: int,
) -> TrajectorySearchOutcome:
    """Evaluate one schedule without invoking a learner or checkpoint stack."""

    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    progress_pct = 0.0
    try:
        observation, _ = environment.reset(seed=0)
        reset_policy = getattr(policy, "reset_episode", None)
        if callable(reset_policy):
            reset_policy()
        for _ in range(max_episode_steps):
            action = policy.act(observation, deterministic=True)
            observation, _, terminated, truncated, info = environment.step(action)
            progress_pct = float(info.get("progress_pct", progress_pct))
            if not (terminated or truncated):
                continue
            finished = info.get("termination_reason") == "finished"
            race_time_ms = float(info.get("race_time_ms", 0.0))
            finish_time_s = race_time_ms / 1_000.0 if finished and race_time_ms > 0.0 else None
            return TrajectorySearchOutcome(finished, finish_time_s, progress_pct)
    except (TimeoutError, ConnectionError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return TrajectorySearchOutcome(False, None, progress_pct, error)
    return TrajectorySearchOutcome(False, None, progress_pct)


@dataclass(frozen=True, slots=True)
class TrajectorySearchOutcome:
    finished: bool
    finish_time_s: float | None
    progress_pct: float
    error: str | None = None

    def __post_init__(self) -> None:
        if self.finished and (self.finish_time_s is None or self.finish_time_s <= 0.0):
            raise ValueError("a finished trajectory trial requires a positive finish time")
        if not 0.0 <= self.progress_pct <= 100.0:
            raise ValueError("trajectory trial progress must be in [0, 100]")


@dataclass(frozen=True, slots=True)
class TrajectorySearchConfig:
    shortening_ticks: tuple[int, ...] = (4, 2, 1)
    minimum_window_ticks: int = 3
    baseline_trials: int = 2
    confirmation_trials: int = 2
    minimum_improvement_s: float = 0.015
    target_time_s: float = 36.0
    max_trials: int = 64
    checkpoint_path: Path | None = None
    journal_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.shortening_ticks or any(ticks < 1 for ticks in self.shortening_ticks):
            raise ValueError("shortening_ticks must contain positive integers")
        if min(self.minimum_window_ticks, self.baseline_trials, self.confirmation_trials) < 1:
            raise ValueError("trajectory search counts must be positive")
        if (
            self.minimum_improvement_s < 0.0
            or self.target_time_s <= 0.0
            or self.max_trials < self.baseline_trials
        ):
            raise ValueError("trajectory search budget is invalid")


@dataclass(frozen=True, slots=True)
class TrajectorySearchRecord:
    trial: int
    kind: str
    window: SlowControlWindow | None
    side: Literal["start", "end"] | None
    ticks: int
    accepted: bool
    outcome: TrajectorySearchOutcome


@dataclass(frozen=True, slots=True)
class TrajectorySearchResult:
    schedule: TrajectorySchedule
    median_finish_time_s: float
    outcomes: tuple[TrajectorySearchOutcome, ...]
    records: tuple[TrajectorySearchRecord, ...]


class SafeTrajectoryOptimizer:
    """Greedily shorten slow expert windows, retaining only confirmed improvements."""

    def __init__(self, config: TrajectorySearchConfig | None = None) -> None:
        self.config = config or TrajectorySearchConfig()

    def optimize(
        self,
        initial: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
    ) -> TrajectorySearchResult:
        records: list[TrajectorySearchRecord] = []
        baseline = self._evaluate_many(initial, evaluate, self.config.baseline_trials)
        self._require_robust_baseline(baseline)
        incumbent, incumbent_outcomes = initial, baseline
        incumbent_time = _median_finished(baseline)
        trial = len(baseline)
        self._checkpoint(incumbent)
        if incumbent_time <= self.config.target_time_s:
            return self._result(incumbent, incumbent_outcomes, records)
        for ticks in self.config.shortening_ticks:
            for window in initial.slow_windows(minimum_ticks=self.config.minimum_window_ticks):
                for side in ("start", "end"):
                    if trial >= self.config.max_trials:
                        return self._result(incumbent, incumbent_outcomes, records)
                    try:
                        candidate = incumbent.shorten(window, side, ticks)
                    except ValueError:
                        continue
                    screening = evaluate(candidate)
                    trial += 1
                    accepted = False
                    outcomes: tuple[TrajectorySearchOutcome, ...] = (screening,)
                    if self._promising(screening, incumbent_time):
                        remaining = self.config.max_trials - trial
                        confirmation_count = min(self.config.confirmation_trials - 1, remaining)
                        confirmations = self._evaluate_many(candidate, evaluate, confirmation_count)
                        trial += len(confirmations)
                        outcomes += confirmations
                        accepted = self._confirmed(outcomes, incumbent_time)
                    for outcome in outcomes:
                        records.append(
                            TrajectorySearchRecord(
                                trial=len(records) + self.config.baseline_trials + 1,
                                kind="shorten_slow_window",
                                window=window,
                                side=side,
                                ticks=ticks,
                                accepted=accepted,
                                outcome=outcome,
                            )
                        )
                        self._journal(records[-1])
                    if accepted:
                        incumbent, incumbent_outcomes = candidate, outcomes
                        incumbent_time = _median_finished(outcomes)
                        self._checkpoint(incumbent)
                        if incumbent_time <= self.config.target_time_s:
                            return self._result(incumbent, incumbent_outcomes, records)
        return self._result(incumbent, incumbent_outcomes, records)

    @staticmethod
    def _evaluate_many(
        schedule: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
        count: int,
    ) -> tuple[TrajectorySearchOutcome, ...]:
        return tuple(evaluate(schedule) for _ in range(count))

    @staticmethod
    def _require_robust_baseline(outcomes: Sequence[TrajectorySearchOutcome]) -> None:
        if not outcomes or any(not outcome.finished or outcome.error for outcome in outcomes):
            raise RuntimeError("trajectory optimization requires a fully finishing baseline")

    def _promising(self, outcome: TrajectorySearchOutcome, incumbent_time: float) -> bool:
        return bool(
            outcome.finished
            and outcome.error is None
            and outcome.finish_time_s is not None
            and outcome.finish_time_s < incumbent_time - self.config.minimum_improvement_s
        )

    def _confirmed(
        self, outcomes: Sequence[TrajectorySearchOutcome], incumbent_time: float
    ) -> bool:
        return bool(
            len(outcomes) == self.config.confirmation_trials
            and all(outcome.finished and outcome.error is None for outcome in outcomes)
            and _median_finished(outcomes) < incumbent_time - self.config.minimum_improvement_s
        )

    def _checkpoint(self, schedule: TrajectorySchedule) -> None:
        if self.config.checkpoint_path is not None:
            schedule.save(self.config.checkpoint_path)

    def _journal(self, record: TrajectorySearchRecord) -> None:
        if self.config.journal_path is None:
            return
        target = self.config.journal_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trial": record.trial,
            "kind": record.kind,
            "window": (
                [record.window.first_segment, record.window.stop_segment]
                if record.window is not None
                else None
            ),
            "side": record.side,
            "ticks": record.ticks,
            "accepted": record.accepted,
            "outcome": {
                "finished": record.outcome.finished,
                "finish_time_s": record.outcome.finish_time_s,
                "progress_pct": record.outcome.progress_pct,
                "error": record.outcome.error,
            },
        }
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    @staticmethod
    def _result(
        schedule: TrajectorySchedule,
        outcomes: tuple[TrajectorySearchOutcome, ...],
        records: list[TrajectorySearchRecord],
    ) -> TrajectorySearchResult:
        return TrajectorySearchResult(
            schedule=schedule,
            median_finish_time_s=_median_finished(outcomes),
            outcomes=outcomes,
            records=tuple(records),
        )


def _median_finished(outcomes: Sequence[TrajectorySearchOutcome]) -> float:
    times = [
        outcome.finish_time_s
        for outcome in outcomes
        if outcome.finished and outcome.finish_time_s is not None
    ]
    if not times:
        raise ValueError("cannot score trajectory trials without a completed lap")
    return float(median(times))
