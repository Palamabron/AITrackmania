"""Sample-efficient control-schedule search for a fixed TrackMania map."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np

from trackmaniarl.core.contracts import Policy, PolicyMode
from trackmaniarl.trackmania.demonstrations import load_demonstration
from trackmaniarl.trackmania.guidance import TrajectoryTrackingDemonstrationPolicy
from trackmaniarl.trackmania.guidance_tracking import (
    TrajectoryTrackingConfig,
    TrajectoryTrackingReference,
)
from trackmaniarl.trackmania.trajectory_journal import append_trajectory_record
from trackmaniarl.trackmania.trajectory_schedule import (
    SCHEDULE_FORMAT as SCHEDULE_FORMAT,
)
from trackmaniarl.trackmania.trajectory_schedule import (
    SlowControlWindow as SlowControlWindow,
)
from trackmaniarl.trackmania.trajectory_schedule import (
    TrajectorySchedule as TrajectorySchedule,
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

    def tracking_config(self) -> TrajectoryTrackingConfig:
        return TrajectoryTrackingConfig(
            self.action_lead_steps,
            self.action_lead_ms,
            self.lateral_gain,
            self.heading_gain,
            self.lateral_velocity_gain,
            self.steering_threshold,
            self.steering_release_threshold,
            self.preview_ms,
            self.minimum_correction_steps,
            self.reversal_neutral_steps,
        )


@dataclass(frozen=True, slots=True)
class _SearchRequest:
    window: SlowControlWindow
    side: Literal["start", "end"]
    ticks: int


@dataclass(frozen=True, slots=True)
class _CandidateEvaluation:
    candidate: TrajectorySchedule
    outcomes: tuple[TrajectorySearchOutcome, ...]
    accepted: bool


@dataclass(slots=True)
class _SearchState:
    incumbent: TrajectorySchedule
    outcomes: tuple[TrajectorySearchOutcome, ...]
    median_time_s: float
    trial: int
    records: list[TrajectorySearchRecord]


@dataclass(slots=True)
class _TrialState:
    observation: Any
    progress_pct: float = 0.0


@dataclass(frozen=True, slots=True)
class _TrialUpdate:
    outcome: TrajectorySearchOutcome | None = None


def build_scheduled_policy(
    demonstration_path: str | Path,
    schedule: TrajectorySchedule,
    config: TrajectoryTrackerConfig | None = None,
) -> TrajectoryTrackingDemonstrationPolicy:
    demonstration = load_demonstration(demonstration_path)
    _validate_schedule(schedule, demonstration.controls)
    selected = config or TrajectoryTrackerConfig()
    return _tracking_policy(demonstration.frames[:-1], schedule.materialize(), selected)


def _validate_schedule(schedule: TrajectorySchedule, controls: np.ndarray) -> None:
    if schedule.step_count != len(controls):
        raise ValueError("trajectory schedule and demonstration lengths do not match")
    if not np.array_equal(schedule.source_controls(), controls):
        raise ValueError("trajectory schedule was created from a different demonstration")


def _tracking_policy(
    frames: np.ndarray, controls: np.ndarray, config: TrajectoryTrackerConfig
) -> TrajectoryTrackingDemonstrationPolicy:
    return TrajectoryTrackingDemonstrationPolicy(
        TrajectoryTrackingReference(frames, controls), config.tracking_config()
    )


def run_trajectory_trial(
    environment: Any,
    policy: Policy,
    max_episode_steps: int,
) -> TrajectorySearchOutcome:
    if max_episode_steps < 1:
        raise ValueError("max_episode_steps must be positive")
    state = _TrialState(None)
    try:
        state.observation = _reset_trial(environment, policy)
        for _ in range(max_episode_steps):
            update = _trial_step(environment, policy, state)
            if update.outcome is not None:
                return update.outcome
    except (TimeoutError, ConnectionError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        return TrajectorySearchOutcome(False, None, state.progress_pct, error)
    return TrajectorySearchOutcome(False, None, state.progress_pct)


def _reset_trial(environment: Any, policy: Policy) -> Any:
    observation, _ = environment.reset(seed=0)
    reset_policy = getattr(policy, "reset_episode", None)
    if callable(reset_policy):
        reset_policy()
    return observation


def _trial_step(environment: Any, policy: Policy, state: _TrialState) -> _TrialUpdate:
    action = policy.act(state.observation, PolicyMode.EVALUATION)
    state.observation, _, terminated, truncated, info = environment.step(action)
    state.progress_pct = float(info.get("progress_pct", state.progress_pct))
    if not (terminated or truncated):
        return _TrialUpdate()
    finished = info.get("termination_reason") == "finished"
    race_time_ms = float(info.get("race_time_ms", 0.0))
    finish_time_s = race_time_ms / 1_000.0 if finished and race_time_ms > 0.0 else None
    return _TrialUpdate(TrajectorySearchOutcome(finished, finish_time_s, state.progress_pct))


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
        state = self._initial_state(initial, evaluate)
        if state.median_time_s <= self.config.target_time_s:
            return self._state_result(state)
        return self._search(initial, evaluate, state)

    def _initial_state(
        self,
        initial: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
    ) -> _SearchState:
        baseline = self._evaluate_many(initial, evaluate, self.config.baseline_trials)
        self._require_robust_baseline(baseline)
        self._checkpoint(initial)
        return _SearchState(initial, baseline, _median_finished(baseline), len(baseline), [])

    def _search(
        self,
        initial: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
        state: _SearchState,
    ) -> TrajectorySearchResult:
        for request, candidate in self._candidate_schedules(initial, state):
            if state.trial >= self.config.max_trials:
                return self._state_result(state)
            evaluation = self._evaluate_candidate(candidate, evaluate, state)
            self._record_evaluation(state, request, evaluation)
            if evaluation.accepted:
                self._accept_candidate(state, evaluation)
                if state.median_time_s <= self.config.target_time_s:
                    return self._state_result(state)
        return self._state_result(state)

    def _candidate_schedules(
        self, initial: TrajectorySchedule, state: _SearchState
    ) -> Iterator[tuple[_SearchRequest, TrajectorySchedule]]:
        for request in self._search_requests(initial):
            try:
                candidate = state.incumbent.shorten(request.window, request.side, request.ticks)
            except ValueError:
                continue
            yield request, candidate

    def _search_requests(self, initial: TrajectorySchedule) -> Iterator[_SearchRequest]:
        windows = initial.slow_windows(minimum_ticks=self.config.minimum_window_ticks)
        for ticks in self.config.shortening_ticks:
            for window in windows:
                yield _SearchRequest(window, "start", ticks)
                yield _SearchRequest(window, "end", ticks)

    def _evaluate_candidate(
        self,
        candidate: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
        state: _SearchState,
    ) -> _CandidateEvaluation:
        screening = evaluate(candidate)
        state.trial += 1
        outcomes: tuple[TrajectorySearchOutcome, ...] = (screening,)
        if self._promising(screening, state.median_time_s):
            outcomes += self._confirm_candidate(candidate, evaluate, state)
        accepted = self._confirmed(outcomes, state.median_time_s)
        return _CandidateEvaluation(candidate, outcomes, accepted)

    def _confirm_candidate(
        self,
        candidate: TrajectorySchedule,
        evaluate: Callable[[TrajectorySchedule], TrajectorySearchOutcome],
        state: _SearchState,
    ) -> tuple[TrajectorySearchOutcome, ...]:
        remaining = self.config.max_trials - state.trial
        count = min(self.config.confirmation_trials - 1, remaining)
        confirmations = self._evaluate_many(candidate, evaluate, count)
        state.trial += len(confirmations)
        return confirmations

    def _record_evaluation(
        self, state: _SearchState, request: _SearchRequest, evaluation: _CandidateEvaluation
    ) -> None:
        for outcome in evaluation.outcomes:
            record = TrajectorySearchRecord(
                trial=len(state.records) + self.config.baseline_trials + 1,
                kind="shorten_slow_window",
                window=request.window,
                side=request.side,
                ticks=request.ticks,
                accepted=evaluation.accepted,
                outcome=outcome,
            )
            state.records.append(record)
            self._journal(record)

    def _accept_candidate(self, state: _SearchState, evaluation: _CandidateEvaluation) -> None:
        state.incumbent = evaluation.candidate
        state.outcomes = evaluation.outcomes
        state.median_time_s = _median_finished(evaluation.outcomes)
        self._checkpoint(state.incumbent)

    @staticmethod
    def _state_result(state: _SearchState) -> TrajectorySearchResult:
        return SafeTrajectoryOptimizer._result(state.incumbent, state.outcomes, state.records)

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
        append_trajectory_record(self.config.journal_path, record)

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
