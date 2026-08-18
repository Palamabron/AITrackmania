from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from trackmaniarl.trackmania.trajectory_optimization import (
    SafeTrajectoryOptimizer,
    TrajectorySchedule,
    TrajectorySearchConfig,
    TrajectorySearchOutcome,
    TrajectoryTrackerConfig,
    build_scheduled_policy,
    run_trajectory_trial,
)


def _controls() -> np.ndarray:
    return np.asarray(
        [
            *([[1.0, 0.0, 0.0]] * 4),
            *([[0.0, 0.0, 1.0]] * 3),
            *([[0.0, 1.0, 1.0]] * 2),
            *([[0.0, 0.0, 1.0]] * 3),
            *([[1.0, 0.0, 1.0]] * 4),
        ],
        dtype=np.float32,
    )


def test_schedule_round_trip_preserves_expert_controls() -> None:
    controls = _controls()

    schedule = TrajectorySchedule.from_controls(controls)

    np.testing.assert_array_equal(schedule.materialize(), controls)


def test_schedule_shortens_a_complete_coast_and_brake_window() -> None:
    schedule = TrajectorySchedule.from_controls(_controls())
    (window,) = schedule.slow_windows()

    shortened = schedule.shorten(window, "start", 1).shorten(window, "end", 2)

    controls = shortened.materialize()
    assert np.count_nonzero(controls[:, 0] < 0.5) == 5
    np.testing.assert_array_equal(controls[:5], [[1.0, 0.0, 0.0]] * 5)
    np.testing.assert_array_equal(controls[-6:], [[1.0, 0.0, 1.0]] * 6)


def test_schedule_rejects_offsets_that_collapse_a_segment() -> None:
    schedule = TrajectorySchedule.from_controls(_controls())
    (window,) = schedule.slow_windows()

    with pytest.raises(ValueError, match="collapse"):
        schedule.shorten(window, "start", 4)


def test_schedule_can_be_saved_and_loaded_atomically(tmp_path: Path) -> None:
    schedule = TrajectorySchedule.from_controls(_controls())
    path = tmp_path / "best-schedule"

    saved = schedule.save(path)
    loaded = TrajectorySchedule.load(saved)

    assert saved == path.with_suffix(".npz")
    np.testing.assert_array_equal(loaded.materialize(), schedule.materialize())


def test_scheduled_policy_uses_physical_action_lead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = _controls()
    frames = np.zeros((len(controls) + 1, 33), dtype=np.float32)
    frames[:, 3] = np.arange(len(frames), dtype=np.float32) * 10.0
    frames[:, 12] = 1.0
    demonstration = SimpleNamespace(frames=frames, controls=controls)
    monkeypatch.setattr(
        "trackmaniarl.trackmania.trajectory_optimization.load_demonstration",
        lambda _: demonstration,
    )

    policy = build_scheduled_policy(
        "demo.npz",
        TrajectorySchedule.from_controls(controls),
        TrajectoryTrackerConfig(action_lead_ms=15.0),
    )

    assert policy.action_lead_ms == 15.0


def test_safe_optimizer_accepts_only_a_confirmed_faster_schedule(tmp_path: Path) -> None:
    initial = TrajectorySchedule.from_controls(_controls())
    calls: dict[tuple[int, ...], int] = {}

    def evaluate(schedule: TrajectorySchedule) -> TrajectorySearchOutcome:
        key = tuple(int(value) for value in schedule.boundary_offsets)
        calls[key] = calls.get(key, 0) + 1
        reduction = int(np.sum(np.abs(schedule.boundary_offsets)))
        return TrajectorySearchOutcome(True, 36.6 - 0.1 * reduction, 100.0)

    optimizer = SafeTrajectoryOptimizer(
        TrajectorySearchConfig(
            shortening_ticks=(1,),
            baseline_trials=2,
            confirmation_trials=2,
            max_trials=6,
            checkpoint_path=tmp_path / "best.npz",
        )
    )

    result = optimizer.optimize(initial, evaluate)

    assert result.median_finish_time_s == pytest.approx(36.4)
    assert np.sum(np.abs(result.schedule.boundary_offsets)) == 2
    assert (tmp_path / "best.npz").is_file()
    assert all(record.accepted for record in result.records)


def test_safe_optimizer_rejects_a_fast_but_unreliable_candidate() -> None:
    initial = TrajectorySchedule.from_controls(_controls())
    candidate_calls = 0

    def evaluate(schedule: TrajectorySchedule) -> TrajectorySearchOutcome:
        nonlocal candidate_calls
        if not np.any(schedule.boundary_offsets):
            return TrajectorySearchOutcome(True, 36.6, 100.0)
        candidate_calls += 1
        if candidate_calls == 1:
            return TrajectorySearchOutcome(True, 36.0, 100.0)
        return TrajectorySearchOutcome(False, None, 70.0)

    optimizer = SafeTrajectoryOptimizer(
        TrajectorySearchConfig(
            shortening_ticks=(1,),
            baseline_trials=1,
            confirmation_trials=2,
            max_trials=3,
        )
    )

    result = optimizer.optimize(initial, evaluate)

    np.testing.assert_array_equal(result.schedule.boundary_offsets, 0)
    assert result.median_finish_time_s == pytest.approx(36.6)
    assert not any(record.accepted for record in result.records)


def test_safe_optimizer_refuses_to_search_from_a_failing_baseline() -> None:
    schedule = TrajectorySchedule.from_controls(_controls())
    optimizer = SafeTrajectoryOptimizer(TrajectorySearchConfig(baseline_trials=1))

    with pytest.raises(RuntimeError, match="fully finishing baseline"):
        optimizer.optimize(
            schedule,
            lambda _: TrajectorySearchOutcome(False, None, 35.0),
        )


class _FinishingEnvironment:
    def __init__(self) -> None:
        self.steps = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        del seed
        self.steps = 0
        return np.zeros(33, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self.steps += 1
        return (
            np.zeros(33, dtype=np.float32),
            0.0,
            True,
            False,
            {
                "progress_pct": 100.0,
                "termination_reason": "finished",
                "race_time_ms": 36_125.0,
            },
        )


class _StraightPolicy:
    def reset_episode(self) -> None:
        return None

    def act(self, observation: np.ndarray, *, deterministic: bool) -> np.ndarray:
        del observation, deterministic
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def test_live_trial_reports_game_timer_instead_of_wall_clock() -> None:
    outcome = run_trajectory_trial(_FinishingEnvironment(), _StraightPolicy(), 10)

    assert outcome.finished
    assert outcome.finish_time_s == pytest.approx(36.125)
