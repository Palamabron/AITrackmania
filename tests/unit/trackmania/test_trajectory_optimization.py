from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trackmaniarl.trackmania.trajectory_optimization import (
    SafeTrajectoryOptimizer,
    TrajectorySchedule,
    TrajectorySearchConfig,
    TrajectorySearchOutcome,
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


def _successful_outcome(schedule: TrajectorySchedule) -> TrajectorySearchOutcome:
    reduction = int(np.sum(np.abs(schedule.boundary_offsets)))
    return TrajectorySearchOutcome(True, 36.6 - 0.1 * reduction, 100.0)


def _confirmed_optimizer(tmp_path: Path) -> SafeTrajectoryOptimizer:
    config = TrajectorySearchConfig(
        shortening_ticks=(1,),
        baseline_trials=2,
        confirmation_trials=2,
        max_trials=6,
        checkpoint_path=tmp_path / "best.npz",
    )
    return SafeTrajectoryOptimizer(config)


class _UnreliableEvaluation:
    def __init__(self) -> None:
        self.candidate_calls = 0

    def __call__(self, schedule: TrajectorySchedule) -> TrajectorySearchOutcome:
        if not np.any(schedule.boundary_offsets):
            return TrajectorySearchOutcome(True, 36.6, 100.0)
        self.candidate_calls += 1
        if self.candidate_calls == 1:
            return TrajectorySearchOutcome(True, 36.0, 100.0)
        return TrajectorySearchOutcome(False, None, 70.0)


def _unreliable_optimizer() -> SafeTrajectoryOptimizer:
    return SafeTrajectoryOptimizer(
        TrajectorySearchConfig(
            shortening_ticks=(1,), baseline_trials=1, confirmation_trials=2, max_trials=3
        )
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


def test_safe_optimizer_accepts_only_a_confirmed_faster_schedule(tmp_path: Path) -> None:
    initial = TrajectorySchedule.from_controls(_controls())
    optimizer = _confirmed_optimizer(tmp_path)

    result = optimizer.optimize(initial, _successful_outcome)

    assert result.median_finish_time_s == pytest.approx(36.4)
    assert np.sum(np.abs(result.schedule.boundary_offsets)) == 2
    assert (tmp_path / "best.npz").is_file()
    assert all(record.accepted for record in result.records)


def test_safe_optimizer_rejects_a_fast_but_unreliable_candidate() -> None:
    initial = TrajectorySchedule.from_controls(_controls())
    optimizer = _unreliable_optimizer()

    result = optimizer.optimize(initial, _UnreliableEvaluation())

    np.testing.assert_array_equal(result.schedule.boundary_offsets, 0)
    assert result.median_finish_time_s == pytest.approx(36.6)
    assert not any(record.accepted for record in result.records)
