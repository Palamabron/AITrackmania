"""Report whether a run's reward actually prefers the faster lap.

Replays recorded demonstration laps through the reward configured in a run
YAML and measures how much return one saved second is worth, how that compares
to the cost of failing a lap, and how much of the difference survives the
training discount.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmrl.core.spec import RunSpec
from tmrl.trackmania.demonstrations import load_demonstration
from tmrl.trackmania.environment import TrackmaniaEnvironmentConfig
from tmrl.trackmania.geometry import BoundaryGeometry
from tmrl.trackmania.reward import TrajectoryReward


@dataclass(frozen=True, slots=True)
class LapReturn:
    finish_time_s: float
    total: float
    discounted: float


@dataclass(frozen=True, slots=True)
class Sensitivity:
    reward_per_second: float
    discounted_per_second: float
    failure_cost: float
    break_even_crash_probability: float


def lap_sensitivity(laps: list[LapReturn], failure_cost: float) -> Sensitivity:
    """Fit reward against lap time; the slope is what the objective pays for speed."""

    if len(laps) < 2 or failure_cost <= 0.0:
        raise ValueError("need at least two laps and a positive failure cost")
    times = np.asarray([lap.finish_time_s for lap in laps])
    if float(times.max() - times.min()) <= 0.0:
        raise ValueError("laps must differ in finish time")
    per_second = -float(np.polyfit(times, [lap.total for lap in laps], 1)[0])
    discounted_per_second = -float(np.polyfit(times, [lap.discounted for lap in laps], 1)[0])
    return Sensitivity(
        reward_per_second=per_second,
        discounted_per_second=discounted_per_second,
        failure_cost=failure_cost,
        break_even_crash_probability=max(0.0, per_second) / failure_cost,
    )


def replay_lap(
    path: Path, config: TrackmaniaEnvironmentConfig, reference: np.ndarray, gamma: float
) -> LapReturn:
    demo = load_demonstration(path)
    reward = TrajectoryReward(reference, **config.reward_kwargs())
    frames = demo.frames
    position = list(config.position_indices)
    velocity = list(config.velocity_indices)
    reward.reset(
        frames[0, position], velocity=frames[0, velocity], race_time_ms=float(frames[0, 3])
    )
    total = 0.0
    discounted = 0.0
    for step, next_frame in enumerate(frames[1:]):
        result = reward.step(
            next_frame[position],
            finish_ui_active=bool(next_frame[2]),
            velocity=next_frame[velocity],
            race_time_ms=float(next_frame[3]),
            steering=float(demo.controls[step, 2]),
        )
        total += result.reward
        discounted += (gamma**step) * result.reward
    return LapReturn(demo.finish_time_s, total, discounted)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("demos", type=Path)
    parser.add_argument(
        "--failure-cost",
        type=float,
        default=78.0,
        help="return forfeited by abandoning a lap halfway (progress, finish and shaping)",
    )
    arguments = parser.parse_args()
    spec = RunSpec.from_yaml(arguments.config)
    if spec.components.environment is None:
        raise ValueError("config has no components.environment")
    base = arguments.config.resolve().parent
    config = TrackmaniaEnvironmentConfig.model_validate(
        spec.components.environment.kwargs["config"]
    )
    geometry_path = config.geometry_path
    if geometry_path is None:
        raise ValueError("config has no geometry_path")
    geometry = BoundaryGeometry(base / geometry_path)
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    paths = sorted(arguments.demos.glob("*.npz"))
    if len(paths) < 2:
        raise ValueError(f"need at least two demonstrations in {arguments.demos}")
    laps = [replay_lap(path, config, reference, spec.training.gamma) for path in paths]
    for path, lap in zip(paths, laps, strict=True):
        print(f"{path.name:<24}{lap.finish_time_s:>8.2f}s  return {lap.total:>9.2f}")
    sensitivity = lap_sensitivity(laps, arguments.failure_cost)
    steps = float(np.mean([len(load_demonstration(path).actions) for path in paths]))
    print(f"\nreward per second saved:            {sensitivity.reward_per_second:>9.3f}")
    print(f"discounted per second saved:        {sensitivity.discounted_per_second:>9.3f}")
    print(f"cost of failing a lap halfway:      {sensitivity.failure_cost:>9.3f}")
    print(
        "break-even crash probability for a one second gain: "
        f"{100 * sensitivity.break_even_crash_probability:.2f}%"
    )
    print(f"weight of the finish reward from the start line: {spec.training.gamma**steps:.3e}")
    print(f"discount horizon: {1 / (1 - spec.training.gamma):.0f} steps")


if __name__ == "__main__":
    main()
