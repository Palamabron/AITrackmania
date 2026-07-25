"""Record human reference laps as replay-ready demonstration transitions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import time_ns
from typing import Any

import numpy as np

from tmrl.core.data import Transition
from tmrl.core.runtime import _instantiate
from tmrl.core.spec import RunSpec
from tmrl.distributed.demos import DEMO_SUFFIX, save_demonstration
from tmrl.trackmania.actions import (
    build_brake_tap_action_table,
    continuous_control_to_discrete_index,
)
from tmrl.trackmania.control import RecordingController
from tmrl.trackmania.environment import OpenPlanetEnvironmentFactory

_HUMAN_RESTART_TIMEOUT_S = 180.0
_INPUT_STEER_INDEX = 30
_INPUT_GAS_INDEX = 31
_INPUT_BRAKE_INDEX = 32


def human_action(observation: Any, table: list[np.ndarray]) -> int:
    """Map the driver's live inputs from a raw telemetry frame to a table action."""

    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    control = np.asarray(
        [values[_INPUT_GAS_INDEX], values[_INPUT_BRAKE_INDEX], values[_INPUT_STEER_INDEX]],
        dtype=np.float32,
    )
    return continuous_control_to_discrete_index(control, table)


def record_demonstrations(
    config_path: Path,
    output_dir: Path,
    *,
    episodes: int,
    status: Callable[[str], None] = print,
) -> list[Path]:
    """Capture finished human laps while a passive controller injects no input."""

    spec = RunSpec.from_yaml(config_path)
    base_dir = config_path.resolve().parent
    if spec.components.environment is None:
        raise ValueError("demonstration recording requires components.environment")
    pipeline = _instantiate(spec.components.feature_pipeline, base_dir=base_dir)
    environment_config = dict(spec.components.environment.kwargs["config"])
    environment_config["start_timeout_s"] = _HUMAN_RESTART_TIMEOUT_S
    factory = OpenPlanetEnvironmentFactory(
        environment_config, controller=RecordingController(), base_dir=base_dir
    )
    environment = factory.create(seed=0)
    _, table = build_brake_tap_action_table()
    saved: list[Path] = []
    try:
        while len(saved) < episodes:
            status(f"Demo {len(saved) + 1}/{episodes}: restart the race now and drive a clean lap.")
            result = _record_episode(
                environment, pipeline, table, spec.training.max_episode_steps, index=len(saved)
            )
            if result is None:
                status("Episode did not finish; discarded. Restart the race to retry.")
                continue
            transitions, finish_time_s = result
            path = output_dir / f"demo-{time_ns()}-{finish_time_s:.2f}s{DEMO_SUFFIX}"
            save_demonstration(
                path,
                transitions,
                {"finish_time_s": finish_time_s, "steps": len(transitions)},
            )
            status(f"Saved {path.name}: {finish_time_s:.2f}s in {len(transitions)} transitions.")
            saved.append(path)
    finally:
        environment.close()
    return saved


def _record_episode(
    environment: Any,
    pipeline: Any,
    table: list[np.ndarray],
    max_steps: int,
    *,
    index: int,
) -> tuple[list[Transition], float] | None:
    observation, _ = environment.reset()
    episode_id = f"demo/{time_ns()}/{index:04d}"
    transitions: list[Transition] = []
    for step in range(max_steps):
        prepared = pipeline.transform_observation(observation)
        action = human_action(observation, table)
        observation, reward, terminated, truncated, info = environment.step(action)
        transitions.append(
            Transition(
                observation=prepared,
                action=action,
                reward=float(reward),
                next_observation=pipeline.transform_observation(observation),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info={"is_demo": True},
                episode_id=episode_id,
                step=step,
            )
        )
        if terminated or truncated:
            if str(info.get("termination_reason")) == "finished":
                return transitions, float(info.get("race_time_ms", 0.0)) / 1_000.0
            return None
    return None
