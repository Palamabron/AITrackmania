"""First-party OpenPlanet environment factory for the current SDK runtime."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from time import monotonic, sleep
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tmrl.trackmania.actions import build_brake_tap_action_table
from tmrl.trackmania.control import Controller, GamepadController
from tmrl.trackmania.geometry import BoundaryGeometry
from tmrl.trackmania.reward import TrajectoryReward
from tmrl.trackmania.session import OpenPlanetSessionClient
from tmrl.trackmania.telemetry import (
    DEFAULT_POSITION_INDICES,
    DEFAULT_TELEMETRY_FIELD_COUNT,
    DEFAULT_VELOCITY_INDICES,
    OpenPlanetClient,
    TelemetryFrame,
)


class TrackmaniaEnvironmentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_path: Path | None = None
    geometry_path: Path | None = None
    expected_map_uid: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=9000, ge=1, le=65535)
    session_port: int = Field(default=9001, ge=1, le=65535)
    field_count: int = Field(default=DEFAULT_TELEMETRY_FIELD_COUNT, ge=3)
    timeout_s: float = Field(default=10.0, gt=0)
    reset_settle_s: float = Field(default=0.5, ge=0.0)
    start_timeout_s: float = Field(default=15.0, gt=0.0)
    start_poll_s: float = Field(default=0.01, ge=0.0)
    action_repeat_frames: int = Field(default=4, ge=1, le=20)
    position_indices: tuple[int, int, int] = DEFAULT_POSITION_INDICES
    velocity_indices: tuple[int, int, int] = DEFAULT_VELOCITY_INDICES
    crash_distance: float = Field(default=25.0, gt=0)
    no_progress_steps: int = Field(default=200, ge=1)
    slow_progress_window_steps: int = Field(default=80, ge=2)
    minimum_progress_per_window_m: float = Field(default=2.0, ge=0.0)
    terminal_failure_penalty: float = Field(default=1.0, ge=0.0)
    collision_penalty: float = Field(default=0.05, ge=0.0)
    collision_cooldown_s: float = Field(default=0.0, ge=0.0)
    minimum_finish_steps: int = Field(default=50, ge=1)
    nearest_forward_points: int = Field(default=500, ge=1)
    nearest_backward_points: int = Field(default=10, ge=0)
    time_penalty_per_second: float = Field(default=0.1, ge=0.0)
    max_time_delta_s: float = Field(default=1.0, gt=0.0)
    progress_reward_full_lap: float = Field(default=10.0, ge=0.0)
    finish_reward: float = Field(default=30.0, ge=0.0)
    potential_progress_weight: float = Field(default=2.0, ge=0.0)
    max_projected_speed_mps: float = Field(default=100.0, gt=0.0)
    velocity_to_mps_scale: float = Field(default=0.001, gt=0.0)
    projected_velocity_scale: float = Field(default=0.0, ge=0.0)
    steering_delta_penalty: float = Field(default=0.0, ge=0.0)
    reward_gamma: float = Field(default=0.995, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _reward_contract_is_valid(self) -> TrackmaniaEnvironmentConfig:
        if self.trajectory_path is None and self.geometry_path is None:
            raise ValueError("either trajectory_path or geometry_path is required")
        for name, indices in (
            ("position_indices", self.position_indices),
            ("velocity_indices", self.velocity_indices),
        ):
            if len(set(indices)) != 3 or any(
                index < 0 or index >= self.field_count for index in indices
            ):
                raise ValueError(f"{name} must be three unique indices inside the telemetry packet")
        values = (
            self.time_penalty_per_second,
            self.max_time_delta_s,
            self.progress_reward_full_lap,
            self.finish_reward,
            self.potential_progress_weight,
            self.max_projected_speed_mps,
            self.velocity_to_mps_scale,
            self.projected_velocity_scale,
            self.steering_delta_penalty,
            self.collision_cooldown_s,
            self.reward_gamma,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("reward values must be finite")
        return self


class OpenPlanetEnvironment:
    def __init__(
        self,
        config: TrackmaniaEnvironmentConfig,
        controller: Controller,
        *,
        evaluation_map: Any | None = None,
    ) -> None:
        self.config, self.controller = config, controller
        self.client = OpenPlanetClient(
            config.host, config.port, field_count=config.field_count, timeout_s=config.timeout_s
        )
        reward_kwargs: dict[str, Any] = {
            "crash_distance": config.crash_distance,
            "no_progress_steps": config.no_progress_steps,
            "slow_progress_window_steps": config.slow_progress_window_steps,
            "minimum_progress_per_window_m": config.minimum_progress_per_window_m,
            "terminal_failure_penalty": config.terminal_failure_penalty,
            "collision_penalty": config.collision_penalty,
            "collision_cooldown_s": config.collision_cooldown_s,
            "minimum_finish_steps": config.minimum_finish_steps,
            "nearest_forward_points": config.nearest_forward_points,
            "nearest_backward_points": config.nearest_backward_points,
            "time_penalty_per_second": config.time_penalty_per_second,
            "max_time_delta_s": config.max_time_delta_s,
            "progress_reward_full_lap": config.progress_reward_full_lap,
            "finish_reward": config.finish_reward,
            "potential_progress_weight": config.potential_progress_weight,
            "max_projected_speed_mps": config.max_projected_speed_mps,
            "velocity_to_mps_scale": config.velocity_to_mps_scale,
            "projected_velocity_scale": config.projected_velocity_scale,
            "steering_delta_penalty": config.steering_delta_penalty,
            "reward_gamma": config.reward_gamma,
        }
        geometry_path = (
            evaluation_map.geometry_path if evaluation_map is not None else config.geometry_path
        )
        expected_map_uid = (
            evaluation_map.expected_map_uid
            if evaluation_map is not None
            else config.expected_map_uid
        )
        self.geometry = (
            BoundaryGeometry(geometry_path, expected_map_uid=expected_map_uid)
            if geometry_path is not None
            else None
        )
        if self.geometry is not None:
            self.reward = TrajectoryReward(self.geometry.reward_center, **reward_kwargs)
        else:
            assert config.trajectory_path is not None
            self.reward = TrajectoryReward.from_file(config.trajectory_path, **reward_kwargs)
        self.evaluation_map = evaluation_map
        self._session = (
            OpenPlanetSessionClient(config.host, config.session_port, timeout_s=config.timeout_s)
            if evaluation_map is not None
            else None
        )
        self._action_count, self._action_table = build_brake_tap_action_table()

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        if self.evaluation_map is not None:
            if self._session is None:
                raise RuntimeError("evaluation map requires an OpenPlanet session channel")
            assert self.geometry is not None
            self.geometry.validate_map(self.evaluation_map.map_path)
            self._session.verify_loaded_map(self.evaluation_map.expected_map_uid)
        frame = self._restart_race()
        if self.config.reset_settle_s:
            sleep(self.config.reset_settle_s)
            frame = self.client.read()
        self.reward.reset(
            frame.values[list(self.config.position_indices)],
            velocity=frame.values[list(self.config.velocity_indices)],
            race_time_ms=float(frame.values[3]),
        )
        self._episode_started_at = monotonic()
        self._last_race_time_ms = float(frame.values[3])
        return frame.values, {"telemetry_health": "ok"}

    def _restart_race(self) -> TelemetryFrame:
        for attempt in range(2):
            previous_race_time_ms = float(self.client.read().values[3])
            self.controller.reset()
            if self.evaluation_map is not None:
                assert self._session is not None
                self._session.confirm_ready(self.evaluation_map.expected_map_uid)
            try:
                return self._wait_for_active_run(previous_race_time_ms)
            except TimeoutError:
                if attempt:
                    raise
        raise AssertionError("unreachable")

    def _wait_for_active_run(self, previous_race_time_ms: float) -> TelemetryFrame:
        deadline = monotonic() + self.config.start_timeout_s
        restart_observed = previous_race_time_ms <= 0.0
        while True:
            frame = self.client.read()
            race_time_ms = float(frame.values[3])
            restart_observed = restart_observed or race_time_ms < previous_race_time_ms
            if restart_observed and race_time_ms > 0.0:
                return frame
            if monotonic() >= deadline:
                raise TimeoutError(
                    "Trackmania did not confirm a new race after reset within "
                    f"{self.config.start_timeout_s:g}s. Check that the virtual gamepad "
                    "restart binding resets the race timer, then restart the loaded map."
                )
            if self.config.start_poll_s:
                sleep(self.config.start_poll_s)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, (int, np.integer)):
            index = int(action)
            if not 0 <= index < self._action_count:
                raise ValueError(f"TrackMania discrete action must be in [0, {self._action_count})")
            control = self._action_table[index]
            discrete_apply = getattr(self.controller, "apply_discrete", None)
            if callable(discrete_apply):
                discrete_apply(control)
            else:
                self.controller.apply(np.where(control == -1.0, 1.0, control))
        else:
            control = np.asarray(action, dtype=np.float32).reshape(-1)
            if control.shape != (3,):
                raise ValueError("TrackMania action must be index or [gas, brake, steer]")
            self.controller.apply(control)
        frame = None
        # Holding a command for several rendered frames prevents an
        # untrained epsilon-greedy policy from flickering throttle and steering
        # every frame, while preserving one replay transition per decision.
        for _ in range(self.config.action_repeat_frames):
            frame = self.client.read()
        assert frame is not None
        position = frame.values[list(self.config.position_indices)]
        collision = self.controller.consume_collision()
        race_time_ms = float(frame.values[3])
        step_race_time_ms = max(0.0, race_time_ms - self._last_race_time_ms)
        self._last_race_time_ms = race_time_ms
        result = self.reward.step(
            position,
            finish_ui_active=bool(frame.values[2]),
            velocity=frame.values[list(self.config.velocity_indices)],
            race_time_ms=race_time_ms,
            collision=collision,
            steering=float(control[2]),
        )
        return (
            frame.values,
            result.reward,
            result.terminated,
            False,
            {
                "termination_reason": result.reason,
                "telemetry_health": "ok",
                "position": position.tolist(),
                "race_time_ms": race_time_ms,
                "step_race_time_ms": step_race_time_ms,
                "control_gas": float(control[0]),
                "control_brake": float(control[1]),
                "control_steer": float(control[2]),
                "episode_elapsed_s": monotonic() - self._episode_started_at,
                "progress_m": self.reward.progress_m,
                "progress_pct": self.reward.progress_pct,
                "reward_time": result.time_reward,
                "reward_pbrs": result.pbrs_reward,
                "reward_progress": result.progress_reward,
                "reward_projected_velocity": result.projected_velocity_reward,
                "reward_steering_delta": result.steering_delta_reward,
                "reward_collision": result.collision_reward,
                "collision": result.collided,
                "collision_detected": result.collision_detected,
                "reward_terminal": result.terminal_reward,
                "potential_progress": result.potential_progress,
                "projected_velocity_mps": result.projected_velocity_mps,
                "projected_velocity_ratio": result.projected_velocity_ratio,
            },
        )

    def close(self) -> None:
        self.controller.close()
        self.client.close()


class OpenPlanetEnvironmentFactory:
    def __init__(
        self,
        config: TrackmaniaEnvironmentConfig | dict[str, Any],
        *,
        controller: Controller | None = None,
        base_dir: str | Path = ".",
    ) -> None:
        parsed = TrackmaniaEnvironmentConfig.model_validate(config)
        trajectory_path = parsed.trajectory_path
        if trajectory_path is not None and not trajectory_path.is_absolute():
            parsed = parsed.model_copy(
                update={"trajectory_path": (Path(base_dir) / trajectory_path).resolve()}
            )
        geometry_path = parsed.geometry_path
        if geometry_path is not None and not geometry_path.is_absolute():
            parsed = parsed.model_copy(
                update={"geometry_path": (Path(base_dir) / geometry_path).resolve()}
            )
        self.config = parsed
        self._controller = controller

    def create(self, *, seed: int, evaluation_map: Any | None = None) -> OpenPlanetEnvironment:
        del seed
        return OpenPlanetEnvironment(
            self.config, self._controller or GamepadController(), evaluation_map=evaluation_map
        )
