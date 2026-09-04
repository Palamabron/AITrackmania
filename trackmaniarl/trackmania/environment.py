"""First-party OpenPlanet environment factory for the current SDK runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter, sleep
from typing import Any

import numpy as np

import trackmaniarl.trackmania.environment_step as environment_step
from trackmaniarl.core.contracts import FeaturePipeline
from trackmaniarl.core.data import Transition
from trackmaniarl.trackmania.actions import select_brake_tap_actions
from trackmaniarl.trackmania.control import Controller, GamepadController, KeyboardController
from trackmaniarl.trackmania.environment_config import (
    TrackmaniaEnvironmentConfig as TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry
from trackmaniarl.trackmania.pace import PaceDemonstrationRequest, ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.session import OpenPlanetSessionClient
from trackmaniarl.trackmania.telemetry import (
    OpenPlanetClient,
    OpenPlanetClientConfig,
    TelemetryFrame,
)


def _validated_live_map_uid(expected_map_uid: str | None, geometry: BoundaryGeometry) -> str:
    map_uid = (expected_map_uid or "").strip()
    if not map_uid or _is_placeholder_uid(map_uid):
        raise ValueError(
            "Trackmania live runs require the real expected_map_uid; replace the scaffold "
            "placeholder with the UID reported by `trackmaniarl track check`"
        )
    _validate_geometry_identity(map_uid, geometry)
    return map_uid


def _is_placeholder_uid(map_uid: str) -> bool:
    return map_uid.startswith("REPLACE_") or (map_uid.startswith("<") and map_uid.endswith(">"))


def _validate_geometry_identity(map_uid: str, geometry: BoundaryGeometry) -> None:
    if not geometry.map_uid.strip():
        raise ValueError("geometry asset is missing its source map UID")
    if geometry.map_uid != map_uid:
        raise ValueError("geometry asset map UID does not match the configured expected_map_uid")
    if not geometry.map_sha256.strip():
        raise ValueError(
            "geometry asset is missing its source map checksum; rebuild it with "
            "`trackmaniarl track build-geometry --map-path ...`"
        )


@dataclass(frozen=True, slots=True)
class _MapContext:
    geometry: BoundaryGeometry
    expected_map_uid: str


class OpenPlanetEnvironment:
    def __init__(
        self,
        config: TrackmaniaEnvironmentConfig,
        controller: Controller,
        *,
        evaluation_map: Any | None = None,
    ) -> None:
        self.config, self.controller = config, controller
        self.client = self._create_client(config)
        map_context = self._map_context(config, evaluation_map)
        self.geometry = map_context.geometry
        self._expected_map_uid = map_context.expected_map_uid
        self.reward = self._create_reward(config, self.geometry)
        self.evaluation_map = evaluation_map
        self._session = self._create_session(config)
        self._action_count, self._action_table = select_brake_tap_actions(config.compact_action_ids)
        self._finish_confirmation_pending = True
        self._last_race_time_ms: float | None = None

    @staticmethod
    def _create_client(config: TrackmaniaEnvironmentConfig) -> OpenPlanetClient:
        client = OpenPlanetClientConfig(
            config.host,
            config.port,
            config.timeout_s,
        )
        return OpenPlanetClient(client)

    @staticmethod
    def _create_session(config: TrackmaniaEnvironmentConfig) -> OpenPlanetSessionClient:
        return OpenPlanetSessionClient(config.host, config.session_port, timeout_s=config.timeout_s)

    @staticmethod
    def _map_context(
        config: TrackmaniaEnvironmentConfig, evaluation_map: Any | None
    ) -> _MapContext:
        geometry_path = evaluation_map.geometry_path if evaluation_map else config.geometry_path
        expected_uid = (
            evaluation_map.expected_map_uid if evaluation_map else config.expected_map_uid
        )
        geometry = BoundaryGeometry(geometry_path)
        return _MapContext(geometry, _validated_live_map_uid(expected_uid, geometry))

    @staticmethod
    def _create_reward(
        config: TrackmaniaEnvironmentConfig, geometry: BoundaryGeometry
    ) -> TrajectoryReward:
        reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
        pace_profile = OpenPlanetEnvironment._pace_profile(config, geometry, reference)
        return TrajectoryReward(reference, config.reward_config(pace_profile))

    @staticmethod
    def _pace_profile(
        config: TrackmaniaEnvironmentConfig,
        geometry: BoundaryGeometry,
        reference: np.ndarray,
    ) -> ReferencePaceProfile | None:
        if config.pace_reference_path is None:
            return None
        request = PaceDemonstrationRequest(
            config.pace_reference_path,
            geometry,
            reference,
            config.velocity_to_mps_scale,
        )
        return ReferencePaceProfile.from_demonstration(request)

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        del seed
        if self.evaluation_map is not None:
            self.geometry.validate_map(self.evaluation_map.map_path)
        self._session.verify_loaded_map(self._expected_map_uid)
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
        previous_race_time_ms = self._last_race_time_ms
        for attempt in range(2):
            self._confirm_finish_if_needed()
            if previous_race_time_ms is None:
                previous_race_time_ms = float(self.client.read().values[3])
            self.controller.reset()
            try:
                frame = self._wait_for_active_run(previous_race_time_ms)
            except TimeoutError:
                self._finish_confirmation_pending = True
                self._recover_reset_timeout()
                if attempt:
                    raise
            else:
                self._session.confirm_ready(self._expected_map_uid)
                self._finish_confirmation_pending = False
                return frame
        raise AssertionError("unreachable")

    def _recover_reset_timeout(self) -> None:
        self.client.close()
        self._confirm_finish_if_needed()
        sleep(0.25)
        self.controller.reset()
        sleep(0.25)

    def _confirm_finish_if_needed(self) -> None:
        if not self._finish_confirmation_pending or not self.config.confirm_finish_before_reset:
            return
        self.controller.confirm_finish()

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
                    f"{self.config.start_timeout_s:g}s. Check that the configured "
                    f"{self.config.control_backend} restart input resets the race timer, "
                    "then restart the loaded map."
                )
            if self.config.start_poll_s:
                sleep(self.config.start_poll_s)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return environment_step.step(self, action, perf_counter)

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
        self.config = _resolve_environment_paths(parsed, Path(base_dir))
        self._controller = controller

    def create(self, *, seed: int, evaluation_map: Any | None = None) -> OpenPlanetEnvironment:
        del seed
        controller = self._controller
        if controller is None:
            controller = (
                KeyboardController()
                if self.config.control_backend == "keyboard"
                else GamepadController(restart_input=self.config.restart_input)
            )
        return OpenPlanetEnvironment(self.config, controller, evaluation_map=evaluation_map)

    def load_demonstration(self, path: str | Path, pipeline: FeaturePipeline) -> list[Transition]:
        from trackmaniarl.trackmania.demonstrations import (
            DemonstrationTransitionContext,
            demonstration_transitions,
        )

        geometry = BoundaryGeometry(
            self.config.geometry_path, expected_map_uid=self.config.expected_map_uid
        )
        context = DemonstrationTransitionContext(self.config, geometry)
        return demonstration_transitions(path, pipeline, context)


def _resolve_environment_paths(
    config: TrackmaniaEnvironmentConfig, base_dir: Path
) -> TrackmaniaEnvironmentConfig:
    resolved = config
    for name in ("geometry_path", "pace_reference_path"):
        path = getattr(resolved, name)
        if path is not None and not path.is_absolute():
            resolved = resolved.model_copy(update={name: (base_dir / path).resolve()})
    return resolved
