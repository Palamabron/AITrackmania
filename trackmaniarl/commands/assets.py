"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trackmaniarl.core.spec import RunSpec
from trackmaniarl.trackmania.assets import (
    BoundaryRecordingRequest,
    TrajectoryRecordingRequest,
    record_boundary,
    record_trajectory,
)
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    DemonstrationSessionConfig,
    DemonstrationSessionRequest,
    record_demonstration_session,
    reject_outliers,
    save_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.geometry_types import GeometryBuildRequest
from trackmaniarl.trackmania.session import OpenPlanetSessionClient
from trackmaniarl.trackmania.telemetry import (
    OpenPlanetClient,
    OpenPlanetClientConfig,
)


def _record_trajectory(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(OpenPlanetClientConfig(args.host, args.port, args.timeout))
    try:
        request = TrajectoryRecordingRequest(args.output, client, args.samples, args.interval)
        path = record_trajectory(request)
    finally:
        client.close()
    print(f"Recorded trajectory: {path}")


def _record_boundary(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(OpenPlanetClientConfig(args.host, args.port, args.timeout))
    try:
        request = BoundaryRecordingRequest(
            args.output, client, args.max_duration, args.minimum_spacing, status=print
        )
        path = record_boundary(request)
    finally:
        client.close()
    print(f"Recorded {args.side} boundary: {path}")


def _trackmania_factory(config_path: Path) -> OpenPlanetEnvironmentFactory:
    spec = RunSpec.from_yaml(config_path)
    component = spec.components.environment
    expected = "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
    if component is None or component.class_path != expected:
        raise ValueError("command requires the first-party TrackMania environment")
    return OpenPlanetEnvironmentFactory(**component.kwargs, base_dir=config_path.parent)


def _record_demo(args: argparse.Namespace) -> None:
    _validate_recording_arguments(args)
    config, geometry = _recording_components(args.config.resolve(), args.start_timeout)
    demonstrations = _record_session(_RecordingSession(config, geometry, args))
    for demonstration in demonstrations:
        validate_recording_quality(demonstration)
    _save_session_demonstrations(args.output, demonstrations, args.max_gap)


def _validate_recording_arguments(args: argparse.Namespace) -> None:
    if args.start_timeout <= 0.0:
        raise ValueError("start timeout must be positive")
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.max_gap < 0.0:
        raise ValueError("max gap must be non-negative")
    if args.sampling_interval_ms < 0.0:
        raise ValueError("sampling interval must be non-negative")


def _recording_components(
    config_path: Path, start_timeout: float
) -> tuple[TrackmaniaEnvironmentConfig, BoundaryGeometry]:
    factory = _trackmania_factory(config_path)
    config = factory.config.model_copy(update={"start_timeout_s": start_timeout})
    if config.expected_map_uid is None:
        raise ValueError("record-demo requires expected_map_uid")
    geometry = BoundaryGeometry(config.geometry_path, expected_map_uid=config.expected_map_uid)
    print(
        "Recorder contract: native sequential telemetry, frame-start controls, "
        "strict 100 Hz quality gate"
    )
    return config, geometry


@dataclass(frozen=True, slots=True)
class _RecordingSession:
    config: TrackmaniaEnvironmentConfig
    geometry: BoundaryGeometry
    args: argparse.Namespace


def _record_session(session: _RecordingSession) -> list[Demonstration]:
    config = session.config
    expected_map_uid = config.expected_map_uid
    if expected_map_uid is None:
        raise RuntimeError("recording session has no validated map UID")
    control = OpenPlanetSessionClient(config.host, config.session_port, timeout_s=config.timeout_s)
    control.verify_loaded_map(expected_map_uid)
    client = OpenPlanetClient(OpenPlanetClientConfig(config.host, config.port, config.timeout_s))
    try:
        return _record_demonstrations(session, client)
    finally:
        client.close()


def _record_demonstrations(
    session: _RecordingSession, client: OpenPlanetClient
) -> list[Demonstration]:
    return record_demonstration_session(
        DemonstrationSessionRequest(
            client,
            session.config,
            session.geometry,
            DemonstrationSessionConfig(
                session.args.count,
                session.args.max_duration,
                session.args.sampling_interval_ms,
                print,
            ),
        )
    )


def _save_session_demonstrations(
    output: Path, demonstrations: list[Demonstration], max_gap_s: float
) -> None:
    kept = reject_outliers(demonstrations, max_gap_s=max_gap_s)
    for rank, demonstration in enumerate(kept, start=1):
        path = save_demonstration(
            output / f"demo-{rank:02d}-{demonstration.finish_time_s:.3f}s", demonstration
        )
        print(f"Saved demonstration: {path} ({len(demonstration.actions)} transitions)")
    discarded = len(demonstrations) - len(kept)
    if discarded:
        best = min(demonstration.finish_time_s for demonstration in demonstrations)
        print(
            f"Discarded {discarded} outlier "
            f"{'lap' if discarded == 1 else 'laps'} "
            f"(slower than {best + max_gap_s:.3f}s)."
        )


def _build_geometry(args: argparse.Namespace) -> None:
    request = GeometryBuildRequest(
        args.output,
        args.left,
        args.right,
        args.map_uid,
        args.map_path,
        args.spacing,
        args.smooth_window,
        args.lookahead_points,
    )
    path = build_geometry_asset(request)
    print(f"Built geometry asset: {path}")


@dataclass(frozen=True, slots=True)
class _TrackCheck:
    frames: list[Any]
    active_map: Any


def _check_track_connection(args: argparse.Namespace) -> None:
    try:
        result = _inspect_track(args)
    except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as error:
        print(f"Trackmania/Openplanet check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    _print_track_check(result)


def _inspect_track(args: argparse.Namespace) -> _TrackCheck:
    client = OpenPlanetClient(
        OpenPlanetClientConfig(
            args.host,
            args.port,
            args.timeout,
        )
    )
    try:
        frames = [client.read() for _ in range(3)]
    finally:
        client.close()
    session = OpenPlanetSessionClient(args.host, args.session_port, timeout_s=args.timeout)
    active = session.inspect_loaded_map()
    expected_map_uid = _expected_map_uid(args, active.map_uid)
    session.confirm_ready(expected_map_uid)
    return _TrackCheck(frames, active)


def _expected_map_uid(args: argparse.Namespace, active_map_uid: str) -> str:
    if args.config is None:
        return active_map_uid
    configured = _trackmania_factory(args.config.resolve()).config.expected_map_uid
    expected = (configured or "").strip()
    placeholder = expected.startswith("REPLACE_") or (
        expected.startswith("<") and expected.endswith(">")
    )
    if not expected or placeholder:
        raise ValueError(
            "config still contains an expected_map_uid placeholder; "
            f"active map UID is {active_map_uid!r}"
        )
    if active_map_uid != expected:
        raise ValueError(
            f"active map UID does not match config: expected {expected!r}, got {active_map_uid!r}"
        )
    return expected


def _print_track_check(result: _TrackCheck) -> None:
    frame = result.frames[-1]
    position = frame.values[4:7].tolist()
    finished = bool(frame.values[2])
    race_time = float(frame.values[3])
    active = result.active_map
    print(
        "Trackmania/Openplanet OK: telemetry_schema=33 float32; frames=3; "
        f"session_protocol={active.protocol_version}; map_uid={active.map_uid!r}; ready=true; "
        f"position={position}; finished={finished}; race_time_ms={race_time}"
    )
