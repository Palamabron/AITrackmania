from __future__ import annotations

import argparse
from pathlib import Path

from trackmaniarl.commands.assets import (
    _build_geometry,
    _check_track_connection,
    _record_boundary,
    _record_demo,
    _record_trajectory,
)
from trackmaniarl.commands.parser_types import CommandParsers


def register_asset_commands(commands: CommandParsers) -> None:
    track = commands.add_parser("track", help="TrackMania asset tools")
    track_commands = track.add_subparsers(dest="track_command", required=True)
    _register_trajectory_recorder(track_commands)
    _register_demo_recorder(track_commands)
    _register_boundary_recorder(track_commands)
    _register_geometry_builder(track_commands)
    _register_track_check(track_commands)


def _add_telemetry_connection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=10.0)


def _register_trajectory_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser("record-trajectory", help="record XYZ points from OpenPlanet")
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--interval", type=float, default=1 / 30)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=_record_trajectory)


def _register_demo_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-demo", help="record finished human laps and drop outliers for replay seeding"
    )
    parser.add_argument("output", type=Path, help="directory that receives the kept .npz laps")
    parser.add_argument("--config", type=Path, required=True)
    _add_demo_recording_options(parser)
    parser.set_defaults(handler=_record_demo)


def _add_demo_recording_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1, help="laps to record in one session")
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="discard laps slower than the best finish by more than this many seconds",
    )
    parser.add_argument("--start-timeout", type=float, default=120.0)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument(
        "--sampling-interval-ms",
        type=float,
        default=0.0,
        help="physical sampling interval; 0 records every new telemetry frame",
    )


def _register_boundary_recorder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "record-boundary", help="record a manually driven left or right boundary"
    )
    parser.add_argument("side", choices=("left", "right"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-duration", type=float, default=300.0)
    parser.add_argument("--minimum-spacing", type=float, default=0.25)
    _add_telemetry_connection(parser)
    parser.set_defaults(handler=_record_boundary)


def _register_geometry_builder(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "build-geometry", help="build a versioned lidar geometry .npz from two boundaries"
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--map-uid", required=True)
    parser.add_argument("--map-path", type=Path, required=True)
    _add_geometry_options(parser)
    parser.set_defaults(handler=_build_geometry)


def _add_geometry_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spacing", type=float, default=2.0)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="odd moving-average window over resampled points (1 disables)",
    )
    parser.add_argument(
        "--lookahead-points",
        type=int,
        default=60,
        help="virtual points past the finish on open tracks so lidar look-ahead stays fresh",
    )


def _register_track_check(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "check", help="verify Openplanet telemetry, protocol, active map, and readiness"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--session-port", type=int, default=9001)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument(
        "--config",
        type=Path,
        help="also require the active map UID to match a first-party run.yaml",
    )
    parser.set_defaults(handler=_check_track_connection)
