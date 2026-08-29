from __future__ import annotations

import argparse
from pathlib import Path

from trackmaniarl.commands.behavior_command import _bc_benchmark, _bc_train
from trackmaniarl.commands.dagger import _dagger_collect
from trackmaniarl.commands.demonstration_benchmark import _demo_benchmark
from trackmaniarl.commands.diagnostics import _diagnose_expert
from trackmaniarl.commands.evaluation import _benchmark
from trackmaniarl.commands.parser_types import CommandParsers
from trackmaniarl.commands.trajectory import (
    _trajectory_optimize,
    _trajectory_stitch,
    _trajectory_synthetic_recovery,
)


def register_trackmania_commands(commands: CommandParsers) -> None:
    _register_benchmark(commands)
    _register_demo_benchmark(commands)
    _register_trajectory_stitch(commands)
    _register_synthetic_recovery(commands)
    _register_trajectory_optimize(commands)
    _register_diagnostics(commands)
    _register_bc_train(commands)
    _register_bc_benchmark(commands)
    _register_dagger(commands)


def _register_benchmark(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "benchmark", help="run the configured Trackmania evaluation release gate"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--target-median", type=float)
    parser.add_argument("--min-finish-rate", type=float)
    parser.set_defaults(handler=_benchmark)


def _register_demo_benchmark(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "demo-benchmark", help="evaluate direct time-synchronised replay of one human demonstration"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("demo", type=Path)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--target-median", type=float)
    parser.add_argument("--min-finish-rate", type=float)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="report a failed replay gate without returning a failing process exit code",
    )
    _add_replay_timing_options(parser)
    _add_replay_mode_options(parser)
    _add_tracker_options(parser)
    parser.set_defaults(phase_locked=False, trajectory_tracking=False, handler=_demo_benchmark)


def _add_replay_timing_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--action-offset-ms",
        type=float,
        default=0.0,
        help="signed open-loop action timestamp offset; positive values delay switching",
    )
    _add_replay_action_lead_options(parser)
    parser.add_argument(
        "--trajectory-schedule",
        type=Path,
        help="optimized schedule produced by trajectory-optimize",
    )


def _add_replay_action_lead_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--action-lead-steps",
        type=int,
        default=1,
        help="expert control look-ahead in native telemetry ticks",
    )
    parser.add_argument(
        "--action-lead-ms",
        type=float,
        help="expert control look-ahead in physical milliseconds; overrides --action-lead-steps",
    )


def _add_replay_mode_options(parser: argparse.ArgumentParser) -> None:
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--open-loop",
        dest="phase_locked",
        action="store_false",
        help="replay strictly by race time (default)",
    )
    modes.add_argument(
        "--phase-locked",
        action="store_true",
        help="enable state matching and steering recovery for diagnostic comparison",
    )
    modes.add_argument(
        "--trajectory-tracking",
        action="store_true",
        help="track raw world-space expert state with feed-forward controls and feedback",
    )


def _add_tracker_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tracker-lateral-gain", type=float, default=0.8)
    parser.add_argument("--tracker-heading-gain", type=float, default=4.0)
    parser.add_argument("--tracker-lateral-velocity-gain", type=float, default=0.03)
    parser.add_argument("--tracker-engage-threshold", type=float, default=0.35)
    parser.add_argument("--tracker-release-threshold", type=float, default=0.15)
    parser.add_argument("--tracker-preview-ms", type=float, default=0.0)
    parser.add_argument("--tracker-minimum-hold-steps", type=int, default=4)
    parser.add_argument("--tracker-reversal-neutral-steps", type=int, default=2)


def _register_trajectory_stitch(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "trajectory-stitch",
        help="splice state-compatible segments from demonstrations with matching time contracts",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory (repeatable)",
    )
    parser.set_defaults(handler=_trajectory_stitch)


def _register_synthetic_recovery(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "trajectory-synthetic-recovery",
        help="generate deterministic counterfactual recovery states around an expert trajectory",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("demo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--action-lead-ms", type=float, default=0.0)
    parser.set_defaults(handler=_trajectory_synthetic_recovery)


def _register_trajectory_optimize(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "trajectory-optimize",
        help="safely optimize expert coast and brake windows on one fixed map",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("demo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-trials", type=int, default=64)
    parser.add_argument("--baseline-trials", type=int, default=3)
    parser.add_argument("--confirmation-trials", type=int, default=2)
    parser.add_argument("--target-time", type=float, default=36.0)
    parser.add_argument("--action-lead-ms", type=float, default=10.0)
    _add_trajectory_search_options(parser)
    parser.set_defaults(handler=_trajectory_optimize)


def _add_trajectory_search_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--shortening-ms",
        type=float,
        nargs="+",
        default=(40.0, 20.0, 10.0),
        metavar="MS",
    )
    parser.add_argument("--minimum-window-ms", type=float, default=30.0)
    parser.add_argument("--minimum-improvement-ms", type=float, default=15.0)


def _register_diagnostics(commands: CommandParsers) -> None:
    diagnose = commands.add_parser("diagnose", help="offline policy diagnostics")
    subcommands = diagnose.add_subparsers(dest="diagnose_command", required=True)
    expert = subcommands.add_parser(
        "expert", help="score resampled demonstrations with the configured IQN action head"
    )
    expert.add_argument("config", type=Path)
    expert.add_argument("checkpoint", type=Path)
    expert.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    expert.set_defaults(handler=_diagnose_expert)


def _register_bc_train(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "bc-train", help="train a compact TrackMania policy from complete demonstrations"
    )
    parser.add_argument("config", type=Path)
    _add_bc_training_data_options(parser)
    parser.set_defaults(handler=_bc_train)


def _add_bc_training_data_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume an exact BC v2 training checkpoint (bc-latest.pt)",
    )
    parser.add_argument(
        "--horizontal-flip-augmentation",
        action="store_true",
        help="add reflected local-frame demonstration laps to behavior-cloning training only",
    )
    _add_bc_recovery_options(parser)


def _add_bc_recovery_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recovery",
        action="append",
        type=Path,
        default=[],
        help="recovery .npz file (repeatable; split into training and validation episodes)",
    )


def _register_bc_benchmark(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "bc-benchmark", help="run closed TrackMania rollouts for a behavior-cloning checkpoint"
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="report a failed rollout gate without returning a failing process exit code",
    )
    parser.add_argument(
        "--minimum-action-hold-steps",
        type=int,
        help="override the BC policy's minimum action duration for this benchmark",
    )
    parser.set_defaults(handler=_bc_benchmark)


def _register_dagger(commands: CommandParsers) -> None:
    parser = commands.add_parser(
        "dagger-collect",
        help="collect student states labelled by a closed-loop trajectory expert",
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("demo", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--teacher-probability", type=float, default=0.15)
    parser.add_argument("--intervention-error", type=float, default=0.8)
    parser.add_argument("--action-lead-ms", type=float, default=0.0)
    parser.set_defaults(handler=_dagger_collect)
