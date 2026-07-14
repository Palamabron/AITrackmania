"""Command-line entrypoint for the current TMRL project workflow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tmrl.core.runtime import resolve_run, validate_resolved_run
from tmrl.core.spec import RunSpec
from tmrl.core.training import Trainer
from tmrl.project.scaffold import create_project
from tmrl.trackmania.assets import record_boundary, record_trajectory
from tmrl.trackmania.geometry import build_geometry_asset
from tmrl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, OpenPlanetClient


def _package_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_").lower()
    if not name or name[0].isdigit():
        raise ValueError("Project package name must start with a letter or underscore")
    return name


def _init(args: argparse.Namespace) -> None:
    package = _package_name(args.package or Path(args.directory).name)
    target = create_project(args.directory, package, template=args.template)
    print(f"Created {target}. Install it with: pip install -e {target}")
    print(f"Then run: tmrl validate {target / 'run.yaml'}")


def _validate(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    print(f"Validated {spec.run_id}: {metrics}")
    print(f"Manifest: {run.run_dir / 'manifest.json'}")


def _train(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    try:
        result = Trainer(run, resume_checkpoint=getattr(args, "checkpoint", None)).train()
    finally:
        run.logger.close()
    print(
        f"Finished {spec.run_id}: {result.transitions} transitions, "
        f"{result.updates} updates, {result.episodes} episodes"
    )
    print(f"Artifacts: {run.run_dir}")


def _smoke(args: argparse.Namespace) -> None:
    """Exercise collection, update, checkpoint restore, and evaluation against the game."""

    spec = RunSpec.from_yaml(args.config)
    transitions = args.transitions
    if transitions < 8:
        raise ValueError("smoke testing requires at least 8 transitions")
    batch_size = min(spec.training.batch_size, max(2, transitions // 4))
    n_step = min(spec.training.n_step, transitions - batch_size + 1)
    ready = batch_size * spec.training.sequence_length + n_step - 1
    if transitions < ready:
        raise ValueError("transitions must cover one complete replay batch")
    training = spec.training.model_copy(
        update={
            "total_transitions": transitions,
            "max_episode_steps": min(spec.training.max_episode_steps, transitions),
            "batch_size": batch_size,
            "n_step": n_step,
            "warmup_transitions": ready,
            "updates_per_transition": 1.0,
            "checkpoint_interval_updates": 1,
        }
    )
    smoke_spec = spec.model_copy(update={"run_id": f"{spec.run_id}-smoke", "training": training})
    base_dir = Path(args.config).parent
    run = resolve_run(smoke_spec, base_dir=base_dir)
    try:
        result = Trainer(run).train()
    finally:
        run.logger.close()
    if result.updates < 1 or not result.checkpoints:
        raise RuntimeError("smoke run did not produce an update and checkpoint")
    resumed = resolve_run(smoke_spec, base_dir=base_dir)
    try:
        Trainer(resumed, resume_checkpoint=result.checkpoints[-1]).train()
    finally:
        resumed.logger.close()
    print(
        f"Smoke test passed: {result.transitions} transitions, {result.updates} updates, "
        f"checkpoint restored from {result.checkpoints[-1]}"
    )


def _record_trajectory(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(
        args.host, args.port, field_count=args.field_count, timeout_s=args.timeout
    )
    try:
        path = record_trajectory(args.output, client, samples=args.samples)
    finally:
        client.close()
    print(f"Recorded trajectory: {path}")


def _record_boundary(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(
        args.host, args.port, field_count=args.field_count, timeout_s=args.timeout
    )
    try:
        path = record_boundary(args.output, client, samples=args.samples)
    finally:
        client.close()
    print(f"Recorded {args.side} boundary: {path}")


def _build_geometry(args: argparse.Namespace) -> None:
    path = build_geometry_asset(
        args.output,
        args.left,
        args.right,
        map_uid=args.map_uid,
        map_path=args.map_path,
        spacing_m=args.spacing,
    )
    print(f"Built geometry asset: {path}")


def _benchmark(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    if (
        spec.evaluation is None
        or spec.evaluation.trials_per_map != 20
        or len(spec.evaluation.maps) != 1
        or spec.evaluation.maps[0].id != "test-3"
    ):
        raise ValueError("benchmark requires exactly one test-3 map with exactly 20 trials")
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    if run.evaluator is None:
        raise ValueError("benchmark requires components.evaluator")
    try:
        run.learner.setup(
            {"seed": spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
        )
        checkpoint = run.checkpoint_codec.load(args.checkpoint)
        learner_state = checkpoint.get("learner", checkpoint)
        run.learner.load_state_dict(learner_state)
        set_checkpoint = getattr(run.evaluator, "set_checkpoint", None)
        if callable(set_checkpoint):
            set_checkpoint(args.checkpoint)
        metrics = dict(run.evaluator.evaluate(run.learner.policy()))
        artifact = json.loads((run.run_dir / "evaluation.json").read_text(encoding="utf-8"))
    finally:
        run.logger.close()
    trials = artifact["trials"]
    if artifact.get("checkpoint") != str(args.checkpoint):
        raise RuntimeError("benchmark artifact checkpoint does not match the evaluated checkpoint")
    if len(trials) != 20 or any(trial["map_id"] != "test-3" for trial in trials):
        raise RuntimeError("benchmark artifact must contain exactly 20 test-3 trials")
    completed = [trial for trial in trials if trial["finished"]]
    telemetry_or_controller_errors = [
        trial
        for trial in trials
        if trial["telemetry_error"] is not None or trial["controller_error"] is not None
    ]
    passed = (
        len(completed) >= 18
        and metrics["eval/median_finish_time_s"] < 37.0
        and not telemetry_or_controller_errors
    )
    if not passed:
        raise RuntimeError(
            "benchmark failed: require >=18/20 finishes, median completed time <37.0s, "
            "and no telemetry/controller errors"
        )
    print(
        f"Benchmark passed: {len(completed)}/20 finishes, median "
        f"{metrics['eval/median_finish_time_s']:.3f}s"
    )


def entrypoint(argv: list[str] | None = None) -> None:
    """Parse cross-platform TMRL SDK commands."""

    parser = argparse.ArgumentParser(prog="tmrl", description="TMRL project tooling")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create an installable local extension project")
    init.add_argument("directory")
    init.add_argument("--package", help="Python package name (defaults to directory name)")
    init.add_argument("--template", choices=("starter", "trackmania"), default="starter")
    init.set_defaults(handler=_init)
    validate = commands.add_parser(
        "validate", help="resolve components and run a synthetic smoke update"
    )
    validate.add_argument("config", type=Path)
    validate.set_defaults(handler=_validate)
    train = commands.add_parser(
        "train", help="collect TrackMania episodes and train a resolved run"
    )
    train.add_argument("config", type=Path)
    train.set_defaults(handler=_train)
    resume = commands.add_parser("resume", help="resume a run from a full training checkpoint")
    resume.add_argument("config", type=Path)
    resume.add_argument("checkpoint", type=Path)
    resume.set_defaults(handler=_train)
    smoke = commands.add_parser(
        "smoke", help="run the bounded live TrackMania release gate and verify checkpoint restore"
    )
    smoke.add_argument("config", type=Path)
    smoke.add_argument("--transitions", type=int, default=100)
    smoke.set_defaults(handler=_smoke)
    benchmark = commands.add_parser("benchmark", help="run the fixed test-3 20-trial release gate")
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("checkpoint", type=Path)
    benchmark.set_defaults(handler=_benchmark)
    track = commands.add_parser("track", help="TrackMania asset tools")
    track_commands = track.add_subparsers(dest="track_command", required=True)
    record = track_commands.add_parser(
        "record-trajectory", help="record XYZ points from OpenPlanet"
    )
    record.add_argument("output", type=Path)
    record.add_argument("--samples", type=int, default=2_000)
    record.add_argument("--host", default="127.0.0.1")
    record.add_argument("--port", type=int, default=9000)
    record.add_argument("--field-count", type=int, default=DEFAULT_TELEMETRY_FIELD_COUNT)
    record.add_argument("--timeout", type=float, default=10.0)
    record.set_defaults(handler=_record_trajectory)
    boundary = track_commands.add_parser(
        "record-boundary", help="record a manually driven left or right boundary"
    )
    boundary.add_argument("side", choices=("left", "right"))
    boundary.add_argument("output", type=Path)
    boundary.add_argument("--samples", type=int, default=2_000)
    boundary.add_argument("--host", default="127.0.0.1")
    boundary.add_argument("--port", type=int, default=9000)
    boundary.add_argument("--field-count", type=int, default=DEFAULT_TELEMETRY_FIELD_COUNT)
    boundary.add_argument("--timeout", type=float, default=10.0)
    boundary.set_defaults(handler=_record_boundary)
    geometry = track_commands.add_parser(
        "build-geometry", help="build a versioned lidar geometry .npz from two boundaries"
    )
    geometry.add_argument("output", type=Path)
    geometry.add_argument("--left", type=Path, required=True)
    geometry.add_argument("--right", type=Path, required=True)
    geometry.add_argument("--map-uid", required=True)
    geometry.add_argument("--map-path", type=Path, required=True)
    geometry.add_argument("--spacing", type=float, default=2.0)
    geometry.set_defaults(handler=_build_geometry)
    args = parser.parse_args(argv)
    args.handler(args)
