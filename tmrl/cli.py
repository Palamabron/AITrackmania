"""Command-line entrypoint for the current TMRL project workflow."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import os
import re
import secrets
import signal
import sys
from math import ceil
from pathlib import Path
from time import sleep, time_ns
from typing import Any

from tmrl.core.runtime import resolve_run, validate_resolved_run
from tmrl.core.spec import RunSpec
from tmrl.project.scaffold import create_project
from tmrl.trackmania.assets import record_boundary, record_trajectory
from tmrl.trackmania.demonstrations import (
    Demonstration,
    record_demonstration_session,
    reject_outliers,
    resolve_demonstration_paths,
    save_demonstration,
)
from tmrl.trackmania.environment import OpenPlanetEnvironmentFactory
from tmrl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from tmrl.trackmania.session import OpenPlanetSessionClient
from tmrl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, OpenPlanetClient


def _configure_process_logging() -> None:
    """Send library INFO logs (progress, episodes, demos) to the console."""

    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("tmrl").setLevel(logging.INFO)


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
    # Validation writes a synthetic checkpoint and must never reserve the
    # artifact directory intended for the real live-training run.
    validation_spec = spec.model_copy(update={"run_id": f"{spec.run_id}-validate-{time_ns()}"})
    run = resolve_run(validation_spec, base_dir=Path(args.config).parent)
    try:
        metrics = validate_resolved_run(run)
    finally:
        run.logger.close()
    print(f"Validated {spec.run_id}: {metrics}")
    print(f"Manifest: {run.run_dir / 'manifest.json'}")


def _load_env_value(config: Path, name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    dotenv = config.resolve().parent / ".env"
    if not dotenv.exists():
        return None
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        key, separator, raw_value = raw_line.partition("=")
        if separator and key.strip() == name:
            return raw_value.strip().strip("'\"") or None
    return None


def _required_token(config: Path) -> str:
    spec = RunSpec.from_yaml(config)
    token = _load_env_value(config, spec.distributed.token_env)
    if not token:
        raise ValueError(
            f"Set {spec.distributed.token_env} in the environment or {config.parent / '.env'}"
        )
    return token


def _spawn_executable() -> str:
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if not virtual_env:
        return sys.executable
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    candidate = Path(virtual_env) / scripts_dir / executable_name
    return str(candidate) if candidate.is_file() else sys.executable


def _spawn_context() -> multiprocessing.context.SpawnContext:
    """Start Windows children with the active virtual environment's interpreter."""

    multiprocessing.set_executable(_spawn_executable())
    return multiprocessing.get_context("spawn")


def _next_versioned_run_id(run_id: str, artifacts_dir: Path) -> str:
    """Return the first free local run identifier for a new training attempt."""

    match = re.fullmatch(r"(?P<base>.+-v\d+)(?P<suffix>[a-z]*)", run_id)
    if match is None:
        index = 1
        while (artifacts_dir / f"{run_id}-{index}").exists():
            index += 1
        return f"{run_id}-{index}"
    base = match.group("base")
    suffix = match.group("suffix")
    index = _alphabetic_suffix_index(suffix) + 1
    while (artifacts_dir / f"{base}{_alphabetic_suffix(index)}").exists():
        index += 1
    return f"{base}{_alphabetic_suffix(index)}"


def _alphabetic_suffix(index: int) -> str:
    """Format a one-based alphabetic sequence number."""

    value = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        value = chr(ord("a") + remainder) + value
    return value


def _alphabetic_suffix_index(value: str) -> int:
    """Parse a possibly empty alphabetic sequence suffix."""

    index = 0
    for character in value:
        index = index * 26 + ord(character) - ord("a") + 1
    return index


def _new_attempt_spec(config: Path, spec: RunSpec, args: argparse.Namespace) -> RunSpec:
    """Assign a distinct run ID when a fresh local attempt would reuse artifacts."""

    fresh_attempt = bool(getattr(args, "reset_replay", False)) or not bool(
        getattr(args, "checkpoint", None)
    )
    artifacts_dir = config.parent / spec.artifacts_dir
    if not fresh_attempt or not (artifacts_dir / spec.run_id).exists():
        return spec
    run_id = _next_versioned_run_id(spec.run_id, artifacts_dir)
    print(f"Run ID {spec.run_id!r} already exists; using {run_id!r} for this new attempt.")
    return spec.model_copy(update={"run_id": run_id})


def _resumed_attempt_spec(config: Path, spec: RunSpec, args: argparse.Namespace) -> RunSpec:
    """Recover an auto-assigned sibling run ID from a local resume checkpoint."""

    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is None or bool(getattr(args, "reset_replay", False)):
        return spec
    path = Path(checkpoint).resolve()
    artifacts_dir = (config.parent / spec.artifacts_dir).resolve()
    run_dir = path.parent.parent
    if path.parent.name != "checkpoints" or run_dir.parent != artifacts_dir:
        return spec
    configured = re.fullmatch(r"(?P<base>.+-v\d+)(?P<suffix>[a-z]*)", spec.run_id)
    resumed = re.fullmatch(r"(?P<base>.+-v\d+)(?P<suffix>[a-z]*)", run_dir.name)
    if configured is None or resumed is None or configured.group("base") != resumed.group("base"):
        return spec
    if run_dir.name != spec.run_id:
        print(f"Resuming checkpoint run ID {run_dir.name!r}.")
    return spec.model_copy(update={"run_id": run_dir.name})


def _train(args: argparse.Namespace) -> None:
    """Launch a spawn-safe local learner and actor pair."""

    source_config = args.config.resolve()
    source_spec = RunSpec.from_yaml(source_config)
    spec = _resumed_attempt_spec(source_config, source_spec, args)
    spec = _new_attempt_spec(source_config, spec, args)
    temporary_config: Path | None = None
    config = source_config
    if spec.run_id != source_spec.run_id:
        temporary_config = source_config.with_name(f".tmrl-{spec.run_id}-{time_ns()}.yaml")
        temporary_config.write_text(spec.to_yaml(), encoding="utf-8")
        config = temporary_config
    token = secrets.token_urlsafe(32)
    target = f"127.0.0.1:{spec.distributed.port}"
    bind = target
    context = _spawn_context()
    shutdown = context.Event()
    learner = context.Process(
        target=_learner_process,
        args=(
            str(config),
            bind,
            token,
            str(args.checkpoint) if getattr(args, "checkpoint", None) else None,
            bool(getattr(args, "reset_replay", False)),
            shutdown,
            tuple(str(path) for path in resolve_demonstration_paths(getattr(args, "demo", ()))),
        ),
        name="tmrl-learner",
    )
    actor = context.Process(
        target=_actor_process,
        args=(str(config), target, "local-actor", token, shutdown),
        name="tmrl-local-actor",
    )
    learner.start()
    actor.start()
    print("Local async training launched:", flush=True)
    print(
        f"  learner_pid={learner.pid}  gradient updates, replay, checkpoints",
        flush=True,
    )
    print(
        f"  actor_pid={actor.pid}      TrackMania rollouts -> learner",
        flush=True,
    )
    print(
        f"  endpoint={target}  gRPC; actor connects here",
        flush=True,
    )
    stopped_by_user = False
    actor_finished_first = False
    try:
        while learner.is_alive() and actor.is_alive():
            sleep(0.25)
        actor_finished_first = not actor.is_alive() and learner.is_alive()
        if actor_finished_first:
            if actor.exitcode == 0:
                print(
                    f"Actor process (pid={actor.pid}) completed rollout collection; "
                    f"waiting for learner (pid={learner.pid}) to drain update credit...",
                    flush=True,
                )
                while learner.is_alive():
                    sleep(0.25)
            else:
                print(
                    f"Actor process (pid={actor.pid}) exited first with code={actor.exitcode}; "
                    f"stopping learner (pid={learner.pid}) gracefully...",
                    flush=True,
                )
    except KeyboardInterrupt:
        stopped_by_user = True
        print("Stopping async training; saving the learner checkpoint...", flush=True)
    finally:
        shutdown.set()
        learner.join(timeout=10)
        actor.join(timeout=10)
        for process in (actor, learner):
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        if temporary_config is not None:
            temporary_config.unlink(missing_ok=True)
    if stopped_by_user:
        return
    failures = [
        f"{name} process exited with code {process.exitcode}"
        for name, process in (("actor", actor), ("learner", learner))
        if process.exitcode not in (0, None)
    ]
    if failures:
        raise RuntimeError("; ".join(failures))
    print(f"Finished async run {spec.run_id}. Artifacts: {config.parent / spec.artifacts_dir}")


def _learner(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    token = _required_token(config)
    spec = RunSpec.from_yaml(config)
    _learner_process(
        str(config),
        args.bind or f"127.0.0.1:{spec.distributed.port}",
        token,
        str(args.checkpoint) if args.checkpoint else None,
        demo_paths=tuple(str(path) for path in resolve_demonstration_paths(args.demo)),
    )


def _actor(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    _actor_process(
        str(config),
        args.connect,
        args.actor_id,
        _required_token(config),
    )


def _learner_process(
    config_path: str,
    bind: str,
    token: str,
    resume_checkpoint: str | None = None,
    reset_replay: bool = False,
    external_stop: Any | None = None,
    demo_paths: tuple[str, ...] = (),
) -> None:
    _configure_process_logging()
    if external_stop is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from tmrl.distributed.coordinator import learner_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install tmrl[distributed] to use distributed training") from exc
    learner_process_entry(
        config_path,
        bind,
        token,
        resume_checkpoint,
        reset_replay,
        external_stop,
        demo_paths,
    )


def _actor_process(
    config_path: str,
    target: str,
    actor_id: str | None,
    token: str,
    external_stop: Any | None = None,
) -> None:
    _configure_process_logging()
    if external_stop is not None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        from tmrl.distributed.actor import actor_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install tmrl[distributed] to use distributed training") from exc
    actor_process_entry(config_path, target, actor_id, token, external_stop)


def _smoke(args: argparse.Namespace) -> None:
    """Run a bounded local async actor/learner release check against the game."""

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
            "checkpoint_interval_updates": 25,
        }
    )
    # A release smoke test proves that live collection, replay, an update, and
    # checkpoint restore work.  It must not launch the configured 20-trial
    # benchmark: a freshly initialized exploratory policy is intentionally not
    # suitable for that evaluation and could hold the game for a long time.
    smoke_components = spec.components.model_copy(update={"evaluator": None})
    smoke_spec = spec.model_copy(
        update={
            # Manifests are immutable, so a failed or interrupted smoke run
            # must never prevent a later retry from creating its own artifact.
            "run_id": f"{spec.run_id}-smoke-{time_ns()}",
            "training": training,
            "components": smoke_components,
            "evaluation": None,
        }
    )
    smoke_spec = smoke_spec.model_copy(
        update={"distributed": smoke_spec.distributed.model_copy(update={"policy_refresh_s": 0.25})}
    )
    base_dir = Path(args.config).resolve().parent
    temporary = base_dir / f".tmrl-{smoke_spec.run_id}.yaml"
    temporary.write_text(smoke_spec.to_yaml(), encoding="utf-8")
    try:
        _train(argparse.Namespace(config=temporary, checkpoint=None))
        _restore_smoke_checkpoint(temporary, smoke_spec)
    finally:
        temporary.unlink(missing_ok=True)
    events_path = base_dir / smoke_spec.artifacts_dir / smoke_spec.run_id / "events.jsonl"
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refreshed = any(
        event.get("event") == "distributed/policy_published"
        and int(event.get("payload", {}).get("policy_version", 0)) > 0
        for event in events
    )
    if not refreshed:
        raise RuntimeError("async smoke completed without refreshing the actor policy")
    print("Async TrackMania smoke passed with a live policy-refresh interval of 0.25s.")


def _restore_smoke_checkpoint(config: Path, spec: RunSpec) -> None:
    from tmrl.distributed.coordinator import Coordinator

    checkpoint_dir = config.parent / spec.artifacts_dir / spec.run_id / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("distributed-update-*.pt"))
    if not checkpoints:
        raise RuntimeError("async smoke did not produce a distributed checkpoint")
    components = spec.components.model_copy(update={"additional_loggers": ()})
    restore_spec = spec.model_copy(update={"components": components})
    run = resolve_run(restore_spec, base_dir=config.parent)
    coordinator = Coordinator(run, bind="127.0.0.1:8787", token="smoke", fingerprint="smoke")
    try:
        run.learner.setup(
            {
                "seed": restore_spec.seed,
                "run_dir": run.run_dir,
                "model_factory": run.model_factory,
            }
        )
        coordinator.restore_checkpoint(checkpoints[-1])
        if coordinator.counters.updates < 1:
            raise RuntimeError("async smoke checkpoint contains no learner updates")
    finally:
        coordinator._checkpoint_writer.close()
        coordinator.journal.close()
        run.logger.close()


def _record_trajectory(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(
        args.host, args.port, field_count=args.field_count, timeout_s=args.timeout
    )
    try:
        path = record_trajectory(
            args.output, client, samples=args.samples, sample_interval_s=args.interval
        )
    finally:
        client.close()
    print(f"Recorded trajectory: {path}")


def _record_boundary(args: argparse.Namespace) -> None:
    client = OpenPlanetClient(
        args.host, args.port, field_count=args.field_count, timeout_s=args.timeout
    )
    try:
        path = record_boundary(
            args.output,
            client,
            max_duration_s=args.max_duration,
            minimum_spacing_m=args.minimum_spacing,
            status=print,
        )
    finally:
        client.close()
    print(f"Recorded {args.side} boundary: {path}")


def _trackmania_factory(config_path: Path) -> OpenPlanetEnvironmentFactory:
    spec = RunSpec.from_yaml(config_path)
    component = spec.components.environment
    expected = "tmrl.trackmania.environment:OpenPlanetEnvironmentFactory"
    if component is None or component.class_path != expected:
        raise ValueError("record-demo requires the first-party TrackMania environment")
    return OpenPlanetEnvironmentFactory(**component.kwargs, base_dir=config_path.parent)


def _record_demo(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    factory = _trackmania_factory(config_path)
    if args.start_timeout <= 0.0:
        raise ValueError("start timeout must be positive")
    if args.count < 1:
        raise ValueError("count must be positive")
    if args.max_gap < 0.0:
        raise ValueError("max gap must be non-negative")
    config = factory.config.model_copy(update={"start_timeout_s": args.start_timeout})
    if config.geometry_path is None or config.expected_map_uid is None:
        raise ValueError("record-demo requires geometry_path and expected_map_uid")
    geometry = BoundaryGeometry(config.geometry_path, expected_map_uid=config.expected_map_uid)
    session = OpenPlanetSessionClient(config.host, config.session_port, timeout_s=config.timeout_s)
    session.verify_loaded_map(config.expected_map_uid)
    client = OpenPlanetClient(
        config.host, config.port, field_count=config.field_count, timeout_s=config.timeout_s
    )
    try:
        demonstrations = record_demonstration_session(
            client,
            config,
            geometry,
            count=args.count,
            max_duration_s=args.max_duration,
            status=print,
        )
    finally:
        client.close()
    _save_session_demonstrations(args.output, demonstrations, args.max_gap)


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
    path = build_geometry_asset(
        args.output,
        args.left,
        args.right,
        map_uid=args.map_uid,
        map_path=args.map_path,
        spacing_m=args.spacing,
        smooth_window=args.smooth_window,
        lookahead_points=args.lookahead_points,
    )
    print(f"Built geometry asset: {path}")


def _check_track_connection(args: argparse.Namespace) -> None:
    """Verify that the installed OpenPlanet plugin is emitting compatible telemetry."""

    client = OpenPlanetClient(
        args.host, args.port, field_count=args.field_count, timeout_s=args.timeout
    )
    try:
        frame = client.read()
    finally:
        client.close()
    position = frame.values[4:7].tolist() if args.field_count >= 7 else None
    finished = bool(frame.values[2]) if args.field_count >= 3 else "n/a"
    race_time = float(frame.values[3]) if args.field_count >= 4 else "n/a"
    print(
        f"OpenPlanet telemetry OK: {args.field_count} float32 fields; "
        f"position={position}; finished={finished}; race_time_ms={race_time}"
    )


def _apply_benchmark_tau_window(learner: Any, tau_min: float | None, tau_max: float | None) -> None:
    """Override the deployed policy's IQN risk window without a new config file."""

    if tau_min is None and tau_max is None:
        return
    window = (
        tau_min if tau_min is not None else float(getattr(learner, "evaluation_tau_min", 0.0)),
        tau_max if tau_max is not None else float(getattr(learner, "evaluation_tau_max", 1.0)),
    )
    if not 0.0 <= window[0] < window[1] <= 1.0:
        raise ValueError("benchmark tau window must satisfy 0 <= tau-min < tau-max <= 1")
    learner.evaluation_tau_min = window[0]
    learner.evaluation_tau_max = window[1]


def _benchmark(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    evaluation = spec.evaluation
    if evaluation is None or not evaluation.maps:
        raise ValueError("benchmark requires an evaluation suite with at least one map")
    if evaluation.target_median_s is None:
        raise ValueError(
            "benchmark requires evaluation.target_median_s "
            "(for example 37.0 for a sub-37s release gate)"
        )
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    if run.evaluator is None:
        raise ValueError("benchmark requires components.evaluator")
    expected_trials = evaluation.trials_per_map * len(evaluation.maps)
    expected_map_ids = {item.id for item in evaluation.maps}
    try:
        run.learner.setup(
            {"seed": spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
        )
        _apply_benchmark_tau_window(run.learner, args.tau_min, args.tau_max)
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
    if len(trials) != expected_trials or {trial["map_id"] for trial in trials} != expected_map_ids:
        raise RuntimeError(
            f"benchmark artifact must contain exactly {expected_trials} trials covering "
            f"{sorted(expected_map_ids)}"
        )
    completed = [trial for trial in trials if trial["finished"]]
    telemetry_or_controller_errors = [
        trial
        for trial in trials
        if trial["telemetry_error"] is not None or trial["controller_error"] is not None
    ]
    required_finishes = ceil(evaluation.min_finish_rate * expected_trials)
    median = float(metrics["eval/median_finish_time_s"])
    _print_benchmark_report(trials, metrics)
    passed = (
        len(completed) >= required_finishes
        and median < evaluation.target_median_s
        and not telemetry_or_controller_errors
    )
    if not passed:
        raise RuntimeError(
            "benchmark failed: require "
            f">={required_finishes}/{expected_trials} finishes, "
            f"median completed time <{evaluation.target_median_s}s, "
            "and no telemetry/controller errors"
        )
    print(f"Benchmark passed: {len(completed)}/{expected_trials} finishes, median {median:.3f}s")


def _print_benchmark_report(trials: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    """Print every benchmark trial before applying the release gate."""

    completed = [trial for trial in trials if trial["finished"]]
    print("Benchmark trials:")
    for trial in trials:
        finish_time = trial["finish_time_s"]
        time_text = "-" if finish_time is None else f"{float(finish_time):.3f}s"
        print(
            f"  trial={trial['trial_index']} map={trial['map_id']} "
            f"finished={trial['finished']} time={time_text} "
            f"progress={float(trial['progress_pct']):.1f}% "
            f"telemetry_error={trial['telemetry_error'] or '-'} "
            f"controller_error={trial['controller_error'] or '-'}"
        )
    print(
        f"Benchmark summary: finishes={len(completed)}/{len(trials)}, "
        f"mean_completed={float(metrics['eval/finish_time_s']):.3f}s, "
        f"median_completed={float(metrics['eval/median_finish_time_s']):.3f}s"
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
    train = commands.add_parser("train", help="start a local asynchronous learner and actor")
    train.add_argument("config", type=Path)
    train.add_argument(
        "--demo",
        action="append",
        type=Path,
        default=[],
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    train.set_defaults(handler=_train)
    resume = commands.add_parser("resume", help="resume a local asynchronous training run")
    resume.add_argument("config", type=Path)
    resume.add_argument("checkpoint", type=Path)
    resume.add_argument(
        "--reset-replay",
        action="store_true",
        help="restore learner state while starting with an empty replay and sampler",
    )
    resume.add_argument(
        "--demo",
        action="append",
        type=Path,
        default=[],
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    resume.set_defaults(handler=_train)
    learner = commands.add_parser("learner", help="run a distributed coordinator/learner")
    learner.add_argument("config", type=Path)
    learner.add_argument("--bind")
    learner.add_argument("--checkpoint", type=Path)
    learner.add_argument(
        "--demo",
        action="append",
        type=Path,
        default=[],
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    learner.set_defaults(handler=_learner)
    actor = commands.add_parser("actor", help="run a remote continuous rollout actor")
    actor.add_argument("config", type=Path)
    actor.add_argument("--connect", required=True)
    actor.add_argument("--actor-id")
    actor.set_defaults(handler=_actor)
    smoke = commands.add_parser(
        "smoke", help="run a bounded local async TrackMania actor/learner release gate"
    )
    smoke.add_argument("config", type=Path)
    smoke.add_argument("--transitions", type=int, default=100)
    smoke.set_defaults(handler=_smoke)
    benchmark = commands.add_parser("benchmark", help="run the fixed test-3 20-trial release gate")
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("checkpoint", type=Path)
    benchmark.add_argument(
        "--tau-min",
        type=float,
        help="lower IQN evaluation quantile; raise above 0 for a risk-seeking policy",
    )
    benchmark.add_argument(
        "--tau-max",
        type=float,
        help="upper IQN evaluation quantile; lower below 1 for a conservative policy",
    )
    benchmark.set_defaults(handler=_benchmark)
    track = commands.add_parser("track", help="TrackMania asset tools")
    track_commands = track.add_subparsers(dest="track_command", required=True)
    record = track_commands.add_parser(
        "record-trajectory", help="record XYZ points from OpenPlanet"
    )
    record.add_argument("output", type=Path)
    record.add_argument("--samples", type=int, default=2_000)
    record.add_argument("--interval", type=float, default=1 / 30)
    record.add_argument("--host", default="127.0.0.1")
    record.add_argument("--port", type=int, default=9000)
    record.add_argument("--field-count", type=int, default=DEFAULT_TELEMETRY_FIELD_COUNT)
    record.add_argument("--timeout", type=float, default=10.0)
    record.set_defaults(handler=_record_trajectory)
    demo = track_commands.add_parser(
        "record-demo",
        help="record finished human laps and drop outliers for replay seeding",
    )
    demo.add_argument("output", type=Path, help="directory that receives the kept .npz laps")
    demo.add_argument("--config", type=Path, required=True)
    demo.add_argument("--count", type=int, default=1, help="laps to record in one session")
    demo.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="discard laps slower than the best finish by more than this many seconds",
    )
    demo.add_argument("--start-timeout", type=float, default=120.0)
    demo.add_argument("--max-duration", type=float, default=180.0)
    demo.set_defaults(handler=_record_demo)
    boundary = track_commands.add_parser(
        "record-boundary", help="record a manually driven left or right boundary"
    )
    boundary.add_argument("side", choices=("left", "right"))
    boundary.add_argument("output", type=Path)
    boundary.add_argument("--max-duration", type=float, default=300.0)
    boundary.add_argument("--minimum-spacing", type=float, default=0.25)
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
    geometry.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="odd moving-average window over resampled points (1 disables)",
    )
    geometry.add_argument(
        "--lookahead-points",
        type=int,
        default=60,
        help="virtual points past the finish on open tracks so lidar look-ahead stays fresh",
    )
    geometry.set_defaults(handler=_build_geometry)
    check = track_commands.add_parser(
        "check", help="verify that OpenPlanet is emitting one compatible telemetry frame"
    )
    check.add_argument("--host", default="127.0.0.1")
    check.add_argument("--port", type=int, default=9000)
    check.add_argument("--field-count", type=int, default=DEFAULT_TELEMETRY_FIELD_COUNT)
    check.add_argument("--timeout", type=float, default=5.0)
    check.set_defaults(handler=_check_track_connection)
    args = parser.parse_args(argv)
    args.handler(args)
