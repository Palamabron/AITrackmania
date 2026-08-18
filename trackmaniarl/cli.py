"""Command-line entrypoint for the current TrackmaniaRL project workflow."""

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
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, exp
from pathlib import Path
from time import sleep, time_ns
from typing import Any, cast

import numpy as np
import torch

from trackmaniarl.core.contracts import Policy
from trackmaniarl.core.pytree import sanitize_finite, tree_map, tree_to_device
from trackmaniarl.core.runtime import prepare_run, resolve_run, validate_resolved_run
from trackmaniarl.core.spec import RunSpec, TrainingSpec
from trackmaniarl.project.scaffold import create_project
from trackmaniarl.trackmania.actions import (
    continuous_control_to_discrete_index,
    select_brake_tap_actions,
)
from trackmaniarl.trackmania.assets import record_boundary, record_trajectory
from trackmaniarl.trackmania.demonstrations import (
    Demonstration,
    demonstration_timing_summary,
    load_demonstration,
    record_demonstration_session,
    reject_outliers,
    resolve_demonstration_paths,
    save_demonstration,
    validate_demonstration,
    validate_recording_quality,
)
from trackmaniarl.trackmania.diagnostics import ExpertActionDiagnostics, aggregate_expert_bins
from trackmaniarl.trackmania.environment import (
    OpenPlanetEnvironmentFactory,
    TrackmaniaEnvironmentConfig,
)
from trackmaniarl.trackmania.evaluation import TrackmaniaEvaluator
from trackmaniarl.trackmania.features import LidarFeaturePipeline
from trackmaniarl.trackmania.geometry import BoundaryGeometry, build_geometry_asset
from trackmaniarl.trackmania.guidance import (
    DemonstrationReplayPolicy,
    PhaseLockedDemonstrationPolicy,
    TrajectoryTrackingDemonstrationPolicy,
)
from trackmaniarl.trackmania.pace import ReferencePaceProfile
from trackmaniarl.trackmania.reward import TrajectoryReward
from trackmaniarl.trackmania.session import OpenPlanetSessionClient
from trackmaniarl.trackmania.synthetic_recovery import (
    SyntheticRecoveryConfig,
    generate_synthetic_recovery_from_path,
)
from trackmaniarl.trackmania.telemetry import DEFAULT_TELEMETRY_FIELD_COUNT, OpenPlanetClient
from trackmaniarl.trackmania.trajectory_optimization import (
    SafeTrajectoryOptimizer,
    TrajectorySchedule,
    TrajectorySearchConfig,
    TrajectorySearchOutcome,
    TrajectoryTrackerConfig,
    build_scheduled_policy,
    run_trajectory_trial,
)
from trackmaniarl.trackmania.trajectory_stitching import (
    build_fastest_compatible_trajectory,
)


def _configure_process_logging() -> None:
    """Send library INFO logs (progress, episodes, demos) to the console."""

    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("trackmaniarl").setLevel(logging.INFO)


def _package_name(value: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_").lower()
    if not name or name[0].isdigit():
        raise ValueError("Project package name must start with a letter or underscore")
    return name


def _init(args: argparse.Namespace) -> None:
    package = _package_name(args.package or Path(args.directory).name)
    target = create_project(args.directory, package, template=args.template)
    print(f"Created {target}. Install it with: uv sync --directory {target}")
    print(f"Then run: trackmaniarl validate {target / 'run.yaml'}")


def _validate(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    # Validation writes a synthetic checkpoint and must never reserve the
    # artifact directory intended for the real live-training run.
    components = spec.components.model_copy(update={"additional_loggers": ()})
    validation_spec = spec.model_copy(
        update={"components": components, "run_id": f"{spec.run_id}-validate-{time_ns()}"}
    )
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
    configured_spec = RunSpec.from_yaml(source_config)
    initialization = getattr(args, "model_initialization_checkpoint", None)
    source_spec = (
        _with_model_initialization_checkpoint(configured_spec, initialization.resolve())
        if initialization is not None
        else configured_spec
    )
    spec = _resumed_attempt_spec(source_config, source_spec, args)
    spec = _new_attempt_spec(source_config, spec, args)
    temporary_config: Path | None = None
    config = source_config
    if spec.run_id != configured_spec.run_id or initialization is not None:
        temporary_config = source_config.with_name(f".trackmaniarl-{spec.run_id}-{time_ns()}.yaml")
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
        name="trackmaniarl-learner",
    )
    actor = context.Process(
        target=_actor_process,
        args=(str(config), target, "local-actor", token, shutdown),
        name="trackmaniarl-local-actor",
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
        _signal_shutdown(shutdown, learner, actor)
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


def _with_model_initialization_checkpoint(spec: RunSpec, checkpoint: Path) -> RunSpec:
    learner = spec.components.learner
    kwargs = dict(learner.kwargs)
    kwargs["model_initialization_checkpoint"] = str(checkpoint)
    learner = learner.model_copy(update={"kwargs": kwargs})
    components = spec.components.model_copy(update={"learner": learner})
    return spec.model_copy(update={"components": components})


def _offline_pretrain(args: argparse.Namespace) -> None:
    """Run configured demonstration-only IQN/DQfD updates in this process."""

    config = args.config.resolve()
    spec = RunSpec.from_yaml(config)
    initialization = args.model_initialization_checkpoint
    if initialization is not None:
        spec = _with_model_initialization_checkpoint(spec, initialization.resolve())
    spec = _new_attempt_spec(config, spec, args)
    if spec.training.offline_pretrain_updates == 0:
        raise ValueError("offline-pretrain requires training.offline_pretrain_updates > 0")
    demonstrations = tuple(resolve_demonstration_paths(args.demo))
    from trackmaniarl.distributed.coordinator import Coordinator
    from trackmaniarl.distributed.protocol import run_fingerprint

    run = resolve_run(spec, base_dir=config.parent)
    try:
        result = Coordinator(
            run,
            bind=f"127.0.0.1:{spec.distributed.port}",
            token=secrets.token_urlsafe(32),
            fingerprint=run_fingerprint(spec, config.parent),
            demo_paths=demonstrations,
        ).run_offline_pretraining()
    finally:
        run.logger.close()
    checkpoint = result.checkpoints[-1]
    print(f"Offline pretraining complete: updates={result.updates}, checkpoint={checkpoint}")


def _signal_shutdown(shutdown: Any, *processes: Any) -> None:
    if any(process.is_alive() for process in processes):
        shutdown.set()


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
        from trackmaniarl.distributed.coordinator import learner_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install trackmaniarl[distributed] to use distributed training") from exc
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
        from trackmaniarl.distributed.actor import actor_process_entry
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install trackmaniarl[distributed] to use distributed training") from exc
    actor_process_entry(config_path, target, actor_id, token, external_stop)


def _smoke(args: argparse.Namespace) -> None:
    """Run a bounded local async actor/learner release check against the game."""

    spec = RunSpec.from_yaml(args.config)
    transitions = args.transitions
    training = _smoke_training(spec.training, transitions)
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
    temporary = base_dir / f".trackmaniarl-{smoke_spec.run_id}.yaml"
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


def _smoke_training(spec: TrainingSpec, transitions: int) -> TrainingSpec:
    """Derive a bounded training schedule that guarantees one learner update."""

    if transitions < 8:
        raise ValueError("smoke testing requires at least 8 transitions")
    n_step = min(spec.n_step, transitions)
    available = transitions - n_step + 1
    batch_capacity = available // spec.sequence_length
    if batch_capacity < 2:
        minimum = spec.sequence_length + n_step
        raise ValueError(f"transitions must be at least {minimum} for a smoke learner update")
    batch_size = min(spec.batch_size, max(1, batch_capacity // 2))
    ready = batch_size * spec.sequence_length + n_step - 1
    return spec.model_copy(
        update={
            "total_transitions": transitions,
            "max_episode_steps": min(spec.max_episode_steps, transitions),
            "batch_size": batch_size,
            "n_step": n_step,
            "warmup_transitions": ready,
            "updates_per_transition": 1.0,
            "checkpoint_interval_updates": 25,
        }
    )


def _restore_smoke_checkpoint(config: Path, spec: RunSpec) -> None:
    from trackmaniarl.distributed.coordinator import Coordinator

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
    expected = "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
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
    if args.sampling_interval_ms < 0.0:
        raise ValueError("sampling interval must be non-negative")
    config = factory.config.model_copy(update={"start_timeout_s": args.start_timeout})
    if config.geometry_path is None or config.expected_map_uid is None:
        raise ValueError("record-demo requires geometry_path and expected_map_uid")
    geometry = BoundaryGeometry(config.geometry_path, expected_map_uid=config.expected_map_uid)
    print(
        "Recorder contract: native sequential telemetry, frame-start controls, "
        "strict 100 Hz quality gate"
    )
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
            sampling_interval_ms=args.sampling_interval_ms,
            status=print,
        )
    finally:
        client.close()
    for demonstration in demonstrations:
        validate_recording_quality(demonstration)
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
    except ConnectionError as error:
        print(f"OpenPlanet telemetry check failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    finally:
        client.close()
    position = frame.values[4:7].tolist() if args.field_count >= 7 else None
    finished = bool(frame.values[2]) if args.field_count >= 3 else "n/a"
    race_time = float(frame.values[3]) if args.field_count >= 4 else "n/a"
    print(
        f"OpenPlanet telemetry OK: {args.field_count} float32 fields; "
        f"position={position}; finished={finished}; race_time_ms={race_time}"
    )


def _benchmark(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config)
    evaluation = spec.evaluation
    if evaluation is None or not evaluation.maps:
        raise ValueError("benchmark requires an evaluation suite with at least one map")
    evaluation_updates = {
        key: value
        for key, value in (
            ("trials_per_map", getattr(args, "trials", None)),
            ("target_median_s", getattr(args, "target_median", None)),
            ("min_finish_rate", getattr(args, "min_finish_rate", None)),
        )
        if value is not None
    }
    if evaluation_updates:
        evaluation = type(evaluation).model_validate(
            {**evaluation.model_dump(), **evaluation_updates}
        )
        spec = spec.model_copy(update={"evaluation": evaluation})
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


def _demo_benchmark(args: argparse.Namespace) -> None:
    if args.trajectory_schedule is not None and not args.trajectory_tracking:
        raise ValueError("--trajectory-schedule requires --trajectory-tracking")
    if args.action_offset_ms and (args.trajectory_tracking or args.phase_locked):
        raise ValueError("--action-offset-ms is only valid for open-loop replay")
    spec = RunSpec.from_yaml(args.config)
    evaluation = spec.evaluation
    if evaluation is None or not evaluation.maps:
        raise ValueError("demo-benchmark requires an evaluation suite with at least one map")
    updates = {
        key: value
        for key, value in (
            ("trials_per_map", args.trials),
            ("target_median_s", args.target_median),
            ("min_finish_rate", args.min_finish_rate),
        )
        if value is not None
    }
    if updates:
        evaluation = type(evaluation).model_validate({**evaluation.model_dump(), **updates})
        spec = spec.model_copy(update={"evaluation": evaluation})
    if evaluation.target_median_s is None:
        raise ValueError("demo-benchmark requires evaluation.target_median_s")
    if not args.phase_locked:
        demonstration = load_demonstration(args.demo)
        interval_ms = demonstration.decision_interval_ms
        if args.trajectory_tracking or (
            interval_ms is None and demonstration.action_repeat_frames == 1
        ):
            validate_recording_quality(demonstration)
            interval_ms = demonstration_timing_summary(demonstration)["interval_median_ms"]
        if interval_ms is not None:
            spec = _with_environment_decision_interval(spec, interval_ms)
            mode = "Trajectory tracking" if args.trajectory_tracking else "Open-loop replay"
            print(f"{mode} interval: {interval_ms:.3f} ms")
    run = resolve_run(spec, base_dir=Path(args.config).parent)
    if not isinstance(run.evaluator, TrackmaniaEvaluator):
        raise ValueError("demo-benchmark requires components.evaluator")
    expected_trials = evaluation.trials_per_map * len(evaluation.maps)
    required_finishes = ceil(evaluation.min_finish_rate * expected_trials)
    try:
        environment_config = getattr(run.environment_factory, "config", None)
        if not isinstance(environment_config, TrackmaniaEnvironmentConfig):
            raise ValueError("demo-benchmark requires OpenPlanetEnvironmentFactory")
        action_ids = environment_config.compact_action_ids
        policy: Policy
        if args.trajectory_tracking:
            tracker = TrajectoryTrackerConfig(
                action_lead_steps=args.action_lead_steps,
                action_lead_ms=args.action_lead_ms,
                lateral_gain=args.tracker_lateral_gain,
                heading_gain=args.tracker_heading_gain,
                lateral_velocity_gain=args.tracker_lateral_velocity_gain,
                steering_threshold=args.tracker_engage_threshold,
                steering_release_threshold=args.tracker_release_threshold,
                preview_ms=args.tracker_preview_ms,
                minimum_correction_steps=args.tracker_minimum_hold_steps,
                reversal_neutral_steps=args.tracker_reversal_neutral_steps,
            )
            policy = (
                build_scheduled_policy(
                    args.demo,
                    TrajectorySchedule.load(args.trajectory_schedule),
                    tracker,
                )
                if args.trajectory_schedule is not None
                else TrajectoryTrackingDemonstrationPolicy.from_path(
                    args.demo,
                    action_lead_steps=args.action_lead_steps,
                    action_lead_ms=args.action_lead_ms,
                    lateral_gain=args.tracker_lateral_gain,
                    heading_gain=args.tracker_heading_gain,
                    lateral_velocity_gain=args.tracker_lateral_velocity_gain,
                    steering_threshold=args.tracker_engage_threshold,
                    steering_release_threshold=args.tracker_release_threshold,
                    preview_ms=args.tracker_preview_ms,
                    minimum_correction_steps=args.tracker_minimum_hold_steps,
                    reversal_neutral_steps=args.tracker_reversal_neutral_steps,
                )
            )
        elif args.phase_locked:
            if not isinstance(run.feature_pipeline, LidarFeaturePipeline):
                raise ValueError("phase-locked demo-benchmark requires LidarFeaturePipeline")
            action_ids = action_ids or tuple(range(78))
            policy = PhaseLockedDemonstrationPolicy.from_path(
                args.demo,
                run.feature_pipeline,
                tuple(action_ids),
                environment_config.decision_interval_ms,
            )
        else:
            policy = DemonstrationReplayPolicy.from_path(
                args.demo, action_ids, action_offset_ms=args.action_offset_ms
            )
            print(
                "Open-loop action timing: "
                f"alignment={demonstration.control_alignment}, "
                f"offset={args.action_offset_ms:+.1f} ms"
            )
        run.evaluator.set_checkpoint(args.demo)
        metrics = dict(run.evaluator.evaluate(policy))
        artifact = json.loads((run.run_dir / "evaluation.json").read_text(encoding="utf-8"))
    finally:
        run.logger.close()
    trials = artifact["trials"]
    _print_benchmark_report(trials, metrics)
    if isinstance(policy, TrajectoryTrackingDemonstrationPolicy):
        _print_trajectory_tracker_diagnostics(policy)
    completed = [trial for trial in trials if trial["finished"]]
    failures = [
        trial
        for trial in trials
        if trial["telemetry_error"] is not None or trial["controller_error"] is not None
    ]
    median = float(metrics["eval/median_finish_time_s"])
    gate_failed = (
        len(completed) < required_finishes or median >= evaluation.target_median_s or bool(failures)
    )
    if gate_failed and args.report_only:
        print("Demonstration replay gate failed; --report-only keeps the diagnostic run successful")
        return
    if gate_failed:
        raise RuntimeError(
            "demonstration replay failed: require "
            f">={required_finishes}/{expected_trials} finishes, "
            f"median completed time <{evaluation.target_median_s}s, "
            "and no telemetry/controller errors"
        )
    print(
        f"Demonstration replay passed: {len(completed)}/{expected_trials} finishes, "
        f"median {median:.3f}s"
    )


def _print_trajectory_tracker_diagnostics(
    policy: TrajectoryTrackingDemonstrationPolicy,
) -> None:
    print(
        "Trajectory tracker diagnostics: "
        f"reference_index={policy.reference_index}, "
        f"correction_events={policy.correction_count}, "
        f"correction_steps={policy.correction_step_count}, "
        f"neutralized_expert_steps={policy.neutralized_expert_step_count}, "
        f"output_steering_switches={policy.output_switch_count}, "
        f"expert_steering_switches={policy.expert_steering_switch_count}, "
        f"opposing_reversals={policy.opposing_switch_count}, "
        f"max_error(position={policy.max_position_error_m:.3f}m, "
        f"lateral={policy.max_abs_lateral_error_m:.3f}m, "
        f"heading={policy.max_abs_heading_error:.4f}, "
        f"lateral_velocity={policy.max_abs_lateral_velocity_error_mps:.3f}m/s)"
    )


def _trajectory_stitch(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    spec = RunSpec.from_yaml(config_path)
    component = spec.components.environment
    if component is None or component.class_path != (
        "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
    ):
        raise ValueError("trajectory-stitch requires OpenPlanetEnvironmentFactory")
    factory = OpenPlanetEnvironmentFactory(base_dir=config_path.parent, **component.kwargs)
    geometry_path = factory.config.geometry_path
    if geometry_path is None:
        raise ValueError("trajectory-stitch requires environment.geometry_path")
    geometry = BoundaryGeometry(geometry_path, expected_map_uid=factory.config.expected_map_uid)
    result = build_fastest_compatible_trajectory(args.demo, geometry)
    output = save_demonstration(args.output.resolve(), result.demonstration)
    print(
        f"Trajectory stitch complete: estimated={result.demonstration.finish_time_s:.3f}s, "
        f"gain={result.estimated_gain_s:.3f}s, joins={len(result.joins)}, output={output}"
    )
    for join in result.joins:
        print(
            f"  join={join.progress_fraction * 100.0:.3f}% "
            f"position_gap={join.position_gap_m:.3f}m "
            f"velocity_gap={join.velocity_gap_mps:.3f}m/s "
            f"heading_gap={join.heading_gap_degrees:.3f}deg"
        )


def _trajectory_synthetic_recovery(args: argparse.Namespace) -> None:
    spec = RunSpec.from_yaml(args.config.resolve())
    dataset = generate_synthetic_recovery_from_path(
        args.demo,
        _compact_action_ids(spec),
        SyntheticRecoveryConfig(
            sample_stride=args.sample_stride,
            action_lead_ms=args.action_lead_ms,
        ),
    )
    output = dataset.save(args.output.resolve())
    print(
        f"Synthetic trajectory recovery: samples={len(dataset.frames)}, "
        f"interventions={int(dataset.interventions.sum())}, output={output}"
    )


def _trajectory_optimize(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    demonstration = load_demonstration(args.demo)
    validate_recording_quality(demonstration)
    interval_ms = _demonstration_interval_ms(demonstration)
    spec = _with_environment_decision_interval(RunSpec.from_yaml(config_path), interval_ms)
    output = _npz_path(args.output.resolve())
    schedule = (
        TrajectorySchedule.load(output)
        if output.exists()
        else TrajectorySchedule.from_controls(demonstration.controls)
    )
    tracker = TrajectoryTrackerConfig(action_lead_ms=args.action_lead_ms)
    search = _trajectory_search_config(args, output, interval_ms)
    build_scheduled_policy(args.demo, schedule, tracker)
    environment = _trajectory_search_environment(spec, config_path, demonstration, args.seed)
    trial = 0

    def evaluate(candidate: TrajectorySchedule) -> TrajectorySearchOutcome:
        nonlocal trial
        trial += 1
        policy = build_scheduled_policy(args.demo, candidate, tracker)
        outcome = run_trajectory_trial(environment, policy, spec.training.max_episode_steps)
        time_text = f"{outcome.finish_time_s:.3f}s" if outcome.finish_time_s else "-"
        print(
            f"Trajectory search trial={trial}/{search.max_trials} "
            f"finished={outcome.finished} time={time_text} "
            f"progress={outcome.progress_pct:.1f}% error={outcome.error or '-'}"
        )
        return outcome

    try:
        result = SafeTrajectoryOptimizer(search).optimize(schedule, evaluate)
    finally:
        environment.close()
    result.schedule.save(output)
    print(
        f"Trajectory optimization complete: median={result.median_finish_time_s:.3f}s, "
        f"live_trials={trial}, schedule={output}"
    )
    if result.median_finish_time_s > search.target_time_s:
        raise RuntimeError(
            f"trajectory target not reached: best confirmed median "
            f"{result.median_finish_time_s:.3f}s > {search.target_time_s:.3f}s"
        )


def _demonstration_interval_ms(demonstration: Demonstration) -> float:
    interval_ms = demonstration.decision_interval_ms
    if interval_ms is None:
        interval_ms = demonstration_timing_summary(demonstration)["interval_median_ms"]
    if interval_ms <= 0.0:
        raise ValueError("trajectory optimization requires a positive demonstration cadence")
    return float(interval_ms)


def _trajectory_search_environment(
    spec: RunSpec,
    config_path: Path,
    demonstration: Demonstration,
    seed: int | None,
) -> Any:
    evaluation = spec.evaluation
    if evaluation is None or len(evaluation.maps) != 1:
        raise ValueError("trajectory-optimize requires exactly one configured evaluation map")
    component = spec.components.environment
    if component is None or component.class_path != (
        "trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory"
    ):
        raise ValueError("trajectory-optimize requires OpenPlanetEnvironmentFactory")
    factory = OpenPlanetEnvironmentFactory(base_dir=config_path.parent, **component.kwargs)
    map_spec = evaluation.maps[0]
    resolved_map = map_spec.model_copy(
        update={
            "map_path": _relative_to(config_path.parent, map_spec.map_path),
            "geometry_path": _relative_to(config_path.parent, map_spec.geometry_path),
        }
    )
    geometry = BoundaryGeometry(
        resolved_map.geometry_path, expected_map_uid=resolved_map.expected_map_uid
    )
    validate_demonstration(demonstration, factory.config, geometry)
    return factory.create(seed=spec.seed if seed is None else seed, evaluation_map=resolved_map)


def _trajectory_search_config(
    args: argparse.Namespace,
    output: Path,
    interval_ms: float,
) -> TrajectorySearchConfig:
    if any(value <= 0.0 for value in args.shortening_ms):
        raise ValueError("--shortening-ms values must be positive")
    if args.minimum_window_ms <= 0.0:
        raise ValueError("--minimum-window-ms must be positive")
    if args.action_lead_ms < 0.0:
        raise ValueError("--action-lead-ms must be non-negative")
    shortening_ticks = tuple(
        dict.fromkeys(max(1, round(value / interval_ms)) for value in args.shortening_ms)
    )
    minimum_window_ticks = max(1, ceil(args.minimum_window_ms / interval_ms))
    return TrajectorySearchConfig(
        shortening_ticks=shortening_ticks,
        minimum_window_ticks=minimum_window_ticks,
        baseline_trials=args.baseline_trials,
        confirmation_trials=args.confirmation_trials,
        minimum_improvement_s=args.minimum_improvement_ms / 1_000.0,
        target_time_s=args.target_time,
        max_trials=args.max_trials,
        checkpoint_path=output,
        journal_path=output.with_suffix(".jsonl"),
    )


def _relative_to(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def _npz_path(path: Path) -> Path:
    return path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")


def _with_environment_decision_interval(spec: RunSpec, interval_ms: float) -> RunSpec:
    environment = spec.components.environment
    if environment is None:
        raise ValueError("demo-benchmark requires components.environment")
    kwargs = dict(environment.kwargs)
    config = dict(kwargs.get("config", {}))
    config.update({"action_repeat_frames": 1, "decision_interval_ms": interval_ms})
    kwargs["config"] = config
    environment = environment.model_copy(update={"kwargs": kwargs})
    components = spec.components.model_copy(update={"environment": environment})
    return spec.model_copy(update={"components": components})


def _diagnose_expert(args: argparse.Namespace) -> None:
    config = args.config.resolve()
    source_spec = RunSpec.from_yaml(config)
    spec = source_spec.model_copy(update={"run_id": f"{source_spec.run_id}-expert-{time_ns()}"})
    run = resolve_run(spec, base_dir=config.parent)
    paths = resolve_demonstration_paths(args.demo)
    try:
        run.learner.setup(_learner_context(run))
        checkpoint = run.checkpoint_codec.load(args.checkpoint)
        learner_state = checkpoint.get("learner", checkpoint)
        run.learner.load_state_dict(learner_state)
        environment_config = _expert_environment_config(run)
        if environment_config.geometry_path is None:
            raise ValueError("expert diagnostics require geometry_path")
        geometry = BoundaryGeometry(
            environment_config.geometry_path,
            expected_map_uid=environment_config.expected_map_uid,
        )
        prepare_run(run)
        reports = [
            _expert_demonstration_report(
                path,
                run.learner,
                run.feature_pipeline,
                environment_config,
                geometry,
            )
            for path in paths
        ]
        payload = {
            "schema_version": "1",
            "checkpoint": str(args.checkpoint),
            "demos": reports,
            "summary": {
                "demonstrations": len(reports),
                "progress_bins": aggregate_expert_bins(
                    report["progress_bins"] for report in reports
                ),
            },
        }
        target = run.run_dir / "expert-diagnostics.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)
    finally:
        run.logger.close()
    print(f"Expert diagnostics: {target}")


def _expert_environment_config(run: Any) -> TrackmaniaEnvironmentConfig:
    environment_config = getattr(run.environment_factory, "config", None)
    if not isinstance(environment_config, TrackmaniaEnvironmentConfig):
        raise ValueError("expert diagnostics require OpenPlanetEnvironmentFactory")
    if environment_config.compact_action_ids is not None:
        raise ValueError("expert diagnostics require the canonical 78-action IQN head")
    if getattr(run.learner.model, "action_count", None) != 78:
        raise ValueError("expert diagnostics require the canonical 78-action IQN head")
    return environment_config


def _expert_demonstration_report(
    path: Path,
    learner: Any,
    pipeline: Any,
    config: TrackmaniaEnvironmentConfig,
    geometry: BoundaryGeometry,
) -> dict[str, Any]:
    demonstration = load_demonstration(path)
    validate_demonstration(demonstration, config, geometry)
    model = learner.model
    if model is None:
        raise RuntimeError("expert diagnostics require an initialized IQN learner model")
    device = learner.device
    model.eval()
    reset = getattr(model, "reset_policy_state", None)
    if callable(reset):
        reset()
    pipeline_reset = getattr(pipeline, "reset_episode", None)
    if callable(pipeline_reset):
        pipeline_reset()
    reference = geometry.racing_line if config.use_racing_line else geometry.reward_center
    pace_profile = (
        ReferencePaceProfile.from_demonstration(config.pace_reference_path, geometry, reference)
        if config.pace_reference_path is not None
        else None
    )
    reward = TrajectoryReward(reference, pace_profile=pace_profile, **config.reward_kwargs())
    reward.reset(
        demonstration.frames[0, list(config.position_indices)],
        velocity=demonstration.frames[0, list(config.velocity_indices)],
        race_time_ms=float(demonstration.frames[0, 3]),
    )
    diagnostics = ExpertActionDiagnostics()
    for action, frame, next_frame in zip(
        demonstration.actions, demonstration.frames[:-1], demonstration.frames[1:], strict=True
    ):
        q_values = _raw_q_values(model, device, pipeline.transform_observation(frame), learner)
        source_action = int(action)
        if not 0 <= source_action < q_values.shape[-1]:
            raise ValueError("demonstration action is outside the raw IQN action head")
        expert_q = float(q_values[source_action])
        greedy_q = float(q_values.max())
        rank = int((q_values > expert_q).sum()) + 1
        result = reward.step(
            next_frame[list(config.position_indices)],
            finish_ui_active=bool(next_frame[2]),
            velocity=next_frame[list(config.velocity_indices)],
            race_time_ms=float(next_frame[3]),
        )
        diagnostics.record(reward.progress_pct, expert_q, greedy_q, rank)
        if result.terminated:
            break
    return {
        "path": str(path),
        "finish_time_s": demonstration.finish_time_s,
        "progress_bins": diagnostics.summary(),
    }


def _raw_q_values(model: Any, device: torch.device, observation: Any, learner: Any) -> torch.Tensor:
    prepare = getattr(model, "prepare_policy_observation", None)
    if callable(prepare):
        observation = prepare(observation)
    observation = tree_to_device(sanitize_finite(observation), device)
    detector = getattr(model, "observation_is_single", None)
    single = bool(detector(observation)) if callable(detector) else observation.ndim == 1
    if single:
        observation = tree_map(
            lambda value: value.unsqueeze(0) if isinstance(value, torch.Tensor) else value,
            observation,
        )
    with torch.inference_mode():
        values = model.q_values(observation, learner.evaluation_quantile_count)
    return cast(torch.Tensor, values).squeeze(0).float().cpu()


def _bc_train(args: argparse.Namespace) -> None:
    from trackmaniarl.trackmania.behavior_cloning import (
        augment_behavior_cloning_laps,
        load_behavior_cloning_laps,
        load_behavior_cloning_recovery,
        split_behavior_cloning_laps,
    )

    config = args.config.resolve()
    spec = _new_attempt_spec(config, RunSpec.from_yaml(config), args)
    paths = resolve_demonstration_paths(args.demo)
    action_ids = _compact_action_ids(spec)
    run = resolve_run(spec, base_dir=config.parent)
    try:
        run.learner.setup(_learner_context(run))
        model = getattr(run.learner, "model", None)
        if model is None or tuple(model.action_ids) != action_ids:
            raise ValueError(
                "model action_ids must exactly match environment.config.compact_action_ids"
            )
        if bool(getattr(run.feature_pipeline, "include_control_inputs", True)):
            raise ValueError(
                "behavior cloning must exclude control inputs to prevent target leakage"
            )
        prepare_run(run)
        environment_config = spec.components.environment
        assert environment_config is not None
        action_repeat_frames = int(
            environment_config.kwargs.get("config", {}).get("action_repeat_frames", 1)
        )
        decision_interval = environment_config.kwargs.get("config", {}).get("decision_interval_ms")
        laps = load_behavior_cloning_laps(
            paths,
            run.feature_pipeline,
            action_ids,
            expected_action_repeat_frames=action_repeat_frames,
            expected_decision_interval_ms=(
                None if decision_interval is None else float(decision_interval)
            ),
            previous_action_conditioning=bool(model.previous_action_conditioning),
        )
        train_laps, validation_laps = split_behavior_cloning_laps(laps, spec.seed)
        recovery_paths = tuple(Path(path).resolve() for path in getattr(args, "recovery", ()))
        if recovery_paths:
            recovery_laps = load_behavior_cloning_recovery(
                recovery_paths,
                run.feature_pipeline,
                action_ids,
            )
            recovery_training, recovery_validation = split_behavior_cloning_laps(
                recovery_laps,
                spec.seed + 1,
            )
            train_laps.extend(recovery_training)
            validation_laps.extend(recovery_validation)
        use_horizontal_flip = bool(
            getattr(args, "horizontal_flip_augmentation", False)
            or getattr(run.learner, "horizontal_flip_augmentation", False)
        )
        if use_horizontal_flip:
            if not getattr(run.feature_pipeline, "local_velocity_features", False):
                raise ValueError("horizontal flip augmentation requires local_velocity_features")
            train_laps = augment_behavior_cloning_laps(train_laps, action_ids)
        _train_behavior_cloning(run, train_laps, validation_laps)
    finally:
        run.logger.close()


def _dagger_collect(args: argparse.Namespace) -> None:
    from trackmaniarl.trackmania.behavior_cloning import (
        BehaviorCloningPolicy,
        save_behavior_cloning_recovery,
    )

    if args.episodes < 1 or not 0.0 <= args.teacher_probability <= 1.0:
        raise ValueError("DAgger episodes and teacher probability are invalid")
    if args.intervention_error <= 0.0:
        raise ValueError("DAgger intervention error must be positive")
    if args.action_lead_ms < 0.0:
        raise ValueError("DAgger action lead must be non-negative")
    config = args.config.resolve()
    source_spec = RunSpec.from_yaml(config)
    spec = source_spec.model_copy(update={"run_id": f"{source_spec.run_id}-dagger-{time_ns()}"})
    run = resolve_run(spec, base_dir=config.parent)
    if run.evaluator is None or spec.evaluation is None or not spec.evaluation.maps:
        raise ValueError("dagger-collect requires a configured evaluation map")
    environment_factory = run.environment_factory
    if not isinstance(environment_factory, OpenPlanetEnvironmentFactory):
        raise ValueError("dagger-collect requires OpenPlanetEnvironmentFactory")
    environment_config = environment_factory.config
    if not isinstance(run.feature_pipeline, LidarFeaturePipeline):
        raise ValueError("dagger-collect requires LidarFeaturePipeline")
    action_ids = environment_config.compact_action_ids
    if action_ids is None:
        raise ValueError("dagger-collect requires compact_action_ids")
    teacher_demonstration = load_demonstration(args.demo)
    validate_recording_quality(teacher_demonstration)
    validate_demonstration(
        teacher_demonstration,
        environment_config,
        run.feature_pipeline.geometry,
    )
    frames: list[np.ndarray] = []
    labels: list[int] = []
    episode_starts: list[bool] = []
    sample_weights: list[float] = []
    student_actions: list[int] = []
    interventions: list[bool] = []
    state_errors: list[float] = []
    finished = 0
    environment = None
    try:
        run.learner.setup(_learner_context(run))
        model = getattr(run.learner, "model", None)
        if model is None or bool(model.previous_action_conditioning):
            raise ValueError("dagger-collect requires BC without previous-action conditioning")
        checkpoint = run.checkpoint_codec.load(args.checkpoint)
        run.learner.load_state_dict(checkpoint["learner"])
        student = run.learner.policy()
        if not isinstance(student, BehaviorCloningPolicy):
            raise ValueError("dagger-collect requires BehaviorCloningPolicy")
        teacher = TrajectoryTrackingDemonstrationPolicy.from_path(
            args.demo,
            action_lead_steps=0,
            action_lead_ms=args.action_lead_ms,
        )
        _, action_table = select_brake_tap_actions(tuple(action_ids))
        map_spec = spec.evaluation.maps[0]
        environment = environment_factory.create(seed=spec.seed, evaluation_map=map_spec)
        generator = np.random.default_rng(spec.seed)
        for episode in range(args.episodes):
            raw, _ = environment.reset(seed=spec.seed + episode)
            run.feature_pipeline.reset_episode()
            student.reset_episode()
            teacher.reset_episode()
            prepared = run.feature_pipeline.transform_observation(raw)
            episode_finished = False
            for step in range(spec.training.max_episode_steps):
                teacher_control = teacher.act(raw, deterministic=True)
                teacher_action = continuous_control_to_discrete_index(teacher_control, action_table)
                student_action = student.act(prepared, deterministic=True)
                state_error = _trajectory_teacher_state_error(teacher)
                intervene = (
                    state_error >= args.intervention_error
                    or generator.random() < args.teacher_probability
                )
                disagree = student_action != teacher_action
                frames.append(np.asarray(raw, dtype=np.float32).copy())
                labels.append(teacher_action)
                episode_starts.append(step == 0)
                sample_weights.append(
                    _dagger_sample_weight(
                        disagree,
                        intervene,
                        state_error,
                        args.intervention_error,
                    )
                )
                student_actions.append(student_action)
                interventions.append(intervene)
                state_errors.append(state_error)
                raw, _, terminated, truncated, info = environment.step(
                    teacher_control if intervene else student_action
                )
                prepared = run.feature_pipeline.transform_observation(raw)
                if terminated or truncated:
                    episode_finished = info.get("termination_reason") == "finished"
                    break
            finished += int(episode_finished)
            print(
                f"DAgger episode {episode + 1}/{args.episodes}: "
                f"finished={episode_finished}, samples={len(frames)}"
            )
        output = save_behavior_cloning_recovery(
            args.output,
            np.asarray(frames, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            np.asarray(episode_starts, dtype=np.bool_),
            tuple(action_ids),
            sample_weights=np.asarray(sample_weights, dtype=np.float32),
            student_actions=np.asarray(student_actions, dtype=np.int64),
            interventions=np.asarray(interventions, dtype=np.bool_),
            state_errors=np.asarray(state_errors, dtype=np.float32),
        )
    finally:
        if environment is not None:
            environment.close()
        run.logger.close()
    print(
        f"DAgger recovery data: {output} ({len(frames)} samples, "
        f"{finished}/{args.episodes} finishes)"
    )


def _trajectory_teacher_state_error(
    teacher: TrajectoryTrackingDemonstrationPolicy,
) -> float:
    return max(
        teacher.last_position_error_m,
        8.0 * abs(teacher.last_heading_error),
        0.5 * abs(teacher.last_lateral_velocity_error_mps),
    )


def _dagger_sample_weight(
    disagreement: bool,
    intervention: bool,
    state_error: float,
    intervention_error: float,
) -> float:
    relative_error = np.clip(state_error / intervention_error, 0.0, 2.0)
    return float(
        np.clip(
            0.25 + 1.75 * disagreement + 2.0 * intervention + relative_error,
            0.25,
            6.0,
        )
    )


def _compact_action_ids(spec: RunSpec) -> tuple[int, ...]:
    environment = spec.components.environment
    if environment is None:
        raise ValueError("behavior cloning requires components.environment")
    raw_ids = environment.kwargs.get("config", {}).get("compact_action_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ValueError("behavior cloning requires environment.config.compact_action_ids")
    return tuple(int(action) for action in raw_ids)


def _learner_context(run: Any) -> dict[str, Any]:
    return {"seed": run.spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}


@dataclass(frozen=True, slots=True)
class _BehaviorCloningSelection:
    minimum_loss: float = float("inf")
    checkpoint_score: float = float("-inf")
    checkpoint_loss: float = float("inf")
    checkpoint_state: dict[str, Any] | None = None
    stale_validations: int = 0


def _behavior_cloning_control_score(metrics: Mapping[str, float]) -> float:
    if metrics.get("intervention_count", 0.0) > 0.0:
        return (
            0.25 * metrics["steering_transition_accuracy"]
            + 0.20 * metrics["transition_accuracy"]
            + 0.15 * metrics["steering_accuracy"]
            + 0.15 * metrics["balanced_accuracy"]
            + 0.15 * metrics["intervention_accuracy"]
            + 0.05 * metrics["weighted_accuracy"]
            + 0.05 * metrics["accuracy"]
        )
    return (
        0.35 * metrics["steering_transition_accuracy"]
        + 0.20 * metrics["transition_accuracy"]
        + 0.15 * metrics["steering_accuracy"]
        + 0.15 * metrics["balanced_accuracy"]
        + 0.10 * metrics["accuracy"]
        + 0.05 * exp(-metrics["loss"])
    )


def _behavior_cloning_checkpoint_improved(
    selection: _BehaviorCloningSelection, loss: float, score: float
) -> bool:
    minimum_loss = min(selection.minimum_loss, loss)
    eligible = loss <= 1.10 * minimum_loss
    selected_eligible = selection.checkpoint_loss <= 1.10 * minimum_loss
    return eligible and (not selected_eligible or score > selection.checkpoint_score + 1.0e-4)


def _train_behavior_cloning(run: Any, training: list[Any], validation: list[Any]) -> None:
    from trackmaniarl.trackmania.behavior_cloning import (
        class_weights,
        clone_state,
        collate_behavior_cloning,
        flatten_behavior_cloning_laps,
    )

    learner = run.learner
    train_observations, train_labels = flatten_behavior_cloning_laps(training)
    validation_observations, validation_labels = flatten_behavior_cloning_laps(validation)
    weights = class_weights(
        train_labels,
        learner.model.head.out_features,
        power=learner.class_weight_power,
    )
    generator = torch.Generator().manual_seed(run.spec.seed)
    selection = _BehaviorCloningSelection()
    best_step = 0
    checkpoint = run.run_dir / "checkpoints" / "bc-best-validation.pt"
    for step in range(1, learner.max_steps + 1):
        indices = torch.randint(
            len(train_labels), (run.spec.training.batch_size,), generator=generator
        )
        observations = [train_observations[int(index)] for index in indices]
        labels = train_labels[indices]
        metrics = learner.train_batch(collate_behavior_cloning(observations), labels, weights)
        run.logger.log("bc/train", metrics, step=step)
        if step % learner.validation_interval == 0:
            selection, improved = _validate_behavior_cloning(
                run,
                validation_observations,
                validation_labels,
                weights,
                step,
                selection,
            )
            if improved:
                assert selection.checkpoint_state is not None
                best_step = step
                run.checkpoint_codec.save(
                    {"learner": clone_state(selection.checkpoint_state)}, checkpoint
                )
            if selection.stale_validations >= learner.early_stopping_patience:
                print(
                    f"Behavior cloning early-stopped at step {step}: "
                    f"lr={learner.current_learning_rate():.2e}"
                )
                break
    if selection.checkpoint_state is None:
        raise RuntimeError("behavior cloning completed without a validation checkpoint")
    learner.load_state_dict(selection.checkpoint_state)
    print(
        f"Behavior cloning complete: best_step={best_step}, "
        f"control_score={selection.checkpoint_score:.5f}, "
        f"checkpoint_loss={selection.checkpoint_loss:.5f}, "
        f"minimum_loss={selection.minimum_loss:.5f}, "
        f"lr={learner.current_learning_rate():.2e}, "
        f"checkpoint={checkpoint}"
    )


def _validate_behavior_cloning(
    run: Any,
    observations: list[Any],
    labels: Any,
    weights: Any,
    step: int,
    selection: _BehaviorCloningSelection,
) -> tuple[_BehaviorCloningSelection, bool]:
    from trackmaniarl.trackmania.behavior_cloning import clone_state, collate_behavior_cloning

    losses: list[float] = []
    correct = total = 0
    transition_correct = transition_total = 0
    steering_correct = steering_total = 0
    steering_transition_correct = steering_transition_total = 0
    weighted_correct = sample_weight_total = 0.0
    intervention_correct = intervention_total = 0
    disagreement_correct = disagreement_total = 0
    action_count = run.learner.model.head.out_features
    per_action_correct = torch.zeros(action_count, dtype=torch.long)
    per_action_count = torch.zeros(action_count, dtype=torch.long)
    for start in range(0, len(labels), run.spec.training.batch_size):
        end = start + run.spec.training.batch_size
        batch = run.learner.evaluate_batch(
            collate_behavior_cloning(observations[start:end]), labels[start:end], weights
        )
        losses.append(batch.loss * batch.total)
        correct += batch.correct
        total += batch.total
        per_action_correct += batch.per_action_correct
        per_action_count += batch.per_action_count
        transition_correct += batch.transition_correct
        transition_total += batch.transition_count
        steering_correct += batch.steering_correct
        steering_total += batch.steering_count
        steering_transition_correct += batch.steering_transition_correct
        steering_transition_total += batch.steering_transition_count
        weighted_correct += batch.weighted_correct
        sample_weight_total += batch.sample_weight_total
        intervention_correct += batch.intervention_correct
        intervention_total += batch.intervention_count
        disagreement_correct += batch.student_disagreement_correct
        disagreement_total += batch.student_disagreement_count
    loss = sum(losses) / total
    learning_rate = run.learner.step_scheduler(loss)
    action_recall = per_action_correct.float() / per_action_count.clamp_min(1)
    observed_actions = per_action_count > 0
    balanced_accuracy = float(action_recall[observed_actions].mean())
    metrics = {
        "loss": loss,
        "accuracy": correct / total,
        "balanced_accuracy": balanced_accuracy,
        "transition_accuracy": transition_correct / max(transition_total, 1),
        "transition_count": float(transition_total),
        "steering_accuracy": steering_correct / max(steering_total, 1),
        "steering_transition_accuracy": steering_transition_correct
        / max(steering_transition_total, 1),
        "steering_transition_count": float(steering_transition_total),
        "weighted_accuracy": weighted_correct / max(sample_weight_total, 1.0e-8),
        "intervention_accuracy": intervention_correct / max(intervention_total, 1),
        "intervention_count": float(intervention_total),
        "student_disagreement_accuracy": disagreement_correct / max(disagreement_total, 1),
        "student_disagreement_count": float(disagreement_total),
        "learning_rate": learning_rate,
    }
    score = _behavior_cloning_control_score(metrics)
    improved = _behavior_cloning_checkpoint_improved(selection, loss, score)
    minimum_loss = min(selection.minimum_loss, loss)
    metrics["control_score"] = score
    metrics["checkpoint_loss_eligible"] = float(loss <= 1.10 * minimum_loss)
    metrics["best"] = float(improved)
    for action_id, recall, count in zip(
        run.learner.model.action_ids,
        action_recall.tolist(),
        per_action_count.tolist(),
        strict=True,
    ):
        metrics[f"action_recall/{action_id}"] = recall
        metrics[f"action_count/{action_id}"] = count
    run.logger.log("bc/validation", metrics, step=step)
    print(
        f"BC validation step={step}: loss={loss:.5f}, accuracy={metrics['accuracy']:.4f}, "
        f"balanced_accuracy={balanced_accuracy:.4f}, "
        f"transition_accuracy={metrics['transition_accuracy']:.4f}, "
        f"steering_accuracy={metrics['steering_accuracy']:.4f}, "
        f"steering_transition_accuracy={metrics['steering_transition_accuracy']:.4f}, "
        f"intervention_accuracy={metrics['intervention_accuracy']:.4f}, "
        f"control_score={score:.5f}, "
        f"lr={learning_rate:.2e}, best={improved}"
    )
    if improved:
        return (
            _BehaviorCloningSelection(
                minimum_loss=minimum_loss,
                checkpoint_score=score,
                checkpoint_loss=loss,
                checkpoint_state=clone_state(run.learner.state_dict()),
            ),
            True,
        )
    return (
        _BehaviorCloningSelection(
            minimum_loss=minimum_loss,
            checkpoint_score=selection.checkpoint_score,
            checkpoint_loss=selection.checkpoint_loss,
            checkpoint_state=selection.checkpoint_state,
            stale_validations=selection.stale_validations + 1,
        ),
        False,
    )


def _bc_benchmark(args: argparse.Namespace) -> None:
    if args.trials < 1:
        raise ValueError("bc-benchmark --trials must be positive")
    spec = RunSpec.from_yaml(args.config)
    if spec.evaluation is None or not spec.evaluation.maps:
        raise ValueError("bc-benchmark requires an evaluation suite with at least one map")
    suite = spec.evaluation.model_copy(update={"trials_per_map": args.trials})
    benchmark_spec = spec.model_copy(
        update={"run_id": f"{spec.run_id}-bc-eval-{time_ns()}", "evaluation": suite}
    )
    run = resolve_run(benchmark_spec, base_dir=args.config.parent)
    if run.evaluator is None:
        raise ValueError("bc-benchmark requires components.evaluator")
    try:
        run.learner.setup(_learner_context(run))
        checkpoint = run.checkpoint_codec.load(args.checkpoint)
        run.learner.load_state_dict(checkpoint["learner"])
        set_checkpoint = getattr(run.evaluator, "set_checkpoint", None)
        if callable(set_checkpoint):
            set_checkpoint(args.checkpoint)
        metrics = dict(run.evaluator.evaluate(run.learner.policy()))
        artifact = json.loads((run.run_dir / "evaluation.json").read_text(encoding="utf-8"))
    finally:
        run.logger.close()
    _print_benchmark_report(artifact["trials"], metrics)
    _print_bc_rollout_gate(artifact["trials"], metrics, suite)


def _print_bc_rollout_gate(
    trials: list[dict[str, Any]], metrics: dict[str, float], suite: Any
) -> None:
    completed = [trial for trial in trials if trial["finished"]]
    target_median_s = suite.target_median_s
    if target_median_s is None:
        print("BC rollout gate: no target_median_s configured")
        return
    required_finishes = ceil(suite.min_finish_rate * len(trials))
    faster_than_target = [
        trial for trial in completed if float(trial["finish_time_s"]) < target_median_s
    ]
    median = float(metrics["eval/median_finish_time_s"])
    go = len(completed) >= required_finishes and bool(faster_than_target)
    full_success = go and median < target_median_s
    print(
        f"BC rollout gate: go={go}, full_success={full_success}, "
        f"finishes={len(completed)}/{len(trials)}, under_target={len(faster_than_target)}, "
        f"target={target_median_s:.3f}s, median={median:.3f}s"
    )


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
    """Parse cross-platform TrackmaniaRL SDK commands."""

    parser = argparse.ArgumentParser(
        prog="trackmaniarl", description="TrackmaniaRL project tooling"
    )
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
    train.add_argument(
        "--model-initialization-checkpoint",
        type=Path,
        help="warm-start the configured learner model from this checkpoint",
    )
    train.set_defaults(handler=_train)
    offline_pretrain = commands.add_parser(
        "offline-pretrain",
        help="run configured IQN/DQfD updates from demonstrations without starting TrackMania",
    )
    offline_pretrain.add_argument("config", type=Path)
    offline_pretrain.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    offline_pretrain.add_argument(
        "--model-initialization-checkpoint",
        type=Path,
        help="warm-start the configured learner model from this checkpoint",
    )
    offline_pretrain.set_defaults(handler=_offline_pretrain)
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
    benchmark = commands.add_parser(
        "benchmark", help="run the configured Trackmania evaluation release gate"
    )
    benchmark.add_argument("config", type=Path)
    benchmark.add_argument("checkpoint", type=Path)
    benchmark.add_argument("--trials", type=int)
    benchmark.add_argument("--target-median", type=float)
    benchmark.add_argument("--min-finish-rate", type=float)
    benchmark.set_defaults(handler=_benchmark)
    demo_benchmark = commands.add_parser(
        "demo-benchmark", help="evaluate direct time-synchronised replay of one human demonstration"
    )
    demo_benchmark.add_argument("config", type=Path)
    demo_benchmark.add_argument("demo", type=Path)
    demo_benchmark.add_argument("--trials", type=int)
    demo_benchmark.add_argument("--target-median", type=float)
    demo_benchmark.add_argument("--min-finish-rate", type=float)
    demo_benchmark.add_argument(
        "--report-only",
        action="store_true",
        help="report a failed replay gate without returning a failing process exit code",
    )
    demo_benchmark.add_argument(
        "--action-offset-ms",
        type=float,
        default=0.0,
        help="signed open-loop action timestamp offset; positive values delay switching",
    )
    replay_mode = demo_benchmark.add_mutually_exclusive_group()
    replay_mode.add_argument(
        "--open-loop",
        dest="phase_locked",
        action="store_false",
        help="replay strictly by race time (default)",
    )
    replay_mode.add_argument(
        "--phase-locked",
        action="store_true",
        help="enable state matching and steering recovery for diagnostic comparison",
    )
    replay_mode.add_argument(
        "--trajectory-tracking",
        action="store_true",
        help="track raw world-space expert state with feed-forward controls and feedback",
    )
    demo_benchmark.add_argument(
        "--action-lead-steps",
        type=int,
        default=1,
        help="expert control look-ahead in native telemetry ticks",
    )
    demo_benchmark.add_argument(
        "--action-lead-ms",
        type=float,
        help="expert control look-ahead in physical milliseconds; overrides --action-lead-steps",
    )
    demo_benchmark.add_argument(
        "--trajectory-schedule",
        type=Path,
        help="optimized schedule produced by trajectory-optimize",
    )
    demo_benchmark.add_argument("--tracker-lateral-gain", type=float, default=0.8)
    demo_benchmark.add_argument("--tracker-heading-gain", type=float, default=4.0)
    demo_benchmark.add_argument("--tracker-lateral-velocity-gain", type=float, default=0.03)
    demo_benchmark.add_argument("--tracker-engage-threshold", type=float, default=0.35)
    demo_benchmark.add_argument("--tracker-release-threshold", type=float, default=0.15)
    demo_benchmark.add_argument("--tracker-preview-ms", type=float, default=0.0)
    demo_benchmark.add_argument("--tracker-minimum-hold-steps", type=int, default=4)
    demo_benchmark.add_argument("--tracker-reversal-neutral-steps", type=int, default=2)
    demo_benchmark.set_defaults(phase_locked=False, trajectory_tracking=False)
    demo_benchmark.set_defaults(handler=_demo_benchmark)
    trajectory_stitch = commands.add_parser(
        "trajectory-stitch",
        help="splice state-compatible segments from demonstrations with matching time contracts",
    )
    trajectory_stitch.add_argument("config", type=Path)
    trajectory_stitch.add_argument("output", type=Path)
    trajectory_stitch.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory (repeatable)",
    )
    trajectory_stitch.set_defaults(handler=_trajectory_stitch)
    synthetic_recovery = commands.add_parser(
        "trajectory-synthetic-recovery",
        help="generate deterministic counterfactual recovery states around an expert trajectory",
    )
    synthetic_recovery.add_argument("config", type=Path)
    synthetic_recovery.add_argument("demo", type=Path)
    synthetic_recovery.add_argument("output", type=Path)
    synthetic_recovery.add_argument("--sample-stride", type=int, default=4)
    synthetic_recovery.add_argument("--action-lead-ms", type=float, default=0.0)
    synthetic_recovery.set_defaults(handler=_trajectory_synthetic_recovery)
    trajectory_optimize = commands.add_parser(
        "trajectory-optimize",
        help="safely optimize expert coast and brake windows on one fixed map",
    )
    trajectory_optimize.add_argument("config", type=Path)
    trajectory_optimize.add_argument("demo", type=Path)
    trajectory_optimize.add_argument("output", type=Path)
    trajectory_optimize.add_argument("--seed", type=int)
    trajectory_optimize.add_argument("--max-trials", type=int, default=64)
    trajectory_optimize.add_argument("--baseline-trials", type=int, default=3)
    trajectory_optimize.add_argument("--confirmation-trials", type=int, default=2)
    trajectory_optimize.add_argument("--target-time", type=float, default=36.0)
    trajectory_optimize.add_argument("--action-lead-ms", type=float, default=10.0)
    trajectory_optimize.add_argument(
        "--shortening-ms",
        type=float,
        nargs="+",
        default=(40.0, 20.0, 10.0),
        metavar="MS",
    )
    trajectory_optimize.add_argument("--minimum-window-ms", type=float, default=30.0)
    trajectory_optimize.add_argument("--minimum-improvement-ms", type=float, default=15.0)
    trajectory_optimize.set_defaults(handler=_trajectory_optimize)
    diagnose = commands.add_parser("diagnose", help="offline policy diagnostics")
    diagnose_commands = diagnose.add_subparsers(dest="diagnose_command", required=True)
    expert = diagnose_commands.add_parser(
        "expert", help="score complete demonstrations with the unmasked IQN action head"
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
    bc_train = commands.add_parser(
        "bc-train", help="train a compact TrackMania policy from complete demonstrations"
    )
    bc_train.add_argument("config", type=Path)
    bc_train.add_argument(
        "--demo",
        action="append",
        type=Path,
        required=True,
        help="demonstration .npz file or directory of .npz files (repeatable)",
    )
    bc_train.add_argument(
        "--horizontal-flip-augmentation",
        action="store_true",
        help="add reflected local-frame demonstration laps to behavior-cloning training only",
    )
    bc_train.add_argument(
        "--recovery",
        action="append",
        type=Path,
        default=[],
        help="recovery .npz file (repeatable; split into training and validation episodes)",
    )
    bc_train.set_defaults(handler=_bc_train)
    bc_benchmark = commands.add_parser(
        "bc-benchmark", help="run closed TrackMania rollouts for a behavior-cloning checkpoint"
    )
    bc_benchmark.add_argument("config", type=Path)
    bc_benchmark.add_argument("checkpoint", type=Path)
    bc_benchmark.add_argument("--trials", type=int, default=30)
    bc_benchmark.set_defaults(handler=_bc_benchmark)
    dagger = commands.add_parser(
        "dagger-collect",
        help="collect student states labelled by a closed-loop trajectory expert",
    )
    dagger.add_argument("config", type=Path)
    dagger.add_argument("checkpoint", type=Path)
    dagger.add_argument("demo", type=Path)
    dagger.add_argument("output", type=Path)
    dagger.add_argument("--episodes", type=int, default=10)
    dagger.add_argument("--teacher-probability", type=float, default=0.15)
    dagger.add_argument("--intervention-error", type=float, default=0.8)
    dagger.add_argument("--action-lead-ms", type=float, default=0.0)
    dagger.set_defaults(handler=_dagger_collect)
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
    demo.add_argument(
        "--sampling-interval-ms",
        type=float,
        default=0.0,
        help="physical sampling interval; 0 records every new telemetry frame",
    )
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
