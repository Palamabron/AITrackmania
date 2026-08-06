"""Component resolution, capability checks, and reproducible run manifests."""

from __future__ import annotations

import importlib
import inspect
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tmrl.core.contracts import (
    CheckpointCodec,
    EnvironmentFactory,
    Evaluator,
    FeaturePipeline,
    Learner,
    ModelFactory,
    ReplayStore,
    RunLogger,
    Sampler,
)
from tmrl.core.data import Transition
from tmrl.core.spec import ComponentSpec, RunSpec


def import_symbol(path: str) -> Any:
    """Import ``package.module:Symbol`` with an actionable error."""

    module_name, separator, symbol_name = path.partition(":")
    if not separator:
        raise ValueError(f"Component path must be 'module:Symbol', got {path!r}")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, symbol_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not import component {path!r}: {exc}") from exc


def _instantiate(spec: ComponentSpec, **injected: Any) -> Any:
    factory = import_symbol(spec.class_path)
    kwargs = dict(spec.kwargs)
    try:
        parameters: dict[str, inspect.Parameter] = dict(inspect.signature(factory).parameters)
    except (TypeError, ValueError):
        parameters = {}
    for key, value in injected.items():
        if key in parameters and key not in kwargs:
            kwargs[key] = value
    try:
        return factory(**kwargs)
    except TypeError as exc:
        raise TypeError(f"Cannot instantiate {spec.class_path} with {kwargs}: {exc}") from exc


def _require(name: str, value: Any, contract: type[Any]) -> None:
    if not isinstance(value, contract):
        raise TypeError(f"{name} ({type(value).__name__}) does not implement {contract.__name__}")


@dataclass(frozen=True, slots=True)
class ResolvedRun:
    """Instantiated, validated components for exactly one immutable ``RunSpec``."""

    spec: RunSpec
    run_dir: Path
    learner: Learner
    environment_factory: EnvironmentFactory | None
    model_factory: ModelFactory | None
    replay_store: ReplayStore
    sampler: Sampler
    feature_pipeline: FeaturePipeline
    logger: RunLogger
    checkpoint_codec: CheckpointCodec
    evaluator: Evaluator | None


def resolve_run(spec: RunSpec, *, base_dir: str | Path = ".") -> ResolvedRun:
    """Instantiate and validate a run without importing the legacy configuration tree."""

    project_dir = Path(base_dir)
    if spec.evaluation is not None:
        maps = tuple(
            item.model_copy(
                update={
                    "map_path": item.map_path
                    if item.map_path.is_absolute()
                    else (project_dir / item.map_path).resolve(),
                    "geometry_path": item.geometry_path
                    if item.geometry_path.is_absolute()
                    else (project_dir / item.geometry_path).resolve(),
                }
            )
            for item in spec.evaluation.maps
        )
        suite = spec.evaluation.model_copy(update={"maps": maps})
        spec = spec.model_copy(update={"evaluation": suite})
    run_dir = project_dir / spec.artifacts_dir / spec.run_id
    pipeline = _instantiate(spec.components.feature_pipeline, base_dir=project_dir)
    environment_factory = None
    if spec.components.environment is not None:
        environment_factory = _instantiate(spec.components.environment, base_dir=project_dir)
    model_factory = None
    if spec.components.model_factory is not None:
        model_factory = _instantiate(spec.components.model_factory)
    store = _instantiate(spec.components.replay_store)
    sampler = _instantiate(spec.components.sampler, pipeline=pipeline, seed=spec.seed)
    learner = _instantiate(
        spec.components.learner,
        seed=spec.seed,
        model_factory=model_factory,
        base_dir=project_dir,
    )
    logger = _instantiate(spec.components.logger, run_dir=run_dir, run_id=spec.run_id)
    if spec.components.additional_loggers:
        from tmrl.core.builtins import CompositeRunLogger

        logger = CompositeRunLogger(
            logger,
            *(
                _instantiate(
                    item,
                    run_dir=run_dir,
                    run_id=spec.run_id,
                    config=spec.model_dump(mode="json"),
                )
                for item in spec.components.additional_loggers
            ),
        )
    codec = _instantiate(spec.components.checkpoint_codec)
    evaluator = None
    if spec.components.evaluator is not None:
        evaluator = _instantiate(
            spec.components.evaluator,
            suite=spec.evaluation,
            environment_factory=environment_factory,
            feature_pipeline=pipeline,
            max_episode_steps=spec.training.max_episode_steps,
            run_dir=run_dir,
        )

    _require("feature_pipeline", pipeline, FeaturePipeline)
    if environment_factory is not None:
        _require("environment", environment_factory, EnvironmentFactory)
    _require("replay_store", store, ReplayStore)
    _require("sampler", sampler, Sampler)
    _require("learner", learner, Learner)
    if model_factory is not None:
        _require("model_factory", model_factory, ModelFactory)
    _require("logger", logger, RunLogger)
    _require("checkpoint_codec", codec, CheckpointCodec)
    if evaluator is not None:
        _require("evaluator", evaluator, Evaluator)

    resolved = ResolvedRun(
        spec=spec,
        run_dir=run_dir,
        learner=learner,
        environment_factory=environment_factory,
        model_factory=model_factory,
        replay_store=store,
        sampler=sampler,
        feature_pipeline=pipeline,
        logger=logger,
        checkpoint_codec=codec,
        evaluator=evaluator,
    )
    return resolved


def prepare_run(run: ResolvedRun) -> None:
    """Seed process RNGs and create the immutable run manifest once."""

    random.seed(run.spec.seed)
    import numpy as np
    import torch

    np.random.seed(run.spec.seed)
    torch.manual_seed(run.spec.seed)
    # Import here to keep data/contracts importable without observability cycles.
    from tmrl.observability.artifacts import write_run_manifest

    write_run_manifest(run)


def validate_resolved_run(run: ResolvedRun) -> dict[str, float]:
    """Execute a deterministic no-game smoke update for ``tmrl validate``."""

    run.learner.setup(
        {"seed": run.spec.seed, "run_dir": run.run_dir, "model_factory": run.model_factory}
    )
    prepare_run(run)
    request = run.spec.training.batch_request()
    transition_count = max(8, request.batch_size + request.sequence_length - 1)
    synthetic = getattr(run.feature_pipeline, "synthetic_observation", None)
    for step in range(transition_count):
        raw_observation = synthetic() if callable(synthetic) else {"speed": float(step)}
        observation = run.feature_pipeline.transform_observation(raw_observation)
        is_demo = step < transition_count // 2
        run.replay_store.append(
            Transition(
                observation=observation,
                action=0.0,
                reward=float(step),
                next_observation=observation,
                terminated=step == transition_count - 1,
                truncated=False,
                info={
                    "is_demo": is_demo,
                    "sampling/projected_lap_time_s": 1.0 if is_demo else float("inf"),
                },
                episode_id="validation",
                step=step,
            )
        )
    batch = run.sampler.sample(run.replay_store, request)
    update = run.learner.update(batch)
    metrics, priority_update = update if isinstance(update, tuple) else (update, None)
    if priority_update is not None:
        run.sampler.update_priorities(priority_update)
    output = {key: float(value) for key, value in metrics.items()}
    run.logger.log("validation/update", output, step=1)
    checkpoint = run.run_dir / "checkpoints" / "validation.json"
    run.checkpoint_codec.save(run.learner.state_dict(), checkpoint)
    run.learner.load_state_dict(run.checkpoint_codec.load(checkpoint))
    return output
