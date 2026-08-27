"""Component resolution, capability checks, and reproducible run manifests."""

from __future__ import annotations

import importlib
import inspect
import random
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any

from trackmaniarl.core.contracts import (
    CheckpointCodec,
    EnvironmentFactory,
    Evaluator,
    EvaluatorRuntimeRequest,
    FeaturePipeline,
    Learner,
    ModelContract,
    ModelFactory,
    ReplayStore,
    RunLogger,
    Sampler,
)
from trackmaniarl.core.spec import ComponentSpec, EvaluationMapSpec, RunSpec


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


def _validate_model_contract(learner: object, model_factory: object | None) -> None:
    if model_factory is None:
        return
    provided = getattr(model_factory, "model_contract", None)
    accepted = getattr(learner, "accepted_model_contracts", None)
    if provided is None or accepted is None:
        return
    try:
        provided_contract = ModelContract(provided)
        accepted_contracts = frozenset(ModelContract(item) for item in accepted)
    except ValueError as exc:
        raise ValueError(f"Unknown model contract: {exc}") from exc
    if provided_contract not in accepted_contracts:
        expected = ", ".join(sorted(item.value for item in accepted_contracts))
        raise ValueError(
            f"{type(learner).__name__} cannot train {type(model_factory).__name__}: "
            f"model contract is {provided_contract.value!r}, expected one of {expected}"
        )


@dataclass(frozen=True, slots=True)
class _TrainingComponents:
    learner: object
    sampler: object
    pipeline: object


def _validate_training_contract(spec: RunSpec, components: _TrainingComponents) -> None:
    _validate_sequence_contract(spec, components)
    _validate_burn_in(spec, components.learner)
    _validate_history_contract(spec, components.pipeline)
    if getattr(components.learner, "on_policy", False) and spec.training.n_step != 1:
        raise ValueError("on-policy training requires training.n_step=1")


def _validate_sequence_contract(spec: RunSpec, components: _TrainingComponents) -> None:
    sequence_length = spec.training.sequence_length
    if sequence_length == 1:
        return
    sampler = components.sampler
    _validate_sequence_support(components.learner, "supports_sequence_training", sequence_length)
    _validate_sequence_support(sampler, "supports_sequence_sampling", sequence_length)
    configured_length = getattr(sampler, "sequence_length", sequence_length)
    if configured_length != sequence_length:
        raise ValueError(
            "training.sequence_length must match sampler sequence_length; "
            f"got {sequence_length} and {configured_length}"
        )
    if spec.training.n_step >= sequence_length:
        raise ValueError("training.n_step must be smaller than training.sequence_length")


def _validate_sequence_support(component: object, attribute: str, sequence_length: int) -> None:
    if getattr(component, attribute, None) is False:
        raise ValueError(
            f"{type(component).__name__} requires training.sequence_length=1; got {sequence_length}"
        )


def _validate_burn_in(spec: RunSpec, learner: object) -> None:
    sequence_length = spec.training.sequence_length
    burn_in = int(getattr(learner, "burn_in", 0))
    if (sequence_length == 1 and burn_in) or burn_in >= sequence_length:
        raise ValueError(
            "learner burn_in must be zero for single-step replay and below sequence_length"
        )


def _validate_history_contract(spec: RunSpec, pipeline: object) -> None:
    if spec.training.sequence_length > 1 and int(getattr(pipeline, "history_length", 1)) > 1:
        raise ValueError(
            "training.sequence_length and feature history_length cannot both exceed one"
        )


def _validate_reward_discount(spec: RunSpec, environment_factory: object | None) -> None:
    if environment_factory is None or (
        type(environment_factory).__module__ != "trackmaniarl.trackmania.environment"
        or type(environment_factory).__name__ != "OpenPlanetEnvironmentFactory"
    ):
        return
    factory: Any = environment_factory
    reward_gamma = float(factory.config.reward_gamma)
    if not isclose(spec.training.gamma, reward_gamma, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(
            "Potential-based reward shaping requires training.gamma to equal "
            f"environment.config.reward_gamma; got {spec.training.gamma} and {reward_gamma}"
        )


def _redact_config(value: Any) -> Any:
    if isinstance(value, dict):
        secret_tokens = ("key", "token", "secret", "password")
        return {
            key: "<redacted>"
            if any(token in key.lower() for token in secret_tokens)
            else _redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_config(item) for item in value]
    return value


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
    project_dir = Path(base_dir)
    return _RunResolver(_resolve_evaluation_paths(spec, project_dir), project_dir).resolve()


def _resolve_evaluation_paths(spec: RunSpec, project_dir: Path) -> RunSpec:
    if spec.evaluation is None:
        return spec
    maps = tuple(_resolve_evaluation_map(item, project_dir) for item in spec.evaluation.maps)
    suite = spec.evaluation.model_copy(update={"maps": maps})
    return spec.model_copy(update={"evaluation": suite})


def _resolve_evaluation_map(item: EvaluationMapSpec, project_dir: Path) -> EvaluationMapSpec:
    return item.model_copy(
        update={
            "map_path": _project_path(item.map_path, project_dir),
            "geometry_path": _project_path(item.geometry_path, project_dir),
        }
    )


def _project_path(path: Path, project_dir: Path) -> Path:
    return path if path.is_absolute() else (project_dir / path).resolve()


class _RunResolver:
    def __init__(self, spec: RunSpec, project_dir: Path) -> None:
        self.spec = spec
        self.project_dir = project_dir
        self.run_dir = project_dir / spec.artifacts_dir / spec.run_id
        self.pipeline = _instantiate(spec.components.feature_pipeline, base_dir=project_dir)
        self.environment_factory = self._instantiate_environment()
        _validate_reward_discount(spec, self.environment_factory)
        self.model_factory = self._instantiate_model_factory()
        self.replay_store = _instantiate(spec.components.replay_store)
        self.sampler = _instantiate(spec.components.sampler, pipeline=self.pipeline, seed=spec.seed)
        self.learner = self._instantiate_learner()
        components = _TrainingComponents(self.learner, self.sampler, self.pipeline)
        _validate_training_contract(spec, components)
        self.logger = self._instantiate_logger()
        self.checkpoint_codec = _instantiate(spec.components.checkpoint_codec)
        self.evaluator = self._instantiate_evaluator()

    def _instantiate_environment(self) -> Any | None:
        component = self.spec.components.environment
        return None if component is None else _instantiate(component, base_dir=self.project_dir)

    def _instantiate_model_factory(self) -> Any | None:
        component = self.spec.components.model_factory
        return None if component is None else _instantiate(component)

    def _instantiate_learner(self) -> Any:
        return _instantiate(
            self.spec.components.learner,
            seed=self.spec.seed,
            model_factory=self.model_factory,
            base_dir=self.project_dir,
        )

    def _instantiate_logger(self) -> Any:
        logger = _instantiate(
            self.spec.components.logger, run_dir=self.run_dir, run_id=self.spec.run_id
        )
        if not self.spec.components.additional_loggers:
            return logger
        from trackmaniarl.core.builtins import CompositeRunLogger

        additional = tuple(
            self._instantiate_additional_logger(item)
            for item in self.spec.components.additional_loggers
        )
        return CompositeRunLogger(logger, *additional)

    def _instantiate_additional_logger(self, component: ComponentSpec) -> Any:
        return _instantiate(
            component,
            run_dir=self.run_dir,
            run_id=self.spec.run_id,
            config=_redact_config(self.spec.model_dump(mode="json")),
        )

    def _instantiate_evaluator(self) -> Any | None:
        component = self.spec.components.evaluator
        if component is None:
            return None
        request = EvaluatorRuntimeRequest(
            self.spec.evaluation,
            self.environment_factory,
            self.pipeline,
            self.spec.training.max_episode_steps,
            self.run_dir,
        )
        return _instantiate(component, request=request)

    def _validate(self) -> None:
        _require("feature_pipeline", self.pipeline, FeaturePipeline)
        if self.environment_factory is not None:
            _require("environment", self.environment_factory, EnvironmentFactory)
        _require("replay_store", self.replay_store, ReplayStore)
        _require("sampler", self.sampler, Sampler)
        _require("learner", self.learner, Learner)
        if self.model_factory is not None:
            _require("model_factory", self.model_factory, ModelFactory)
        _validate_model_contract(self.learner, self.model_factory)
        _require("logger", self.logger, RunLogger)
        _require("checkpoint_codec", self.checkpoint_codec, CheckpointCodec)
        if self.evaluator is not None:
            _require("evaluator", self.evaluator, Evaluator)

    def resolve(self) -> ResolvedRun:
        self._validate()
        return ResolvedRun(
            spec=self.spec,
            run_dir=self.run_dir,
            learner=self.learner,
            environment_factory=self.environment_factory,
            model_factory=self.model_factory,
            replay_store=self.replay_store,
            sampler=self.sampler,
            feature_pipeline=self.pipeline,
            logger=self.logger,
            checkpoint_codec=self.checkpoint_codec,
            evaluator=self.evaluator,
        )


def prepare_run(run: ResolvedRun) -> None:
    """Seed process RNGs and create the immutable run manifest once."""

    random.seed(run.spec.seed)
    import numpy as np
    import torch

    np.random.seed(run.spec.seed)
    torch.manual_seed(run.spec.seed)
    # Import here to keep data/contracts importable without observability cycles.
    from trackmaniarl.observability.artifacts import write_run_manifest

    write_run_manifest(run)


def validate_resolved_run(run: ResolvedRun) -> dict[str, float]:
    """Execute a deterministic no-game smoke update for ``trackmaniarl validate``."""

    from trackmaniarl.core.runtime_validation import validate_resolved_run as validate

    return validate(run)
