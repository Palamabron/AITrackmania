"""Stable contracts and local training runtime for TMRL 1.0."""

from typing import Any

from tmrl.core.contracts import (
    CheckpointCodec,
    EnvironmentFactory,
    Evaluator,
    FeaturePipeline,
    Learner,
    ModelFactory,
    Policy,
    ReplayStore,
    RunLogger,
    Sampler,
)
from tmrl.core.data import (
    BatchRequest,
    EpisodeArtifact,
    PriorityUpdate,
    SampleBatch,
    TrainingBatch,
    Trajectory,
    Transition,
    TransitionId,
)
from tmrl.core.replay import (
    DemoMixSampler,
    InMemoryReplayStore,
    PrioritizedSampler,
    SequenceSampler,
    UniformSampler,
)
from tmrl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec, RunSpec

__all__ = [
    "BatchRequest",
    "CheckpointCodec",
    "DemoMixSampler",
    "EnvironmentFactory",
    "EpisodeArtifact",
    "EvaluationMapSpec",
    "EvaluationSuiteSpec",
    "Evaluator",
    "FeaturePipeline",
    "InMemoryReplayStore",
    "Learner",
    "ModelFactory",
    "Policy",
    "PrioritizedSampler",
    "PriorityUpdate",
    "ReplayStore",
    "ResolvedRun",
    "RunLogger",
    "RunSpec",
    "SampleBatch",
    "Sampler",
    "SequenceSampler",
    "Trainer",
    "TrainingBatch",
    "TrainingResult",
    "Trajectory",
    "Transition",
    "TransitionId",
    "UniformSampler",
    "resolve_run",
    "validate_resolved_run",
]


def __getattr__(name: str) -> Any:
    """Load orchestration classes lazily to keep contract imports side-effect free."""

    if name in {"ResolvedRun", "resolve_run", "validate_resolved_run"}:
        from tmrl.core import runtime

        return getattr(runtime, name)
    if name in {"Trainer", "TrainingResult"}:
        from tmrl.core import training

        return getattr(training, name)
    raise AttributeError(name)
