"""Stable contracts and local training runtime for TrackmaniaRL 1.0."""

from typing import Any

from trackmaniarl.core.contracts import (
    BehaviorPolicy,
    CheckpointCodec,
    EnvironmentFactory,
    Evaluator,
    ExploratoryPolicy,
    FeaturePipeline,
    Learner,
    ModelContract,
    ModelFactory,
    Policy,
    ReplayStore,
    ReplicablePolicy,
    RunLogger,
    Sampler,
)
from trackmaniarl.core.data import (
    BatchRequest,
    EpisodeArtifact,
    PriorityUpdate,
    SampleBatch,
    TrainingBatch,
    Trajectory,
    Transition,
    TransitionId,
)
from trackmaniarl.core.replay import (
    DemoMixSampler,
    InMemoryReplayStore,
    OnPolicySequenceSampler,
    PrioritizedSampler,
    SequenceSampler,
    UniformSampler,
)
from trackmaniarl.core.spec import EvaluationMapSpec, EvaluationSuiteSpec, RunSpec

__all__ = [
    "BatchRequest",
    "BehaviorPolicy",
    "CheckpointCodec",
    "DemoMixSampler",
    "EnvironmentFactory",
    "EpisodeArtifact",
    "EvaluationMapSpec",
    "EvaluationSuiteSpec",
    "Evaluator",
    "ExploratoryPolicy",
    "FeaturePipeline",
    "InMemoryReplayStore",
    "Learner",
    "ModelContract",
    "ModelFactory",
    "OnPolicySequenceSampler",
    "Policy",
    "PrioritizedSampler",
    "PriorityUpdate",
    "ReplayStore",
    "ReplicablePolicy",
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
        from trackmaniarl.core import runtime

        return getattr(runtime, name)
    if name in {"Trainer", "TrainingResult"}:
        from trackmaniarl.core import training

        return getattr(training, name)
    raise AttributeError(name)
