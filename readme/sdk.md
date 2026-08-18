# TrackmaniaRL SDK Guide

TrackmaniaRL has one runtime: an explicit `run.yaml` is parsed into `RunSpec`, its
components are imported, then `Trainer` performs TrackMania collection and
off-policy updates. The same commands work in PowerShell, bash, WSL and CI.

```bash
uv sync --group dev --extra algorithms
uv run trackmaniarl init my-trackmania-agent
cd my-trackmania-agent
uv sync
uv run trackmaniarl validate run.yaml
uv run trackmaniarl train run.yaml
```

## Start with bundled components

`trackmaniarl.builtins` is the public catalogue. Learners are selected directly by
their stable descriptive class paths:

```yaml
components:
  learner:
    class_path: trackmaniarl.algorithms.implicit_quantile_q_learning:ImplicitQuantileQLearning
  model_factory:
    class_path: my_agent.models:MyIqnModelFactory
  feature_pipeline:
    class_path: trackmaniarl.builtins.features:TransitionFeaturePipeline
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
  sampler:
    class_path: trackmaniarl.core.replay:PrioritizedSampler
```

Each learner consumes a `TrainingBatch` with explicit n-step bootstrap discounts,
termination/truncation flags, PER weights and monotonic transition IDs. This lets
a user replace one component at a time without an adapter to a legacy runtime.

Set `training.n_step`, `training.gamma`, `training.sequence_length`, and optional
`training.beta` in `run.yaml`; the local trainer forwards these values in every
`BatchRequest`. Discounting is intentionally owned by this replay request rather
than by individual learner constructors.

`UniformSampler`, `PrioritizedSampler`, `SequenceSampler` and `DemoMixSampler`
are separate from `InMemoryReplayStore`. PER receives `PriorityUpdate` from a
learner, sequence sampling accepts only contiguous episode windows, and demo
mixing enforces explicit min/max fractions.

## Implement only what changes

An extension project may supply any of these protocols: `Learner`, `Policy`,
`ModelFactory`, `ReplayStore`, `Sampler`, `FeaturePipeline`, `Evaluator`,
`RunLogger`, `CheckpointCodec`, and an environment factory with `create(seed=)`.
Use `module:Symbol` paths in `run.yaml`; do not modify the TrackmaniaRL package for an
experiment.

`validate` does not start TrackMania. It resolves components, writes the
redacted manifest and runs a deterministic synthetic update. `train` requires
`components.environment`; it collects bounded episodes, writes compressed
reference-only artifacts, samples replay, applies updates, checkpoints and
runs an optional evaluator.

## Namespaces

| Namespace | Responsibility |
| --- | --- |
| `trackmaniarl.core` | contracts, run spec, trainer, data and reference replay |
| `trackmaniarl.builtins` | supported algorithms, models, buffers and feature components |
| `trackmaniarl.trackmania` | TrackMania environment collection adapter |
| `trackmaniarl.observability` | manifest, JSONL events, artifacts and optional adapters |
| `trackmaniarl.experiments` | evaluation suites and study strategies |
| `trackmaniarl.project` | generated local extension project |

Older module locations are internal migration details, not documented runtime
API or compatibility targets.
