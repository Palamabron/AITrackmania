# TrackmaniaRL SDK Guide

TrackmaniaRL has one runtime: an explicit `run.yaml` is parsed into `RunSpec`, its
components are imported, then the local or distributed coordinator performs
collection and off-policy updates. The same commands work in PowerShell, bash,
WSL and CI.

```bash
uv tool install trackmaniarl
trackmaniarl init my-trackmania-agent --template trackmania
cd my-trackmania-agent
uv sync
uv run trackmaniarl validate run.yaml
```

For library development from a clone, follow
[development.md](development.md#repository-setup) instead. Do not mix changes to
the reusable package with one experiment: the generated project is the intended
place for custom components and run configurations.

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

Bundled model factories declare a `ModelContract`, and bundled learners declare
the contracts they accept. This keeps an encoder choice independent from runtime
or controller selection while preventing invalid objective/head combinations.
For example, both the GRU lidar model and the Mamba lidar model implement
`discrete_quantile` and can be used by any learner accepting that contract;
TQC requires `continuous_quantile_actor_critic`, and behavior cloning requires
`categorical_policy`. Custom components without declarations retain structural
protocol validation, while declared incompatible pairs fail during resolution.

## Implement only what changes

An extension project may supply any of these protocols: `Learner`, `Policy`,
`ModelFactory`, `ReplayStore`, `Sampler`, `FeaturePipeline`, `Evaluator`,
`RunLogger`, `CheckpointCodec`, and an environment factory with `create(seed=)`.
Use `module:Symbol` paths in `run.yaml`; do not modify the TrackmaniaRL package for an
experiment.

The normal extension loop is:

1. generate an installable project with `trackmaniarl init`;
2. implement or subclass one component under `src/<package>/`;
3. point the matching `components` entry at its import path;
4. add a deterministic test under the generated project's `tests/`;
5. run `uv run trackmaniarl validate run.yaml`;
6. run the game connection check and bounded smoke test only when the offline
   contract passes.

For example, a custom feature pipeline implements transformation for actor-time
observations and collation for replay samples:

```python
from typing import Any

import numpy as np

from trackmaniarl.core.data import Transition


class SpeedFeaturePipeline:
    def transform_observation(self, observation: Any) -> np.ndarray:
        return np.asarray([observation["speed"]], dtype=np.float32)

    def collate(self, transitions: list[Transition]) -> dict[str, np.ndarray]:
        return {
            "observations": np.stack([item.observation for item in transitions]),
            "actions": np.asarray([item.action for item in transitions]),
            "rewards": np.asarray([item.reward for item in transitions], dtype=np.float32),
        }
```

Reference it without editing the library:

```yaml
components:
  feature_pipeline:
    class_path: my_trackmania_agent.features:SpeedFeaturePipeline
```

The exact batch structure is a contract between the pipeline/sampler and the
learner. Keep it typed and deterministic, and test one synthetic transition
round trip before a live run.

`validate` does not start TrackMania. It resolves components, writes the
redacted manifest and runs a deterministic synthetic update. `train` requires
`components.environment`; it collects bounded episodes, writes compressed
reference-only artifacts, samples replay, applies updates, checkpoints and
runs an optional evaluator.

`validate` is game-free, not code-free: it imports every configured component
and invokes constructors and a learner update. Never validate an untrusted
configuration or Python package.

## Component responsibilities

| Contract | Required behavior | State to persist |
| --- | --- | --- |
| `Policy` | deterministic or exploratory inference | model/policy tensors when replicated |
| `Learner` | setup, update, policy access and state round trip | model, optimizer, schedules and algorithm statistics |
| `ModelFactory` | construct the configured train-time model | none unless the factory is stateful |
| `EnvironmentFactory` | create one isolated environment per seed | environment state is normally episode-local |
| `FeaturePipeline` | transform one observation and collate transitions | normalization statistics, if learned |
| `ReplayStore` | append and retrieve transitions by monotonic ID | stored transitions and ID watermark |
| `Sampler` | select batches and apply priority updates | RNG, priorities and annealing state |
| `Evaluator` | evaluate a policy against the configured suite | best-result/selection state if it affects training |
| `RunLogger` | receive neutral events and close resources | remote run identity if resume needs it |
| `CheckpointCodec` | atomically save and safely load learner state | format/version metadata |

If resume behavior would change after recreating a component, its relevant state
belongs in the checkpoint. Do not hide configuration in module globals or read
environment variables inside hot-path objects.

## RunSpec layout

The root fields are intentionally small:

- `api_version`: serialized contract version, currently `1.2`;
- `run_id`, `seed`, `artifacts_dir`: identity and local output ownership;
- `components`: import paths and constructor keyword arguments;
- `training`: batch, replay, update, evaluation and checkpoint schedule;
- `distributed`: chunking, timeouts, message limits, exploration profiles and
  the name of the token environment variable;
- `evaluation`: immutable map/geometry suite and acceptance thresholds;
- `metadata`: descriptive, serializable experiment metadata only.

Unknown fields fail validation. Start a new run directory when immutable
configuration, component source or contracts change; resume only a compatible
run.

## Adding code to the library itself

Promote a component from an extension project only when it is reusable across
runs. Put generic mechanisms in `core`, algorithms in `algorithms`, network
modules in `models`, and Trackmania-only behavior in `trackmania`. Expose a
stable built-in entry point only after deterministic contract, checkpoint and
resume coverage. Distributed changes additionally need idempotency, size-limit
and slow-learner tests.

See [architecture.md](architecture.md) for dependency direction and
[development.md](development.md#adding-a-public-component) for the acceptance
checklist.

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
