# TMRL

TMRL is a TrackMania reinforcement-learning library with ready-to-use
algorithms, model families, replay components and feature pipelines. It also
lets a project replace any one of those components through an explicit import
path. Users should be able to train a bundled baseline first, then change only
the piece they are researching.

## One cross-platform workflow

The commands are identical on Windows, Linux, WSL and CI:

```bash
uv sync --group dev --extra algorithms
uv run tmrl init my-trackmania-agent
cd my-trackmania-agent
uv sync
uv run tmrl validate run.yaml
uv run tmrl train run.yaml
```

`tmrl init` creates a commented, installable agent project. `tmrl validate`
checks imports, contracts and a synthetic update without starting the game.
`tmrl train` owns collection, replay updates, local events, artifacts,
checkpoints and optional evaluation.

## Runtime

```text
run.yaml -> RunSpec -> resolved components -> Trainer
                                          -> collect -> replay -> update
                                          -> checkpoint -> evaluate
                                          -> manifest + JSONL + artifacts
```

There is no global runtime configuration, feature-flag routing or mandatory
external tracker. A run is fully described by `run.yaml` and its referenced,
installed Python components.

## Bundled components

`tmrl.builtins` is the supported catalogue for components included with TMRL:

- algorithms: `soft_actor_critic`, `randomized_ensemble_sac`,
  `truncated_quantile_critic`, `implicit_quantile_q_learning` and
  `stable_discrete_soft_actor_critic`;
- models: replaceable encoders, actor heads and critics;
- replay: uniform, prioritized, episode-sequence and demonstration-mixing samplers;
- TrackMania collection adapters plus typed telemetry and track-geometry model inputs.

Use `tmrl.trackmania` for the neutral TrackMania collection adapter. Game-specific
environment factories belong in the local extension project, so offline validation
does not require a running game or optional game dependencies.

Use the learner class directly in a component spec, for example
`tmrl.algorithms.implicit_quantile_q_learning:ImplicitQuantileQLearning`.
A learner receives a typed `TrainingBatch`, including n-step bootstrap discounts,
separate termination/truncation flags, PER weights and stable transition IDs.

## Extensions and observability

The stable contracts in `tmrl.core` are `Learner`, `Policy`, `ModelFactory`,
`ReplayStore`, `Sampler`, `FeaturePipeline`, `Evaluator`, `RunLogger` and
`CheckpointCodec`. Hot-path objects are slots dataclasses and PyTrees;
Pydantic is only used at the configuration boundary.

Every run records a redacted immutable manifest, local JSONL events, checkpoints
and bounded compressed episode artifacts. W&B, Captum, Gemini and Optuna are
optional extras:

```bash
uv sync --extra wandb --extra explain --extra orchestrator
```

Read the [SDK guide](readme/sdk.md) for the component schema and a built-in
run example, and the [TrackMania workflow](readme/trackmania.md) for the
optional OpenPlanet/gamepad integration and release smoke checklist.

## Development

Use the same commands on Windows and Linux; Poe is installed by the `dev` group:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```
