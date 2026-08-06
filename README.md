# TMRL

TMRL is a TrackMania reinforcement-learning library with ready-to-use
algorithms, model families, replay components and feature pipelines. It also
lets a project replace any one of those components through an explicit import
path. Users should be able to train a bundled baseline first, then change only
the piece they are researching.

## One cross-platform workflow

The commands are identical on Windows, Linux, WSL and CI:

```bash
uv sync
uv run tmrl init my-trackmania-agent
cd my-trackmania-agent
uv sync
uv run tmrl validate run.yaml
uv run tmrl train run.yaml
```

`tmrl init` creates a commented, installable agent project. `tmrl validate`
checks imports, contracts and a synthetic update without starting the game.
`tmrl train` starts a coordinator/learner and one local actor as independent
Windows-safe `spawn` processes. Collection stays continuous while the learner
updates replay and publishes policy snapshots asynchronously.

The TrackMania project uses a fresh API `1.2` run (`v6`); do not reuse an old
immutable artifact directory. With the game and OpenPlanet plugin running, use
the bounded integration check:

```bash
uv run tmrl track check
uv run tmrl smoke run.yaml --transitions 100
```

On Windows (the platform that runs TrackMania), `uv sync` installs the locked
CUDA PyTorch wheel by default via `[tool.uv.sources]`. On other platforms it
installs the CPU wheel. The CUDA wheel does not need the same locally installed
CUDA Toolkit version, and a newer NVIDIA driver remains compatible. ROCm hosts
require the matching AMD Torch build, while macOS MPS uses the normal PyPI Torch
wheel. `device: auto` then resolves CUDA, ROCm, MPS or CPU from the installed
Torch build and fails early when visible accelerator hardware cannot be used.

The smoke command starts the same local async learner/actor pair as training,
checks a live policy refresh, and writes a checkpoint.

## Runtime

```text
run.yaml -> coordinator/learner -> SQLite WAL -> replay -> update -> checkpoint
              ^       |
              |       +---- safetensors policy snapshot
              |
              +---- local or remote actors -> durable rollout spool
```

There is no global runtime configuration, feature-flag routing or mandatory
external tracker. A run is fully described by `run.yaml` and its referenced,
installed Python components.

For multiple machines, put the same `TMRL_DISTRIBUTED_TOKEN` in `.env` and use
an encrypted tunnel. The learner intentionally accepts loopback connections
only, so its bearer token and rollout data never traverse the network in clear
text:

```bash
# training machine
uv run tmrl learner run.yaml --bind 127.0.0.1:8787

# each TrackMania machine: create a tunnel to the training machine first
ssh -N -L 8787:127.0.0.1:8787 TRAINING_MACHINE
uv run tmrl actor run.yaml --connect 127.0.0.1:8787 --actor-id PC-1
```

Only the learner needs W&B credentials. Training loads `WANDB_API_KEY` from
the environment or project `.env`; a separate `wandb login` is unnecessary
when that variable is already present.

The handshake rejects mismatched configs, models, feature/action definitions,
map UIDs and geometry. Rollouts use Protobuf/gRPC with Zstandard compression;
network model state is encoded with safetensors and never pickle.

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

For the concrete `tmrl-test` OpenPlanet installation, telemetry ports, map
preparation, boundary recording and geometry commands, see the
[agent OpenPlanet guide](my-trackmania-agent/openplanet/README.md).

## Development

Use the same commands on Windows and Linux; Poe is installed by the `dev` group:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```
