# TrackmaniaRL

TrackmaniaRL is an independent reinforcement-learning library for training
agents in Trackmania 2020. It provides ready-to-use
algorithms, model families, replay components and feature pipelines. It also
lets a project replace any one of those components through an explicit import
path. Users should be able to train a bundled baseline first, then change only
the piece they are researching.

The project originated from TMRL and has since been substantially redesigned.
It is not affiliated with or endorsed by Ubisoft, Nadeo, or the TMRL
maintainers. Trackmania is a trademark of Nadeo/Ubisoft. See [NOTICE](NOTICE)
for attribution.

## One cross-platform workflow

The commands are identical on Windows, Linux, WSL and CI:

```bash
git clone https://github.com/Palamabron/AITrackmania.git
cd AITrackmania
uv sync
uv run trackmaniarl init my-trackmania-agent
cd my-trackmania-agent
uv sync
uv run trackmaniarl validate run.yaml
uv run trackmaniarl train run.yaml
```

`trackmaniarl init` creates a commented, installable agent project. `trackmaniarl validate`
checks imports, contracts and a synthetic update without starting the game or contacting
optional remote trackers.
`trackmaniarl train` starts a coordinator/learner and one local actor as independent
Windows-safe `spawn` processes. Collection stays continuous while the learner
updates replay and publishes policy snapshots asynchronously.

The TrackMania project uses a fresh API `1.2` run (`v6`); do not reuse an old
immutable artifact directory. With the game and OpenPlanet plugin running, use
the bounded integration check:

```bash
uv run trackmaniarl track check
uv run trackmaniarl smoke run.yaml --transitions 100
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

To start from a published release instead of a checkout, install the package,
then generate the extension project:

```bash
uv tool install "trackmaniarl[distributed]"
trackmaniarl init my-trackmania-agent
cd my-trackmania-agent
uv sync
trackmaniarl validate run.yaml
```

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

For multiple machines, put the same `TRACKMANIARL_DISTRIBUTED_TOKEN` in `.env` and use
an encrypted tunnel. The learner intentionally accepts loopback connections
only, so its bearer token and rollout data never traverse the network in clear
text:

```bash
# training machine
uv run trackmaniarl learner run.yaml --bind 127.0.0.1:8787

# each TrackMania machine: create a tunnel to the training machine first
ssh -N -L 8787:127.0.0.1:8787 TRAINING_MACHINE
uv run trackmaniarl actor run.yaml --connect 127.0.0.1:8787 --actor-id PC-1
```

Only the learner needs W&B credentials. Training loads `WANDB_API_KEY` from
the environment or project `.env`; a separate `wandb login` is unnecessary
when that variable is already present.

The handshake rejects mismatched configs, models, feature/action definitions,
map UIDs and geometry. Rollouts use Protobuf/gRPC with Zstandard compression;
network model state is encoded with safetensors and never pickle.

## Bundled components

`trackmaniarl.builtins` is the supported catalogue for components included with TrackmaniaRL:

- algorithms: `soft_actor_critic`, `randomized_ensemble_sac`,
  `truncated_quantile_critic`, `implicit_quantile_q_learning` and
  `stable_discrete_soft_actor_critic`;
- models: replaceable encoders, actor heads and critics;
- replay: uniform, prioritized, episode-sequence and demonstration-mixing samplers;
- TrackMania collection adapters plus typed telemetry and track-geometry model inputs.

Use `trackmaniarl.trackmania` for the neutral TrackMania collection adapter. Game-specific
environment factories belong in the local extension project, so offline validation
does not require a running game or optional game dependencies.

Use the learner class directly in a component spec, for example
`trackmaniarl.algorithms.implicit_quantile_q_learning:ImplicitQuantileQLearning`.
A learner receives a typed `TrainingBatch`, including n-step bootstrap discounts,
separate termination/truncation flags, PER weights and stable transition IDs.

## Extensions and observability

The stable contracts in `trackmaniarl.core` are `Learner`, `Policy`, `ModelFactory`,
`ReplayStore`, `Sampler`, `FeaturePipeline`, `Evaluator`, `RunLogger` and
`CheckpointCodec`. Hot-path objects are slots dataclasses and PyTrees;
Pydantic is only used at the configuration boundary.

Every run records a redacted immutable manifest, local JSONL events, checkpoints
and bounded compressed episode artifacts. W&B, Captum, Gemini and Optuna are
optional extras:

```bash
uv sync --extra wandb --extra explain --extra orchestrator
```

Read the [SDK guide](https://github.com/Palamabron/AITrackmania/blob/main/readme/sdk.md)
for the component schema and a built-in run example, and the
[TrackMania workflow](https://github.com/Palamabron/AITrackmania/blob/main/readme/trackmania.md) for the
optional OpenPlanet/gamepad integration and release smoke checklist.

For the concrete `trackmaniarl-test` OpenPlanet installation, telemetry ports,
map preparation, boundary recording and geometry commands, see the
[agent OpenPlanet guide](https://github.com/Palamabron/AITrackmania/blob/main/my-trackmania-agent/openplanet/README.md).

## Development

Use the same commands on Windows and Linux; Poe is installed by the `dev` group:

```bash
uv run poe fmt
uv run poe types
uv run poe test
```
