# Migrating from TrackmaniaRL 1.x to 2.0

TrackmaniaRL 2.0 is a breaking runtime and checkpoint boundary. Do not point a
2.0 process at a 1.x run directory or rollout journal. Create a new generated
project, copy reviewed user components into it, validate the new configuration,
and use warm-start only where the guide explicitly permits it.

## RunSpec and component paths

RunSpec now requires `api_version: "2.0"` and rejects unknown fields. The public
flow for off-policy training is always:

```text
RunSpec 2.0 → actor spool → authenticated WAL → replay → learner → checkpoint
```

Generate a current project and compare configuration structurally rather than
editing the old YAML until it parses:

```bash
trackmaniarl init my-agent-v2 --template trackmania
cd my-agent-v2
uv sync
uv run trackmaniarl validate run.yaml
```

User components remain explicit `module:attribute` imports. Install the local
extension package before validation; no source-tree path injection or global
configuration fallback is supported.

The old monolithic replay module was split into stable imports under
`trackmaniarl.core.replay`. Select the store and sampler independently. A custom
sampler that feeds a value learner must now provide explicit `gamma`, `n_step`
and priority transition IDs in batch metadata. Sequential samplers provide
one-step context, a single n-step bootstrap at the trained position and the
full target observation history.

## Value models and algorithms

Scalar Q, QR-DQN, IQN and FQF now use
`trackmaniarl.algorithms.value_based:DiscreteValueLearner` with a composed
encoder, temporal core, head and value strategy. Replace a 1.x monolithic IQN
constructor with `CompositeValueModelFactory`; the SDK guide contains complete
component examples.

TQC renamed `top_quantiles_to_drop` to
`top_quantiles_to_drop_per_critic`. The value is per critic, matching the paper;
the learner removes `critic_count × value` atoms from the pooled target
mixture. The old name is rejected so a configuration cannot silently change
truncation strength.

Continuous SAC, REDQ, TQC and stable discrete SAC currently require
`training.sequence_length: 1`. Recurrent sequence training is supported by the
composed discrete-value path and PPO's separate on-policy sampler. Invalid
continuous sequence configurations fail during resolution rather than during a
tensor operation.

`SimbaV2Backbone`, `AdaptiveGradientClipper` and native Mamba remain opt-in
experimental components. Their presence in 2.0 does not make them default or a
guaranteed improvement. Compare each one against the identical seed, replay,
update budget and evaluation suite.

## Checkpoints and data

Distributed resume requires checkpoint schema 2.0, an exact architecture/run
fingerprint and the matching journal identity. It restores model and target
parameters, optimizers, value-strategy state, schedules, scaler/RNG state,
replay, sampler and runtime counters. A checkpoint older than the journal's
durable prune frontier is rejected; rollback cannot guess at already-pruned
rollouts.

A 1.x IQN checkpoint is not a valid resume or warm-start input. Establish a new
2.0 baseline, then use current compressed checkpoints for named-submodule
warm-starts. Review the generated tensor match report; only tensors with exact
names, shapes and dtypes are copied.

Local PPO checkpoints resume at an episode boundary. Environment and recurrent
actor state are not serializable, so partial on-policy replay is deliberately
discarded and the next episode receives a new index. Behavior cloning has a
separate v2 exact-resume artifact, `bc-latest.pt`, bound to its dataset
fingerprint.

The renamed offline-learning package is
`trackmaniarl.trackmania.imitation_learning`. Update imports from
`trackmaniarl.trackmania.behavior_cloning`; there is no compatibility alias.

## Trackmania and Openplanet

Install the signed **TrackmaniaRL Connect** plugin (`SAC_GetData`) through
Openplanet's Plugin Manager and enable School Mode. The bundled AngelScript is a
developer reference, not the installation path. The first-party telemetry
contract is exactly 33 little-endian `float32` values on `127.0.0.1:9000`; the
session protocol is version `2` JSON Lines on `127.0.0.1:9001`.

Live training now fails closed when the active map UID, readiness, map checksum
or geometry provenance does not match configuration. Run `trackmaniarl track
check --config run.yaml` before smoke/training. Existing boundary arrays without
map provenance should be re-recorded and rebuilt rather than relabelled.

## Observability

Local `events.jsonl` remains the complete event stream. W&B no longer exports
every flattened progress bin or raw heartbeat. It uses semantic axes
(`trainer/update`, `env/transitions`, `env/episode`, `eval/batch` and
`system/elapsed_s`) and a bounded metric catalogue. Existing dashboards using
the old `training/*`, `actor/*` or global-step series must be recreated from the
mapping in [Observability and W&B](observability.md).

## Migration gate

Before a long run, require all of the following:

```bash
uv run trackmaniarl validate run.yaml
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
```

Also verify one checkpoint/resume cycle, inspect the immutable manifest and run
the deterministic evaluation suite. Keep the 1.x run directory read-only until
the 2.0 warm-start or new baseline has passed those gates.
