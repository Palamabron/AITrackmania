# TrackMania lidar training and release workflow

For demonstration recording, behavior cloning, DAgger recovery, exact BC
resume and the required closed-loop gate, see the
[imitation-learning workflow](imitation-learning.md).

Install the game integration only on a machine that has TrackMania, OpenPlanet,
the compatible telemetry plugin, and a virtual gamepad driver:

```bash
uv sync
uv run trackmaniarl init --template trackmania my-agent
cd my-agent
uv sync
uv run trackmaniarl validate run.yaml
uv run trackmaniarl smoke run.yaml
uv run trackmaniarl train run.yaml
```

The generated `run.yaml` selects the control device explicitly:

```yaml
components:
  environment:
    kwargs:
      config:
        control_backend: gamepad
```

Use `gamepad` for analog steering and rumble-based collision detection. Select
`keyboard` when a virtual gamepad is unavailable. The keyboard backend converts
analog model output to digital gas/brake and left/right input, with a steering
dead zone, and cannot provide rumble collision signals. The choice belongs to
the environment, not the model, so the same policy can drive either backend;
expect different driving dynamics after analog-to-digital conversion.

Before `smoke`, `train`, `learner` or `actor`, generate one random distributed
token and store it as `TRACKMANIARL_DISTRIBUTED_TOKEN` in the project's ignored
`.env` file:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

The same requirement applies to local training because the local actor and
learner are separate authenticated processes.

Generated TrackMania agents select the project's tested PyTorch CUDA runtime by
default on Windows and Linux; a newer NVIDIA driver stays compatible. macOS
falls back to its normal PyPI/MPS Torch wheel. ROCm hosts require the matching
AMD Torch build.

The generated `.npz` is a structural placeholder only. Before training, record
the two map boundaries by hand and build a UID-bound asset. Do not reuse an
asset from another map:

```bash
uv run trackmaniarl track record-boundary left assets/trackmaniarl-test-left.npy
uv run trackmaniarl track record-boundary right assets/trackmaniarl-test-right.npy
uv run trackmaniarl track build-geometry assets/trackmaniarl-test.geometry.npz \
  --left assets/trackmaniarl-test-left.npy --right assets/trackmaniarl-test-right.npy \
  --map-uid <trackmaniarl-test-map-uid> --map-path maps/trackmaniarl-test.Map.Gbx
```

Set that same UID in both `feature_pipeline.kwargs.expected_map_uid` and
`evaluation.maps[].expected_map_uid`. Before a live evaluation, load that local
`.Map.Gbx` manually in TrackMania. The documented OpenPlanet API exposes the
active map UID but not a safe API to load an arbitrary local map. The bundled
plugin's second local command port (default `9001`) therefore verifies the
already loaded UID before every episode, then confirms an active player after
the controller reset using protocol version `2`; a timeout, disconnect or UID
mismatch aborts the run.

The reference baseline is a 78-action dueling IQN (`13` steering levels × `2`
gas levels × continuous brake, full brake, or brake tap), not TQC. Its
observation is a fixed 20-feature projection of the documented 33-field
`TrackmaniaRL_GrabData` packet, plus 15 left + 15 right car-local boundary samples.
The local frame comes from `api.Position` and `vis.Dir`; it does not require
aim-yaw telemetry. TQC remains an optional example only.

Model factories publish their train-time contract and learners publish the
contracts they accept. Composed Q/QR-DQN/IQN/FQF models expose `discrete_value`, the telemetry
TQC baseline exposes `continuous_quantile_actor_critic`, and behavior cloning
exposes `categorical_policy`. `trackmaniarl validate` rejects a mismatched pair
before model setup instead of failing later on a missing head. Models remain
interchangeable between algorithms that consume the same contract; algorithms
with different objectives require a matching model head.

## Value algorithm selection

All discrete value experiments use `DiscreteValueLearner`. The YAML composition
selects the algorithm:

| Experiment | Head | Strategy |
| --- | --- | --- |
| scalar baseline | `ScalarQHead` | `ScalarValueStrategy` |
| QR-DQN | `FixedQuantileHead` | `FixedQuantileStrategy` |
| IQN | `ImplicitQuantileHead` | `RandomQuantileStrategy` |
| FQF | `ImplicitQuantileHead` | `LearnedFractionStrategy` |

For recurrent experiments select `GruTemporalCore` or `MambaTemporalCore`, set
`training.sequence_length`, and configure learner `burn_in`. The lidar encoder
is frame-only: the model vectorizes `[B,T]` into `[B*T]` before encoding and
restores `[B,T,D]` before the temporal core.

FQF creates its own fraction optimizer from
`LearnedFractionStrategy.auxiliary_parameters()`. Monitor fraction entropy and
boundary spacing as well as TD/quantile metrics. The target network contains
its own fraction proposal network, and target quantiles are evaluated only for
the action selected by online Double-DQN.

To initialize FQF from a proven IQN run, use the warm-start loader for named
`encoder`, `temporal` and compatible `head` tensors. Do not use a 1.x IQN
checkpoint as resume state and do not silently copy mismatched shapes. Preserve
the generated match report with the experiment artifacts.

## Mamba temporal core

`MambaTemporalCore` is an opt-in temporal component. Use it as a named experiment
after recording an identical GRU baseline with the same seed, replay, update
budget and evaluation suite.

The `torch` backend is portable across Windows, Linux, CPU and CUDA. The
`native` backend requires a working `mamba-ssm` selective-scan kernel; `auto`
probes native forward/backward and records the Pure PyTorch fallback reason.
Both backends use the same model parameters and checkpoint fingerprint.

Install the `mamba` extra only when testing the native kernel. The `torch`
backend is implemented locally with standard Torch operations and works without
`mamba-ssm`:

```bash
uv sync --extra mamba
```

Select the model explicitly in `run.yaml` and make `training.sequence_length`
match `history_length`:

```yaml
components:
  model_factory:
    class_path: trackmaniarl.models.factory:CompositeValueModelFactory
    kwargs:
      encoder:
        class_path: trackmaniarl.trackmania.encoders:LidarSensorEncoder
        kwargs: {telemetry_dim: 26, spatial_bins: 12, output_dim: 256}
      temporal:
        class_path: trackmaniarl.models.temporal:MambaTemporalCore
        kwargs: {input_dim: 256, backend: auto, d_state: 16, d_conv: 4, expand: 2}
      head:
        class_path: trackmaniarl.models.heads:ImplicitQuantileHead
        kwargs: {feature_dim: 256, action_count: 78, cosine_count: 64, dueling: true}
      strategy:
        class_path: trackmaniarl.models.strategies:LearnedFractionStrategy
        kwargs: {feature_dim: 256, fraction_count: 32}
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs: {burn_in: 4}
training:
  sequence_length: 16
```

The sensor encoder processes the 16 frames as one vectorized `B*T` batch. The
Mamba core consumes `[B,T,D]`; `burn_in: 4` builds its initial recurrent state
without gradients and excludes those positions from losses and priorities.

The optional dependency is imported only when the native backend is probed, so
normal imports, GRU runs and Pure PyTorch Mamba remain independent of
`mamba-ssm`. On Windows, CPU or an unsupported CUDA build, use `backend: torch`
or let `auto` record its fallback. Treat every new deployment platform as
unsupported until it passes the bounded live smoke test on that exact host;
offline contract tests alone are not evidence of game compatibility.

Every run writes `manifest.json`, versioned `events.jsonl`, compressed episode
artifacts, checkpoints and study records. Resume a stopped run with:

```bash
uv run trackmaniarl resume run.yaml artifacts/<run-id>/checkpoints/distributed-update-XXXXXXXX.pt
```

`trackmaniarl smoke` is the required Windows preflight. It collects a bounded number of
real actions, completes at least one update, verifies a live policy refresh,
and restores the produced checkpoint. It will operate the virtual gamepad:

```bash
uv run trackmaniarl smoke run.yaml --transitions 100
```

The release benchmark is deterministic only in the sense that it repeats the
same local map and assets. It does not claim game-engine seed control. It uses
the `trials_per_map`, `min_finish_rate`, and `target_median_s` thresholds in
`run.yaml`, writes `evaluation.json` with per-trial status, latency/FPS and map
UID, and fails when any configured acceptance threshold is missed:

```bash
uv run trackmaniarl benchmark run.yaml artifacts/trackmania-iqn-lidar/checkpoints/distributed-update-XXXXXXXX.pt
```

The remaining manual release gate is a four-hour Windows soak on the real game,
with periodic checkpoints and at least one successful `trackmaniarl resume`. A failed
benchmark or soak blocks release.
