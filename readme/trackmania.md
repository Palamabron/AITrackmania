# TrackMania lidar training and release workflow

For demonstration recording, behavior cloning, DAgger recovery, exact BC
resume and the required closed-loop gate, see the
[imitation-learning workflow](imitation-learning.md).

Install the released CLI, then create the game project only on a machine that
has Trackmania, Openplanet and a virtual gamepad driver:

```powershell
uv tool install --index https://download.pytorch.org/whl/cpu --with "torch==2.11.0+cpu" "trackmaniarl==1.1.0"
trackmaniarl init my-agent --template trackmania
cd my-agent
uv sync
uv run trackmaniarl validate run.yaml
```

> **Windows driver boundary:** the generated TrackMania project installs
> `vgamepad`, whose normal installer may install or repair the system-wide
> ViGEmBus driver. Review the pinned `vgamepad` source before provisioning a
> host. `VGAMEPAD_SKIP_VIGEMBUS_INSTALL=true` skips that driver installer for
> CI or an already provisioned machine; it does not provide a driver, so the
> gamepad backend still requires a compatible ViGEmBus installation. Use the
> keyboard backend when no virtual gamepad driver should be installed.

In Openplanet's **Plugin Manager**, install the signed
[**TrackmaniaRL Connect**](https://openplanet.dev/plugin/sac_getdata) plugin
(identifier `SAC_GetData`) and verify version **2.4.0**. Enable
[School Mode](https://openplanet.dev/docs/school-mode) on Openplanet 1.26.0 or
newer, which blocks online play and official leaderboard submissions while the
plugin is active. Do not copy the bundled developer-reference `.as` file into
`Scripts` or enable an unmanaged loose TrackmaniaRL script alongside the managed
plugin.

<p align="center">
  <img src="../docs/diagrams/trackmania-integration-preview.svg" alt="Trackmania and signed Openplanet plugin integration boundary" width="900">
</p>

[Editable integration diagram](../docs/diagrams/trackmania-integration.excalidraw) ·
[local preview](../docs/diagrams/trackmania-integration-preview.html)

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

Before starting separate `learner` or `actor` commands, generate one random
distributed token and store the same value as
`TRACKMANIARL_DISTRIBUTED_TOKEN` in each project's ignored `.env` file:

```powershell
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Local `train` and `smoke` still authenticate their actor/learner processes, but
the launcher generates an ephemeral token internally; they do not require this
environment variable.

Generated TrackMania agents select the project's tested PyTorch CUDA runtime by
default on Windows and Linux; a newer NVIDIA driver stays compatible. macOS
falls back to its normal PyPI/MPS Torch wheel. ROCm hosts require the matching
AMD Torch build.

The generated `.npz` is a structural placeholder only. Before training, record
the two map boundaries by hand and build a UID-bound asset. Do not reuse an
asset from another map:

```powershell
uv run trackmaniarl track record-boundary left assets/trackmaniarl-test-left.npy
uv run trackmaniarl track record-boundary right assets/trackmaniarl-test-right.npy
uv run trackmaniarl track build-geometry assets/trackmaniarl-test.geometry.npz --left assets/trackmaniarl-test-left.npy --right assets/trackmaniarl-test-right.npy --map-uid TRACKMANIARL_TEST_MAP_UID --map-path maps/trackmaniarl-test.Map.Gbx
```

Replace `TRACKMANIARL_TEST_MAP_UID` with the UID reported by `track check` and
set that same UID in `environment.kwargs.config.expected_map_uid`,
`feature_pipeline.kwargs.config.expected_map_uid` and
`evaluation.maps[].expected_map_uid`. Load that local `.Map.Gbx` manually in
Trackmania, enter it with a visible vehicle, and run:

```powershell
uv run trackmaniarl track check --config run.yaml
uv run trackmaniarl smoke run.yaml --transitions 100
uv run trackmaniarl train run.yaml
```

Openplanet does not expose a documented safe API for loading an arbitrary local
map. The plugin's second localhost port (default `9001`) therefore verifies the
already loaded UID and protocol before every training, smoke and evaluation
episode, then confirms a ready local player after controller reset. A timeout,
disconnect, UID mismatch or readiness rejection terminates the actor with a
failing process status. The protocol does not expose the plugin package's
signature or version; verify those properties in Plugin Manager.

The reference baseline is a 78-action dueling IQN (`13` steering levels × `2`
gas levels × no brake, full brake, or timed brake tap), not TQC. With default
feature settings, each observation contains 20 normalized telemetry values, a
`[4, 60]` car-local boundary tensor and a 60-element validity mask, all derived
from the documented 33-field `SAC_GetData` packet and the geometry
asset. The four lidar channels are the lateral/forward coordinates of the left
and right boundaries. The local frame comes from `api.Position` and `vis.Dir`;
it does not require aim-yaw telemetry. TQC remains an optional example only.

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
`encoder`, `temporal` and compatible `head` tensors from a current compressed
checkpoint. Pre-2.0 checkpoints are rejected, and tensors with changed names,
shapes or dtypes are reported without being copied. Preserve the generated
match report with the experiment artifacts.

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

Select the model explicitly in `run.yaml` and set
`training.sequence_length > 1`. In the composed replay path, keep
`LidarFeaturePipeline` config `history_length: 1`: the sampler creates `[B,T,...]`
sequences, `FrameBatchAdapter` flattens their frames for the sensor encoder and
the temporal core receives the restored sequence. Do not also stack history in
the feature pipeline.

```yaml
components:
  model_factory:
    class_path: trackmaniarl.models.factory:CompositeValueModelFactory
    kwargs:
      encoder:
        class_path: trackmaniarl.trackmania.encoders:LidarSensorEncoder
        kwargs:
          config: {telemetry_dim: 20, spatial_bins: 12, output_dim: 256}
      temporal:
        class_path: trackmaniarl.models.temporal:MambaTemporalCore
        kwargs: {input_dim: 256, backend: auto, d_state: 16, d_conv: 4, expand: 2}
      head:
        class_path: trackmaniarl.models.heads:ImplicitQuantileHead
        kwargs:
          config: {feature_dim: 256, action_count: 78, cosine_count: 64, dueling: true}
      strategy:
        class_path: trackmaniarl.models.strategies:LearnedFractionStrategy
        kwargs: {feature_dim: 256, fraction_count: 32}
  learner:
    class_path: trackmaniarl.algorithms.value_based:DiscreteValueLearner
    kwargs: {burn_in: 4}
training:
  sequence_length: 16
```

The sensor encoder processes the 16 replay frames as one vectorized `B*T`
batch. The Mamba core consumes `[B,T,D]`; `burn_in: 4` builds its initial
recurrent state without gradients and excludes those positions from losses and
priorities.

The optional dependency is imported only when the native backend is probed, so
normal imports, GRU runs and Pure PyTorch Mamba remain independent of
`mamba-ssm`. On Windows, CPU or an unsupported CUDA build, use `backend: torch`
or let `auto` record its fallback. Treat every new deployment platform as
unsupported until it passes the bounded live smoke test on that exact host;
offline contract tests alone are not evidence of game compatibility.

Every run writes `manifest.json`, versioned `events.jsonl`, compressed episode
artifacts, checkpoints and study records. Resume a stopped run with:

```bash
uv run trackmaniarl resume run.yaml artifacts/trackmania-iqn/checkpoints/distributed-update-XXXXXXXX.pt
```

Replace `trackmania-iqn` if your `run_id` differs.

W&B is optional. The generated project logs locally until you run
`uv add "trackmaniarl[trackmania,distributed,wandb]"` and add an
explicit `WandbTracker` under `components.additional_loggers`. Supply
`WANDB_API_KEY` only through a private environment or ignored `.env` file.
The generated project retains its vetted `vgamepad` source during this update;
an existing project must retain the same direct source pin documented in the
[installation guide](../README.md#install-and-create-an-agent).

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
with periodic checkpoints and at least one successful `trackmaniarl resume`.
Run the final benchmark against the newer post-resume checkpoint, then generate
`soak-report.json` with `scripts/verify_soak.py`; the verifier requires that
`evaluation.json` be bound to that checkpoint and rejects every telemetry or
controller error. A failed benchmark or soak blocks release.

## Connection troubleshooting

- **Port 9000 refuses the connection:** keep only signed TrackmaniaRL Connect
  2.4.0 enabled in Plugin Manager, enable School Mode, and enter the local map.
- **Port 9000 connects but sends no complete frame:** make sure a local vehicle
  is visible. The supported plugin waits for real player and vehicle-visual
  state instead of zero-filling missing fields.
- **Port 9001, protocol or readiness fails:** reload the managed plugin and
  return to the local map. Protocol 2 and a ready local player are required.
- **The active UID differs:** use the UID printed by `track check`, replace all
  three UID settings, and rebuild geometry from that exact `.Map.Gbx`.
- **Geometry checksum is missing or different:** re-run `track build-geometry`
  with `--map-path`; the generated `.npz` is intentionally only a placeholder.
- **Reset times out:** confirm that the configured `gamepad` or `keyboard`
  backend actually restarts the race timer before increasing any timeout.
