# Imitation learning

`trackmaniarl.trackmania.imitation_learning` is the public package for offline
imitation workflows. Behavior cloning (BC) is its supervised training method.
It shares the lidar
encoder and temporal components with value-based models, but it does not write
rollouts to WAL or replay. Use BC to initialize a policy, verify it in closed
loop, then transfer compatible encoder and temporal weights into an RL run.

<p align="center">
  <img src="../docs/diagrams/imitation-learning-preview.svg" alt="Behavior cloning, closed-loop gate and RL warm-start" width="900">
</p>

[Editable diagram](../docs/diagrams/imitation-learning.excalidraw) ·
[local preview](../docs/diagrams/imitation-learning-preview.html)

The timing path has a separate diagram because timestamp calibration, label
lead and closed-loop correction solve different problems:

<p align="center">
  <img src="../docs/diagrams/demonstration-timing-preview.svg" alt="Demonstration telemetry, decision windows, action lead and replay offset" width="900">
</p>

[Editable timing diagram](../docs/diagrams/demonstration-timing.excalidraw) ·
[local timing preview](../docs/diagrams/demonstration-timing-preview.html)

## Data contract

Record at least three complete laps on one map. Each demonstration records its
map UID, geometry hash, action timing, telemetry frames, controls and finish
time. `bc-train` rejects:

- a different map, geometry, action set or decision interval;
- recordings that start late, have sparse telemetry or use non-frame-start
  control alignment;
- demonstrations containing actions outside `compact_action_ids`;
- fewer than three complete human laps.

The split is deterministic by seed and occurs at lap level. The fastest lap is
kept in training and train/validation identities are always disjoint. Recovery
archives are split by episode when at least three episodes exist. One or two
recovery episodes are train-only, so they cannot make human-lap validation
optimistic.

Archives are NumPy files loaded with `allow_pickle=False`. Treat datasets from
other people as untrusted input and retain the driver's consent, provenance and
license outside the model checkpoint. Every BC run writes
`bc-dataset-manifest.json` with file hashes, sizes, action IDs, feature/model
configuration, split membership and a dataset fingerprint.

The current archive format stores `frames` with one more row than `actions`,
continuous `controls` with shape `(transitions, 3)`, `finish_time_s`,
`action_repeat_frames`, optional `decision_interval_ms`, and
`control_alignment`. Controls are `[gas, brake, steer]`; the discrete action
must equal deterministic quantization of that row. Race timestamps must be
strictly increasing, only the final frame may carry the finish flag, and the
finish metadata must agree with it within 50 ms.

Native recording targets 100 Hz telemetry: first frame at most 15 ms after the
race starts, median interval 8–12 ms, p95 at most 12 ms and maximum 20 ms. The
more general quality gate allows p95 at most 25 ms and maximum 50 ms. Sparse or
late recordings cannot preserve short steering or brake-tap inputs and are
rejected rather than silently resampled.

## Configuration

The RunSpec must use API 2.0, a feature pipeline without control inputs, a
categorical model factory and the BC learner. Starting from the generated
Trackmania configuration, make these entries agree (the replay, sampler,
evaluator and geometry settings can remain in place):

```yaml
api_version: "2.0"

components:
  learner:
    class_path: trackmaniarl.trackmania.imitation_learning:BehaviorCloningLearner
    kwargs:
      learning_rate: 3.0e-4
      validation_interval: 100
      early_stopping_patience: 30
      execution: {device: auto, precision: bfloat16}

  environment:
    class_path: trackmaniarl.trackmania.environment:OpenPlanetEnvironmentFactory
    kwargs:
      config:
        geometry_path: assets/trackmaniarl-test.geometry.npz
        expected_map_uid: <map-uid>
        compact_action_ids: [0, 1, 3, 39, 72, 73, 75]

  model_factory:
    class_path: trackmaniarl.trackmania.imitation_learning:LidarBehaviorCloningModelFactory
    kwargs:
      action_ids: [0, 1, 3, 39, 72, 73, 75]
      telemetry_dim: 17
      spatial_bins: 12
      history_length: 8
      previous_action_conditioning: false

  feature_pipeline:
    class_path: trackmaniarl.trackmania.features:LidarFeaturePipeline
    kwargs:
      geometry_path: assets/trackmaniarl-test.geometry.npz
      expected_map_uid: <map-uid>
      history_length: 8
      include_control_inputs: false

training:
  batch_size: 256
  metrics_interval_updates: 50
```

`action_ids` must exactly match
`components.environment.kwargs.config.compact_action_ids`, and the model's
`history_length`, `telemetry_dim` and `lidar_channels` must match the feature
pipeline output. The minimal configuration above produces 17 telemetry values
and four lidar channels. If
`previous_action_conditioning` is enabled, human and recovery data use expert
previous actions during training; inference uses the policy's previous
prediction. DAgger collection deliberately requires conditioning to be off.

Horizontal reflection is opt-in and only accepts the versioned local
8-channel lidar/46-feature telemetry schema. That schema requires local
velocity, track-relative, pace, racing-line, finish, dynamics and goal features
with control inputs excluded; set the model to `lidar_channels: 8`,
`telemetry_dim: 46` and `telemetry_group_dims: [23, 5, 4, 14]`. It mirrors
steering labels and known directional fields, preserves other tensor fields,
and fails explicitly for an incompatible schema.

## Commands

```powershell
uv run trackmaniarl track record-demo demonstrations --config run.yaml --count 3
uv run trackmaniarl validate run.yaml
uv run trackmaniarl bc-train run.yaml --demo demonstrations
# Requires the 8-channel/46-feature schema described above.
uv run trackmaniarl bc-train run.yaml --demo demonstrations --recovery recovery.npz --horizontal-flip-augmentation
uv run trackmaniarl bc-train run.yaml --demo demonstrations --recovery recovery.npz --resume artifacts/my-bc-run/checkpoints/bc-latest.pt
uv run trackmaniarl bc-benchmark run.yaml artifacts/my-bc-run/checkpoints/bc-best-validation.pt --trials 30
```

Replace `my-bc-run` with the configured `run_id`.

Training uses contiguous feature tensors, deterministic batch sampling,
class/sample/transition weighting, optional focal and steering losses, AMP when
requested, gradient clipping, ReduceLROnPlateau and early stopping. Validation
loss uses the global effective weight denominator, so changing validation
batch size does not change the result.

## Artifacts and metrics

- `manifest.json`: immutable redacted RunSpec and execution environment;
- `bc-dataset-manifest.json`: data and preprocessing attribution;
- `events.jsonl`: interval training and validation metrics;
- `checkpoints/bc-latest.pt`: exact-resume state, including RNG and trainer
  selection state;
- `checkpoints/bc-best-validation.pt`: best open-loop policy candidate.

Accuracy is exact compact-action accuracy. Balanced accuracy averages recall
over observed actions. Transition metrics cover action changes, steering
metrics collapse actions to left/neutral/right, and intervention metrics cover
teacher interventions. The control score ranks eligible open-loop candidates;
it is not evidence that the car drives safely or finishes.

Always run `bc-benchmark` before promotion. Compare finish rate first, then
median finish time/progress and intervention/recovery behavior, and use
open-loop metrics only as tie-breakers. A checkpoint that improves frame-level
accuracy may regress in closed loop because prediction errors compound.

By default, a failed `bc-benchmark` or `demo-benchmark` gate exits with failure.
Use `--report-only` only for diagnostics whose failure must not stop an outer
experiment script.

## Timing and latency calibration

TrackmaniaRL uses the race clock, not wall-clock arrival time, as the physical
label axis. For an observation selected at race time `t`:

```text
demonstration_action_lead_ms = L >= 0
label time = t + L
```

A positive lead chooses a future expert command as the label for the current
observation, so the learned policy switches earlier. `0` keeps the command
aligned to the transition's start frame. The parameter is a manual, constant
calibration; the library does not estimate control latency automatically.
Without aggregation, lookup uses the first recorded timestamp at or after
`t + L`. Beyond the recorded tail it deliberately holds the final action.

When `demonstration_control_aggregation: true`, each online decision window
`[t, t + decision_interval_ms)` instead integrates recorded gas, brake and
steering over `[t + L, t + decision_interval_ms + L)` and quantizes the mean
control. Brake taps use their physical duty duration. A shifted tail holds the
last recorded control. Aggregation requires:

- `control_alignment: frame_start` in the demonstration;
- `action_repeat_frames: 1`;
- a positive `decision_interval_ms` no greater than 250 ms;
- the `gamepad` controller backend;
- the same decision interval for online training and data preparation.

`demo-benchmark --action-offset-ms O` is a separate open-loop diagnostic. It
adds `O` to the timestamps at which replay switches actions. A positive offset
delays switching; a negative offset advances it. It does not rewrite training
labels, and it cannot be combined with phase-locked or trajectory-tracking
replay. Once the best diagnostic offset is understood, express an anticipatory
training correction with the non-negative label lead and record the decision,
instead of leaving an unexplained benchmark-only offset.

A reproducible calibration procedure is:

1. Record at least three clean laps on the same map and verify their cadence.
2. Hold the RunSpec, map, controller backend and demo fixed. Run at least three
   open-loop trials at offsets such as `-40`, `-20`, `0`, `+20`, `+40` ms.
3. Compare finish rate first, then median finish time and action-transition
   agreement. Expand around the best signed offset with a finer grid.
4. Repeat on the retained laps; reject a value that works for only one lap.
5. Configure a non-negative `demonstration_action_lead_ms`, rebuild the BC
   dataset, train with the same seed/budget and run the closed-loop BC gate.
6. Keep the zero-lead baseline and report the exact grid, trial count and
   confidence-relevant spread. Do not call a single successful replay automatic
   latency estimation.

Example diagnostic grid:

```powershell
uv run trackmaniarl demo-benchmark run.yaml demonstrations/lap-01.npz --trials 3 --action-offset-ms -20
uv run trackmaniarl demo-benchmark run.yaml demonstrations/lap-01.npz --trials 3 --action-offset-ms 0
uv run trackmaniarl demo-benchmark run.yaml demonstrations/lap-01.npz --trials 3 --action-offset-ms 20
```

Four mechanisms must not be conflated:

- **timestamp replay** selects commands only from race time and exposes timing
  drift directly;
- **phase locking** matches current feature state to a nearby reference phase;
- **trajectory tracking** finds a nearby forward world-space reference state,
  takes feed-forward expert controls at an optional lead, and adds steering
  feedback from lateral, heading and lateral-velocity error;
- **DAgger** visits student-induced states and asks the closed-loop trajectory
  teacher for labels.

Phase locking and trajectory tracking can recover from state drift, but that
can hide a raw timestamp error. Use them as separate diagnostic comparisons.
Action-label lead changes the supervised dataset; DAgger changes the state
distribution.

## DAgger, recovery and trajectory tools

[DAgger](https://proceedings.mlr.press/v15/ross11a.html) addresses compounding
covariate shift by collecting states visited by the student and labelling them
with the trajectory-tracking teacher. The current command requires a BC policy
without previous-action conditioning, a compact action set, lidar features and
a configured evaluation map:

```powershell
uv run trackmaniarl dagger-collect run.yaml artifacts/my-bc-run/checkpoints/bc-best-validation.pt demonstrations/lap-01.npz recovery/dagger.npz --episodes 10
```

`--teacher-probability` mixes teacher control into collection;
`--intervention-error` triggers an intervention from tracking error; and
`--action-lead-ms` controls the teacher's non-negative feed-forward look-ahead.
The resulting recovery episodes can be passed repeatedly with `--recovery` to
`bc-train`. Recovery splitting is episode-level when at least three episodes
exist; one or two episodes remain train-only.

Recovery archives use the fail-closed `trackmaniarl-bc-recovery-v3` contract.
Every archive records the map UID, geometry SHA-256, action-repeat or decision
interval, `frame_start` control alignment and a SHA-256 over the source
demonstration's canonical metadata and arrays. DAgger archives also record the
student-checkpoint file SHA-256 for audit attribution. `bc-train` compares the
contract with the active feature geometry and parsed environment, and requires
the canonical source digest to match one of its `--demo` inputs, before
building any observations. It cannot independently re-hash the historical
student checkpoint unless that file is retained. Version 1 and 2 archives lack
sufficient provenance and are rejected with an instruction to regenerate
them; they are not silently upgraded.

Synthetic recovery perturbs states around an expert trajectory and produces
deterministic counterfactual labels. It is useful coverage, not evidence that
the states are dynamically reachable, so every generated state is an
independent recovery episode rather than a fabricated temporal rollout. The
CLI resamples label windows to the parsed environment decision cadence and its
v3 archive binds map, geometry and timing to that consumer while retaining the
canonical identity of the input lap:

```powershell
uv run trackmaniarl trajectory-synthetic-recovery run.yaml demonstrations/lap-01.npz recovery/synthetic.npz
```

Trajectory stitching joins state-compatible segments from demonstrations that
share map, geometry and the complete timing contract, including control
alignment:

```powershell
uv run trackmaniarl trajectory-stitch run.yaml demonstrations/stitched.npz --demo demonstrations
```

Trajectory optimization evaluates bounded coast/brake schedule changes in the
live environment. Keep its baseline, confirmation trials and safety limits; an
optimized schedule is not a human demonstration and needs its own provenance:

```powershell
uv run trackmaniarl trajectory-optimize run.yaml demonstrations/lap-01.npz schedules/optimized.npz --baseline-trials 3 --confirmation-trials 2
```

For demonstration-guided value learning, set
`training.offline_pretrain_updates` to a positive update budget and use:

```powershell
uv run trackmaniarl offline-pretrain run.yaml --demo demonstrations
uv run trackmaniarl diagnose expert run.yaml artifacts/my-rl-run/checkpoints/latest.pt --demo demonstrations
```

The public `DemonstrationMarginObjective` and
`DemonstrationCrossEntropyObjective` are DQfD-inspired auxiliary losses; they
respect `policy_action_ids`, but TrackmaniaRL does not claim an exact
reproduction of [Deep Q-learning from
Demonstrations](https://ojs.aaai.org/index.php/AAAI/article/view/11757).

## RL handoff and limitations

Warm-start only named compatible submodules. Encoder and temporal weights can
move into IQN/FQF or another composed model; a categorical BC head is not a
quantile head. Warm-start reports must show matched tensors, while RL resume
still requires an exact 2.0 architecture fingerprint.

The default BC-to-RL warm start transfers only `encoder` and `temporal` named
submodules. It never copies the categorical action head into a scalar,
quantile, fraction-proposal or actor-critic head. Use
`trackmaniarl train run.yaml --model-initialization-checkpoint ...` for a named
warm start; use `resume` only for an exact checkpoint from the same immutable
run contract.

BC remains sensitive to expert quality, class imbalance and covariate shift.
It does not optimize lap return or recovery trajectories directly. Use DAgger
or demonstration-aware RL objectives for states outside the expert
distribution, and never promote a model solely from open-loop validation.

Behavior cloning here follows the supervised imitation-learning setup
described in [Learning to act by watching others](https://www.cse.unsw.edu.au/~claude/papers/MI15.pdf),
but the data contract, weighted losses, lidar model and release gates are
TrackmaniaRL-specific.
