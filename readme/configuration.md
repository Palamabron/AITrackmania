# Configuration reference

`run.yaml` is the only public runtime configuration boundary. It is parsed by
Pydantic as RunSpec 2.0, rejects unknown fields and NaN/Inf, then resolves each
`module:attribute` component from the local environment. It is executable
configuration: use only files and extension packages you trust.

Start with the [generated Trackmania project](../README.md#install-and-create-an-agent)
for a real run. The complete, game-free
[`builtin-smoke.yaml`](examples/builtin-smoke.yaml) is kept under test and can
be checked without Trackmania:

```bash
uv run trackmaniarl inspect-config readme/examples/builtin-smoke.yaml
uv run trackmaniarl validate readme/examples/builtin-smoke.yaml
```

`inspect-config` performs only safe YAML loading and RunSpec validation. It
prints every nested `class_path`, its YAML location and whether its module is
first-party or external, without importing any component. The command never
blocks a namespace and is not a sandbox; review the output before using the
trusted `validate`, `train`, `learner` or `actor` paths.

Algorithm blocks belong under `components.*.kwargs`; see the
[algorithm support matrix](algorithms.md). Replay and recurrent constraints
have one canonical explanation in [Replay and sequences](replay-and-sequences.md),
and reward fields are derived in [Rewards](rewards.md).

## RunSpec root and components

| YAML path | Type; default | Contract |
| --- | --- | --- |
| `api_version` | literal `"2.0"`; required | No implicit 1.x migration. |
| `run_id` | non-empty safe identifier; required | Names the immutable artifact directory. Start a new ID after a contract-changing edit. |
| `seed` | integer; `0` | Seeds model construction, samplers, actors and deterministic splits. |
| `artifacts_dir` | path; `artifacts` | Root for manifests, events, replay/WAL state and checkpoints. |
| `metadata` | mapping; `{}` | User attribution only; it does not alter training. Never put secrets here. |
| `components.learner` | `ComponentSpec`; required | Learner selected by `class_path`; its model contract must match `model_factory`. |
| `components.environment` | optional component | Required by `train`, Trackmania smoke and live evaluation; not required by offline validation. |
| `components.model_factory` | optional component | Required by built-in Torch learners unless the learner receives a model directly in Python. |
| `components.replay_store` | component; required | `InMemoryReplayStore(capacity=100000)` is the built-in bounded store. |
| `components.sampler` | component; required | Must support the requested `sequence_length` and learner lifecycle. |
| `components.feature_pipeline` | component; required | Owns observation transformation and batch collation. |
| `components.logger` | component; JSONL logger | Local events are authoritative. |
| `components.additional_loggers` | tuple; empty | Optional projections such as W&B; failures must not replace the local stream. |
| `components.checkpoint_codec` | component; `TorchCheckpointCodec` | Bounded Zstandard + `torch.load(weights_only=True)` checkpoint format. |
| `components.evaluator` | optional component | Mandatory when scheduled evaluation is enabled. |
| `training` | `TrainingSpec`; defaults below | Budgets, replay request shape, update ratio, checkpoints and evaluation stopping. |
| `distributed` | `DistributedSpec`; defaults below | Local/remote actor transport, durability, exploration and actor execution. |
| `evaluation` | suite or null; null | Versioned local maps and release acceptance thresholds. |

A component has exactly `class_path` and optional `kwargs`:

```yaml
components:
  replay_store:
    class_path: trackmaniarl.core.replay:InMemoryReplayStore
    kwargs: {capacity: 1000000}
```

Relative asset paths are resolved against the directory containing
`run.yaml`, not the caller's current working directory.

## `training`

| Field | Type; default | Unit/range and effect |
| --- | --- | --- |
| `total_transitions` | positive int; `10000` | PPO uses an exact divisible budget. Asynchronous off-policy training stops after reaching this target but accepts already durable whole chunks, so the final count can be higher; use the recorded count as the actual budget. |
| `max_episode_steps` | positive int; `2000` | Decision steps. Too small creates artificial truncations; too large delays episode-level diagnostics. |
| `batch_size` | positive int; `256` | Transitions or sequences per update. Memory grows roughly linearly. |
| `sequence_length` | positive int; `1` | Replay timesteps per sampled sequence. Values above one require sequence-capable replay and temporal learner support. |
| `n_step` | positive int; `1` | Bellman horizon. It must be smaller than `sequence_length` for recurrent replay; terminal stops bootstrap, truncation does not. |
| `gamma` | float; `0.99` | Discount in `[0,1]`. For first-party Trackmania rewards it must equal `components.environment.kwargs.config.reward_gamma`. |
| `beta` | float or null; null | Request-time PER importance exponent. When set, it overrides the sampler's constructor `beta`. |
| `warmup_transitions` | int; `1000` | Replay items collected before the first off-policy update; no deferred update debt is accumulated. |
| `offline_pretrain_updates` | int; `0` | Updates performed from loaded demonstration replay before actors start. |
| `updates_per_transition` | finite float `>0`; `1.0` | Off-policy update-to-data ratio. High values improve reuse but amplify stale-policy and overfitting risk. |
| `checkpoint_interval_updates` | positive int or null; `1000` | Periodic exact-state checkpoint cadence; null disables periodic saves. |
| `checkpoint_keep_last` | positive int or null; null | Retains only the newest count within each checkpoint family after a successful save. Regular and evaluation-leader checkpoints are pruned independently; null disables pruning. |
| `save_final_checkpoint` | bool; `true` | Atomically writes the latest counters/state even when no new optimizer update followed the last periodic save. |
| `metrics_interval_updates` | positive int; `50` | Learner diagnostic cadence; it does not change optimization. |
| `per_beta_final` | `[0,1]` or null; null | Linear final PER beta; requires `training.beta`. |
| `per_beta_anneal_transitions` | positive int or null | Anneal duration; defaults to `total_transitions`. |
| `evaluate_every_episodes` | positive int or null | Schedules evaluation and therefore requires `components.evaluator`. |
| `evaluation_stop_min_finish_rate` | float `[0,1]` or null; null | Minimum finish rate for early stopping. Requires the next two stop fields, scheduled evaluation and an `evaluation` suite. |
| `evaluation_stop_median_s` | positive float or null; null | Maximum qualifying median finish time in seconds. It is evaluated only from finished trials. |
| `evaluation_stop_consecutive_batches` | positive int or null; null | Number of consecutive evaluation batches that must satisfy both thresholds before training stops. |
| `max_episode_artifacts` | positive int; `100` | Retention bound for compressed local episode artifacts. |

Scheduled distributed evaluation maintains two leaders. `best-eval-*` requires
the suite's `min_finish_rate` and ranks policies by finish rate first, then lower
median finish time, then lower best finish time. `fastest-eval-*` ranks policies
by lower best finish time first, then finish rate, then lower median finish time,
so it preserves the best individual completed lap even when its batch does not
pass the reliability threshold. When one batch improves both leaders, each
family writes its own exact-policy checkpoint. Promotion durability and
checkpoint retention are independent for the two families.

## `distributed`

All participants use the same resolved RunSpec fingerprint and a bearer token
of at least 32 characters from `token_env`. The token authenticates but does
not encrypt; bind to loopback and use an encrypted tunnel.

| Field | Default | Unit/range and effect |
| --- | --- | --- |
| `port` | `8787` | TCP port `1..65535` used by the local launcher. Remote `--bind/--connect` can override the address. |
| `rollout_chunk_transitions` | `128` | Maximum transitions per durable actor chunk. |
| `rollout_flush_s` | `2.0` | Seconds before a partial chunk is spooled. Lower values reduce latency but increase I/O/RPC overhead. |
| `policy_refresh_s` | `5.0` | Minimum seconds between ordinary actor snapshot pulls. |
| `heartbeat_s` / `actor_timeout_s` | `5.0` / `20.0` s | Liveness cadence and learner timeout; timeout should comfortably exceed heartbeat plus network jitter. |
| `actor_stall_timeout_s` | `null` or positive seconds | Optional actor-side collection watchdog. When no environment step completes within the limit, the actor stops and hard-exits after a 60 s grace period so an external launcher can restart it. |
| `max_inflight_chunks` | `4` | Parallel sender workers. Receipts make retries idempotent. |
| `spool_max_bytes` | `2147483648` | Per-actor durable backpressure cap. One encoded chunk must fit this cap. |
| `max_message_bytes` | `16777216` | Compressed and decompressed wire bound. |
| `soft_policy_lag_updates` / `hard_policy_lag_updates` | `1000` / `5000` | Refresh warning and rejection thresholds measured in learner updates. |
| `max_update_credit` | `512` | Caps accumulated learner update debt after bursts. |
| `epsilon_profiles` | `[1,.4,.1,.02]` | Multipliers assigned by stable actor ID. Each value is in `[0,1]`. |
| `epsilon_start` / `epsilon_final` | `.5` / `.05` | Base epsilon schedule in `[0,1]`. |
| `epsilon_decay_transitions`, `epsilon_decay_updates` | `1500000`, null | Transition decay is the default; a positive update count switches the epsilon schedule axis to learner updates. |
| `actor_execution` | null | Optional actor replica `device`, `precision` and `torch_threads`; defaults remain CPU/float32 at this boundary. |
| `token_env` | `TRACKMANIARL_DISTRIBUTED_TOKEN` | Environment-variable name only. Values are never written into manifests. |

### Actor execution override

| Field | Default | Effect |
| --- | --- | --- |
| `device` | `cpu` | Actor replica backend: `auto`, `cuda`, `rocm`, `mps` or `cpu`. Keep remote actors on CPU unless policy inference is measured as the bottleneck. |
| `precision` | `float32` | Actor inference precision: `auto`, `bfloat16`, `float16` or `float32`; unsupported backend/precision pairs fail validation. |
| `torch_threads` | null | Optional positive intra-op thread count for the actor process. Null preserves Torch's process default. |

## `evaluation`

`evaluation` is a versioned local asset suite, not a random-seed benchmark.
`maps[]` contains `id`, `map_path`, `geometry_path` and `expected_map_uid`.
Map IDs are unique and every path is bound into evaluation provenance.

| Field | Default | Effect |
| --- | --- | --- |
| `name`, `version` | required strings | Human- and machine-readable suite identity. |
| `maps` | required, non-empty tuple | Immutable evaluation maps. Omit the entire `evaluation` section when no suite is configured. |
| `trials_per_map` | `1` | Closed-loop attempts per map. |
| `time_buckets_s` | `[40,38,36]` s | Positive strict finish-time thresholds used for rates. |
| `target_median_s` | null | Positive release target; BC's mandatory gate fails when absent unless `--report-only` is explicit. |
| `min_finish_rate` | `.9` | Required fraction in `[0,1]`. |

Each `maps[]` item has an immutable local asset identity:

| Field | Default | Effect |
| --- | --- | --- |
| `id` | required | Unique safe identifier used in evaluation results and artifact names. |
| `map_path` | required | Local `.Map.Gbx` loaded for this evaluation case. |
| `geometry_path` | required | Matching versioned geometry used for feature/reward contracts. |
| `expected_map_uid` | required | UID that the live session must report; a mismatch fails before driving. |

## Trackmania environment

The paths below are under `components.environment.kwargs.config`.
`geometry_path` is required because it binds map UID/hash and track boundaries.

### Connection, controls and termination

| Field | Default | Unit/range and effect |
| --- | --- | --- |
| `host`, `port`, `session_port` | `127.0.0.1`, `9000`, `9001` | OpenPlanet telemetry/control and session endpoints. |
| `timeout_s`, `start_timeout_s`, `start_poll_s`, `reset_settle_s` | `10`, `15`, `.01`, `0` s | I/O timeout, start deadline, polling cadence and optional post-reset wait. |
| `confirm_finish_before_reset` | `true` | Send Enter before reset for normal play result screens; set `false` when School Mode requires editor validation. |
| `restart_input` | `gamepad` | With gamepad driving, use its Give Up binding or select `keyboard` to send Delete for editor validation. |
| `action_repeat_frames` | `4` | Native telemetry frames per decision, `1..20`. Must be `1` when `decision_interval_ms` is set. |
| `decision_interval_ms` | null | Physical decision grid `(0,250]` ms. The generated Trackmania template uses 50 ms and repeat 1. |
| `control_backend` | `gamepad` | `gamepad` preserves analog controls; `keyboard` digitizes them. |
| `compact_action_ids` | null | Explicit subset of the 78-action brake-tap table; model and BC IDs must match exactly. |
| `position_indices`, `velocity_indices` | protocol defaults | Three unique telemetry indices each. |
| `expected_map_uid` | null | Optional active-map UID assertion for training/smoke. Configure it for every release run. |
| `crash_distance` | `25` m | Distance threshold for off-track failure. |
| `finish_progress` | `0.995`, range `(0,1]` | Required accepted lap fraction before finish UI can end the episode successfully. |
| `no_progress_steps` | `200` decisions | Consecutive stall limit. Cadence changes alter elapsed time, so retune intentionally. |
| `slow_progress_window_steps` | `80` decisions | Rolling progress window, at least two. |
| `minimum_progress_per_window_m` | `2` m | Required arc progress in the rolling window. |
| `minimum_finish_steps` | `50` decisions | Prevents start/finish false positives. |
| `nearest_forward_points`, `nearest_backward_points` | `500`, `10` | Local projection search window. Too small loses fast motion; too large increases folded-track ambiguity. |
| `limit_progress_by_kinematics` | `true` | Caps reward progress by measured displacement and elapsed-time speed bounds. Disable only for an adapter with intentional teleport/reset semantics; this is independent of the feature-pipeline switch below. |
| `maximum_race_time_s` | null | Optional physical timeout. `TrajectoryReward` treats it as the natural `time_limit` terminal and applies `terminal_failure_penalty`; transport/game interruption is a truncation instead. |

Reward fields in this same mapping are listed with equations and ranges in
[Rewards](rewards.md).

### Demonstration timing

| Field | Default | Contract |
| --- | --- | --- |
| `demonstration_action_lead_ms` | `0`, range `[0,250]` ms | Chooses a future expert label on race time; it is manual and constant, not an estimator. |
| `demonstration_control_aggregation` | `false` | Integrates controls over each `decision_interval_ms` window, then quantizes. Requires gamepad and an explicit decision interval. |

See the [imitation timing procedure](imitation-learning.md#timing-and-latency-calibration)
before changing either field.

## Lidar feature pipeline

These paths are under `components.feature_pipeline.kwargs.config` for
`LidarFeaturePipeline`.

| Field | Default | Unit/range and effect |
| --- | --- | --- |
| `geometry_path` | required | Versioned boundary asset; `expected_map_uid` fails closed on the wrong map. |
| `samples_per_side` | `60` | At least 2 lookahead samples per boundary. |
| `max_distance_m` | `300` m | Normalization/clipping distance. Too small saturates; too large compresses useful variation. |
| `history_length` | `1` | Left-padded online frame history. Keep it `1` when replay sequences already provide temporal context. |
| `include_track_relative` | `false` | Adds progress/lateral/heading/projected-velocity fields. |
| `include_control_inputs` | `true` | Adds current steer/gas/brake. BC normally sets false; otherwise it must mask current controls to prevent target leakage. |
| `mask_current_control_inputs` | `false` | Zeros current controls and requires them to be present. |
| `local_velocity_features` | `false` | Rotates velocity into car coordinates; required by horizontal BC reflection. |
| `use_racing_line` | `false` | Uses the asset racing line instead of reward center where supported. |
| `max_speed_mps`, `velocity_to_mps_scale` | `80`, `.001` | Physical speed normalization and native velocity-unit conversion. |
| `max_time_delta_s` | `1` s | Rejects stale finite-difference dynamics. |
| `limit_progress_by_kinematics` | `false` | Opt-in physical bound for feature progress projection. Reward projection has the independent environment setting above. |
| `nearest_forward_points`, `nearest_backward_points` | `128`, `10` | Feature projection search window. |
| `pace_reference_path`, `pace_debt_clip_s` | null, `10` s | Optional compatible human reference and clipped debt features. |
| `reference_speed_offsets_m` | `[0,20,40,80]` m | Future reference-speed lookaheads; ignored without a pace profile. |
| `include_racing_line_channels` | `false` | Adds two racing-line lidar channels. |
| `include_finish_channels` | `false` | Adds two finish-relative lidar channels. |
| `include_dynamics` | `false` | Adds elapsed time, yaw rate and local acceleration. |
| `include_goal_features` | `false` | Adds 14 finish-gate geometry features. |

Model `telemetry_dim`, `lidar_channels`, history layout and feature output must
match exactly. `validate` executes a synthetic update to catch these contracts.

## Built-in learner `kwargs`

The tables below name every stable learner option accepted below
`components.learner.kwargs`. The runtime supplies `model_factory` from
`components.model_factory` and `seed` from the RunSpec root unless they are
explicitly overridden. `base_dir` is also supplied to the discrete learner for
relative warm-start paths. Prefer the RunSpec fields so one value remains
authoritative.

### `DiscreteValueLearner`

| Field | Default | Effect and constraint |
| --- | --- | --- |
| `learning_rate` | `1e-4` | Main optimizer rate; positive. |
| `fraction_learning_rate` | `1e-7` | Separate FQF proposal optimizer rate; positive and ignored by non-FQF strategies. |
| `target_update_interval` | `1000` | Hard-copy cadence when `target_tau` is zero; positive. |
| `target_tau` | `0.0` | Polyak coefficient in `[0,1]`; a positive value updates every learner step. |
| `gradient_clip_norm` / `fraction_gradient_clip_norm` | `10` / `10` | Positive main and FQF gradient-norm caps. |
| `burn_in` | `0` | Recurrent prefix positions used only to reconstruct state; non-negative and below `training.sequence_length`. |
| `exploration_epsilon` | `0.1` | Standalone policy epsilon in `[0,1]`; distributed actors apply their assigned profile to the runtime schedule. |
| `policy_action_ids` | null | Unique non-negative global action IDs allowed by the policy; model output and environment mapping must agree. |
| `online_quantile_distortion` / `evaluation_quantile_distortion` | `neutral` / `neutral` | `neutral` or `upper_cvar`; controls action selection, not the TD target distribution. |
| `upper_cvar_alpha` | `0.25` | Upper-tail mass in `(0,1]` used by either `upper_cvar` distortion. |
| `value_rescaling` | `false` | Applies the signed-square-root transform and its inverse around value targets. |
| `adaptive_gradient_clipper` | null | Optional `ComponentSpec` for `AdaptiveGradientClipper`; replaces the fixed main clip and is checkpointed. |
| `diagnostics_interval_updates` | `100` | Positive cadence for detailed Q, TD, return and action-distribution metrics. |
| `objectives` | empty | Ordered `ComponentSpec` sequence of auxiliary value objectives; their weighted losses add to TD loss. |
| `action_selector` | null | Optional custom action-selection component; it must preserve the configured action mask and value contract. |
| `model_initialization_checkpoint` | null | Warm-start checkpoint path, resolved relative to `run.yaml`; this is not resume. |
| `warm_start_submodules` | `[encoder, temporal]` | Named model prefixes eligible for transfer. |
| `warm_start_required_tensors` | empty | Exact tensor names that must be present and shape-compatible. |
| `freeze_warm_start_during_offline_pretraining` | `false` | Freezes transferred parameters only during the configured demonstration pretraining phase. |
| `execution` | null | Torch device, precision and determinism policy described below. |

Built-in auxiliary objectives accept: `DemonstrationMarginObjective(margin=0.8,
weight=1.0, steering_switch_weight=1.0)`,
`DemonstrationCrossEntropyObjective(weight=1.0, steering_switch_weight=1.0)`, and
`PolicyAnchorObjective(weight=1.0)`. All values are non-negative. A switch
weight above one emphasizes demonstration positions where the steering bin
changes, while one preserves uniform weighting. The policy anchor requires a
custom batch producer for `policy_anchor_q_values`.

### SAC, REDQ, TQC and stable discrete SAC

| Learner/field | Default | Effect and constraint |
| --- | --- | --- |
| all: `learning_rate` | `3e-4` | Actor, critic and learned-temperature optimizer rate; positive. |
| all: `target_tau` | `0.005` | Polyak coefficient in `(0,1]`. |
| all: `entropy_coefficient` | `0.2` | Positive fixed value or initial learned alpha. |
| SAC/TQC/SD-SAC: `target_entropy` | null | Desired policy entropy; null derives the continuous target from action width and uses the learner default for discrete SAC. |
| SAC/TQC/SD-SAC: `learn_entropy_coefficient` | `true` | Enables checkpointed log-alpha optimization; false keeps alpha fixed. |
| REDQ: `target_subset_size` | `2` | Positive number of target critics sampled; cannot exceed ensemble size. |
| REDQ: `policy_update_interval` | `20` | Positive critic updates between actor updates. |
| TQC: `top_quantiles_to_drop_per_critic` | `2` | Non-negative global upper-quantile truncation per critic; cannot remove every target quantile. |
| SD-SAC: `q_clip_epsilon` | `0.5` | Non-negative critic-target clipping width. |
| SD-SAC: `entropy_penalty_coefficient` | `0.5` | Non-negative entropy-change penalty weight. |
| all: `execution` | null | Torch device, precision and determinism policy described below. |

SAC-family feature pipelines may collate observations and next observations as
one tensor or as a homogeneous tensor PyTree (for example, a mapping of image
and telemetry tensors). The supplied actor and critics must accept that same
structure. Scalar critics may return `[B]` or `[B,1]`; other shapes fail before
loss computation. TQC critics must return a non-empty `[B,Q]` matrix.

### `ProximalPolicyOptimization`

| Field | Default | Effect and constraint |
| --- | --- | --- |
| `learning_rate` | `3e-4` | Initial optimizer rate before linear annealing; positive. |
| `clip_epsilon` / `value_clip_epsilon` | `0.2` / `0.2` | Policy-ratio and value-change clips, each strictly between zero and one. |
| `gae_lambda` | `0.95` | GAE coefficient in `[0,1]`. |
| `entropy_coefficient` / `value_coefficient` | `0.01` / `0.5` | Non-negative entropy bonus and value-loss weights. |
| `max_gradient_norm` | `0.5` | Positive gradient-norm cap. |
| `update_epochs` | `10` | Positive passes over each on-policy rollout. |
| `minibatch_size` | `256` | Positive optimization minibatch size. |
| `target_kl` | `0.02` | Positive early-stop threshold; null disables KL stopping. |
| `normalize_observations` / `normalize_rewards` | `true` / `true` | Enables checkpointed running normalization. |
| `observation_clip` / `reward_clip` | `10` / `10` | Positive post-normalization absolute clips. |
| `execution` | null | Torch device, precision and determinism policy described below. |

## Torch execution and stability

Built-in learner `kwargs.execution` accepts `device` (`auto`, `cuda`, `rocm`,
`mps`, `cpu`), `precision` (`auto`, `bfloat16`, `float16`, `float32`),
and `deterministic`. Defaults are auto/auto with deterministic execution. Each
initialized process segment appends its resolved backend, precision and scaler
to `manifest-attempts.jsonl`; `manifest.json` remains the config identity.

Experimental `AdaptiveGradientClipper` is applied after AMP unscaling and
before `optimizer.step`; its EMA/warmup state is checkpointed. Its constructor
is `decay=0.995` (`[0,1)`), `warmup_steps=100` (non-negative), and
`clip_factor=2.0` (positive). Experimental
[SimBaV2](https://arxiv.org/abs/2502.15280) and
[Mamba](https://arxiv.org/abs/2312.00752) blocks remain opt-in reusable model
components; their presence does not reproduce the papers' full experiment
setups. Change one experimental variable at a time and compare against an
identical seeded baseline.

`SimbaV2Backbone` requires `input_dim` and `hidden_dim`, and accepts
`block_count=2`, `expansion=4`, and `input_shift=1.0`. Dimensions and expansion
must be positive; block count is non-negative. A custom learner using it must
call `project_hyperspherical_weights(model)` after each optimizer step. The
built-in discrete learner performs that projection automatically.

## CLI boundary

```text
trackmaniarl init DIRECTORY [--template starter|trackmania]
trackmaniarl inspect-config RUN.yaml
trackmaniarl validate RUN.yaml
trackmaniarl train RUN.yaml [--model-initialization-checkpoint CHECKPOINT]
trackmaniarl resume RUN.yaml CHECKPOINT [--reset-replay]
trackmaniarl learner RUN.yaml --bind 127.0.0.1:8787
trackmaniarl actor RUN.yaml --connect 127.0.0.1:8787 [--actor-id ID]
trackmaniarl smoke RUN.yaml --transitions 100
trackmaniarl benchmark RUN.yaml CHECKPOINT
```

Use `trackmaniarl COMMAND --help` as the syntax authority. The
[Trackmania guide](trackmania.md), [imitation guide](imitation-learning.md)
and [algorithm matrix](algorithms.md) list specialized commands.
