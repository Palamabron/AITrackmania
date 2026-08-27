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
| `total_transitions` | positive int; `10000` | Physical environment transitions that bound a run. PPO requires divisibility by `sequence_length`. |
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
| `save_final_checkpoint` | bool; `true` | Atomically writes the latest counters/state even when no new optimizer update followed the last periodic save. |
| `metrics_interval_updates` | positive int; `50` | Learner diagnostic cadence; it does not change optimization. |
| `per_beta_final` | `[0,1]` or null; null | Linear final PER beta; requires `training.beta`. |
| `per_beta_anneal_transitions` | positive int or null | Anneal duration; defaults to `total_transitions`. |
| `evaluate_every_episodes` | positive int or null | Schedules evaluation and therefore requires `components.evaluator`. |
| `evaluation_stop_*` | rate, seconds, batch count; all null | All three must be supplied together, with scheduled evaluator and `evaluation`; stops only after consecutive qualifying batches. |
| `max_episode_artifacts` | positive int; `100` | Retention bound for compressed local episode artifacts. |

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
| `max_inflight_chunks` | `4` | Parallel sender workers. Receipts make retries idempotent. |
| `spool_max_bytes` | `2147483648` | Per-actor durable backpressure cap. One encoded chunk must fit this cap. |
| `max_message_bytes` | `16777216` | Compressed and decompressed wire bound. |
| `soft_policy_lag_updates` / `hard_policy_lag_updates` | `1000` / `5000` | Refresh warning and rejection thresholds measured in learner updates. |
| `max_update_credit` | `512` | Caps accumulated learner update debt after bursts. |
| `epsilon_profiles` | `[1,.4,.1,.02]` | Multipliers assigned by stable actor ID. Each value is in `[0,1]`. |
| `epsilon_start` / `epsilon_final` | `.5` / `.05` | Base epsilon schedule in `[0,1]`. |
| `epsilon_decay_transitions` | `1500000` | Default schedule axis; `epsilon_decay_updates` switches the axis to updates. |
| `actor_execution` | null | Optional actor replica `device`, `precision` and `torch_threads`; defaults remain CPU/float32 at this boundary. |
| `token_env` | `TRACKMANIARL_DISTRIBUTED_TOKEN` | Environment-variable name only. Values are never written into manifests. |

## `evaluation`

`evaluation` is a versioned local asset suite, not a random-seed benchmark.
`maps[]` contains `id`, `map_path`, `geometry_path` and `expected_map_uid`.
Map IDs are unique and every path is bound into evaluation provenance.

| Field | Default | Effect |
| --- | --- | --- |
| `name`, `version` | required strings | Human- and machine-readable suite identity. |
| `maps` | empty tuple | Immutable evaluation maps. Live benchmark commands require at least one. |
| `trials_per_map` | `1` | Closed-loop attempts per map. |
| `time_buckets_s` | `[40,38,36]` s | Positive strict finish-time thresholds used for rates. |
| `target_median_s` | null | Positive release target; BC's mandatory gate fails when absent unless `--report-only` is explicit. |
| `min_finish_rate` | `.9` | Required fraction in `[0,1]`. |

## Trackmania environment

The paths below are under `components.environment.kwargs.config`.
`geometry_path` is required because it binds map UID/hash and track boundaries.

### Connection, controls and termination

| Field | Default | Unit/range and effect |
| --- | --- | --- |
| `host`, `port`, `session_port` | `127.0.0.1`, `9000`, `9001` | OpenPlanet telemetry/control and session endpoints. |
| `timeout_s`, `start_timeout_s`, `start_poll_s`, `reset_settle_s` | `10`, `15`, `.01`, `0` s | I/O timeout, start deadline, polling cadence and optional post-reset wait. |
| `action_repeat_frames` | `4` | Native telemetry frames per decision, `1..20`. Must be `1` when `decision_interval_ms` is set. |
| `decision_interval_ms` | null | Physical decision grid `(0,250]` ms. The generated Trackmania template uses 50 ms and repeat 1. |
| `control_backend` | `gamepad` | `gamepad` preserves analog controls; `keyboard` digitizes them. |
| `compact_action_ids` | null | Explicit subset of the 78-action brake-tap table; model and BC IDs must match exactly. |
| `position_indices`, `velocity_indices` | protocol defaults | Three unique telemetry indices each. |
| `crash_distance` | `25` m | Distance threshold for off-track failure. |
| `no_progress_steps` | `200` decisions | Consecutive stall limit. Cadence changes alter elapsed time, so retune intentionally. |
| `slow_progress_window_steps` | `80` decisions | Rolling progress window, at least two. |
| `minimum_progress_per_window_m` | `2` m | Required arc progress in the rolling window. |
| `minimum_finish_steps` | `50` decisions | Prevents start/finish false positives. |
| `nearest_forward_points`, `nearest_backward_points` | `500`, `10` | Local projection search window. Too small loses fast motion; too large increases folded-track ambiguity. |
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

These paths are under `components.feature_pipeline.kwargs` for
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
| `limit_progress_by_kinematics` | `false` | Opt-in physical bound for feature progress projection. Reward projection is always bounded separately. |
| `nearest_forward_points`, `nearest_backward_points` | `128`, `10` | Feature projection search window. |
| `pace_reference_path`, `pace_debt_clip_s` | null, `10` s | Optional compatible human reference and clipped debt features. |
| `reference_speed_offsets_m` | `[0,20,40,80]` m | Future reference-speed lookaheads; ignored without a pace profile. |
| `include_racing_line_channels` | `false` | Adds two racing-line lidar channels. |
| `include_finish_channels` | `false` | Adds two finish-relative lidar channels. |
| `include_dynamics` | `false` | Adds elapsed time, yaw rate and local acceleration. |
| `include_goal_features` | `false` | Adds 14 finish-gate geometry features. |

Model `telemetry_dim`, `lidar_channels`, history layout and feature output must
match exactly. `validate` executes a synthetic update to catch these contracts.

## Torch execution and stability

Built-in learner `kwargs.execution` accepts `device` (`auto`, `cuda`, `rocm`,
`mps`, `cpu`), `precision` (`auto`, `bfloat16`, `float16`, `float32`),
and `deterministic`. Defaults are auto/auto with deterministic execution. The
resolved backend, precision, and scaler are recorded in the manifest.

Experimental `AdaptiveGradientClipper` is applied after AMP unscaling and
before `optimizer.step`; its EMA/warmup state is checkpointed. Experimental
[SimBaV2](https://arxiv.org/abs/2502.15280) and
[Mamba](https://arxiv.org/abs/2312.00752) blocks remain opt-in reusable model
components; their presence does not reproduce the papers' full experiment
setups. Change one experimental variable at a time and compare against an
identical seeded baseline.

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
