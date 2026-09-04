# Observability and W&B

Every run writes `artifacts/<run-id>/events.jsonl`. This local stream is the
complete audit record and remains available when W&B is disabled or unavailable.
W&B is an optional, bounded projection of that stream: it keeps metrics that
answer a training or operations question and omits high-cardinality detail.

Configuration passed to remote trackers is recursively redacted for keys whose
names contain `key`, `token`, `secret` or `password`. Do not place secrets under
misleading names. The asynchronous adapter has a bounded queue; remote failure
does not stop training, and `health/tracker_dropped_events` plus
`health/tracker_worker_errors` expose incomplete remote telemetry.

Configure it under `components.additional_loggers` with these kwargs:

| Field | Default | Effect |
| --- | --- | --- |
| `project` | required | W&B project name. |
| `entity` | null | Optional W&B account or team. |
| `queue_size` | `10000` | Positive bounded asynchronous event queue. A full queue drops remote projections and increments health counters; local JSONL remains complete. |
| `attempt_id` | generated UUID | Optional explicit process-segment identity. |
| `resumed_from` | null | Optional checkpoint or prior-attempt attribution. |
| `run_dir`, `run_id`, `config` | runtime supplied | Artifact location, stable run group and recursively redacted RunSpec. Do not duplicate these in YAML. |

Set `WANDB_API_KEY` in the environment or ignored project `.env`; it is never a
component kwarg and must not be committed.

## Semantic axes

W&B's internal `_step` is transport bookkeeping, not a training clock.
TrackmaniaRL defines these custom axes with `define_metric` and includes the
matching axis value in every projected event:

| Axis | Advances when | Use it for |
| --- | --- | --- |
| `trainer/update` | an optimizer update is completed | loss, gradients, replay and learner timing |
| `env/transitions` | accepted transitions enter the run | collection, lag, queues and effective UTD |
| `env/episode` | a training episode summary is accepted | return, progress, safety and exploration |
| `eval/batch` | a deterministic evaluation batch completes | policy quality and release gates |
| `runtime/elapsed_s` | wall time since this process segment started | health and resource incidents |

A resumed run starts a new W&B attempt in the stable `group=<run_id>`. The
attempt ID and optional source checkpoint are stored in configuration. This
keeps segments comparable without pretending that two process lifetimes form
one uninterrupted time series.

## Default workspace

Build one saved view with six sections. Keep W&B's native CPU, RAM, disk,
network, GPU utilization and device-memory panels in the System section; do not
duplicate them as application metrics.

1. **Evaluation outcome** — `evaluation/finish_rate`,
   `evaluation/finish_time_median_s`, `evaluation/failure_progress_median_pct`
   and `evaluation/collision_rate`, all on `eval/batch`.
2. **Training episodes** — `episode/return`, `episode/progress_pct`,
   `episode/finish_time_s`, termination indicators and
   `episode/exploration_epsilon`, all on `env/episode`.
3. **Learning stability** — total/objective losses, gradient norm and clipping,
   TD-error scale, Q/target scale and action entropy on `trainer/update`.
4. **Data pipeline** — transitions/s, updates/s, throughput ratio, update
   backlog, replay fill, policy lag and queue delay.
5. **Timing** — replay sample/wait, host-to-device, forward, backward,
   optimizer, policy publication and checkpoint snapshot durations.
6. **Health** — active actors, spool bytes, actor timeouts, rejected rollouts,
   WAL recovery/errors, checkpoint completion/failure and tracker drops.

Use raw points or a short moving median for incident diagnosis. Add a longer
moving average only as a second series; smoothing must not replace the raw
signal because it can hide stalls and non-finite spikes.

## Metric catalogue

The frequency below describes remote W&B projection. The complete neutral
payload, including per-progress-bin diagnostics, remains in local JSONL.

### Evaluation and episodes

| Metric | Unit and frequency | Interpretation and decision |
| --- | --- | --- |
| `evaluation/trials`, `evaluation/finished_trials` | count per evaluation batch | Establish sample size before acting on a rate. |
| `evaluation/finish_rate` | fraction per evaluation batch | Primary reliability outcome; compare only identical maps, seeds and trial counts. |
| `evaluation/finish_time_median_s` | seconds per batch, finishes only | Primary pace outcome; never treat zero as a fast finish. |
| `evaluation/finish_time_mean_s`, `evaluation/finish_time_best_s` | seconds per batch, finishes only | Detect heavy tails and preserve the best observed deterministic run. |
| `evaluation/failure_progress_mean_pct`, `evaluation/failure_progress_median_pct`, `evaluation/failure_progress_best_pct` | percent per batch, failed trials only | Distinguish an early-collapse regression from failures close to the finish. |
| `evaluation/collision_rate`, `evaluation/off_track_rate`, `evaluation/telemetry_error_rate` | fraction per batch | Separate unsafe policy behavior from integration failure. Any telemetry error requires investigation before comparing policy quality. |
| `evaluation/projected_velocity_ratio_mean` | ratio per batch | Detect systematic under-speed or an invalid pace reference. |
| `evaluation/q_margin_start_mean` | value units per batch | Track confidence at the common start state; a scale jump without outcome gain can indicate value drift. |
| `evaluation/policy_version` | update index per batch | Binds results to the exact evaluated snapshot. |
| `evaluation/action_latency_ms` | milliseconds per policy step | Measures only `policy.act`; it is not physical controller-to-engine latency. |
| `evaluation/controller_apply_ms`, `evaluation/telemetry_wait_ms` | milliseconds per step | Splits the local controller backend call from the subsequent wait for an accepted telemetry frame. Neither is an engine acknowledgement. |
| `evaluation/telemetry_skipped_frames_total`, `evaluation/telemetry_skipped_frames_mean`, `evaluation/telemetry_skipped_frames_max`, `evaluation/telemetry_steps_with_skipped_frames_fraction` | frames and fraction per batch | Quantifies complete OpenPlanet frames discarded by latest-frame draining; trailing packet fragments are not counted. |
| `episode/return` | reward per episode | Training signal only; compare with evaluation rather than optimizing the chart in isolation. |
| `episode/progress_pct`, `episode/progress_m` | percent and metres per episode | Shows whether failures move farther along the track. |
| `episode/duration_s`, `episode/race_time_s`, `episode/finish_time_s` | seconds per episode | Diagnose stalls and pace. `finish_time_s` is meaningful only when `episode/finished=1`. |
| `episode/finished` | 0/1 per episode | Training finish rate input. Use deterministic evaluation for release decisions. |
| `episode/termination_no_progress`, `episode/termination_slow_progress`, `episode/termination_off_track`, `episode/termination_max_steps`, `episode/termination_telemetry_error` | 0/1 per episode | Reason-coded failure distribution. Telemetry errors are system incidents, not policy failures. |
| `episode/collision_count` | count per episode | Safety proxy; inspect together with pace and finish rate. |
| `episode/exploration_epsilon`, `episode/policy_version` | fraction and update index per episode | Explains behavior changes and detects stale actor policies. |
| `episode/q_margin_mean`, `episode/q_margin_min`, `episode/q_margin_start_mean` | value units per episode | Detect ambiguous action choice and collapsing value separation. |
| `episode/velocity_ratio_mean` | ratio per episode | Compares projected velocity with the configured reference. |
| `episode/timing_policy_inference_ms_mean`, `episode/timing_policy_inference_ms_max` | milliseconds per episode | Actor latency; sustained maxima near the environment tick budget make collection actor-bound. |
| `episode/controller_apply_ms_mean`, `episode/controller_apply_ms_max`, `episode/telemetry_wait_ms_mean`, `episode/telemetry_wait_ms_max` | milliseconds per episode | Separates time inside the controller backend from time waiting for the accepted telemetry frame. |
| `episode/telemetry_skipped_frames_total`, `episode/telemetry_skipped_frames_mean`, `episode/telemetry_skipped_frames_max`, `episode/telemetry_steps_with_skipped_frames_fraction` | frames and fraction per episode | Detects producer/consumer cadence mismatch without treating a partial network packet as a dropped frame. |
| `episode/telemetry_error` | count/fraction per episode | Non-zero means integration health must be fixed before evaluating policy quality. |

A true apply-to-engine measurement would require a new signed OpenPlanet
protocol carrying command sequence IDs, timestamps and an acknowledgement.
Current timing metrics deliberately make no such claim.

### Learner and replay

| Metric | Unit and frequency | Interpretation and decision |
| --- | --- | --- |
| `learner/loss_<objective>` | objective units per metrics window | Algorithm-specific optimization loss. Check finiteness and trends; absolute scales are not comparable across objectives. |
| `learner/gradients_norm`, `learner/gradient_norm`, `learner/gradient_norm_max` | L2 norm per window | Detect explosions or vanishing gradients relative to the run's own baseline. |
| `learner/gradients_fraction_norm` | L2 norm per FQF window | Confirms the fraction proposal network receives an isolated gradient. |
| `learner/gradients_adaptive_ema_norm`, `learner/gradients_adaptive_coefficient`, `learner/gradients_adaptive_clipped` | norm, multiplier and 0/1 per window | Shows AdaptiveGradientClipper state; persistent clipping near 1 needs learning-rate or objective inspection. |
| `learner/gradient_clipped_fraction`, `learner/gradient_clip_coefficient` | fraction and multiplier per window | Quantifies fixed clipping rather than merely reporting a threshold. |
| `learner/td_abs_mean`, `learner/td_abs_max` | value units per window | Drives PER and exposes isolated extreme targets. |
| `learner/q_selected_mean`, `learner/q_selected_max`, `learner/q_selected_abs_max`, `learner/q_selected_std_mean` | value units per window | Detects value-scale drift and loss of distributional spread. |
| `learner/target_mean`, `learner/target_abs_max`, `learner/target_std_mean` | value units per window | Separates target instability from online-network instability. |
| `learner/reward_mean`, `learner/reward_abs_max` | reward units per window | Detects reward-pipeline changes or outliers. |
| `learner/action_entropy`, `learner/action_unique_fraction` | nats and fraction per window | Detects action collapse; interpret against the configured action mask and exploration schedule. |
| `learner/importance_weight_mean`, `learner/importance_weight_min` | fraction per PER window | Shows how strongly importance correction changes effective samples. |
| `learner/demonstration_action_accuracy` | fraction per demo window | Monitors supervised compatibility of demo samples without replacing closed-loop evaluation. |
| `replay/size`, `replay/fill_fraction` | transitions and fraction per window | Determines warm-up/fill state and whether eviction pressure is expected. |
| `replay/per_beta` | fraction per window | Confirms the configured importance-sampling schedule. |

### Pipeline, timing and system health

| Metric | Unit and frequency | Interpretation and decision |
| --- | --- | --- |
| `performance/transitions_per_s`, `performance/cumulative_transitions_per_s` | transitions/s per window | Current and lifetime actor throughput. A drop with stable learner speed is actor/integration-bound. |
| `performance/updates_per_s`, `performance/target_updates_per_s`, `performance/update_throughput_ratio` | updates/s and ratio per window | A ratio below 1 with growing backlog means the learner cannot meet configured UTD. |
| `pipeline/update_credit`, `pipeline/update_backlog_s` | owed updates and seconds per window | Measures training debt. Sustained growth is actionable even when GPU utilization looks high. |
| `pipeline/rollout_queue_depth`, `pipeline/queue_delay_s` | chunks and seconds per window | Detects ingestion backpressure and stale experience. |
| `pipeline/policy_lag_updates`, `pipeline/policy_version` | updates per window/event | Detects actors running stale snapshots; compare with configured soft/hard limits. |
| `pipeline/utd` | updates/transition per window | Effective update-to-data ratio; compare with the requested ratio after warm-up. |
| `performance/replay_sample_s`, `performance/replay_wait_s`, `performance/learner_update_s` | seconds per window | Splits input preparation from GPU work. `replay_wait_s > learner_update_s` is input-bound. |
| `performance/host_to_device_s`, `performance/forward_s`, `performance/backward_s`, `performance/gradient_clip_s`, `performance/optimizer_s` | seconds per window | Locates learner-side stalls and host/device synchronization. |
| `performance/policy_publish_s`, `performance/checkpoint_snapshot_s`, `performance/logging_s` | seconds per event/window | Finds periodic pauses caused by snapshots, persistence or telemetry. |
| `system/torch_cuda_memory_bytes` | bytes per window | PyTorch-allocated memory; use W&B native metrics for total device memory and utilization. |
| `health/active_actors`, `health/spool_bytes` | count and bytes per window | No active actor or monotonically growing spool is a collection/transport incident. |
| `health/actor_timeouts`, `health/max_heartbeat_age_s` | cumulative count and seconds per incident | Compare age with `heartbeat_s` and `actor_timeout_s`. |
| `health/rollouts_rejected`, `health/wal_errors`, `health/wal_recoveries` | cumulative counts per incident | Any increase requires reason-coded local-event inspection before trusting the run. |
| `health/wal_pending_rows`, `health/wal_pending_payload_bytes`, `health/wal_receipt_rows`, `health/wal_database_bytes`, `health/wal_bytes` | rows and bytes per metrics window | Shows unconsumed rollout pressure, durable idempotency receipts, the SQLite database size and its separate `-wal` file size. Size is observed rather than capped. |
| `health/checkpoints_queued`, `health/checkpoints_completed`, `health/checkpoint_failures` | cumulative counts per event | A queued checkpoint without completion is not a durable recovery point. |
| `health/tracker_dropped_events`, `health/tracker_worker_errors` | cumulative counts per window | Non-zero means W&B is incomplete; local JSONL remains authoritative. |
| `health/run_failures` | cumulative count per incident | Marks the W&B attempt failed and closes it with non-zero exit status. |

### Behavior cloning

`imitation_train/{loss,accuracy,balanced_accuracy,learning_rate}` and
`imitation_validation/{loss,accuracy,balanced_accuracy,weighted_accuracy,transition_accuracy,steering_accuracy,steering_transition_accuracy,intervention_accuracy,student_disagreement_accuracy,control_score,learning_rate,best}`
are logged on `trainer/update`. Counts and per-action recall remain in local
JSONL because they create high-cardinality W&B series. Promotion still depends
on the closed-loop `bc-benchmark`, not supervised accuracy alone.

## Incident signals

Configure alerts or external monitors from these explicit conditions:

- no new `env/transitions` for longer than the environment reset budget;
- `health/max_heartbeat_age_s > 2 × heartbeat_s`, critical at
  `actor_timeout_s`;
- any telemetry error, WAL error, checkpoint failure, non-finite learner value
  or rejected rollout;
- policy lag beyond the configured soft limit, critical beyond the hard limit;
- update credit growing for five consecutive metric windows;
- queued checkpoint without a matching completion event;
- tracker drops or worker errors greater than zero;
- a finish-rate regression only after the configured evaluation trial count,
  never from a single training episode.

There is no universal gradient-norm or loss threshold across algorithms and
reward scales. Establish a seeded healthy baseline, alert on non-finite values
immediately, and treat sustained deviations together with outcome metrics.

References: [W&B custom axes](https://docs.wandb.ai/models/track/log/customize-logging-axes),
[system metrics](https://docs.wandb.ai/models/ref/python/experiments/system-metrics),
and [alerts](https://docs.wandb.ai/models/runs/alert).
