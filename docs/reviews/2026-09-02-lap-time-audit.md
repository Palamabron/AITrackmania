# Lap-time audit of the RunSpec 2.0 branch (PR #29) and the v104f setup

Date: 2026-09-02. Scope: the `feature/trackmaniarl-2.0-modular-refactor` branch at 7254afc and the
live baseline `sub37-iqn-gnn-simba-v104f-s17-online-scratch` (IQN + neighbour-GNN + SimbaV2, from scratch,
20 Hz, 78 gamepad actions). Objective of the project: fastest deterministic lap on `trackmaniarl-test`
(best retained human demonstration 36.035 s, agent 45.46 s median, target < 37 s).

## Verification status

This is a useful hypothesis inventory, not an approved experiment recipe. Independent review of PR #30
found no external reviews or CI checks. The bundled `sub37-v105-candidate.yaml` is not runnable from
`docs/reviews` because configuration-relative asset paths point to missing directories, and it changes
many learning variables at once. The project owner selected a bundle-first discovery run, followed by
backward ablations if it beats V104F. Use the corrected runnable copy and gates recorded in
`my-trackmania-agent/agent-notes/SUB37.md`, not this documentation-relative file.

Corrections made during integration: the cause of the v104f process interruption remains unproven; its
durable fastest checkpoint ends in update 70676; the canonical geometry contains 60 m rather than 120 m
of stored finish extension; the v2 pipeline now adds any missing distance in memory; held exploration now
preserves epsilon as an exploratory-step fraction; and the expert preset uses measured joint controls
rather than independent marginals. Feature v2 is intentionally incompatible with v1 encoder checkpoints
and must not be used with native demonstration pretraining until its frame-start control echo is realigned.

## How this was produced

* Static review by 12 dimension-specific reviewers (observation, model, reward, learner, replay,
  distributed runtime, environment/timing, evaluation/exploration, hyperparameters vs literature,
  tests/docs, PR-diff regressions, geometry), merged into 85 distinct findings, then adversarially
  re-verified per area. 70 of 85 verdicts completed before the budget ran out; the 15 unverified
  findings are all in the observation/model area and are marked "(unverified)" below. They are,
  however, backed by the measurements in the next two bullets.
* Empirical replay of Kuba's 10 expert laps (36.57-37.37 s, `demo-*.npz` on the
  `codex/experimental-backbones` branch) through the exact v104f `TrajectoryReward` and
  `BoundaryGraphFeaturePipeline`, plus a time-stretched 45.5 s copy of the same path.
* W&B history of the v104f run (`ivi7ihxk`, 15,740 rows) and the inventory of all 417 runs in
  `dsc-pjatk-warsaw/my-trackmania-agent`.
* Literature reference: pb4git/linesight (IQN, world-record TMNF), trackmania-rl/tmrl, IQN/QR-DQN,
  Ape-X, R2D2, SimbaV2 (values with file:line citations were fetched from the sources).

Finding ids (F001...) refer to the merged list; "measured" = reproduced numerically here.

## 1. Diagnosis: why 45 s and not 37 s

1. **Under-trained, then interrupted.** v104f is on the same trajectory as the only from-scratch run that
   ever went under 40 s (1.x v35a: 46.08 s at 301k transitions, 38.12 s at 1.44M). v104f was at
   45.63 s at 180k and 45.46 s at 333k when the run stopped producing episodes for 4.37 h after episode 401
   (evaluation batch 14 was due); the final queued checkpoint separately remained a 0-byte
   `.pt.tmp`, with no traceback establishing the shutdown cause. It had collected 14% of its 2.5M budget. The single most valuable change is
   therefore reliability + throughput so a run reaches 1.5M+ transitions (measured, W&B).
2. **The objective is lap-time-blind.** Undiscounted, a 36.6 s and a 45.5 s lap on the same path
   score 321.17 vs 320.98 (0.062% apart): progress (200/lap) and the projected-velocity term
   (0.0576·v·dt, integrates to distance, 112/lap) are fixed per lap; the time penalty is 0.02/s
   (0.23% of the return). All time pressure comes from γ = 0.99 per 50 ms (5 s horizon), which
   values a 1 s stall at 18% of the remaining value but makes the finish bonus (10) worth 0.006 from
   the start and only 3.7 five seconds out. W&B confirms: finished training laps returned 309.9 at
   74.6 s and 322.1 at 43.8 s, and the whole difference is collisions, not time (measured, F001/F022).
3. **Every evaluation lap collided** (`evaluation/collision_rate` = 1.0 from batch 2 on). A collision
   costs 2 with a 2 s cooldown the policy cannot observe (≈0.24 s of lap time equivalent); the
   optimum of this reward is a cautious, wall-touching lap (measured, F023).
4. **The policy cannot see what it needs to drive at the limit.** The 60-dim physics vector has 7
   constant-zero slots, no velocity direction (only |v|), no yaw rate, no previous control (masked
   twice), a raw global yaw that wraps 34 times per 10 laps, and unnormalised scales (km/h up to 313
   next to [-1, 1] curvature). Yaw rate is the most variable hidden quantity on the expert laps
   (std 0.93 rad/s) and is unrecoverable from one frame with an Identity temporal core (measured;
   F002/F004/F005/F006 unverified by a second reviewer).
5. **Exploration cannot discover a later braking point.** ε-greedy draws a *single 50 ms* random
   action (hold 1) from a distribution with 45% braking and 80% intermediate steering, whereas the
   expert brakes 5.5% of the time, steers only at 0/±1 and holds actions for 160 ms median / 324 ms
   mean. The schedule decays on learner updates, so ε was still 0.139 when the run died and would
   have hit 0.002 at 22% of the budget (verified, F007/F008/F034).
6. **Learning speed is capped by one 20 Hz actor and UTD 0.25**, not by the GPU (25% busy): the
   learner is credit-bound at 4.92 updates/s and 90% of each update is Python replay sampling.
   Evaluation on the sole collector costs 13-17% of wall-clock (verified, F009/F017).
7. **Architecture** is a second-order issue: the model is correct (dueling/IQN/double-Q verified),
   but the GNN mean-pools 44 ordered points into an order-agnostic summary and the per-node
   LayerNorm erases near/far magnitude; a flat MLP over the ordered points (what Linesight uses)
   is at least as expressive (F003/F029, unverified).

## 2. Top changes ranked by expected gain per unit of effort

| # | Change | Why | Effect | Effort | Risk |
|---|---|---|---|---|---|
| 1 | Fail fast on actor stalls, clean recoverable temporary-write failures, and checkpoint every 5k updates | v104f produced no new episodes for 4.4 h and its final queued write left a 0-byte temporary; any hot-fix now blocks resume (F015/F016/F048/F014) | preserve more work and surface a stopped collector; automatic restart remains a follow-up | S-M | low |
| 2 | Time-sensitive reward + longer horizon: `time_penalty_per_second` 1.0, `finish_reward` 70, γ 0.995, n-step 5, collision penalty per contact step without cooldown | objective currently blind to 8.9 s of lap time; finish is a value cliff; hidden cooldown state (F001/F022/F023) | optimum becomes the fast lap; faster laps become visible in the training return | S (config) | medium: Q scale doubles; watch `learner/q_target_max` |
| 3 | Observation v2 (`graph_iqn_v2`): body-frame velocity, yaw rate, input echo, relative heading, normalised scales | 7 dead inputs, no rotation/direction signal (F002/F004/F005/F006/F051) | test whether a fresh online policy learns fewer collisions and sharper braking; tensor shapes stay fixed but feature semantics and encoder state do not | S (in this PR) | medium: online scratch only; native demo controls leak same-frame targets |
| 4 | Temporally extended, expert-shaped exploration: `exploration_hold_steps` 6, ε floor 0.01 on a transition schedule, neighbour-steering half the time; later make the mode weights configurable | 50 ms pulses cannot change a line; global samples should match the expert's measured joint control distribution (F007/F008/F034/F038) | discovers later braking points / different lines | S (config) + M (weights) | low |
| 5 | Truncate n-step returns at exploratory actions (Linesight `discard_non_greedy_actions_in_nsteps`) | 36-72% of 3-step targets contain a random action (F035) | cleaner targets | M | low |
| 6 | Take evaluation off the collector: 2 trials while training, full 5 only when a candidate beats the leader; evaluate on a cadence in updates | 13-17% of wall-clock idle learner (F017) | +15% data per hour | M | low |
| 7 | Replay sampling off the critical path (real prefetch thread / vectorised PER materialisation), then UTD 0.5 | GPU 75% idle; UTD is the cheapest learning-speed lever once sampling is not 90% of the update (F009/F044) | 2× updates per transition | M | medium |
| 8 | Drop the adaptive gradient clipper (or `clip_factor` 2.0) | damped 39.6% of updates by 0.69 on average, an unintended ~12% lr cut main never had (F043) | cleaner ablations | S | low |
| 9 | Compact action set: remove the 26 brake-tap actions (and later the 39 gas-off × intermediate-steer actions the expert never uses) via `compact_action_ids` + a head with a configurable `action_count` | expert uses 7 of 78 actions; taps are 1-2 physics ticks (F012/F055) | faster credit assignment | M (head is hard-coded to 78) | medium |
| 10 | Flat MLP / 1-D conv over the ordered lookahead instead of mean-pooled GNN; keep SimbaV2 + IQN | order-agnostic readout (F003/F029, unverified) | unknown; run as an ablation after 1-4 | M | medium |

Not recommended now: Linesight's fixed-horizon "mini-race" objective (a large rewrite of replay,
inputs and inference); revisit only if 2 does not make lap time visible in the return.

## 3. Bugs and defects to fix regardless of the experiment plan

### (i) Training signal

* `trackmaniarl/trackmania/reward_step.py:257` / `reward_components.py:208`: return is lap-time-blind
  (§1.2). Config fix in `sub37-v105-candidate.yaml`; code follow-up: continuous (interpolated)
  progress instead of 2 m stations (quantisation error std 0.077 = 27% of the per-step progress
  term, F026).
* `trackmaniarl/trackmania/reward_components.py:23-40`: collision penalty with hidden 2 s cooldown
  (F023). Config fix: cooldown 0, per-step penalty.
* `trackmaniarl/trackmania/reward_components.py:43-48` + `reward_step.py:114`: the monotone max
  progress index drives the off-track test and tangent; a bounce/spin > 37 m behind the max index
  is killed as `off_track` (measured on the asset, F025). Config mitigation: `nearest_backward_points`
  40; code follow-up: separate non-monotonic tracker shared with the feature pipeline.
* `trackmaniarl/experiments/graph_iqn.py:23-28, 253-304`: masked previous control, 7 dead dims,
  wrapping global yaw, no body-frame velocity / yaw rate, unnormalised scales (F002/F004-F006/F051).
  Fixed opt-in by `graph_iqn_v2.py` in this PR.
* `trackmaniarl/experiments/graph_iqn.py:317-323`: the pipeline slices the asset to
  `recorded_count` and discards its 60-point (60 m in the canonical 1 m asset) finish extension,
  so the lookahead collapses near the finish (measured, F027). **Fixed in `graph_iqn_v2.py`**:
  the v2 pipeline keeps the stored extension and extrapolates any distance missing from its 110 m request.
* `trackmaniarl/core/replay/store_materialization.py:101-112`: n-step returns are not truncated at
  exploratory actions (F035). Follow-up: `act_with_info` → `explored` flag column → truncate.
* `trackmaniarl/algorithms/optimization.py:66`: adaptive clipper at `clip_factor` 1.0 (F043).
  Config fix: `adaptive_gradient_clipper: null`.
* `trackmaniarl/distributed/coordinator_rpc.py:286-296`: `epsilon_decay_updates` silently overrides
  `epsilon_decay_transitions`; `learner.exploration_epsilon` is dead in distributed runs (F034).
  Config fix; follow-up: warn in `trackmaniarl validate` when both axes are set.

### (ii) Durability and wall-clock

* **Silent actor stall (the v104f "crash").** Heartbeats come from a separate thread
  (`actor_background.py:247-273`) that keeps the actor "alive" while the collection thread is
  not advancing. No log establishes where the collection thread was blocked after episode 401;
  an evaluation was due, but the reset boundary is only a hypothesis. The learner only logs on updates, so
  4.37 h passed with no data and no alarm (`coordinator_learning.py:84-94`, F015).
  **Partly fixed in this PR**: `distributed.actor_stall_timeout_s` (opt-in) makes the heartbeat
  thread compare the age of the last environment step (`actor_watchdog.py`) and, past the limit,
  request a stop and hard-exit the actor 60 s later, so the launcher sees a dead actor and the
  learner shuts down with a final checkpoint instead of idling. Follow-ups: `health/seconds_since_last_ingest`
  on the coordinator and a launcher that restarts a dead actor instead of stopping the learner.
* `trackmaniarl/core/builtins.py:171-183`: checkpoint temp stays 0 bytes for the whole pickle phase
  and was never unlinked on a recoverable write failure (F016). **Fixed in this PR** for Python
  exceptions and replace failures. A forced process kill or power loss cannot execute cleanup.
  Follow-ups: the launcher should wait for `train/checkpoint_completed` before `terminate()`
  (`commands/training.py:231-240`); `checkpoint_interval_updates` 5000.
* `trackmaniarl/core/fingerprint.py:19-57`: the run fingerprint hashes every `.py` file and the
  whole spec, so any bug fix or knob edit forbids resume (F014). v104f *is* resumable today from
  `fastest-eval-policy-00070434-at-update-00070676.pt` may contain resumable state, but exact resume
  remains unverified and requires unchanged code and config. Fix: fingerprint only data-compatibility semantics
  (feature pipeline, action table, model architecture, reward config, replay class) and add
  `--allow-fingerprint-mismatch`.
* `trackmaniarl/distributed/coordinator_checkpoint.py:95-102, 194-218`: leader checkpoints are full
  replay checkpoints that prune the journal and clear Adam state; only the newest checkpoint of any
  family is resumable (F032/F046). Fix: policy-only leader checkpoints that never prune.
* `trackmaniarl/distributed/actor_collection.py:88-107`: evaluation blocks the sole collector
  (F017). See §2 item 6.
* `trackmaniarl/distributed/coordinator_submission.py:82-92`: hard policy lag is measured against
  the episode-frozen version; the margin to `hard_policy_lag_updates` 1000 is 25% at 150 s episodes
  (max observed 748, F033 low). Config: 2000.

### (iii) Minor / verified but low impact

* `trackmaniarl/trackmania/actions.py:22` + `control.py:112-114`: the brake tap is a 10 ms
  blocking sleep = 1-2 physics ticks; the expert never brakes < 40 ms (F012). Audit tap vs no-tap
  outcomes offline before removing the 26 tap actions.
* `trackmaniarl/trackmania/environment_step.py:63`: the 150 s time limit is a penalised
  termination in the environment but a truncation in the actor (F042; 2 episodes in v104f).
* `trackmaniarl/distributed/actor_background.py:143-156`: a permanently rejected spool file is
  re-queued on every restart (F013). Unreachable with the local launcher; add UNKNOWN/INTERNAL to
  the retryable codes and move rejected files to `spool/rejected/`.
* `trackmaniarl/algorithms/value_based/update_helpers.py:113-124`: one of four encoder passes per
  update is unused and two run with autograd (F073). Learner is not the bottleneck; cleanup only.
* Quantile-Huber loss is 1/64 of the Dopamine scale and PER weights are normalised by their sum
  (F071/F072): identical to `main`, invariant under Adam; do not "fix" without re-tuning clipping.

## 4. Observation vector v2 (implemented in `trackmaniarl/experiments/graph_iqn_v2.py`)

Shapes stay `physics` (60,) and `track` (3, 88), so the encoder, head and checkpoints keep their
layout; only the meaning of the 60 scalars changes.

| slot | name | source | scale | why |
|---|---|---|---|---|
| 0 | speed | field 16 | m/s / 100 | replaces km/h (313 max) |
| 1 | forward_velocity | v·heading (fields 7,9 vs 10,12) | / 100 | direction of motion, confirmed missing |
| 2 | lateral_velocity | v×heading | / 20, clip ±1 | slip/drift signal (expert p95 1.2 m/s, max 17) |
| 3 | progress | station / N | 0..1 | unchanged |
| 4 | yaw_rate | Δyaw / Δt (wrapped) | / 3 rad/s, clip ±1 | most variable hidden quantity (std 0.93 rad/s) |
| 5 | acceleration | Δspeed / Δt | / 50 m/s², clip ±1 | replaces dt-dependent "accel per 10 ms tick" |
| 6-7 | heading_sin/cos | heading vs local track tangent | unit | replaces global yaw that wraps at ±π |
| 8 | gear | field 18 | / 5 | unchanged |
| 9 | pitch | asin(dir_y) | rad | unchanged |
| 10-11 | input_gas, input_brake | game input echo (fields 31, 32) | 0/1 | previous control; v1 masked it |
| 12-13 | front slips | fields 19, 20 | 0/1 | unchanged |
| 14 | input_steer | field 30 | -1..1 | previous control |
| 15-58 | curvature[44] | as v1 | clip ±1 | unchanged |
| 59 | skidding_wheels | field 27 | / 4 | non-zero on 17% of expert frames |

Removed: jerk (noise), global yaw, the masked action slot. Not added (no information on this map,
measured): adherence (always 1), flying_duration (always 0), surface (constant 16). Nice-to-have
later: rear slips, rpm, race time / remaining distance, a short history stack (2-3 frames) if the
Identity core is kept. The encoder v2 drops the control mask and the joint LayerNorm(384) in front
of SimbaV2 (which defeated its shift channel, F052) and scales the track tensor to O(1) metres/50.

## 5. Architecture

Keep SimbaV2 + dueling IQN (all verified correct: dueling math to 6e-8, IQN taus and double-Q as
in the paper, 1.48M params, 1.6 ms CPU inference). First model ablation after the config/feature
changes: replace `TrackNeighborGraph` (mean pool over 44 nodes, per-node LayerNorm) with a flat
MLP over the ordered 264 coordinates (Linesight's design; ~50k params) or a 1-D conv with stride
over the 44 stations; the GNN's symmetric neighbour aggregation is reverse-invariant, so "which
point is 10 m ahead vs 100 m ahead" survives only through coordinate magnitude, which the per-node
LayerNorm removes (F003/F029, unverified by a second reviewer). Fallback: keep the GNN but remove
the per-node LayerNorm and add a learned positional embedding. Also drop `deterministic: True`
(`algorithms/execution.py:24`) once UTD is raised: it forces the slow CUDA `index_add_` path.

## 6. Reward and horizon (v105a)

Per-step reward at the expert pace (55 m/s, 50 ms): progress 0.255 + projected velocity 0.158 −
time 0.050 = **+0.364** (v104f: +0.412 − 0.001). At 10 m/s: +0.025. Net per-step reward stays
positive above about 6.7 m/s, so ordinary racing remains preferable to terminating (the
failure mode of "just raise the time penalty" under γ < 1, F001).

| lap | progress | velocity | time | finish | total |
|---|---|---|---|---|---|
| 36.6 s | 200 | 112 | −36.6 | 70 | 345.4 |
| 45.5 s | 200 | 112 | −45.5 | 70 | 336.5 |
| crash at mid-lap | ~100 | ~56 | −18 | −2 | ~136 |

The two laps now differ by 8.9 (2.6%) undiscounted instead of 0.2 (0.06%); a crash still costs
~200. With γ 0.995 (10 s horizon, n-step 5) the continuation value at the line is
0.364/(1−0.995) ≈ 73, hence `finish_reward` 70: no value cliff at the finish (F022). If γ is changed,
re-derive `finish_reward` = per-step reward / (1−γ). Expected Q scale ≈ 70-80 (v104f: 37); watch
`learner/q_target_max` and `learner/td_abs_mean` for the first 50k updates.

## 7. Exploration and evaluation protocol

* Exploration: ε on a *transition* schedule 0.30 → 0.01 over 500k, `exploration_hold_steps` 6
  (300 ms, between the expert's median and mean action run length), neighbour-steering exploration
  for half the events. The selector converts epsilon to an event probability so a six-step hold
  still spends approximately epsilon of timesteps exploring. **In this PR**:
  `exploration_weights_preset: expert_keyboard` (`actions.py`) draws globally random actions from
  the joint controls measured over all 23 accepted demonstrations; neighbour moves remain separate.
  Follow-up: truncate
  n-step targets at exploratory steps.
* Evaluation: 5-trial medians of a deterministic policy have ≥ 2.4 s spread and no confirmation
  re-run (F039). Protocol: 2 trials every 50 episodes while training; when a batch beats the leader,
  immediately re-run 5 trials with the same snapshot and promote only on the confirmed median; log
  min/max/spread; keep `evaluation_stop_consecutive_batches` 2.

## 8. Ablation order

The bundled v105 YAML predates independent review and is not itself a causal ablation. The project
owner selected its corrected runnable copy as a discovery run: first test whether the combined proposal
beats V104F, then remove or restore one mechanism per run to identify the gain. Follow the exact current
decision and gates in `SUB37.md`; V104H is paused while the bundle-first run is active.

Treat the PR ideas as separate later branches:

| branch | sole experimental mechanism | first decision |
|---|---|---|
| reliability baseline | watchdog plus more frequent durable checkpoints; learning semantics unchanged | the run stops cleanly on a reproduced stall and preserves the newest completed checkpoint |
| objective | time-sensitive reward and its coupled horizon/finish-value calibration | equal-transition finish rate, median, Q scale and collision count |
| observation | online-scratch feature/encoder v2, with no demonstrations or warm start | equal-transition finish rate, median and collision count |
| exploration | corrected held-event probability plus measured joint keyboard preset | action fractions, TD error and equal-transition median |
| ordered encoder | flat MLP or 1-D convolution over lookahead | equal-transition median |
| throughput | replay prefetch, then UTD 0.5 | updates/s and backlog before policy quality |

Stop rule for any run: `evaluation/finish_rate` < 0.6 for 3 consecutive batches after 200k
transitions, or `learner/q_target_max` > 3× the per-step-reward/(1−γ) bound.

## 9. Operator prompt (for an autonomous iteration cycle)

```
You operate the TrackmaniaRL sub-37 experiment on map trackmaniarl-test (repo Palamabron/TrackmaniaRL,
branch feature/trackmaniarl-2.0-modular-refactor). Objective: lowest deterministic
evaluation/finish_time_median_s with evaluation/finish_rate = 1.0 over 5 trials. Baselines: v104f
45.46 s at 333k transitions; 1.x v35a 38.12 s at 1.44M; retained human best 36.035 s. Current
decision and runnable config are recorded in my-trackmania-agent/agent-notes/SUB37.md; this audit
and docs/reviews/sub37-v105-candidate.yaml are hypothesis references only.

Each cycle:
1. Pull the W&B run (project dsc-pjatk-warsaw/my-trackmania-agent) and read, per evaluation batch:
   evaluation/finish_time_median_s, evaluation/finish_rate, evaluation/collision_rate,
   evaluation/policy_version; per update: learner/q_selected_mean, learner/q_target_max,
   learner/td_abs_mean, learner/loss_total, learner/gradients_norm, pipeline/utd,
   pipeline/update_backlog_s, performance/updates_per_s, performance/cumulative_transitions_per_s;
   per episode: episode/return, episode/collision_count, episode/exploration_epsilon,
   episode/timing_step_race_ms_mean, episode/termination_* counts; health/checkpoints_queued vs
   health/checkpoints_completed and the age of the last train/episode row.
2. Health first. If no train/episode row for > 2 x max episode length (300 s) while
   health/active_actors = 1: treat the actor as stalled. The watchdog stops the actor but does not
   restart it. Resume explicitly from the newest compatible completed checkpoint only after
   checking that checkpoints_completed matches checkpoints_queued.
   If a .pt.tmp exists with 0 bytes, delete it. Never edit code or config on a run you intend to
   resume (fingerprint).
3. Learning. Compare the eval median at equal transitions with the previous run of the same
   ablation lineage. Roll back the last single change if the median is worse by > 1.5 s at 300k
   transitions or finish_rate < 0.6 for 3 batches after 200k. If learner/q_target_max exceeds
   3 x per-step-reward/(1-gamma) (about 220 for v105a), halve time_penalty_per_second and restart.
   If episode/collision_count is not below 3 by 300k, raise collision_penalty to 0.3 per step.
4. Treat v105a as the one explicitly approved bundle-first discovery run. If it beats V104F, use
   backward ablations that remove or restore one mechanism per run. Record for each run: config diff,
   transitions reached, best confirmed eval median, collisions/episode at the end, and the reason it
   stopped, in docs/reviews/experiment-log.md.
5. Promote a checkpoint only after a 5-trial confirmation re-evaluation of the same snapshot; the
   release gate is 2 consecutive batches with finish_rate 1.0 and median <= 37.0 s.
```

## 10. Refuted or not worth pursuing (checked, do not re-investigate)

* Decision period "drifts to 53 ms": the human recordings are 75 Hz frames with 10/20 ms ticks;
  the live actor logs `episode/timing_step_race_ms_mean` 50.0-50.2 ms, p99 57-61 ms (W&B).
* Q over-estimation: `q_selected_mean` tracks `q_target_mean` within 0.1, max-Q ≈ r/(1−γ), no spikes.
* Exploration collapse: ε was still 0.139 at the end of v104f.
* OOM / gRPC size limits / W&B exceptions as the crash cause: replay is 0.5 GB at 356k, all
  payloads < 6 MB of a 16 MiB cap, the tracker is fire-and-forget.
* `velocity_to_mps_scale` / `limit_progress_by_kinematics` defaults changed on the RunSpec 2.0 branch, but
  v104f sets both explicitly (1.0 / false): no effect on the project; template users beware.
* Windowed nearest-point search losing the car: 0 mismatches vs brute force over 6,953 expert
  decision frames; the lookahead always contained the braking apex (max 73 m of 110 m).
* Action-space resolution: the expert uses exactly 0/±1 steering and binary gas/brake (7 of 78
  actions), so finer bins are not the bottleneck.
* Adaptive clipper causing the crash (no non-finite gradient in 76,583 updates), bf16 precision
  (v104f is fp32), PER weight normalisation, loss scale, unused encoder pass: correct as facts,
  no lap-time effect.
* Demonstrations by themselves: across 74 evaluated runs, runs using demos had a worse
  median-of-best (41.1 s) than runs without (38.3 s); every result under 38 s came from anchored
  fine-tuning of the v35a lineage.

## 11. What the W&B history says about what worked (74 evaluated runs)

Best lap-time lineage: 1.x R2D2 recurrent (sequence 64), n-step 8, γ 0.999, UTD 0.005-0.02,
`action_repeat_frames` 2, racing-line features on, reward with `time_penalty_per_second` 0.5,
`time_attack_bonus_scale` 25, `steering_delta_penalty` 0.02, `collision_penalty` 1.5. The
78-action / 50 ms / gamepad / from-scratch group (v97a-v104f) is the weakest group (3 of 12 ever
qualified; best 44.37 s). This is confounded with lineage and budget, but it is consistent with §1:
the good lineage had a time-sensitive reward and ran 4× longer.
