# v27: deterministic-evaluation stability

Successor to the v26 experiment. Target metric order: deterministic finish rate
first (>= 90%), then deterministic mean below 45 s, then stable sub-40 s.

## Diagnosis of the v26 collapse

v26 evidence: training windows improved monotonically (32% -> 82% -> 84%
finish, best 46.01 s), while deterministic evaluation went 9/10 (update 17.5k)
-> 8/10 (33.5k) -> 1/10 (47.4k) with failures at 3-34% progress.

1. **Replay narrowing is the primary cause and its timing matches exactly.**
   At UTD 0.35, update 33.5k corresponds to ~96k collected transitions: the
   100k FIFO buffer still contained every early high-epsilon, crash-rich,
   diverse trajectory, and greedy evaluation held 8/10. Update 47.4k
   corresponds to ~135k transitions: the first ~35k transitions had just been
   evicted, leaving only recent near-greedy fast laps (epsilon ~0.017). The
   Q-function then only sees a narrow "tube" of states around the behavior
   policy. A frozen greedy rollout from the standing start follows a subtly
   different line, exits the tube early (3-34% progress), and off-distribution
   Q extrapolation picks arbitrary actions. Training rollouts cannot show this
   because the behavior policy is, by construction, always in-distribution.
2. **Mid-episode policy swaps hid the problem.** The actor refreshed weights
   every ~5 s, so one training lap blended ~10 policy versions plus epsilon.
   No single frozen policy was ever measured during training; rollout finish
   rate was therefore not evidence of greedy quality.
3. **The learner ingested rollouts only when out of update credit, one chunk
   at a time.** Any transient stall (checkpoint, evaluation, W&B) created a
   standing queue that could never shrink (~50-58 chunks, 100-115 s delay).
   Consequences: the learner trained on ~2-minute-old data, and the
   `policy_lag_updates` metric (measured at ingest) reported ~1000 while the
   actor's true weight lag was ~35-70 updates.
4. **Gradient clipping at 100% is cosmetic, not a defect.** The IQN loss sums
   over 64 prediction quantiles, so a norm of ~210-247 against clip 10 simply
   rescales gradients by a near-constant factor; Adam's normalization cancels
   a constant rescale. Q/target alignment (~41) and TD (~1.05) are healthy.
   Do not retune the optimizer in v27; isolate the data/stability hypothesis.

## Code changes in this branch

- Coordinator drains the entire rollout backlog every loop iteration; update
  pacing via `update_credit` is unchanged. Queue depth and ingest delay drop
  to ~0, and the lag metric becomes the actor's true weight lag.
- The actor freezes one policy snapshot per training episode. Every episode
  now measures a single policy version (stamped in the episode summary and
  console line), making training finish rate an honest per-version signal.
- `_IQNPolicy` reports the greedy action gap (`last_q_margin`, top-1 minus
  top-2 Q). Episode and evaluation summaries log `q_margin/mean`,
  `q_margin/min` and `q_margin/start_mean` (first 50 decisions) so start-state
  brittleness is visible directly.
- Every evaluation batch is aggregated into `eval/summary` (finish rate,
  mean/median/best time, sub-40 rate, policy version) and any strictly better
  batch (finish rate, then mean time) writes a checkpoint immediately and
  logs `eval/best_checkpoint`. The final deliverable is the best-eval
  checkpoint, not the last one.
- `InMemoryReplayStore.load_state_dict` accepts restoring into a larger
  capacity (shrinking still fails), so v27 can resume the v26 checkpoint with
  a bigger buffer. PER priorities re-seed at the stored maximum after growth.

## v27 configuration delta (apply to a copy of the local v26 YAML)

Create `trackmania-iqn-lidar-v27.yaml` as a copy of the v26 file and change
only:

| Key | v26 | v27 | Why |
| --- | --- | --- | --- |
| `run_id` | `...-v26` | `...-v27` | new experiment |
| `components.replay_store.kwargs.capacity` | 100000 | 300000 | keep diverse data; eviction of early data triggered the collapse (~3-4 GB RAM at 300k with the 8-frame stack; use 200000 if RAM-bound) |
| `distributed.epsilon_final` | ~0.02 (effective ~0.017) | 0.04 | keep the behavior tube wide enough that the greedy line stays in-distribution |
| `distributed.max_inflight_chunks` | 4 | 1 | in-order chunk delivery locally; removes the out-of-order episode-step class the replay safeguard defends against |

Reward, model, LR, clip, gamma, n-step, batch size, UTD and PER stay exactly
at v26 values: v27 tests one hypothesis (data diversity + measurement
honesty), not a new optimizer or reward.

Optional, config-only: `training.evaluate_every_episodes: 50` for a denser
deterministic signal (~10 min of eval per ~45 min of collection).

## Resume decision

**Resume from the latest v26 checkpoint (~50k updates), without
`--reset-replay`.** The behavior policy still completed 46-48 s laps at 47k;
the defect is the greedy argmax off-tube, which continued training with a
wider buffer and a higher epsilon floor repairs. Keeping the replay preserves
100k transitions of recent good laps; new diverse data accumulates next to
them in the grown buffer. Counters resume past the epsilon decay, so the
effective epsilon becomes the new 0.04 floor immediately. `--reset-replay`
would also zero all counters and re-enter the 40k-transition epsilon=1.0
warmup; do not use it here.

PowerShell, from the repository root:

```powershell
git merge feat/v27-deterministic-stability
uv sync --group dev
uv run poe fmt; uv run poe types; uv run poe test
cd my-trackmania-agent
Copy-Item trackmania-iqn-lidar-v26.yaml trackmania-iqn-lidar-v27.yaml
# apply the key delta above, then:
uv run tmrl validate trackmania-iqn-lidar-v27.yaml
$ckpt = Get-ChildItem artifacts\trackmania-iqn-lidar-v26\checkpoints\distributed-update-*.pt |
    Sort-Object Name | Select-Object -Last 1
uv run tmrl resume trackmania-iqn-lidar-v27.yaml $ckpt.FullName
```

Fallback: if the resumed policy does not reach >= 6/10 deterministic finishes
within ~15k further updates, start v27 from scratch (same YAML,
`uv run tmrl train trackmania-iqn-lidar-v27.yaml`) and accept the slower ramp.

## What to watch

- `distributed/ingest`: `rollout_queue_depth` ~0-2 (was 50-58),
  `queue_delay_s` < 3 s (was 100-115 s), `policy_lag_updates` ~30-100 (was
  ~1000). If these do not drop, the drain fix is not active.
- Console episode lines: `policy=<version>` constant per episode;
  `q_margin(start=..)` trending up over training. Near-zero start margins
  while evaluation fails at the start confirm residual brittleness.
- `eval/summary` finish rate is the primary metric; `eval/best_checkpoint`
  marks the artifact to keep. Gate 1: >= 9/10 on two consecutive evaluations.
  Gate 2: deterministic mean < 45 s. Gate 3: sub-40 rate, then the 20-trial
  release benchmark in `docs/benchmark-test-3.md`.
- If greedy evaluation still degrades while q_margin stays healthy and the
  queue metrics are clean, the next isolated candidates for v28 are: an EMA
  weight copy used only for evaluation/deployment, and a learning-rate decay
  for the polishing phase. Change one at a time.
