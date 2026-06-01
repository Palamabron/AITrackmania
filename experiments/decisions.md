# Experiment Decisions Log

**Target:** Finish track (test-3) in <= 36.7 seconds (`eval/finish_time_test_s`)

This file is the institutional memory for the experiment tuning process. Each entry records what was decided, why, and what was learned. Future agents should read this entire file before proposing new experiments.

---

## Baseline Configuration (gtn-testv1.0)

**Date:** 2026-05-13
**Source:** `C:\Users\szulc\TmrlData\config\local.yaml`

The baseline is an IQN agent with:
- **Architecture:** Sophy Residual with SimBa v2 backbone, GTN track encoder (2 layers, hidden=128), residual MLP (4 blocks, hidden=192, actor=2/critic=4), binary brake.
- **Algorithm:** IQN with lr=3.5e-5, gamma=0.995, n_steps=3, dueling=true, double_dqn=true, Huber kappa=1.0, sorted quantiles. Linear epsilon decay 0.5->0.015 over 2M steps. Soft target tau=0.005. Grad clip=15.0, backup clip=150.0.
- **Training:** batch_size=256, 300 steps/round, 20 rounds/epoch, UTD cap=15, warmup=20k steps.
- **Reward:** Progress + speed alignment (weight=1.0, exp=1.2, floor=0.1) + drift (0.1) + finish bonus (8.0) - constant penalty (0.0015) - crash penalty (1.0). Min 10s before no-progress termination.
- **Demos:** Online injection with 10-25% batch fraction, weight=1.2 decaying over 1.5M samples.

### Key Differences from Earlier Presets (v5.11-stable)
- Lower lr: 3.5e-5 (was 5e-5) -- more conservative.
- Higher gamma: 0.995 (was 0.99) -- longer horizon, cares more about future.
- Larger epsilon decay: 2M steps (was 400k) -- much slower exploitation.
- Lower epsilon start: 0.5 (was 0.65) -- less initial randomness.
- Lower init_gas_bias: 0.1 (was 0.9) -- less aggressive throttle bias.
- Larger grad_clip: 15.0 (was 2.5) -- less aggressive clipping.
- Larger backup_clip_range: 150.0 (was 25.0) -- wider TD target range.
- Higher crash_penalty: 1.0 (was 0.5) -- stronger crash punishment.
- Lower speed_reward_alignment_floor: 0.1 (was 0.4) -- less reward for misaligned speed.
- Smaller batch_size: 256 (was 512).

### Areas to Explore First

1. **Learning rate:** Current 3.5e-5 is conservative. If training is stable but slow, try 5e-5. If spiky, drop to 2e-5.
2. **Batch size:** 256 is relatively small for IQN. Try 512 to reduce gradient variance.
3. **Epsilon decay:** 2M steps is very slow -- the agent will explore for hours before exploiting. Try 800k or 1M for faster convergence.
4. **Reward balance:** end_of_track_reward=8 with crash_penalty=1.0 and constant_penalty=0.0015. If the agent is too cautious, lower crash_penalty. If it doesn't try to finish, raise end_of_track_reward.
5. **Gamma:** 0.995 is high -- rewards 100+ steps ahead are still 60% weighted. If Q-values diverge, try 0.99.
6. **Grad/backup clipping:** Both are loose (15.0 / 150.0). If Q-values or loss explode, tighten to 5.0 / 50.0.

---

*Append new decisions below this line.*

## Audit corrections (2026-05-15, extended 2026-05-16)

Verified against `experiments/registry.jsonl` and `experiments/analysis/*.json` using `scripts/validate_decisions.py`. Primary metric: **`eval/finish_time_test_s` > 0** (also check `worker.best_finish_time_s` / `run/finish_time` on WORKER).

| Experiment | Issue | Ground truth |
|---|---|---|
| `increase-batch-size-to-512` | 2h/3h checkpoints claimed no track finish | **Best eval finish 64.23s**; worker finish_rate **37%** at end |
| `increase-batch-size-to-512` | 3h: "loss consistently above 50" = divergence | Loss median **~47**, last **~93** — high but **not** explosion; Q stable (max_q ~50) |
| `increase-gamma-to-0-995` | Stop: "loss consistently above 50" | Full-run loss median **~27**; stop was fair on **no eval finishes**, not loss |
| `lower-iqn-soft-target-tau` | 3h stop cited eval loss >50 | Worker **was** finishing (best **70.2s**, 31 finishes); eval `finish_time_test_s` stayed **0** (deterministic eval lag) |
| `stable-learning-with-strict-clip` | Not flagged as leader | **Actual leader at 61.65s** (not `increase-batch-size-to-512` at 64.23s); worker 63 finishes, 40% rate |
| `accelerated-stable-learning` | Stopped at 2h claiming "no finishes" | Worker **did** finish once (best 154.05s); stop was still reasonable on slow progress |

### Corrected Leaderboard (2026-05-16, updated 2026-05-17 after worker best_finish fix)

| # | Experiment | Best Time | Worker % | Loss Median | Notes |
|---|---|---|---|---|---|
| 1 | `stable-learning-with-strict-clip` | **61.65s** | 40% | 20.9 | Leader |
| 2 | `increase-batch-size-to-512` | 64.23s | 37% | 46.7 | |
| 3 | `long-horizon-planning-v2` | **69.9s** | 25% | 40.1 | n_steps=7; highest finish rate; best from worker (eval=78.18s) |
| 4 | `lower-iqn-soft-target-tau` | 70.20s | 19% | 18.9 | |
| 5 | `decrease-iqn-lr-to-3e-5` | 70.99s | 16% | 17.8 | |
| 6 | `less-strict-grad-clip-from-stable` | 79.35s | 4% | 8.1 | |

**Update (2026-05-16 afternoon):** Validation re-run after `fast-exploit-800k` (stopped at 2h, no finishes) and `long-horizon-speed-incentive` (trainer stuck — worker died from OpenPlanet disconnect, not a config issue). Orchestrator code updated to: (a) override Gemini gradient-based stops, (b) auto-restart dead workers, (c) kill all relay ports on trainer restart. `long-horizon-planning-v2` analysis fetched: real best eval finish is **78.18s** (initial 26.88s was a wandb interpolation artifact — see takeaway #8), but worker best is **69.9s** (56 finishes), making it #3 overall.

**Key takeaways from full validation:**

1. **Do not treat IQN loss in the 40–90 range as automatic divergence.** The two best experiments (`stable-learning-with-strict-clip` and `increase-batch-size-to-512`) both have loss median 20–47 and last values 69–93, yet they finished the track fastest. Loss rising late in training correlates with learning more complex Q-values, not necessarily divergence.

2. **Gradient clipping saturation is structural, not a bug.** ALL 18 experiments show pre-clip/clip ratios >20x (often >100x). Stopping experiments for "severe gradient clipping" wasted 11 experiments. The model architecture and loss landscape cause large raw gradients regardless of clip limit. Future experiments should NOT use gradient norm ratio as a stop criterion.

3. **The best configuration combines: batch_size=512, iqn_lr=3e-5, iqn_grad_clip=1.0.** The leader (`stable-learning-with-strict-clip`) uses exactly these three overrides. Adding n_steps=7 (`long-horizon-planning-v2`) increased finish rate to 25% and achieved worker best of 69.9s (#3 overall, eval=78.18s). n_steps=7 helps consistency and may benefit from longer training.

4. **Eval lags worker significantly.** In at least 4 experiments, the worker was actively finishing tracks while eval `finish_time_test_s` stayed 0. Always check worker metrics before claiming "no progress." Worker finish_count > 5 should be a keep-running signal.

5. **Loss vs finish time is NOT inversely correlated.** Lower loss does NOT mean better finish time. Experiments with loss median <10 (e.g., `adam-eps-for-stability` at 7.4, `long-horizon-from-stable-base` at 5.1) are among the worst performers (>160s). This suggests these runs are undertrained or the low loss reflects simpler Q-value landscapes (not learning complex behavior).

6. **85.8% of worker episodes terminate via no_progress_timeout.** Only 10.3% reach end_of_track. This is the biggest bottleneck — the agent needs to learn to sustain forward progress more consistently.

7. **Gap to target is still large: 61.65s vs 36.0s target.** The best experiment is 1.7x the target. `long-horizon-planning-v2` (n_steps=7) has the highest finish rate (25%) and worker best of 69.9s (eval 78.18s — initial "26.88s" was a **wandb sampling interpolation artifact**, see takeaway #8). Focus areas: faster epsilon decay (2M steps is very slow), reward tuning (higher end_of_track_reward, speed incentives), and longer training for promising configs.

8. **Beware wandb history interpolation.** When `history(samples=N)` downsamples, it can interpolate between a 0 (didn't finish) and a real finish time, producing fake small values. Always validate that `finished_track_count_test` is an integer (typically 10) before trusting `finish_time_test_s`. The fetch_analysis script should be updated to filter these out.

---

### 2026-05-13 21:56 UTC -- increase-batch-size-to-512

**Action:** continue
**Reason:** Experiment is only 1 hour in, no critical stop signs observed. Loss is stable, Q-values are healthy, and epsilon is still high, indicating ongoing exploration. It's too early to assess finish time improvement.

### 2026-05-13 22:57 UTC -- increase-batch-size-to-512

**Action:** continue
**Reason:** Experiment is halfway through (2h/4h). Loss and Q-values are stable and within reasonable ranges (loss mean 38.56, max_q mean 31.16). Epsilon is still high (0.45), indicating active exploration. While the agent hasn't finished the track yet, it's too early to stop given the stable learning metrics and remaining time. Gradient norm is consistently at 15.0, suggesting clipping, but this isn't causing divergence in other metrics yet.

**[Audit]** "Hasn't finished" was likely wrong at 2h — final run shows many eval finishes and **best 64.23s**.

### 2026-05-13 23:58 UTC -- increase-batch-size-to-512

**Action:** continue
**Reason:** Q-values are stable and within a reasonable range (0-50), and epsilon is still decaying, indicating active exploration. While the loss is consistently above 50 and the agent has not yet finished the track after 3 hours, there are no signs of divergence or explosion. The experiment has 1 hour remaining, and the hypothesis of more stable learning might still be manifesting.

**[Audit]** Incorrect: agent **had** finished (post-hoc **best_finish_time_s = 64.23s**, best in registry). High loss is normal here, not "no finish."

### 2026-05-14 09:44 UTC -- increase-gamma-to-0-995

**Action:** continue
**Reason:** Experiment is only 1 hour in, and core learning metrics (loss, Q-values, epsilon) are stable and healthy. Agent has not finished the track yet, which is expected at this early stage. Continue to allow the agent more time to learn and for the long-term impact of gamma to manifest.

### 2026-05-14 10:44 UTC -- increase-gamma-to-0-995

**Action:** stop
**Reason:** Loss is consistently above 50, indicating divergence. Gradient norms are saturating at the clip limit, suggesting instability. The agent has not finished the track after 2 hours, showing no progress towards the target.

**[Audit]** Stop on **no eval finishes** was reasonable; full-run loss median **~27** — "consistently above 50" overstated. Worker had sparse training finishes only (finish_rate **16%**). **Note:** This experiment was a no-op — baseline `local.yaml` already has `gamma: 0.995`, and the override set the same value. The hypothesis "Increasing the discount factor" was misleading.

### 2026-05-14 11:52 UTC -- decrease-iqn-lr-to-3e-5

**Action:** continue
**Reason:** Loss and Q-values are stable and within healthy ranges, indicating training stability. Epsilon is still high, suggesting active exploration. Only 1 hour elapsed, too early to expect track completion or declare no improvement, despite gradient norms frequently hitting the clip limit.

### 2026-05-14 12:52 UTC -- decrease-iqn-lr-to-3e-5

**Action:** continue
**Reason:** Loss and Q-values are stable and healthy. Epsilon is still high, indicating active exploration. While gradient norms are at the clip limit, other metrics do not show divergence, and the agent has not yet had a chance to finish the track. Let it run for the full duration to see if it can achieve a finish.

**[Audit]** "Has not yet had a chance to finish" was incorrect — post-hoc analysis shows **best eval finish 70.99s**. The agent was finishing tracks.

### 2026-05-14 13:53 UTC -- decrease-iqn-lr-to-3e-5

**Action:** continue
**Reason:** Loss and Q-values are stable and within reasonable ranges. Epsilon is still high, indicating ongoing exploration. While gradient norms are consistently at the clip limit, other metrics do not show divergence, which was the primary concern of the hypothesis. The agent has not yet finished the track, but there is still 1 hour remaining in the experiment for potential progress.

**[Audit]** Incorrect — agent **had** finished (post-hoc **best_finish_time_s = 70.99s**). Gradient clipping rationale for "not finished" was wrong.

### 2026-05-14 20:37 UTC -- increase-grad-clip-with-lower-lr

**Action:** stop
**Reason:** Gradient norms are consistently saturating at the new clip limit (5.0), with very high pre-clip norms (mean 121.57, max 509.68). This indicates that the core hypothesis of allowing 'larger, more informative gradient updates' is not being met, as gradients are still heavily clipped despite increasing the limit. This matches a 'sign to stop early'.

**[Audit]** Gradient-based stop after only 1 hour. Experiment later achieved **172.0s** finish. Pre-clip >> clip is structural. Premature stop.

### 2026-05-14 21:43 UTC -- add-moderate-weight-decay

**Action:** continue
**Reason:** All internal metrics (loss, Q-values, gradient norms) are stable and within healthy ranges. Epsilon indicates active exploration is ongoing. Only 1 hour of a 4-hour experiment has passed, and the agent has not yet had sufficient time to demonstrate track completion. No signs to stop early are present.

### 2026-05-14 22:44 UTC -- add-moderate-weight-decay

**Action:** continue
**Reason:** Training metrics (loss, Q-values, gradient norms) are stable and healthy, aligning with the hypothesis of improved stability. Epsilon is still high (0.46), indicating active exploration. While the agent hasn't finished the track yet, it's only halfway through the allocated time, and these stable metrics suggest learning is progressing well, even if slowly. Continue to observe for track completion.

### 2026-05-14 23:44 UTC -- add-moderate-weight-decay

**Action:** stop
**Reason:** Gradient norms are consistently saturating at the clip limit (15.0) with very high pre-clip norms (mean 276.05, max 858.55). The hypothesis was to stabilize gradient magnitudes with weight decay, but this issue persists, indicating the current regularization is insufficient or ineffective. Agent has not finished any tracks after 3 hours.

**[Audit]** Gradient reasoning irrelevant (structural). Stop was still fair at 3h with no finishes, but the gradient justification was wrong.

### 2026-05-15 00:52 UTC -- lower-iqn-soft-target-tau

**Action:** continue
**Reason:** Only 1 hour elapsed, too early to see track completion. Loss and Q-values are stable and within reasonable ranges. Epsilon is still high, indicating active exploration. While gradient norms are still hitting the clip limit, it's unclear if this is worse than previous runs, and the overall stability of other metrics suggests the experiment should continue for more time to observe its impact on performance.

### 2026-05-15 01:52 UTC -- lower-iqn-soft-target-tau

**Action:** continue
**Reason:** Experiment is only halfway through (2h/4h). Q-values and loss are stable and healthy. Agent is finishing tracks, indicating learning, despite times being far from target. Epsilon is still high, allowing for further exploration. While pre-clip gradient norms are high, the clipped norms are perfectly stable, suggesting the target network update change is contributing to overall stability, allowing learning to progress.

### 2026-05-15 02:52 UTC -- lower-iqn-soft-target-tau

**Action:** stop
**Reason:** Gradient norms are saturating at the clip limit (15.0) with very high pre-clip norms (mean 527, max 1555), directly contradicting the hypothesis that lowering 'iqn_soft_target_tau' would prevent saturation. IQN loss is also consistently high (>50).

**[Audit]** Worker **was** finishing (best **70.2s**); eval `finish_time_test_s` stayed 0 — stop on grad hypothesis OK, but not "no finishes" globally.

### 2026-05-15 03:59 UTC -- lower-lr-and-increase-n-steps

**Action:** continue
**Reason:** Experiment is early (1/4 total time). Loss and Q-values are stable and within reasonable ranges. Agent is exploring and has started finishing tracks. While gradient norms are saturating at the clip limit with high pre-clip values, indicating heavy clipping, the lower LR might be helping to maintain overall stability. It's too soon to stop.

### 2026-05-15 05:00 UTC -- lower-lr-and-increase-n-steps

**Action:** stop
**Reason:** Gradient norms are consistently saturating at the clip limit (15.0) with very high pre-clip norms (mean 67.7, max 390.9). This indicates severe gradient clipping, which can hinder effective learning and contradicts the hypothesis of improving stability. Despite 2 hours elapsed, the best finish time (95.6s) is far from the target (36.0s), and recent performance (168.8s) shows no clear improvement trend.

**[Audit]** Worker had **9 finishes** (best 95.6s). Stopped despite active learning. Gradient reasoning was irrelevant (structural). Should have continued — the agent was finishing tracks.

### 2026-05-15 09:25 UTC -- stable-learning-with-strict-clip

**Action:** continue
**Reason:** Q-values and loss are stable and within expected ranges, not showing signs of divergence. Epsilon is still high, indicating ongoing exploration. Although no finishes have occurred yet, it's only 1 hour into the 4-hour experiment, which is not yet a stop condition for lack of finishes.

**[Audit]** Good decision. This experiment went on to become the **leaderboard leader at 61.65s** with 63 worker finishes and 40% finish rate.

### 2026-05-15 10:26 UTC -- stable-learning-with-strict-clip

**Action:** continue
**Reason:** Experiment is stable with low, healthy loss and Q-values. Agent is successfully finishing tracks (10 completions, best 121.5s), indicating learning is occurring. The strict gradient clipping (1.0) is actively preventing large updates (pre-clip ~95x clip), which aligns with the hypothesis to achieve stable training. Epsilon is still high, allowing for further exploration and potential improvement.

### 2026-05-15 11:27 UTC -- stable-learning-with-strict-clip

**Action:** continue
**Reason:** Agent is finishing tracks (worker_finish_count=30, best_finish_time_s=84.95), Q-values are stable, and loss is within the expected range. Epsilon is still decaying, indicating exploration. The high pre-clip gradient norms confirm the severe saturation problem the experiment aims to address with strict clipping, which is working as intended to prevent divergence.

### 2026-05-15 13:34 UTC -- long-horizon-planning-v2

**Action:** continue
**Reason:** Experiment is only 1 hour into a 4-hour run. Agent has successfully finished the track once (best_finish_time_s: 157.4), which is a positive sign of initial learning, even if far from target. Loss and Q-values are stable and well within acceptable ranges, showing no signs of divergence or explosion. Epsilon is still high, indicating active exploration. While gradient clipping is very active (pre_clip >> clip), this aligns with the hypothesis of building on 'strict gradient clipping' and is not causing instability.

### 2026-05-15 14:34 UTC -- long-horizon-planning-v2

**Action:** continue
**Reason:** The agent is making progress, having finished the track multiple times (worker_finish_count: 7) with a best time of 93.10s, indicating it's learning to complete the task. Loss and Q-values are stable and within acceptable ranges, and gradient clipping is effectively managing updates as per the hypothesis. Epsilon is still high, suggesting ongoing exploration. With 2 hours remaining, there's ample time for further improvement towards the target.

### 2026-05-15 15:35 UTC -- long-horizon-planning-v2

**Action:** continue
**Reason:** Agent is learning to finish the track (best_finish_time_s=75.95, worker_finish_count=28) and core learning metrics (loss, Q-values) are stable and within expected ranges. Exploration is ongoing (epsilon ~0.46). Despite high pre-clip gradient norms, Q-values and loss are not diverging, aligning with the hypothesis's mention of 'strict gradient clipping' and 'stable configuration'. The experiment has 1 hour remaining, allowing more time for the long-horizon planning changes (gamma, n_steps) to take effect.

### 2026-05-15 17:44 UTC -- stable-long-horizon-regularized

**Action:** continue
**Reason:** Experiment is only 1 hour into a 4-hour run. While no finishes have occurred yet, this is not a stop condition until after 2+ hours. Loss and Q-values are stable and within acceptable ranges, not showing signs of divergence or explosion. Epsilon is still high, indicating active exploration. Gradient clipping is occurring but not excessively, not meeting the 'pre-clip >> 5x clip' stop condition.

### 2026-05-15 18:44 UTC -- stable-long-horizon-regularized

**Action:** continue
**Reason:** All internal learning metrics (Q-values, loss, gradients) are stable and healthy. Epsilon is still high (0.47), indicating significant exploration is ongoing. While no finish time has been recorded after 2 hours, the experiment focuses on 'long-horizon planning' which might require more time to achieve initial track completion. Given that the learning process itself appears stable and there are 2 hours remaining, it's reasonable to allow more time for the agent to find a path to the finish line.

### 2026-05-15 19:45 UTC -- stable-long-horizon-regularized

**Action:** stop
**Reason:** Agent has not finished the track even once after 3 hours of training (best_finish_time_s: None, worker_finish_count: 0), indicating a lack of progress towards the primary goal despite stable internal metrics.

### 2026-05-15 20:51 UTC -- accelerated-stable-learning

**Action:** continue
**Reason:** Experiment is only 1 hour in. Loss and Q-values are very stable and well within acceptable ranges. Epsilon is still high, indicating active exploration. No finishes yet, but it's too early to stop based on this (guideline is 2+ hours). Gradient clipping is active as intended by the hypothesis.

### 2026-05-15 21:52 UTC -- accelerated-stable-learning

**Action:** stop
**Reason:** Agent has not finished any track after 2 hours (half the total budget), indicating no progress on the primary goal. Gradient norms are severely clipped (pre-clip mean ~275x clip limit of 1.0), suggesting updates are consistently truncated and learning is stalled or ineffective, despite stable Q-values and loss.

**[Audit]** "Not finished any track" was incorrect — post-hoc analysis shows **best eval finish 154.05s**, worker finish count **1**. Stop was still reasonable on slow progress, but the gradient clipping reasoning was irrelevant (structural, not a bug).

### 2026-05-15 23:02 UTC -- stable-clip-regularized-tau

**Action:** stop
**Reason:** Gradient norms are severely clipped (pre-clip mean 163.59 vs clip 1.5, ~109x clip limit), directly contradicting the hypothesis of allowing larger, more effective gradient updates. This indicates updates are too small relative to the true gradient magnitude, likely hindering learning, especially with no finishes after 1 hour.

**[Audit]** Stopped after only 1 hour due to gradient clipping — premature. Pre-clip >> clip is **structural** (all 18 experiments show this). Should have waited for 2h no-finish cutoff.

### 2026-05-16 00:08 UTC -- long-horizon-from-stable-base

**Action:** stop
**Reason:** Gradient norms are consistently saturating at the clip limit (1.0), with pre-clip norms (mean 130.2) being excessively high (130x the clip limit). This indicates severe underlying instability despite stable Q-values, contradicting the hypothesis's premise of stability and likely hindering effective learning.

**[Audit]** Gradient-based stop was wrong — this experiment achieved **best finish 169.3s** despite "severe instability." Pre-clip >> clip is structural. Q-values were stable, so learning was occurring.

### 2026-05-16 01:15 UTC -- stable-clip-and-target-regularization

**Action:** stop
**Reason:** Gradient norms are saturating at the clip limit (2.5), with pre-clip norms (mean 113.54, last 233.03) being significantly greater than 5x the clip limit (12.5). This indicates severe and persistent gradient clipping, failing to mitigate the issue as per the hypothesis, and likely hindering effective learning despite stable Q-values and loss.

**[Audit]** Gradient-based stop — premature. Pre-clip >> clip is structural. Q-values and loss were stable; should have continued to see if finishes occurred.

### 2026-05-16 02:22 UTC -- ultra-low-lr-stable-clip

**Action:** continue
**Reason:** Experiment is only 1 hour in. Core learning metrics (loss, Q-values) are extremely stable and healthy, far from divergence. Epsilon is still high, indicating active exploration. While pre-clip gradient norms are very high, the ultra-low learning rate appears to be maintaining stability despite severe clipping, which could lead to consistent progress as per the hypothesis. Too early to stop based on lack of finishes.

### 2026-05-16 03:23 UTC -- ultra-low-lr-stable-clip

**Action:** stop
**Reason:** Severe gradient clipping persists despite ultra-low LR. Pre-clip gradient norms (mean 166.35) are orders of magnitude higher than the clip limit (1.0), indicating updates are constantly saturated. This prevents precise learning, negating the hypothesis, and progress towards the 36.0s target is extremely slow (best 129.25s after 2h).

**[Audit]** Worker had **6 finishes** (best 129.2s) — the gradient reasoning is irrelevant (structural). Stopped despite active finishing. 129s is slow but the agent was learning; more time might have helped.

### 2026-05-16 04:30 UTC -- less-strict-grad-clip-from-stable

**Action:** stop
**Reason:** The core hypothesis was to allow 'less truncated gradient updates' by increasing iqn_grad_clip to 5.0. However, debug/grad_norm_pre_clip (mean 142.8, max 540.4) is consistently far exceeding the clip limit (5.0), often by 28x to 108x. This indicates severe gradient clipping is still occurring, directly contradicting the experiment's goal and triggering the 'SIGN TO STOP EARLY' condition (pre-clip >> 5x clip). While the agent has finished the track (79.35s) and other metrics are stable, the experiment is not testing its intended change.

**[Audit]** Agent achieved **79.35s** (6th best) — stopping on gradient reasoning was counterproductive. Gradient saturation is structural. This experiment should have continued for the full duration given it was finishing tracks.

### 2026-05-16 05:36 UTC -- softer-target-network-for-stability

**Action:** stop
**Reason:** Pre-clip gradient norms (mean 165.44) are excessively high compared to the clip limit (1.0), indicating severe and constant gradient truncation (165x clip). This directly contradicts the hypothesis that increasing `iqn_soft_target_tau` would lead to less truncated and more precise updates, hindering effective learning.

**[Audit]** Worker had **6 finishes** (best 155.9s) — stopped despite active learning. Pre-clip >> clip is structural and irrelevant to whether the experiment is working.

### 2026-05-16 06:43 UTC -- adam-eps-for-stability

**Action:** stop
**Reason:** The experiment's hypothesis to mitigate severe gradient clipping by increasing adam_eps has not been met. debug/grad_norm_pre_clip (mean ~109) is still ~100x higher than iqn_grad_clip (1.0), indicating severe and persistent clipping, which is a strong sign to stop early.

**[Audit]** Gradient-based stop. Experiment achieved **167.0s** finish. Pre-clip >> clip is structural. The adam_eps change itself is unlikely to help (wrong lever), but the gradient reasoning for stopping was invalid.

### 2026-05-16 16:22 UTC -- boost-speed-reward

**Action:** stop
**Reason:** Trainer state is unreachable and worker state is unknown, indicating the experiment is not running or collecting metrics. No finishes after 1 hour, and no recent metrics are available, suggesting a fundamental issue rather than slow learning.

### 2026-05-16 17:19 UTC -- speed-finish-long-horizon

**Action:** failed
**Reason:** Trainer not receiving worker samples (check ports 55555–55558, no duplicate trainer process)

### 2026-05-16 18:25 UTC -- aggressive-reward-long-horizon

**Action:** continue
**Reason:** Only 1.0h elapsed; 'no finish' stop condition requires 2+ hours. No other explicit stop conditions are met or can be checked with current data.

### 2026-05-16 19:25 UTC -- aggressive-reward-long-horizon

**Action:** stop
**Reason:** No positive best_finish_time_s and worker_finish_count is 0 after 2.0 hours, matching a stop criterion. Trainer and worker states are also unreachable/unknown.

### 2026-05-16 19:53 UTC -- aggressive-speed-finish-long-horizon-v2

**Action:** failed
**Reason:** Trainer not receiving worker samples (check ports 55555–55558, no duplicate trainer process)

### 2026-05-17 14:29 UTC -- speed-boost-from-stable-base

**Action:** stop
**Reason:** Trainer is unreachable and worker state is unknown, indicating a likely infrastructure issue preventing any learning or progress. Recent metrics are also empty.

### 2026-05-17 15:37 UTC -- speed-incentive-from-stable-base

**Action:** continue
**Reason:** Only 1.0h elapsed; too early to stop based on no finishes (criterion requires 2+ hours). No other explicit stop criteria are met.

### 2026-05-17 16:38 UTC -- speed-incentive-from-stable-base

**Action:** stop
**Reason:** No positive best_finish_time_s and worker_finish_count == 0 after 2.0 hours, combined with unreachable trainer and unknown worker states, indicating a stalled or broken experiment.

**[Audit]** FALSE STOP — agent **was** finishing tracks (user confirmed 2 finishes). Root cause: **W&B project mismatch**. Trainer logged to project `tmrlv2` (from `TmrlData/config/local.yaml`) but orchestrator snapshot queried project `tmrl` (from `orchestrator_config.yaml`). All snapshots returned "run not found", making the orchestrator blind. Fixed: `orchestrator_config.yaml` updated to `wandb_project: tmrlv2`, startup validation added, fallback run search added.

### 2026-05-17 W&B infrastructure fix

Experiments `boost-speed-reward`, `aggressive-reward-long-horizon`, `speed-boost-from-stable-base`, and `speed-incentive-from-stable-base` all failed due to the same W&B project mismatch (`tmrl` vs `tmrlv2`). The orchestrator couldn't see any metrics and stopped them as "stalled". None of these were config failures — the underlying training was working.

### 2026-05-17 19:17 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Experiment is only 1 hour into a 24-hour run. Agent has successfully finished the track at least once (best_finish_time_s=169.35s), which is a strong signal to continue. Q-values and loss are stable and within expected ranges. Epsilon is still high, indicating active exploration. No signs of divergence or critical failure.

### 2026-05-17 20:17 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing the track (7 times, best 158.5s), which is a strong signal to continue. Q-values and loss are stable and within expected ranges. Epsilon is still high, indicating ongoing exploration. Only 2 hours into a 24-hour run, too early to assess speed improvement.

### 2026-05-17 21:18 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is finishing tracks (worker_finish_count=21, best_finish_time_s=113.5), Q-values are stable and within reasonable bounds, and epsilon is still decaying. The slightly elevated IQN loss is not a critical stop signal given stable Q-values and the explicit note on loss interpretation. Only 3 hours elapsed in a 24-hour run.

### 2026-05-17 22:18 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is learning to finish the track (worker_finish_count=43, best_finish_time_s=100.25s). Q-values are stable and not exploding. Epsilon is still decaying, indicating ongoing exploration. While IQN loss is >100, Q-values are not exploding and there's no clear upward trend with collapsing returns, which per instructions means it's not a sole stop criterion. Only 4 hours elapsed in a 24-hour run.

### 2026-05-17 23:19 UTC -- best-config-speed-long-run

**Action:** extended
**Reason:** Reached 62.75s (<= 80.0s), extended from 24h to 30h

### 2026-05-17 23:19 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing the track (worker_finish_count=73, best_finish_time_s=62.75s), which is a strong signal to continue. Q-values are elevated but not exploding (max_q < 200), and epsilon is still decaying. Experiment is early (5h/30h).

### 2026-05-18 00:20 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent has a strong worker_finish_count (99) and a positive best_finish_time_s (62.75s), indicating it's learning to finish the track. While IQN loss is high (210.9), Q-values are not exploding (max_q 93.0, not >200), which per project notes means it's not a stop criterion. Epsilon is still decaying, and only 6.1h of 30h max have elapsed.

### 2026-05-18 01:21 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Despite high IQN loss (222.14), the agent is consistently finishing tracks (worker_finish_count=107) and has achieved a competitive best time (62.75s), which is a strong signal to continue. Q-values are elevated but not exploding, and exploration is ongoing. The experiment is still early in its intended long duration (7.1h/30h), and the higher speed reward might naturally lead to higher Q-values and associated loss.

### 2026-05-18 02:22 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing tracks (worker_finish_count=141, best_finish_time_s=62.75), which is a strong signal to continue. Q-values are not exploding (max_q < 200, min_q > -50). Epsilon is still decaying, indicating ongoing exploration. While IQN loss is > 100, Q-values are stable and not diverging, and the agent is making progress in completing episodes.

### 2026-05-18 03:24 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing the track (worker_finish_count=177, best_finish_time_s=62.75). While IQN loss is high, Q-values are stable and not exploding (max_q=109, min_q=-0.5), which is consistent with the increased speed reward weight. Epsilon is still decaying, indicating ongoing exploration. Plenty of time remains in the 30h run for further improvement.

### 2026-05-18 04:26 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is still actively learning to finish tracks (worker_finish_count=214, best_finish_time_s=62.75s). Q-values are not exploding (max_q=117.44 < 200), and returns are high. While loss is high (298), it's not indicative of divergence without exploding Q-values or collapsing returns, as per project notes. It's only 10.1h into a 30h run, plenty of time for the speed incentive to take effect.

### 2026-05-18 05:27 UTC -- best-config-speed-long-run

**Action:** extended
**Reason:** Reached 57.05s (<= 60.0s), extended from 30h to 36h

### 2026-05-18 05:27 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent has finished the track 241 times with a best time of 57.05s, which is a strong signal to continue learning. While IQN loss is high (326.27) and max_q (125.6) is above the typical stable range, Q-values are not yet exploding (max_q < 200) and returns are stable, not collapsing. Epsilon is still decaying, indicating ongoing exploration. Only 11.2h of a 36h run have elapsed, and the speed incentive is a new variable that needs more time to stabilize.

### 2026-05-18 06:29 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing tracks (worker_finish_count=275) and has achieved a new best finish time (56.7s) better than the previous leader (61.65s), indicating positive progress with the speed incentive. Although IQN loss is high, Q-values are not exploding and performance is improving, suggesting the loss scale has shifted due to the increased speed reward weight rather than divergence. Epsilon is still decaying, allowing for continued exploration. The experiment is only 12.2h into a 36h max run.

### 2026-05-18 07:31 UTC -- best-config-speed-long-run

**Action:** continue
**Reason:** Agent is consistently finishing tracks (worker_finish_count=300) and has achieved a new best finish time (56.7s), which is better than the previous leader (61.65s). This is a strong signal to continue, despite the unreachable trainer state.

### 2026-05-18 17:53 UTC -- more-speed-slower-decay

**Action:** failed
**Reason:** Trainer not receiving worker samples (check ports 55555–55558, no duplicate trainer process)

### 2026-05-18 18:14 UTC -- speed-incentive-slower-decay-v2

**Action:** failed
**Reason:** Trainer stuck during initialization after retries

### 2026-05-18 18:35 UTC -- slower-epsilon-decay-with-speed-reward

**Action:** failed
**Reason:** Trainer stuck during initialization after retries
