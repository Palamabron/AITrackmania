# Experiments and Hyperparameter Changes Report

This document summarizes all tried experiments, hyperparameter changes, and their effects for the TQC/SophyResidual Trackmania training (SophyResidual_runv23 series and related configs).

---

## 1. Baseline (Pre–Run A)

**Run:** SophyResidual_runv23 TRAINER (initial logs).

**Observed problems:**
- `entropy_coef` increased from ~0.015 to ~0.22 over 370 rounds.
- `debug/q_a1`, `debug/backup` drifted strongly negative (~-8 to ~-131).
- `losses/loss_actor` grew from ~1.7 to ~62+.
- `return_train` flat or declining; `episode_length_train` often ~140 (early failure).
- `debug/q1` and `debug/q2` were **identical** (twin critics not independent).

**Relevant config (before any fixes):**
- `LEARN_ENTROPY_COEF`: true  
- `TARGET_ENTROPY`: -0.5  
- `LR_ENTROPY`: 0.0003  
- `ALPHA`: 0.01  
- `GAMMA`: 0.995  
- `MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP`: 25  
- `ENVIRONMENT_STEPS_BEFORE_TRAINING`: 500  
- `TOP_QUANTILES_TO_DROP`: 2  
- Reward: `MAX_STRAY` 100, `SPEED_SAFE_DEVIATION_RATIO` 0.6, `RECKLESS_SPEED_THRESHOLD` 120, `RECKLESS_PENALTY_FACTOR` 0.002  

**Conclusion:** Entropy autotune too aggressive for reward scale; twin critics shared weights; train/data ratio and warmup suboptimal.

---

## 2. First Wave of Fixes (Before Run A)

**Code:**
- **Twin critics:** `q2` initialized with `seed+1` so `q1` and `q2` differ (in `Sophy.py`).

**Config (docs/config_tqcgrab_1actor_recommended.json and/or user config):**
- `TARGET_ENTROPY`: -0.5 → **-3.0**
- `LR_ENTROPY`: 0.0003 → **5e-05**
- `MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP`: 25 → **10**
- `ENVIRONMENT_STEPS_BEFORE_TRAINING`: 500 → **5000**
- `TOP_QUANTILES_TO_DROP`: 2 → **5**

**Result:** Not run as a single named experiment; these became the base for Run A and later runs. Purpose: stabilize alpha, reduce overfitting to replay, improve critic diversity.

---

## 3. Run A — Fixed Entropy (Freeze Alpha)

**Config vs baseline:**
- `LEARN_ENTROPY_COEF`: **false**
- `ALPHA`: **0.027** (fixed)

**Results (first ~79–80 rounds):**
- No entropy explosion; `entropy_coef` absent from logs (fixed alpha).
- Q still drifted negative: `debug/q_a1` ~-0.79 → ~-7.22.
- `loss_actor` 0.88 → 7.31; `loss_critic` low (~0.01).
- `return_train` 5.1 → 11.2 (early window); last_point ~16.7.
- `debug/q1` vs `debug/q2`: 1/80 identical (twin critics now distinct).

**Conclusion:** Stability improved (no alpha runaway), but value drift and actor loss growth remained. Fixed alpha was kept for all following runs.

---

## 4. Run B — Bootstrap and Alpha Tweak

**Config vs Run A:**
- `ALPHA`: 0.027 → **0.0125**
- `GAMMA`: 0.995 → **0.99**
- `MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP`: **6** (stricter train/data ratio).

**Results (80 rounds):**
- Q drift much smaller: `debug/q_a1` +0.14 → -1.83 (vs Run A -0.79 → -7.22).
- `debug/backup_std` 0.21 → 0.89 (vs Run A ~1.15).
- `loss_actor` -0.10 → 1.87; `loss_critic` ~0.004.
- `return_train` 4.67 → 6.97; `q1`/`q2` not identical (0/80 close).
- `episode_length_train` first 314, last 590 (improvement).

**Conclusion:** Clear improvement. Lower alpha + gamma + train ratio reduced value drift and kept learning stable. This setup was used as base for Run C and D.

---

## 5. Run C — More Quantile Truncation

**Config vs Run B:**
- `TOP_QUANTILES_TO_DROP`: 5 → **8**
- `ALPHA`: **0.0125** (unchanged).

**Results (80 then 100 rounds):**
- Matched 80: Q a bit more negative than B (`debug/q_a1` -2.34 vs -1.83); losses slightly higher.
- Full 100: `return_train` 7.36 → 11.81; last_point ~19.2; `episode_length_train` up to ~964.
- Stable; no divergence.

**Conclusion:** Slightly more conservative value estimates; return trend positive. Settings kept for Run D.

---

## 6. Run D — Lower Alpha

**Config vs Run C:**
- `ALPHA`: 0.0125 → **0.01**

**Results (149 rounds):**
- Stable: `debug/q_a1` ~-0.12 → ~-2.65; `debug/backup_std` ~1.11.
- `return_train` 8.79 → 7.83 (slight decline over run); last_point ~1.19.
- `episode_length_train` last 143 — still many short episodes.

**Conclusion:** Training stable but performance plateau; many episodes still end early (e.g. wall-hug or early crash). Focus shifted to reward shaping and demos.

---

## 7. Run E — Stronger Dense Reward and More Demos

**Config vs Run D:**
- `SPEED_BONUS`: 0.0004 → **0.0008**
- `PLAYER_RUNS.MAX_FILES_PER_UPDATE`: **5**
- (Possibly `CONSTANT_PENALTY`, `FAILURE_COUNTDOWN` etc. as in user’s Run E config.)

**Results (359 rounds):**
- `return_train` 12.88 → 10.59; last_point ~1.12.
- `debug/r` a bit higher than before but still small.
- `episode_length_train` median ~473; last 140 — still many early terminations.
- No divergence; `loss_critic` ~0.0024.

**Conclusion:** Slightly better reward signal and demo throughput, but policy still not consistently finishing the track; wall-hugging and low returns remained an issue.

---

## 8. Run F — Anti–Wall-Hug Reward Hardening

**Config (REWARD_CONFIG):**
- `MAX_STRAY`: 100 → **60**, then **50**
- `SPEED_SAFE_DEVIATION_RATIO`: 0.6 → **0.2**, then **0.15**
- `RECKLESS_SPEED_THRESHOLD`: 120 → **45**
- `RECKLESS_PENALTY_FACTOR`: 0.002 → **0.008**
- `FAILURE_COUNTDOWN`: 10 → **7**
- `CONSTANT_PENALTY`: 0 → **0.001**
- `MAX_FILES_PER_UPDATE`: 5 (unchanged).

**Results (99 rounds):**
- `debug/r` dropped: ~0.051 → ~0.019 (reward signal damped).
- `debug/q_a1_std`, `debug/backup_std` rose to ~1.53–1.54 (vs Run E ~1.06).
- Q more negative (~-2.62); `loss_actor` up to ~3.0.
- `return_train` 5.29 → 8.15; last_point 1.18.

**Conclusion:** Wall-hug penalties were too strong; net reward signal weakened and variance increased. Recommended rollback: reduce `RECKLESS_PENALTY_FACTOR` to 0.005, `CONSTANT_PENALTY` to 0.0005, `FAILURE_COUNTDOWN` to 9 (Run G).

---

## 9. Imitation Bias (Code + Config)

**Code:**
- Player-run samples tagged with `info["is_demo"]=True` (and optional `demo_run_id`, `demo_source`).
- Demo injection repeat: `PLAYER_RUNS.DEMO_INJECTION_REPEAT` (default 1); trainer appends demo buffer multiple times when injecting.
- R2D2 memory: `PLAYER_RUNS.DEMO_SAMPLING_WEIGHT` multiplies episode weight when choosing episode for sampling (demo episodes chosen more often).
- New metric: `debug/demo_fraction_in_batch` (fraction of last batch that came from demo).

**Config (recommended):**
- `DEMO_INJECTION_REPEAT`: **2**
- `DEMO_SAMPLING_WEIGHT`: **2.5**
- `MAX_FILES_PER_UPDATE`: **5** (or higher for more demo throughput).

**Result:** Not yet evaluated in a dedicated long run. Expected: higher `debug/demo_fraction_in_batch` early on and better imitation of human trajectories if enough diverse demos are provided.

---

## 10. Final Reward Overhaul (Anti–Wall-Hug in Reward Logic)

**Code (compute_reward.py):**
- **Proximity reward shaping:** Progress reward scaled down when `min_dist > _speed_safe_deviation`, using `PROXIMITY_REWARD_SHAPING` (default 0.5). Rewards staying on the ideal line more than riding the wall.
- **Wall-hug penalty:** If `best_index == cur_idx` (no trajectory progress), speed > `WALL_HUG_SPEED_THRESHOLD` (10 km/h), and `min_dist > _speed_safe_deviation`: accumulate `_no_progress_steps`, add a penalty that grows with steps and distance. If `_no_progress_steps` exceeds `nb_zero_rew_before_failure * 3`, episode is terminated (`wall_hug_no_progress`).
- **Off-track speed penalty:** Any step with `min_dist > _speed_safe_deviation` and speed > 5 km/h gets an extra penalty proportional to speed and distance (discourages fast driving off the line).
- **Deviation penalty:** The 50% reduction when `distance_gained > 0` was removed; full deviation penalty always applied when off threshold.
- **Reward scale:** Raw reward multiplied by `REWARD_SCALE` (default 3.0) before `tanh`, to increase contrast of the signal.

**New REWARD_CONFIG keys (and defaults):**
- `WALL_HUG_SPEED_THRESHOLD`: 10.0  
- `WALL_HUG_PENALTY_FACTOR`: 0.005  
- `PROXIMITY_REWARD_SHAPING`: 0.5  
- `REWARD_SCALE`: 3.0  

**Recommended config (docs/config_tqcgrab_1actor_recommended.json):**
- `MAX_STRAY`: 50  
- `SPEED_SAFE_DEVIATION_RATIO`: 0.15  
- `RECKLESS_SPEED_THRESHOLD`: 45  
- `RECKLESS_PENALTY_FACTOR`: 0.006  
- `CONSTANT_PENALTY`: 0.0005  
- `FAILURE_COUNTDOWN`: 9  
- Plus the four new keys above.

**Result:** Not yet run. Expected: fewer wall-hugging episodes, earlier termination of “stuck on wall” behavior, and stronger preference for fast on-track driving.

---

## Summary Table

| Experiment / change           | Main param changes                                      | Effect |
|------------------------------|---------------------------------------------------------|--------|
| Baseline                     | Default TARGET_ENTROPY -0.5, LR_ENTROPY 0.0003, twin q same seed | Alpha and Q explode; return flat; q1==q2 |
| First fixes                  | TARGET_ENTROPY -3, LR_ENTROPY 5e-5, train ratio 10, warmup 5k, TOP_QUANTILES 5, q2 seed+1 | Base for Run A |
| Run A                        | LEARN_ENTROPY_COEF false, ALPHA 0.027                   | No alpha runaway; Q still drifts |
| Run B                        | ALPHA 0.0125, GAMMA 0.99, MAX_TRAINING_STEPS_PER_ENV 6  | Much smaller Q drift; stable; better return trend |
| Run C                        | TOP_QUANTILES_TO_DROP 8                                 | Slightly more conservative; stable; return up to ~11.8 |
| Run D                        | ALPHA 0.01                                              | Stable; return plateau; many short episodes |
| Run E                        | SPEED_BONUS 0.0008, MAX_FILES_PER_UPDATE 5              | Slightly better signal; still plateau and short eps |
| Run F                        | Stricter REWARD_CONFIG (MAX_STRAY 50, safe ratio 0.15, reckless 45/0.008, FAILURE 7, CONSTANT 0.001) | Reward signal damped; variance up; penalties too strong |
| Run G (suggested)            | RECKLESS 0.005, CONSTANT 0.0005, FAILURE 9              | Intended to soften Run F |
| Imitation bias               | DEMO_INJECTION_REPEAT 2, DEMO_SAMPLING_WEIGHT 2.5, is_demo tagging, sampling weight in R2D2 | Pending long-run evaluation |
| Final reward overhaul        | Proximity shaping, wall-hug penalty/termination, off-track speed penalty, no 50% deviation discount, REWARD_SCALE 3, new REWARD_CONFIG keys | Pending first run |

---

## 11. Training Stability Overhaul (Code Changes)

**Motivation (Run Hv2):**
- 40 human demos, 1.5h demo-only phase, then worker.
- `debug/demo_fraction_in_batch` stayed ~0.93 throughout (demos dominated buffer).
- `debug/q_a1` drifted from ~+10 to **-15.6** (severe Q divergence).
- `losses/loss_actor` exploded to **+15.6**.
- `return_train` collapsed to **-77, -162** at the end despite good mid-run episodes (return ~22, episode_length ~1500).

**Root causes identified:**
1. **No gradient clipping** — explosive gradients in actor/critic accelerated Q drift.
2. **No backup (target Q) clipping** — Q targets could drift without bounds, causing self-reinforcing divergence.
3. **Fixed demo sampling weight** — demos never lost influence, causing permanent distribution mismatch between training data and actual policy behavior.
4. **No best checkpoint saving** — when training collapsed, good intermediate models were lost.

**Code changes:**

### A) Gradient clipping (custom_algorithms.py)
- Added `torch.nn.utils.clip_grad_norm_()` after `backward()` for **both** critic and actor.
- Config: `ALG.GRAD_CLIP_ACTOR` (default 1.0), `ALG.GRAD_CLIP_CRITIC` (default 1.0). Set to 0 to disable.
- New WandB metrics: `debug/critic_grad_norm`, `debug/actor_grad_norm` for monitoring.

### B) Backup (target Q) clipping (custom_algorithms.py)
- Backup values (target Q for critic loss) clamped to `±BACKUP_CLIP_RANGE`.
- Config: `ALG.BACKUP_CLIP_RANGE` (default 100.0). With `tanh(reward)` in [-1,1] and `gamma=0.99`, theoretical max return is 100, so this clips only pathological values.

### C) Demo sampling weight decay (memory.py)
- `demo_sampling_weight` now **decays linearly** from the configured value towards 1.0 as the buffer fills.
- Formula: `weight = 1.0 + (initial_weight - 1.0) * (1.0 - min(1, total_samples / decay_samples))`.
- Config: `PLAYER_RUNS.DEMO_WEIGHT_DECAY_SAMPLES` (default 200000). After 200k samples in buffer, demo_sampling_weight reaches 1.0 (no bias).
- New WandB metric: `debug/demo_sampling_weight` tracks the current effective weight.

### D) Best checkpoint saving (training_offline.py)
- At the end of each epoch, if mean `return_train` exceeds previous best (and is positive), actor weights are saved to `TmrlData/weights/best_actor.pth`.
- Logs: `New best return_train=X.XX at epoch Y -> saved ...`.
- After training collapse, load `best_actor.pth` to resume from the best known policy.

**New config keys (ALG section):**
```json
"GRAD_CLIP_ACTOR": 1.0,
"GRAD_CLIP_CRITIC": 1.0,
"BACKUP_CLIP_RANGE": 100.0
```

**New config keys (PLAYER_RUNS section):**
```json
"DEMO_WEIGHT_DECAY_SAMPLES": 200000
```

**Expected effect:**
- Gradient clipping + backup clipping should prevent Q divergence and actor loss explosion.
- Demo weight decay should naturally reduce demo fraction over time (from ~0.93 early to ~0.5 mid-run to ~1.0 weight late = uniform sampling).
- Best checkpoint preserves the best model found during training.
- Combined with existing settings (ALPHA=0.01, GAMMA=0.99, TOP_QUANTILES_TO_DROP=8), this should enable stable long training runs where the agent improves continuously.

---

## Recommendations for Next Runs

1. **Use the final reward overhaul** (proximity shaping, wall-hug penalty, REWARD_SCALE, new REWARD_CONFIG) with the recommended values in this report.
2. **Keep algorithm settings** from Run B/C/D: `LEARN_ENTROPY_COEF=false`, `ALPHA=0.01`, `GAMMA=0.99`, `MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP=6`, `TOP_QUANTILES_TO_DROP=8`, `N_STEPS=1`.
3. **Enable imitation bias** and add 10–20 diverse human runs (clean laps + recoveries).
4. **Gradient clipping and backup clipping are now on by default** (GRAD_CLIP 1.0, BACKUP_CLIP 100.0). Monitor `debug/critic_grad_norm` and `debug/actor_grad_norm` — if they frequently hit 1.0, gradients are being clipped (good, prevents divergence).
5. **Demo weight decay** is automatic. Monitor `debug/demo_sampling_weight` — it should decrease over time. If it stays high, increase `DEMO_WEIGHT_DECAY_SAMPLES` or record more worker episodes.
6. **Monitor:** `debug/r`, `return_train`, `episode_length_train`, `debug/q_a1`, `losses/loss_actor`, `debug/critic_grad_norm`, `debug/actor_grad_norm`, `debug/demo_sampling_weight`. If Q and loss remain bounded, training is stable.
7. **If training collapses**, load `TmrlData/weights/best_actor.pth` — this is the best actor found before the collapse.
