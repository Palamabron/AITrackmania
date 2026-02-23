# Best inputs, architecture, and buffer for 2-actor TrackMania pipeline

## 1. Best inputs for the model

For a **2-actor** setup (limited env steps), the best inputs are:

- **LIDAR or LIDARPROGRESS** (recommended for sample efficiency): Compact observation (speed + lidar history, or speed + progress + lidar). Low dimensionality, fast stepping. Use `RTGYM_INTERFACE`: `LIDAR` or `LIDARPROGRESS`.
- **IMPALA-style (API + images)**: Rich state (next_checkpoints, speed, acceleration, jerk, race_progress, inputs, gear, steer_angle, slip, failure_counter) plus image history. Best for full driving policy but higher compute; use with high UTD to compensate for only 2 workers. Use `RTGYM_INTERFACE`: `BEST`, `BEST_TQC`, or `MTQC` with images.
- **TrackMap**: speed, gear, rpm, track_information (60), acceleration, steering, slipping, crash, failure_counter — no images. Good if you want geometry without vision.
- **Sophy (no images)**: Track points (left/center/right) + API state only — for API-only policies.

**Recommendation**: Start with **LIDARPROGRESS** (speed + progress + lidar) for fastest iteration and REDQ; move to **IMPALA** (BEST) + images if you need vision and can afford the compute.

---

## 2. Best model architecture (2 actors)

- **Depth over width**: Use **residual MLP blocks** (Dense → LayerNorm → Swish) with **depth 4–8 blocks**, width 256. Paper (2503.14858v4): depth scales better than width; LayerNorm and Swish are essential.
- **Actor and critic**: Scale **both** with the same residual architecture.
- **Weight init**: Standard (e.g. orthogonal or Kaiming) for linear layers; residual branch can use small init for stability.
- **Input normalization**: Normalize observation (e.g. speed, lidar) to zero mean / unit variance or to [0,1] where appropriate (configurable in preprocessor).
- **For images**: Keep existing CNN (IMPALA/EffNet) as encoder; add **residual MLP heads** (same blocks) after the encoder for actor and critic.
- **No RNN required** for 2 actors with LIDAR history (stacked frames or lidar history in obs); RNN adds cost. Use temporal stacking (e.g. img_hist_len=4) instead.
- **Algorithm**: **REDQ-SAC** for sample efficiency (high UTD) with 2 actors.

---

## 3. Best buffer sampling for 2 actors

- **Uniform replay** is the default and works well with REDQ-SAC: large buffer (500k–1M+), uniform sampling, high **max_training_steps_per_env_step** (UTD 20–40).
- **R2D2-style memory** (`MemoryR2D2` / `MemoryR2D2woImages`) is used in this codebase for **MTQC** (image-based or Sophy): it handles **sequences** and rewind for recurrent/attention models. Use R2D2 when you use MTQC + images or Sophy; for plain SAC/REDQ with MLP or residual MLP, **uniform** replay (e.g. `MemoryTMLidar`, `MemoryTMLidarProgress`, `MemoryTMBest`) is simpler and sufficient.
- **Recommendation**: For REDQ + Lidar or REDQ + (BEST + images with non-RNN model): use **uniform** buffer (`MemoryTMLidarProgress` or `MemoryTMBest`) with large size and high UTD. Use **R2D2** only when training MTQC with sequences.

---

## 4. Reward function (stability and config)

- **Bounded output**: `RewardFunction.compute_reward()` (in `tmrl/custom/tm/utils/compute_reward.py`) applies `math.tanh(reward)` at the end, so per-step reward is in **(-1, 1)**. This keeps value scales stable for SAC/TQC.
- **Config**: Set **`ENV.END_OF_TRACK_REWARD`** (top-level in `ENV`) so the finish bonus is used; `REWARD_CONFIG.END_OF_TRACK` is used by the interface for `finish_reward`, while the trajectory-based reward uses `cfg.END_OF_TRACK_REWARD` for the near-finish bonus. Use the same value (e.g. 100.0) for both if you want a strong finish signal.
- **No double-counting**: Speed bonus and backward penalty are applied once per step (inside the `episode_reward != 0` block). Avoid adding a second speed-reward block in the same function.

---

## 5. Discrete actions + controller (vibrations) + InfoNCE

- **Controller kept for vibrations**: The gamepad API is used so that `crash_callback` (e.g. in `TM2020Interface`) still receives vibration (large_motor) when the vehicle hits the guardrail. Do not switch to keyboard-only if you need this signal.
- **Discrete action space**: The policy can output a **discrete** action index; it is then mapped to continuous control `[gas, brake, steer]` and passed to the same `send_control` / `control_gamepad` path. So the game still gets analog gamepad input and vibrations still work.
  - **Implementation**: `tmrl/custom/tm/utils/discrete_control.py` — `build_discrete_to_continuous()`, `discrete_index_to_control()`. Use e.g. 5 steer × 3 gas × 2 brake = 30 actions.
  - **Interface hook**: On `TM2020Interface`, set `self.discrete_action_table` (e.g. in your interface subclass or after building the table). Then when the policy returns a single integer (or shape (1,) array), `send_control` will convert it to continuous via `discrete_index_to_control` before calling `control_gamepad`. So controller and crash_callback (vibrations) are unchanged.
- **InfoNCE loss**: Optional contrastive (goal-conditioned) loss for auxiliary training.
  - **Implementation**: `tmrl/custom/info_nce.py` — `StateActionGoalEncoders` (phi(s,a), psi(g)), `info_nce_loss()`, `info_nce_loss_from_encoders()`. Define goals g (e.g. next checkpoint or future state); sample negatives from other trajectories; add the InfoNCE loss to your training step (e.g. alongside SAC critic loss) and optionally maximize f(s,a,g) for the policy.
  - To use with SAC: train encoders with InfoNCE on (s, a, g_pos, g_neg); optionally add an auxiliary reward or policy loss term that encourages high f(s,a,g) for the current goal.

---

## 6. 1-actor recommended (TQCGRAB + new model architecture)

For **single-actor** training with the TQC_GrabData plugin (API-only, no images):

- **Recommended**: Use **Residual Sophy** (paper 2503.14858v4): deep residual MLP backbone (LayerNorm + SiLU) + optional attention. Set in config: `MODEL.USE_RESIDUAL_SOPHY`: `true`, `RESIDUAL_MLP_HIDDEN_DIM`: `256`, `RESIDUAL_MLP_NUM_BLOCKS`: `8`, `BATCH_SIZE`: `512`. Algorithm: TQC. Copy `docs/config_tqcgrab_1actor_recommended.json` to `TmrlData/config/config.json` to test.
- **Alternative (LIDAR only)**: Use `RTGYM_INTERFACE`: `LIDARPROGRESS`, `USE_RESIDUAL_MLP`: `true`, REDQ, same residual depth/width and high UTD.

For **images + vector** (e.g. BEST or MTQC with images):

- **Frozen EfficientNet + residual head**: Set `USE_FROZEN_EFFNET`: `true` and `ALGORITHM`: `SAC`. A small frozen EfficientNet produces image embeddings; they are concatenated with the API vector and passed through a deep residual MLP head (SiLU, LayerNorm). Config: `FROZEN_EFFNET_EMBED_DIM`, `FROZEN_EFFNET_WIDTH_MULT`, `RESIDUAL_MLP_HIDDEN_DIM`, `RESIDUAL_MLP_NUM_BLOCKS`.

---

## 7. Images + LIDAR (LIDARPROGRESSIMAGES)

To use **both camera images and LIDAR** in one model:

- **Interface**: Set `ENV.RTGYM_INTERFACE` to **`LIDARPROGRESSIMAGES`**. This uses `TM2020InterfaceLidarProgressImages`: one screenshot per step → LIDAR from `Lidar.lidar_20(img)` and a resized (optionally grayscale) image. Observation: `(speed, progress, lidar_history, image_history)`.
- **Model**: Automatically uses **Frozen EfficientNet + residual head** (SAC only): image at index 3, vector = speed + progress + flattened LIDAR history.
- **Config**: Use `ALGORITHM`: `SAC`. Image size via `ENV.IMG_WIDTH`, `ENV.IMG_HEIGHT`; `ENV.IMG_GRAYSCALE`: `true` recommended. Reward trajectory must be recorded for progress (see below).

**Recording points for LIDAR and reward:**  
LIDAR is **not** from recorded map points; it is computed from the game screenshot each step. **Reward trajectory** (progress/checkpoints) and **track boundaries** (left/right) are recorded with scripts and saved under `TmrlData/`. See **[Recording track and LIDAR](RECORDING_TRACK_AND_LIDAR.md)** for scripts (`record_reward.py`, `record_track.py`) and exact paths (`reward/reward_<MAP_NAME>.pkl`, `track/track_<MAP_NAME>_left.pkl`, `_right.pkl`).
