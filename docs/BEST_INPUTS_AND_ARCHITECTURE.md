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

## 4. Discrete actions + controller (vibrations) + InfoNCE

- **Controller kept for vibrations**: The gamepad API is used so that `crash_callback` (e.g. in `TM2020Interface`) still receives vibration (large_motor) when the vehicle hits the guardrail. Do not switch to keyboard-only if you need this signal.
- **Discrete action space**: The policy can output a **discrete** action index; it is then mapped to continuous control `[gas, brake, steer]` and passed to the same `send_control` / `control_gamepad` path. So the game still gets analog gamepad input and vibrations still work.
  - **Implementation**: `tmrl/custom/tm/utils/discrete_control.py` — `build_discrete_to_continuous()`, `discrete_index_to_control()`. Use e.g. 5 steer × 3 gas × 2 brake = 30 actions.
  - **Interface hook**: On `TM2020Interface`, set `self.discrete_action_table` (e.g. in your interface subclass or after building the table). Then when the policy returns a single integer (or shape (1,) array), `send_control` will convert it to continuous via `discrete_index_to_control` before calling `control_gamepad`. So controller and crash_callback (vibrations) are unchanged.
- **InfoNCE loss**: Optional contrastive (goal-conditioned) loss for auxiliary training.
  - **Implementation**: `tmrl/custom/info_nce.py` — `StateActionGoalEncoders` (phi(s,a), psi(g)), `info_nce_loss()`, `info_nce_loss_from_encoders()`. Define goals g (e.g. next checkpoint or future state); sample negatives from other trajectories; add the InfoNCE loss to your training step (e.g. alongside SAC critic loss) and optionally maximize f(s,a,g) for the policy.
  - To use with SAC: train encoders with InfoNCE on (s, a, g_pos, g_neg); optionally add an auxiliary reward or policy loss term that encourages high f(s,a,g) for the current goal.
