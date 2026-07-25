# v28: speed program

v27 restored honest measurement and infrastructure health. v28 answers the
remaining question: why the agent is not below 40 s yet, and what closes the
gap to the ~36 s historical result on test-3 (2278.5 m; 40 s = 57.0 m/s
average, 36 s = 63.3 m/s).

## Why it is not sub-40 yet

The v26 run collected ~170k transitions (~150 episodes, roughly 3 hours of
driving) before the deterministic collapse ended it. That is **8.5% of the
configured 2M-transition budget**, and the training curve was still improving
steeply when it stopped: window means 80.2 -> 58.6 -> 50.8 s, window bests
62.4 -> 50.2 -> 46.0 s per 50 episodes. No run so far has reached a
converged plateau above 40 s; every run died of infrastructure or stability
causes first. The historical ~36 s IQN run on this track proves the game,
model class and reward family support the target. The plan is therefore:
raise the convergence ceiling and speed (this document), then give the run
the time it was never given.

## What v28 adds

### 1. Demonstration seeding (`tmrl track record-demos`, `--demos`)

The fastest way to teach 57-63 m/s lines is to put them in the replay
directly instead of waiting for epsilon noise to discover them. The recorder
captures a human (or any external driver) lapping the configured map: a
passive controller injects no input, actions are reconstructed from the
`input_steer/gas/brake` telemetry fields and snapped to the 78-action table,
and rewards/features are computed by the same pipeline the learner uses.
Only finished laps are saved (`.tmdemo` files). Loading them at learner
startup marks every transition `is_demo`, grants **no** update credit and
does not advance the epsilon schedule; PER inserts them at maximum priority,
so they are sampled heavily until their TD errors settle (DQfD-lite).

Notes:

- Record with the final config: demo rewards are labeled by the recording
  run's reward settings. Re-record after any reward change (it takes
  minutes).
- The old ~36 s SOTA policy can be recorded the same way if it still runs in
  the legacy codebase: it drives through the virtual gamepad, and the
  recorder only reads telemetry, so its laps become demonstrations too.
- Pass `--demos` on the **first** launch of a run only. The demonstrations
  are checkpointed inside the replay; passing the flag again on a later
  resume would load duplicates.
- Aim for 6-12 clean laps, ideally below 42 s. Slower clean laps still help;
  crashed laps are discarded automatically.

### 2. Neighbor exploration in the IQN policy

Uniform table exploration is the wrong tool at epsilon 0.02-0.04: a random
no-throttle or full-lock action at 60 m/s mostly terminates the lap. Half of
all exploratory decisions now shift the greedy action by exactly one steering
bin (same throttle/brake mode, reflecting at the table edges), which perturbs
the racing line instead of the survival of the lap. The other half keeps the
throttle-biased global sample so brake taps and recovery modes still get
explored. This is what turns a surviving 46 s groove into a refined 40 s one.

### 3. Control and timing diagnostics

Every training and evaluation episode now reports:

| Metric | Meaning | Healthy sub-40 signature |
| --- | --- | --- |
| `timing/step_race_ms_mean` | true decision period from race-clock deltas | ~50-70 ms (matches the historical 20 Hz setup) |
| `control/gas_fraction` | fraction of decisions with full throttle | >= ~0.85 |
| `control/brake_fraction` | fraction with any brake (incl. taps) | ~0.05-0.15 |
| `control/steer_abs_mean` | mean absolute steering command | trending down at equal speed |

Interpretation rules: a gas fraction stuck near 0.6-0.7 means the pace
incentive is too weak relative to risk (see the v29 ladder); a step period
well above 80 ms means the effective decision rate is coarser than the
historical setup and the action-rate lever applies.

## Runbook (PowerShell, from `my-trackmania-agent/`)

```powershell
# 1. Record reference laps (drive them yourself; restart the race when prompted)
uv run tmrl track record-demos trackmania-iqn-lidar-v27.yaml demos --episodes 8

# 2. Resume the healthiest checkpoint with demonstrations seeded (first launch only)
$ckpt = Get-ChildItem artifacts\trackmania-iqn-lidar-v27\checkpoints\distributed-update-*.pt |
    Sort-Object Name | Select-Object -Last 1
uv run tmrl resume trackmania-iqn-lidar-v27.yaml $ckpt.FullName --demos demos
```

No config change is required for v28: the neighbor exploration and
diagnostics are code-side, and demonstrations are a launch flag. Starting
fresh instead (`tmrl train ... --demos demos`) is also valid and gives the
cleanest thesis ablation, at the cost of the 40k-transition warmup.

## Run schedule and decision gates

Judge progress on `eval/summary` in ~100k-transition windows, nothing
shorter. Let the run live: overnight and multi-day sessions with `resume`
are the intended mode now that reconnect, spooling and best-eval checkpoints
are in place. Order-of-magnitude expectations for a healthy run with demos:

- deterministic finish rate >= 9/10 by ~250-400k transitions,
- deterministic mean < 45 s by ~400-700k transitions,
- sub-40 laps appearing between ~0.7-1.5M transitions.

If the deterministic mean stalls for three consecutive evaluation windows
(improvement < 1 s), apply exactly **one** lever per new experiment, in this
order, and re-record demos whenever the reward changes:

1. `projected_speed_bonus_scale` 8 -> 12 (pace pressure; needs
   `--reset-replay` because stored rewards go stale).
2. `collision_penalty` 1.5 -> 0.75 if evaluation shows wall-shy cornering
   with low `control/gas_fraction`.
3. Learning-rate decay 1e-4 -> 3e-5 once the deterministic mean is < 43 s
   (polish phase).
4. `action_repeat_frames` 4 -> 3 with `gamma`/`reward_gamma`
   0.99 -> 0.9925 (= 0.99^(3/4)) if `timing/step_race_ms_mean` > 80 ms.

The final thesis numbers come from the `eval/best_checkpoint` artifact run
through the 20-trial gate in `docs/benchmark-test-3.md`.
