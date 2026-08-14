# v38: make lap time the objective

The agent drives test-3 in about 38 s. Ten of the owner's own laps, 36.57 s to
37.37 s, are committed in `my-trackmania-agent/demos/` and imported into the
replay buffer. The agent is slower than the data it already has, so the
question is not "does it need more training" but "what does this objective
actually reward".

Everything below was measured by replaying those ten human laps through the
live reward code with the v37 coefficients, not argued from theory.

## Measurement 1: the objective is almost blind to lap time

Return decomposition of the ten human laps, and how each component responds to
lap time (slope in reward per second saved, correlation across the ten laps):

| Component | Spread over the 10 laps | Corr. with lap time | Reward per second saved |
| --- | ---: | ---: | ---: |
| time penalty | 0.59 | **-1.000** | **0.744** |
| progress (75.0 budget) | 0.08 | 0.253 | -0.028 |
| projected velocity | 0.28 | 0.118 | -0.054 |
| projected speed bonus | 0.21 | -0.267 | 0.081 |
| finish reward | 0.00 | - | 0.000 |
| **total return (~93.5)** | **0.75** | -0.697 | **0.743** |

Read the last column. Saving a full second of lap time is worth **0.74 reward
out of a return of 93.5** - eight tenths of one percent. And every bit of that
comes from `time_penalty_per_second`; the two "speed" terms contribute 0.081
per second between them, with the projected-velocity term actually pointing the
wrong way.

The reason is structural. Every dense term is proportional to distance, not to
time: the progress budget pays out 75.0 for reaching the finish however slowly,
and a reward shaped as `v * dt` integrates to `scale * path length`. Measured,
the progress term is 75.00 on every single lap and the projected-velocity term
sits at 15.8-16.1 whether the lap took 36.57 s or 37.37 s. **A reward
proportional to speed cannot create time pressure, because speed times time is
distance.** Twenty versions of tuning `projected_velocity_scale` and
`projected_speed_bonus_scale` were tuning coefficients that do not select for a
faster lap; they only make early learning dense.

Now put that next to the cost of failing. A lap abandoned halfway forfeits the
remaining progress, the finish reward, the remaining shaping and the terminal
penalty: about 78 reward. So:

> Under v37, driving one second faster is worth 0.74, and it is rejected unless
> it raises the chance of failing the lap by less than **0.73%**.

No racing driver operates to that standard - the ten demos in the repo are the
laps the human kept, not the ones he crashed. The agent is not underperforming
its objective. It is obeying it. A cautious 38 s lap is the optimum of this
reward, and more training, more demos or a better network cannot move it.

## Measurement 2: the discount hides the finish line entirely

The demos run at a median 10 ms and mean 13.5 ms of race clock per decision
(~74 decisions per second, `action_repeat_frames: 2`). v37 trains with
`gamma: 0.99`.

- Effective horizon: `1/(1-0.99)` = 100 steps = **1.35 seconds**.
- Weight of the finish reward seen from the start line: `0.99^2729` =
  **1.06e-12**. Not small - absent.
- Discounted return of the fastest lap: **-0.026**; of the slowest: **-0.072**.
  The entire lap-time signal after discounting is 0.045 reward, which is noise.

A 1.35 s horizon cannot express the central skill of racing: giving up speed
now for a better exit two seconds later. The agent is a greedy local speed
maximiser, which is exactly what "quick but not fast" looks like. Note that
gamma must be chosen per unit of time, not per step: R2D2's 0.997 was at about
15 decisions per second, a 22 s horizon; the same 0.997 at 74 decisions per
second is only 4.5 s, and v36/v37's 0.99 is 1.35 s.

## Measurement 3: the action space is NOT the bottleneck (hypothesis refuted)

Worth recording, because it is the tempting explanation and it is wrong. The
human's raw analog inputs are stored in the demo files. He drives on a
keyboard:

- steering is exactly 0.0 on 35.2% of decisions and exactly +-1.0 on 64.8%;
  there is no intermediate value anywhere in 27437 samples,
- throttle and brake are equally binary (82.6% full throttle, 5.5% any brake,
  and 11.9% coasting with neither),
- quantising his controls onto the 78-action table produces a mean error of
  **0.0000** - the table represents his driving exactly.

So finer steering bins, a continuous action head, or a larger table would buy
nothing against this reference. Do not spend a run on it.

## Measurement 4: the track is shorter than assumed

The geometry asset gives a racing line of **1982.6 m** (reward centre line
2158.5 m). Every earlier note used 2278.5 m. Corrected targets: 36 s is 55.1
m/s average, 38 s is 52.2 m/s, and the measured mean speed of the human laps is
54.3 m/s, which matches 1982.6 / 36.57 exactly. The remaining gap to the human
is 5.5% of average speed, not the 3% previously assumed.

## The change

`TrajectoryReward` gains `finish_time_bonus_per_second` and
`finish_reference_time_s`. Finishing now pays
`finish_reward + bonus * max(0, reference - lap_time)`, so the terminal payout
falls linearly with lap time, and the discount is set from the decision rate so
that payout is actually visible from the start line. `v38-timevalue.yaml` uses
bonus 5.0 against a 45 s reference, `gamma`/`reward_gamma` 0.9994 (a 22.5 s
horizon at 74 Hz, matching R2D2's horizon in seconds), and enables
`value_rescaling` because the returns are now much larger. It also restores
`demonstration_margin_weight` (0.25), which v37 dropped, silently disabling the
DQfD margin loss.

Replaying the same ten human laps through the new objective:

| | v37 | v38 |
| --- | ---: | ---: |
| value of one second saved | 0.57 | **5.57** |
| break-even crash probability for a 1 s gain | 0.73% | **4.64%** |
| discounted signal separating fastest from slowest lap | 0.045 | **2.395** |
| visibility of the finish reward from the start line | 1.1e-12 | **0.19** |

Same laps, same code, same demos: the objective now prefers the faster lap by a
margin the learner can actually see, and tolerates a racing driver's risk
instead of demanding near-certainty.

## Runbook

The stored rewards of every transition already in the replay were computed with
the old objective, so the buffer must be reset; demo rewards are recomputed on
import, so the demos come back correctly. Keep the learned weights.

```powershell
$ckpt = Get-ChildItem artifacts\trackmania-iqn-r2d2-racing-v37a\checkpoints\distributed-update-*.pt |
    Sort-Object Name | Select-Object -Last 1
uv run tmrl resume trackmania-iqn-r2d2-racing-v38-timevalue.yaml $ckpt.FullName `
    --reset-replay --demo demos
```

Expect the reported return to jump (the scale changed) and the deterministic
evaluation to dip briefly while the critic relearns values under the new
objective. Judge only `eval/summary`.

## One number to check on the training machine

The demos were recorded in a tight telemetry loop with no inference: 13.5 ms
per decision. The agent's loop additionally runs the lidar feature transform
and a GRU forward pass on the GPU, and `client.read()` always drains to the
newest frame, so a slow loop does not fall behind in wall-clock time - it
simply makes fewer decisions per lap and holds each control longer.

Compare `timing/step_race_ms_mean` from any recent run against the demos'
13.5 ms. If the agent's period is materially longer, the demonstrations
describe a different control problem than the one the agent lives in (the same
action held for a different duration), and its control is coarser than the
human's by that ratio. That would be a hard ceiling worth its own fix, and it
costs one glance at W&B to rule in or out.

## Held in reserve, one lever per experiment

- Risk-window evaluation (`--tau-min`) from the open v37-flywheel PR: free
  seconds from checkpoints you already have, no training.
- `collision_penalty` 0.5 currently equals 0.9 s of lap time under v38's scale;
  revisit only after the objective change has been measured.
- Self-imitation flywheel (same PR) once the agent's own laps beat the demos.
- Rebalance `time_penalty_per_second` upward only if the finish bonus alone
  proves too coarse a signal early in a lap.
