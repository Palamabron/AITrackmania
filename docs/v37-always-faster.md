# v37: always faster

There is no finish line: the run should keep converting compute into lap time
indefinitely. This document states what the stack already covers, adds the two
missing state-of-the-art mechanisms, and defines the standing iteration loop.

## Scorecard: what the v36 stack already is

Distributional IQN with Double-DQN and dueling; R2D2 recurrence (16-step
sequences, burn-in 8, all-post-burn-in-step training, mixed max/mean sequence
priorities, optional value rescaling); prioritized replay with a pace-based
elite boost; DQfD demonstrations with eviction protection and sub-37 reference
laps committed in `my-trackmania-agent/demos/`; n-step 5; minimum-curvature
racing-line features and reward; action repeat 2 with a matched gamma;
line-refinement exploration; deterministic evaluation with time buckets,
policy-version stamps and best-eval checkpoints. For a single-machine
TrackMania setup this is a genuinely modern stack. The remaining upside is not
another algorithm - it is deployment risk, frontier consolidation, and
protocol.

## Addition 1: risk-window deployment (IQN's free knob)

The IQN head is tau-conditioned, so the deployed risk profile is an inference
parameter. `q_values` accepts a `[tau_min, tau_max]` window: the neutral
default averages the full quantile grid, an upper window selects actions by
their optimistic tail (later braking, tighter lines), a lower window is
conservative. Training and Double-DQN target selection stay risk-neutral.
Sweep existing checkpoints through the release gate - no training required:

```powershell
uv run tmrl benchmark <config> $ckpt                 # neutral baseline
uv run tmrl benchmark <config> $ckpt --tau-min 0.25
uv run tmrl benchmark <config> $ckpt --tau-min 0.5
uv run tmrl benchmark <config> $ckpt --tau-max 0.75  # consistency variant
```

Read the results as a frontier and pick the best gated median. Repeat the
sweep over the top two or three `eval/best_checkpoint` artifacts. With value
rescaling enabled the window mean lives in h-space (a monotone distortion);
the sweep is empirical, so this changes the window's shape, not the method.
Expect `[0.25, 1.0]` to be the sweet spot and `[0.5, 1.0]` to risk the
finish-rate gate.

## Addition 2: the self-imitation flywheel

Static demonstrations stop teaching the moment the agent outruns them. With
`self_imitation_window_s` set, every finished training lap within that window
of the best-so-far is promoted in place to protected demonstration status:
the margin loss anchors its actions, eviction protection keeps it, and the
elite boost already samples it harder. The anchor therefore tracks the moving
frontier - beat your best lap and you become the new teacher. Guardrails:
promotion stops while demonstrations exceed `self_imitation_max_demo_fraction`
of the buffer, and the mechanism belongs on resumed polish runs, not fresh
ones (a cold run would anchor its first slow finishes).

## Addition 3: restore the demonstration margin (open question for v36)

v36 dropped `demonstration_margin_weight` from the learner kwargs, which
silently disabled the DQfD margin loss (default 0.0) right as the sub-37 demo
files landed - they currently shape sampling but not the objective. v37
restores `demonstration_margin: 0.8` at weight 0.25, which the flywheel also
depends on. If the drop was deliberate because the margin hurt, run v37 with
weight 0.1 instead and compare - one change per experiment.

## Runbook

```powershell
$ckpt = Get-ChildItem artifacts\trackmania-iqn-r2d2-racing-v36a\checkpoints\distributed-update-*.pt |
    Sort-Object Name | Select-Object -Last 1
uv run tmrl resume trackmania-iqn-r2d2-racing-v37-flywheel.yaml $ckpt.FullName
```

Reward is unchanged from v36, so the replay stays valid and `--reset-replay`
is not needed. The standing loop, repeated forever:

1. Train until `eval/summary` improves.
2. Tau-sweep the top checkpoints through the 20-trial gate; publish the best
   gated median.
3. Keep `elite_time_s` about 1.5 s under the current deterministic median
   (38.5 fits a ~40 s median; move toward 37.5 once the median is below 38.5).
4. Tighten `self_imitation_window_s` if promotions become too frequent
   (`demo/self_imitation` events and `replay/elite_sample_fraction` show it).

## Held in reserve, one per experiment

- A second actor (second PC or game instance via `tmrl actor --connect`):
  the distributed runtime already supports it and it roughly doubles the
  data rate; epsilon profiles per actor allow one greedy and one exploratory.
- EMA evaluation weights if adjacent checkpoints disagree at the gate.
- 17-bin steering head (from scratch; only if fast sweepers show visible
  line quantization).
- `action_repeat_frames: 1` with gamma re-matched (latency budget permitting).
