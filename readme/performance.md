# Performance and GPU tuning

GPU utilization is an outcome, not the optimization target. TrackmaniaRL first
separates collection, ingestion/replay, host-to-device transfer and learner
time, then changes one measured bottleneck without changing the seed, replay,
update budget or evaluation suite.

## Locate the limiting stage

Use the local `events.jsonl` stream or the panels defined in
[Observability and W&B](observability.md). Interpret a metrics window as follows:

| Evidence | Bound | First action |
| --- | --- | --- |
| low `transitions_per_s`, idle learner and no backlog | environment/actor | inspect telemetry cadence, policy inference and actor count |
| `replay_wait_s > learner_update_s` with growing update credit | replay/host preparation | inspect sampler complexity and duplicate pin/copy work |
| high `host_to_device_s` with low forward/backward time | transfer | verify one pinned owner and one non-blocking transfer path |
| forward/backward dominates and GPU is busy | learner compute | test a larger batch or supported mixed precision |
| periodic stalls match policy/checkpoint timing | persistence | inspect snapshot frequency and storage latency |

Do not increase batch size merely to make a utilization chart larger. It changes
optimization dynamics and memory pressure. Treat it as an experiment and keep
the effective update-to-data ratio and evaluation budget fixed.

## Audit measurements

The 2026-08-23 audit used Windows 11, Python 3.12.12, PyTorch 2.11 with CUDA
12.8, an RTX 4090, an i9-13900K and 128 GB RAM. Trackmania and an existing
learner remained active, so CUDA results are paired diagnostic measurements,
not hardware-neutral claims.

The last 50 metric windows of that already-running process, which had loaded
the pre-fix code, showed:

| Metric | Mean | Median | Range |
| --- | ---: | ---: | ---: |
| updates/s | 6.096 | 6.299 | 4.751–6.544 |
| transitions/s | 14.716 | 15.864 | 0–19.361 |
| replay sample/wait | 91.7 ms | 87.3 ms | 83.0–119.0 ms |
| learner update | 65.1 ms | 64.1 ms | 61.7–74.6 ms |
| update backlog | 671.7 s | 650.1 s | 593.6–859.8 s |
| update credit | 4,072 | 4,092 | 3,853–4,095 |

Replay preparation was slower than GPU learning and credit was saturated. That
is direct evidence for an input-bound learner and explains why roughly 50% GPU
utilization could coexist with a large training backlog.

A paired synthetic learner benchmark used batch 512, 78 actions and 32 online ×
32 target IQN quantiles. Removing the coordinator's speculative global prefetch
path, which re-pinned and recopied complete batches, produced:

| Encoder workload | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| identity temporal + MLP | 58.39 updates/s | 74.78 updates/s | 1.28× |
| synthetic lidar, hidden size 192 | 36.13 updates/s | 50.96 updates/s | 1.41× |

An earlier interleaved repeat under a different momentary GPU load measured
53.69→82.22 updates/s and 31.42→42.38 updates/s respectively. Both repeats
agree on the direction; their spread is why the absolute rates are not treated
as stable hardware benchmarks. The final lidar profile reduced self CPU time
from 197.0 ms to 137.6 ms across five updates while CUDA time remained about
21 ms.

This is a learner microbenchmark, not a claim that lap time or end-to-end live
throughput improved by the same ratio. A restarted process and the normal
Trackmania evaluation gate are required to measure that outcome.

For 50,000 replay transitions and a batch of 64 sequences of length 32, the old
full valid-window scan took 544.958 ms. The revision-aware index takes 242.842 ms
to build cold, 0.486 µs to reuse warm and 29.262 ms for a complete warm sample,
while producing the same 49,969 valid windows.

The W&B projection path was measured separately with 50,000 update events and
a local no-network fake run: 7.25 µs enqueue time per event, 7.67 µs including
worker drain, and zero drops. This verifies the application-side projection and
queue overhead; it does not estimate remote service latency. The bounded worker
isolates that latency and reports drops/errors while local JSONL remains the
authority.

## Supported acceleration choices

- `device: auto` selects ROCm, CUDA, MPS or CPU from the installed Torch build
  and fails when accelerator hardware is visible through driver tools but the
  Torch build cannot use it.
- `precision: auto` probes supported precision. Float16 CUDA/ROCm uses a
  checkpointed `GradScaler`; bfloat16 does not require scaling. Resume restores
  scaler state rather than restarting its dynamic range.
- Non-blocking CUDA transfer is used only after pinning and on the learner's
  transfer stream. A second generic prefetch/copy layer is deliberately not
  enabled.
- `execution.compile: true` is supported only by the legacy IQN learner. Other
  learners reject it during resolution instead of silently ignoring it.
- CUDA graphs are not a supported runtime contract. Random IQN/FQF supports,
  PER feedback, dynamic PyTrees and checkpoint/evaluation control flow require
  a dedicated fixed-shape design before graphs can be safe.

The profiler still observes 33 scalar `.item()` calls per value update and one
device-to-host priority copy. Some calls implement bounded metrics or PER
feedback, so removing them requires a measured asynchronous aggregation design,
not a blind rewrite.

## Comparison protocol

For every performance change:

1. use the same seed, replay contents, batch request, model, precision and update
   count;
2. warm up kernels before timing and report wall-clock updates/s, not only CUDA
   kernel time;
3. record replay wait, transfer, forward, backward, optimizer, synchronization
   and checkpoint timing;
4. repeat in a fresh process because loaded Python code does not change when the
   worktree changes;
5. run `trackmaniarl benchmark` on the same checkpoint policy and evaluation
   suite before claiming a Trackmania improvement.

Experimental Mamba, SimBaV2 and adaptive clipping remain one-variable-at-a-time
experiments. A throughput win cannot promote them without deterministic resume
coverage and a bounded live comparison of finish rate, median finish time,
pace and safety.
