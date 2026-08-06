# Changelog

## Unreleased

- Recurrent IQN training now updates every post-burn-in timestep in a sequence (R2D2-style) instead of only the final step, and sequence priorities use a mixed max/mean TD error.
- Added optional R2D2 value rescaling and a DQfD-style demonstration margin loss to `ImplicitQuantileQLearning`; demonstration transitions are protected from FIFO eviction.
- Progress rewards bound per-step index advance to a physically reachable arc length, preventing hairpin cuts through folded reference lines.
- Lidar features keep the last valid horizontal heading through vertical moments instead of aborting the actor.
- Distributed run safety: journal pruning after checkpoints, refusal to silently re-ingest stale journals on fresh starts, bounded coordinator rollout queue with backpressure, actor threads that stop the process on unexpected failure, telemetry stalls that truncate episodes instead of killing the run, spool-cap pause instead of crash, thread-safe JSONL logging, safer checkpoint loading (`weights_only`), and resume-friendly manifests.
- `tmrl benchmark` is config-driven via `evaluation.target_median_s` / `min_finish_rate` instead of a hardcoded `tmrl-test` release gate.
- Packaging: `setuptools>=77` for SPDX licenses, OS classifiers, stricter mypy import overrides, Windows CI, and broader `.gitignore` coverage for sqlite/event leftovers.

- `tmrl track record-demo` now records a whole session: `--count` laps in one go, discards outliers slower than the best finish by more than `--max-gap` seconds (default 1s), saves the rest into the output directory at the end, and mid-lap restarts discard only the partial lap instead of failing the recording.
- Lidar telemetry now scales velocity and speed by the configured `velocity_to_mps_scale / max_speed_mps` instead of a hardcoded 1/1000, so those observation channels carry usable signal; retrain checkpoints that relied on the previous scaling.
- Prioritized sequence sampling builds full n-step returns only for the timestep the learner bootstraps from, cutting redundant replay work for recurrent batches.

- Coordinator ingests the entire rollout backlog every learner iteration, removing the standing queue that trained on minutes-old transitions and inflated the reported policy lag.
- The distributed actor freezes one policy snapshot per training episode, so episode metrics measure a single policy version instead of a refresh mixture.
- IQN policies report the greedy action gap; episode and evaluation summaries log `q_margin/mean`, `q_margin/min` and `q_margin/start_mean`.
- Evaluation batches aggregate into `eval/summary`, and strictly better batches write an immediate best-eval checkpoint (`eval/best_checkpoint`).
- Replay checkpoints can restore into a larger configured capacity, enabling resume-with-bigger-buffer experiments; see `docs/v27-deterministic-stability.md`.

## 1.0.0

- Introduced the explicit `RunSpec → ResolvedRun` SDK runtime.
- Added independently replaceable learner, policy, replay, sampling, feature, evaluation, tracker and checkpoint contracts.
- Added portable `tmrl init` and `tmrl validate` commands.
- Made local manifests, JSONL events and compressed episode artifacts the default observability path; external integrations are optional extras.
- Removed compatibility guarantees for previous configuration, runtime and checkpoint formats.
