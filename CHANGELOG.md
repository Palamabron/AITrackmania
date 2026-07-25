# Changelog

## Unreleased

- Added human demonstration seeding: `tmrl track record-demos` captures finished reference laps from live telemetry inputs into `.tmdemo` files, and `--demos` on `train`/`resume`/`learner` loads them into replay as `is_demo` transitions without granting update credit; see `docs/v28-speed-program.md`.
- IQN exploration now mixes one-steering-bin neighbor nudges of the greedy action with the throttle-biased global sample, so low-epsilon exploration refines the racing line instead of injecting random brake or no-throttle at speed.
- TrackMania episodes report applied-control usage and the true decision period (`control/gas_fraction`, `control/brake_fraction`, `control/steer_abs_mean`, `timing/step_race_ms_mean`).

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
