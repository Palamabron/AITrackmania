# Changelog

## 1.0.3 - 2026-08-19

- Add the opt-in `TemporalMambaTrackGeometryEncoder` and
  `LidarMambaModelFactory` for causal lidar sequence modeling on Linux CUDA
  learners, with focused contract coverage and explicit dependency errors.
- Document the Mamba training contract, supported deployment split and RunSpec
  wiring, with a new editable model data-flow diagram.
- Enable the pinned vgamepad fork and its `libevdev` dependency on Linux, and
  select the tested CUDA Torch index for Windows and Linux Trackmania hosts.

- Bound distributed wire messages by their decompressed size and added a
  regression test for highly compressible oversized payloads.
- Require distributed bearer tokens to contain at least 32 characters.
- Require `setuptools>=83` for TrackmaniaRL and generated project builds after
  the packaging audit identified CVE-2026-59890.
- Add generated `.gitignore` and `.env-example` files, an architecture guide, a
  development/extension workflow and a dated security audit.
- Add editable Excalidraw diagrams and SVG/HTML previews for the runtime,
  extension workflow and distributed security boundaries.
- Split behavior cloning into model, learner and data package entry points, and
  move shared lidar encoding out of the IQN-specific module.
- Validate declared model/learner contracts during RunSpec resolution.
- Expose gamepad or keyboard control in the generated Trackmania configuration;
  keyboard control digitizes analog model actions with a steering dead zone.

## 1.0.2 - 2026-08-18

- Made the published PyPI installation path the primary README workflow.
- Added release, Python, CI, license and development-status badges.
- Documented package extras, platform requirements, the Trackmania template,
  distributed runtime and the temporary vgamepad source pin.
- Added documentation, issue tracker and security links to the package metadata.
- Moved the Windows vgamepad source pin to the patched `Palamabron/vgamepad`
  revision while upstream PR #47 remains unmerged.

## 1.0.1 - 2026-08-18

- Fixed Linux CI type checking for Windows input APIs and made checkpoint-path tests portable.
- Pin the Windows `vgamepad` integration in uv environments to commit `90f95e3` from upstream PR #47.
- Skip ViGEmBus driver installation on GitHub Actions runners; normal Windows installs retain it.

## 1.0.0 - 2026-08-18

- Renamed the distribution, Python package and CLI to TrackmaniaRL / `trackmaniarl`.
- Added attribution, trademark disclaimer and a security reporting policy.
- Made the generated Trackmania project valid TOML, W&B-free by default, and benchmark-ready.
- Validated actions through each learner policy and made discrete SAC emit Python action indices.
- Removed automatic unsafe checkpoint unpickling and added wheel-level CI verification.

- Recurrent IQN training now updates every post-burn-in timestep in a sequence (R2D2-style) instead of only the final step, and sequence priorities use a mixed max/mean TD error.
- Added optional R2D2 value rescaling and a DQfD-style demonstration margin loss to `ImplicitQuantileQLearning`; demonstration transitions are protected from FIFO eviction.
- Progress rewards bound per-step index advance to a physically reachable arc length, preventing hairpin cuts through folded reference lines.
- Lidar features keep the last valid horizontal heading through vertical moments instead of aborting the actor.
- Distributed run safety: journal pruning after checkpoints, refusal to silently re-ingest stale journals on fresh starts, bounded coordinator rollout queue with backpressure, actor threads that stop the process on unexpected failure, telemetry stalls that truncate episodes instead of killing the run, spool-cap pause instead of crash, thread-safe JSONL logging, safer checkpoint loading (`weights_only`), and resume-friendly manifests.
- `trackmaniarl benchmark` is config-driven via `evaluation.target_median_s` / `min_finish_rate` instead of a hardcoded `trackmaniarl-test` release gate.
- Packaging: `setuptools>=77` for SPDX licenses, OS classifiers, stricter mypy import overrides, Windows CI, and broader `.gitignore` coverage for sqlite/event leftovers.

- `trackmaniarl track record-demo` now records a whole session: `--count` laps in one go, discards outliers slower than the best finish by more than `--max-gap` seconds (default 1s), saves the rest into the output directory at the end, and mid-lap restarts discard only the partial lap instead of failing the recording.
- Lidar telemetry now scales velocity and speed by the configured `velocity_to_mps_scale / max_speed_mps` instead of a hardcoded 1/1000, so those observation channels carry usable signal; retrain checkpoints that relied on the previous scaling.
- Prioritized sequence sampling builds full n-step returns only for the timestep the learner bootstraps from, cutting redundant replay work for recurrent batches.

- Coordinator ingests the entire rollout backlog every learner iteration, removing the standing queue that trained on minutes-old transitions and inflated the reported policy lag.
- The distributed actor freezes one policy snapshot per training episode, so episode metrics measure a single policy version instead of a refresh mixture.
- IQN policies report the greedy action gap; episode and evaluation summaries log `q_margin/mean`, `q_margin/min` and `q_margin/start_mean`.
- Evaluation batches aggregate into `eval/summary`, and strictly better batches write an immediate best-eval checkpoint (`eval/best_checkpoint`).
- Replay checkpoints can restore into a larger configured capacity, enabling resume-with-bigger-buffer experiments; see `docs/v27-deterministic-stability.md`.
