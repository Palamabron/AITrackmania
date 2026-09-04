# Changelog

## 1.1.0 - 2026-09-04

- Require RunSpec API 2.0 and compose discrete value models from a frame-only
  encoder, temporal core, head and value strategy.
- Make generated starter model initialization follow the declared run seed and
  expose the composed discrete-value learner through the public built-in registry.
  Generated projects now partition PyTorch CPU and CUDA indexes by platform so
  a fresh `uv sync` resolves from both an editable checkout and a release wheel.
- Add one `DiscreteValueLearner` for Standard Q, QR-DQN, IQN and FQF, including
  selected-action Double-DQN targets and an isolated FQF fraction optimizer.
- Add portable Mamba selective scan with `auto`, `native` and Pure PyTorch
  backends sharing one checkpoint-compatible parameter layout.
- Add checkpoint schema 2.0, architecture fingerprints and safe
  named-submodule warm-start reports. The unified `DiscreteValueLearner` is the
  sole scalar Q, QR-DQN, IQN and FQF training path.
- Persist AMP scaler state for every Torch learner, reject unsupported sequence
  configurations before training, and make composite priority validation
  atomic with the optimizer update.
- Make uniform and prioritized sequence replay share one raw-context/n-step
  contract, cache valid sequence windows by replay revision and correct TQC and
  upper-CVaR reduction semantics.
- Make the authenticated SQLite WAL the ordered source of truth, bind
  checkpoints to journal identity and a contiguous applied frontier, retain
  conflict receipts after pruning and fsync durable state before deletion.
- Snapshot mutable transition trees on ingest, make on-policy partial resumes
  fail closed, honor final-checkpoint policy and stop evaluation only after
  consecutive successful target batches.
- Make behavior-cloning splits disjoint, seed model construction, aggregate
  weighted validation correctly, quality-gate recordings and support exact BC
  resume bound to an immutable dataset manifest.
- Tensorize behavior-cloning data once, use the shared Torch execution policy,
  remove per-update CUDA transfer synchronization and reduce logging/replay
  bookkeeping overhead.
- Remove speculative full-batch prefetch that regressed learner throughput;
  the final paired RTX 4090 microbenchmark improved IQN updates/s by 1.28x for
  the MLP fixture and 1.41x for the lidar fixture under the recorded noisy
  workload, with the same direction in an interleaved repeat.
- Fail Trackmania startup closed on telemetry, protocol, readiness, map UID and
  geometry mismatches, and make actor startup failures terminate nonzero
  instead of leaving the learner waiting indefinitely.
- Allow School Mode runs in editor validation to skip the normal-play Enter
  confirmation and use the keyboard restart binding while retaining gamepad driving.
- Add bounded asynchronous W&B projection with semantic axes and health events
  while keeping the local JSONL stream authoritative.
- Rename the public offline-learning package from the too-narrow
  `trackmaniarl.trackmania.behavior_cloning` to
  `trackmaniarl.trackmania.imitation_learning`; BC class and CLI names remain
  precise, while DAgger and recovery artifacts now have an accurate namespace.
- Rewrite architecture, SDK, Trackmania, imitation-learning and development
  documentation for 2.0 and regenerate the editable runtime, model-composition,
  imitation-learning and local/remote deployment diagrams. Add explicit
  Trackmania integration and checkpoint/resume diagrams; remove the decorative
  extension-workflow diagram in favor of the concise SDK checklist.
- Reject non-finite RunSpec values, malformed geometry and pace profiles,
  backward reward clocks and physically unreachable progress jumps. Require the
  built-in reward discount to equal `training.gamma`, report terminal and
  time-attack reward components without double counting, and use the documented
  OpenPlanet velocity conversion for pace and projected-velocity diagnostics.
- Sample recurrent replay only from complete, unique, episode-local histories;
  preserve n-step boundaries and elite weighting in both optimized and fallback
  PER paths. Resume now snapshots replay and sampler state before asynchronous
  checkpoint serialization.
- Align generated online control and demonstration aggregation to a 50 ms,
  repeat-one decision grid; support recurrent BC inference, make failed
  closed-loop BC gates exit nonzero by default, bind stitched trajectories to
  control alignment and limit BC-to-RL warm start to encoder/temporal modules.
- Require BC recovery artifacts bound to map UID, geometry, timing, control
  alignment and a verified source demonstration digest, with DAgger checkpoint
  hashes retained for audit. Synthetic counterfactual states are now
  episode-independent instead of pretending to be a dynamically reachable
  sequence.
- Route PPO through the local on-policy trainer, correct Trackmania TQC action
  bounds, validate policy action masks and export evaluated actor state for SAC,
  REDQ-SAC, TQC and stable-discrete-SAC-inspired learners.
- Validate authenticated rollout semantics before WAL append. Actor spools now
  fsync before atomic publication, recover orphaned temporary files, reject a
  single chunk larger than the spool cap and retry only transient gRPC status
  codes; permanent failures retain queued data and stop the actor loudly.
- Apply the 32-character distributed-token minimum in every public runtime
  entry point, keep the game-free starter free of controller dependencies and
  make Zstandard a core checkpoint dependency.
- Gate tagged publishing on formatting, Ruff, strict types and the full test
  suite, then compare every packaged source file byte-for-byte with the release
  checkout before upload. Package metadata and README links now use the
  canonical `Palamabron/TrackmaniaRL` repository URL.
- Package every `trackmaniarl` subpackage, keep base CLI/import paths independent
  of optional gRPC modules, pin the build backend and test the wheel against the
  complete source tree. CI actions are commit-pinned with least privilege and
  CPU-only quality gates avoid downloading the CUDA development stack.
- Correct recurrent replay after eviction, interleaved actor episodes and
  terminal-first checkpoint restore while keeping eviction refresh bounded by
  sequence length. Validate finite scalar priorities before sampler mutation.
- Correct SAC, REDQ-SAC and TQC scalar/quantile shapes, preserve structured
  observation PyTrees and batch single CHW observations during policy inference.
- Bind local exact resume and distributed handshakes to the semantic RunSpec,
  declared and resolved component-package source, and geometry/pace-reference
  contents. Suppress obsolete warm-start loading during state restoration and
  record resolved execution separately for every process attempt. Pytest now
  uses an ignored repository-local base temp directory on Windows.
- Validate documented component constructor kwargs and public configuration
  field coverage, correct the RunSpec examples and defaults, and distinguish the
  off-policy architecture diagrams from the local PPO lifecycle.
- Add a bounded actor-stall watchdog and fail closed when Trackmania stops
  producing environment steps instead of leaving the learner running forever.
- Add opt-in keyboard-informed exploration, body-frame GraphV2 observations and
  directional predecessor/successor graph message passing for lap-time experiments.
- Label completed online episodes with their measured finish pace and support
  optional elite replay weighting without discarding prioritized-replay TD errors.
- Validate episode ownership, outcome and finish-time semantics before durable
  ingest, including the case where an episode summary arrives before its chunks.
- Persist elite replay labels in checkpoint schema v2, restore schema v1 safely,
  and expose elite replay activity through local and W&B metrics.
- Let an unpacked source tree report an unknown development version when package
  metadata is absent; built wheels still require the exact release metadata.
- Package only the reviewed release scripts in source archives so ignored local
  helpers and caches cannot enter an sdist through a broad file pattern.
- Add a measured lap-time audit and an opt-in sub-37 candidate configuration;
  experimental performance claims remain separate from the library release gate.

## 1.0.4 - 2026-08-19

- Use absolute GitHub image URLs in the package README so architecture diagrams
  render on PyPI as well as locally and on GitHub.

## 1.0.3 - 2026-08-19

- Strengthen the release gate with tag/version matching and validation of wheel
  metadata, typing markers, bundled OpenPlanet assets, legal files and sdist
  test sources.
- Use PNG previews for the architecture diagrams embedded in GitHub README,
  while retaining SVG and editable Excalidraw sources.
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
- Replay checkpoints can restore into a larger configured capacity, enabling resume-with-bigger-buffer experiments.
