# Security audit — 2026-08-19

## Scope and method

This review covered the published Python package, CLI boundaries, YAML and
component loading, checkpoint/dataset deserialization, generated projects,
local files and secrets, subprocess usage, OpenPlanet sockets, distributed gRPC
authentication/transport, compression limits, packaging configuration and the
locked direct/transitive dependency set.

The review combined manual source inspection, targeted pattern searches,
existing security and contract tests, a tracked-file secret signature scan and
`pip-audit` against requirements exported from `uv.lock`. Secret values in the
local `.env` were intentionally not read.

## Findings and remediation

### TM-S01 — decompression limit did not cap decoded gRPC data

Severity: medium. Status: fixed.

The wire codec rejected compressed payloads larger than `max_message_bytes` but
allowed decompressed output up to 32 times that limit. With the default setting,
one authenticated request could force allocation of up to 512 MiB before its
shape or contents were validated.

Encoding now rejects raw messages above the configured limit and decoding uses
the same limit as `max_output_size`. Invalid or oversized Zstandard frames are
reported as a value error. A regression test covers a highly compressible
oversized payload.

### TM-S02 — package builders allowed a vulnerable setuptools range

Severity: medium for release engineering, low for normal runtime. Status: fixed
for TrackmaniaRL builds and generated projects.

The dependency audit reported `CVE-2026-59890` / `GHSA-h35f-9h28-mq5c` for
setuptools before 83.0.0. On normalization-preserving macOS filesystems, an sdist
exclusion written in one Unicode normalization form can fail to exclude a file
stored in another form, potentially publishing an intended private file.

Both TrackmaniaRL and projects created by `trackmaniarl init` now require
`setuptools>=83` in build isolation. The locked runtime graph may still contain
an older setuptools selected as a PyTorch runtime dependency; the reviewed
advisory concerns building an sdist, and TrackmaniaRL's own build environment no
longer uses that version. This transitive pin should be rechecked when updating
PyTorch.

### TM-S03 — distributed tokens had no minimum strength

Severity: low because the server is loopback-only. Status: fixed.

The CLI accepted any non-empty bearer token. Distributed commands now require
at least 32 characters and report a portable `secrets.token_urlsafe(32)`
generation command. This does not replace transport encryption.

### TM-S04 — trusted-code nature of run configuration was under-documented

Severity: informational. Status: documented.

YAML parsing itself is safe, but component resolution imports arbitrary
installed `module:attribute` paths and invokes constructors. README, architecture
and the security policy now state that `run.yaml`, extension packages and custom
checkpoint codecs are trusted code boundaries.

## Controls confirmed

- PyYAML uses `safe_load` and Pydantic rejects unknown RunSpec fields.
- NumPy datasets and geometry load with `allow_pickle=False`.
- The default Torch checkpoint loader uses `weights_only=True` with an explicit
  safe-global list and atomic writes.
- Policy transfer uses safetensors-compatible tensor encoding rather than
  executable pickle payloads.
- gRPC authenticates with constant-time token comparison, applies send/receive
  limits and refuses non-loopback learner binds.
- Run fingerprints bind component source, geometry and feature/action contracts.
- Rollout WAL entries use parameterized SQLite queries and idempotent sequence
  keys.
- Subprocess calls use argument arrays, fixed commands and timeouts; no
  `shell=True`, `eval` or `exec` path was found in the library.
- `.env`, local databases, artifacts, recordings and common credential files are
  excluded from version control; no high-confidence secret signature was found
  in tracked source files.
- Manifests and tracker configuration recursively redact conventional secret
  key names.

## Residual risks and operating requirements

- Bearer-token gRPC is intentionally plaintext on loopback. A remote deployment
  must use SSH, WireGuard or an equivalent authenticated encrypted tunnel.
- A malicious installed Python component can execute with the user's privileges.
  The library does not sandbox extension code.
- Restricted checkpoint loading prevents common pickle code execution but does
  not make huge or semantically malicious local files harmless. Accept
  checkpoints, demonstrations and geometry only from trusted sources and use
  filesystem quotas for shared environments.
- OpenPlanet and virtual-gamepad integration can control the game. Live smoke,
  training and evaluation should run on a dedicated user session.
- The audit tool could not match the CUDA-local PyTorch version identifier to a
  PyPI advisory record. PyTorch and the pinned vgamepad Git revision require
  separate review during dependency upgrades.
- This was a repository review, not a penetration test of a live remote tunnel,
  Trackmania process, W&B account or host operating system.

## Recommended recurring checks

For each release: scan exported locked dependencies, inspect wheel and sdist
contents, run the secret scan, execute the deterministic suite on Windows and
Linux, perform the bounded live Trackmania smoke test and verify a checkpoint
resume. Revisit this document whenever serialization, network exposure,
credential handling or package publishing changes.
