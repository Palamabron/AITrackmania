# Security policy

Do not publish suspected vulnerabilities, credentials, checkpoint files, or
telemetry captures in public issues. Report them privately to the repository
maintainer with the affected version, a minimal reproduction, and the expected
impact.

Only load checkpoints produced by TrackmaniaRL or supplied by a trusted source.
The default checkpoint loader uses PyTorch's `weights_only=True` mode and
rejects checkpoints that require executable pickle payloads.
