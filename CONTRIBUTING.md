# Contributing to TrackmaniaRL

TrackmaniaRL is an SDK, not a collection of hard-coded training modes. Read the
[architecture](readme/architecture.md) and
[development guide](readme/development.md) before changing a public contract.

## Local quality gate

Install and run the same commands used by CI:

```bash
uv sync --group dev
uv run poe fmt
uv run poe types
uv run poe test
```

Do not use another package manager or formatter. Keep `uv.lock` synchronized
when dependencies change and describe any optional dependency or platform
impact in the pull request.

## Where changes belong

- stable interfaces, data and runtime mechanics: `trackmaniarl.core`;
- reusable bundled implementations: `trackmaniarl.algorithms`,
  `trackmaniarl.models` and `trackmaniarl.builtins`;
- Trackmania/OpenPlanet behavior only: `trackmaniarl.trackmania`;
- actor/learner transport and durability: `trackmaniarl.distributed`;
- logging/artifacts: `trackmaniarl.observability`;
- experiment orchestration: `trackmaniarl.experiments`;
- application-specific behavior: a generated extension project.

Keep the public CLI portable: `trackmaniarl init` and `trackmaniarl validate`
must behave the same on Windows and Linux. Avoid local file access, environment
reads and optional imports during `import trackmaniarl`. Validate configuration
at the boundary; keep Pydantic models out of rollout and sampling hot paths.

Every bundled component needs deterministic contract coverage. A distributed
change also needs fake actor/slow learner coverage; a Trackmania change needs an
offline test and should pass the bounded Windows smoke test before release.

Security reports follow [SECURITY.md](SECURITY.md), not the public issue tracker.
