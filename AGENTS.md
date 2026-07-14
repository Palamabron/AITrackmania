# TMRL development guide

## Commands

```bash
uv sync --group dev
uv run poe fmt
uv run poe types
uv run poe test
uv run tmrl init my-trackmania-agent
uv run tmrl validate run.yaml
```

These commands deliberately use `uv` without platform-specific virtualenv or shell branches. They must work unchanged on Windows, Linux, WSL and CI.

## Architecture

The public flow is `RunSpec → ResolvedRun → validation → collection/training`. `tmrl.core` contains contracts, data, replay, runtime and built-ins. `tmrl.trackmania` contains the game adapter only. `tmrl.observability` owns manifests, events and artifacts; `tmrl.experiments` owns suites and study strategies; `tmrl.project` owns the generated extension project.

User components are loaded by explicit `module:attribute` paths from an installable local project. Configuration is Pydantic at the CLI/runtime boundary. Transitions and batches are slot dataclasses and PyTrees in hot paths. Do not add global configuration, import-time side effects, feature flags or a mandatory external tracker.

All built-ins and generated user components need deterministic contract tests. Trackmania itself is exercised only in optional smoke tests.
