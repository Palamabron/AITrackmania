# Changelog

## 1.0.0

- Introduced the explicit `RunSpec → ResolvedRun` SDK runtime.
- Added independently replaceable learner, policy, replay, sampling, feature, evaluation, tracker and checkpoint contracts.
- Added portable `tmrl init` and `tmrl validate` commands.
- Made local manifests, JSONL events and compressed episode artifacts the default observability path; external integrations are optional extras.
- Removed compatibility guarantees for previous configuration, runtime and checkpoint formats.
